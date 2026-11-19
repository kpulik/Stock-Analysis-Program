"""Convert the Aenlle CSV archive into conservative, ticker-linked research events."""
import argparse
import csv
from datetime import date, datetime, time, timedelta
import hashlib
import io
import json
import zipfile
from zoneinfo import ZoneInfo

import pandas as pd

from forecast import universe
from main import write_csv
from news import headline_score

ZONE = ZoneInfo("America/New_York")


def following_midnight(day):
    return datetime.combine(day + timedelta(days=1), time.min, ZONE).isoformat()


def import_rows(rows, symbols, start, end):
    records = {}
    eligible = 0
    for row in rows:
        symbol = row["stock"].strip()
        if symbol not in symbols:
            continue
        day = date.fromisoformat(row["date"].strip()[:10])
        if not start <= day < end:
            continue
        title = row["headline"].strip()
        if not title:
            continue
        eligible += 1
        identity = row["url"].strip() or f"{day.isoformat()} {title.casefold()}"
        key = hashlib.sha256(identity.encode()).hexdigest()
        pair = (symbol, key)
        # Keep one conservative snapshot per URL and symbol.
        record = (day, title)
        if pair not in records or record > records[pair]:
            records[pair] = record
    events = [(f"aenlle-{key}", symbol, "news", following_midnight(day), headline_score(title))
              for (symbol, key), (day, title) in sorted(records.items())]
    return pd.DataFrame(events, columns=["event_id", "symbol", "kind", "available_at", "value"]), eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--universe", default="examples/universe.csv")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="Exclusive source date")
    parser.add_argument("--out", default="data/archive-news.csv")
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("end must be later than start")
    with zipfile.ZipFile(args.archive) as archive, archive.open("raw_analyst_ratings.csv") as raw:
        frame, eligible = import_rows(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")),
                                      universe(args.universe), args.start, args.end)
    if frame.empty:
        parser.error("No matching headlines found")
    write_csv(frame, args.out)
    print(json.dumps({"eligible_source_rows": eligible, "unique_article_symbol_pairs": len(frame),
                      "directional": int(frame.value.ne(0).sum()), "symbols": int(frame.symbol.nunique())}))


if __name__ == "__main__":
    main()
