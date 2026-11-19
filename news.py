"""Collect symbol-tagged headlines and export point-in-time research signals."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd


# An interpretable first baseline. Headline direction is a hypothesis to evaluate.
POSITIVE = re.compile(r"\b(?:beats? (?:estimates|expectations)|raises? (?:its )?(?:guidance|outlook)|upgrades?|upgraded|record profit)\b", re.I)
NEGATIVE = re.compile(r"\b(?:miss(?:es)? (?:estimates|expectations)|(?:cuts?|lowers?) (?:its )?(?:guidance|outlook)|downgrades?|downgraded|files for bankruptcy)\b", re.I)


def headline_score(text):
    return int(bool(POSITIVE.search(text))) - int(bool(NEGATIVE.search(text)))


def utc(value):
    if value.tzinfo is None:
        raise ValueError("News timestamps must include a timezone")
    return value.astimezone(timezone.utc)


def iso(value):
    return utc(value).isoformat(timespec="microseconds")


def request_bytes(url, headers=None):
    request = Request(url, headers={"User-Agent": "StockForecastResearch/1.0", **(headers or {})})
    with urlopen(request, timeout=20) as response:
        return response.read()


def rss_rows(raw, symbol, seen):
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed RSS for {symbol}") from exc
    channel = root.find("channel")
    if channel is None:
        raise ValueError(f"Invalid RSS channel for {symbol}")
    for item in channel.findall("item"):
        headline = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        key = (item.findtext("guid") or link).strip()
        if not headline or not key or not item.findtext("pubDate"):
            continue
        published = utc(parsedate_to_datetime(item.findtext("pubDate")))
        if not seen - timedelta(days=1) < published <= seen:
            continue
        yield ("yahoo", key, symbol, headline, link, iso(published), iso(seen))


def yahoo(pairs):
    rows = []
    for symbol in pairs:
        query = urlencode({"s": symbol, "region": "US", "lang": "en-US"})
        raw = request_bytes(f"https://feeds.finance.yahoo.com/rss/2.0/headline?{query}")
        seen = datetime.now(timezone.utc)
        rows.extend(rss_rows(raw, symbol, seen))
    return rows


def historical(pairs, start, end):
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not (key and secret):
        raise ValueError("Alpaca history needs APCA_API_KEY_ID and APCA_API_SECRET_KEY")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    # Interpret YYYY-MM-DD as UTC midnight. The requested end is exclusive.
    first = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    last = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if last <= first:
        raise ValueError("Expected start < exclusive end")
    query = {"start": first.isoformat(), "end": last.isoformat(), "symbols": ",".join(pairs),
             "limit": 50, "sort": "asc", "include_content": "false"}
    seen_tokens = set()
    while True:
        url = "https://data.alpaca.markets/v1beta1/news?" + urlencode(query)
        page = json.loads(request_bytes(url, headers))
        for article in page["news"]:
            headline = article["headline"].strip()
            published = utc(datetime.fromisoformat(article["created_at"]))
            revised = utc(datetime.fromisoformat(article["updated_at"]))
            available = max(published, revised)
            if not headline or not first <= available < last:
                continue
            key = str(article["id"])
            for symbol in sorted(set(article["symbols"]) & pairs.keys()):
                yield ("alpaca", key, symbol, headline, article.get("url") or "",
                       iso(published), iso(available))
        token = page.get("next_page_token")
        if not token:
            return
        if token in seen_tokens:
            raise ValueError("Provider repeated a news pagination token")
        seen_tokens.add(token)
        query["page_token"] = token


def collect(args, pairs, write_csv):
    if args.source == "alpaca" and not (args.start and args.end):
        raise ValueError("Supply --start and --end for the historical archive")
    if args.source == "yahoo" and (args.start or args.end):
        raise ValueError("Yahoo RSS collects currently visible stories. Use Alpaca for a date range.")
    path = Path(args.db or f"data/{args.source}-news.sqlite")
    out = Path(args.out or f"data/{args.source}-news.csv")
    if path.resolve() == out.resolve():
        raise ValueError("Archive and CSV paths must differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (historical(pairs, args.start, args.end) if args.source == "alpaca" else yahoo(pairs))
    with closing(sqlite3.connect(path, timeout=30)) as db, db:
        db.execute("""CREATE TABLE IF NOT EXISTS articles (
            source TEXT, article_id TEXT, symbol TEXT, headline TEXT NOT NULL, link TEXT,
            published_at TEXT NOT NULL, available_at TEXT NOT NULL,
            PRIMARY KEY(source, article_id, symbol))""")
        db.execute("BEGIN IMMEDIATE")
        # Keep the first collected version. Repeated polls cannot backdate or relabel it.
        db.executemany("INSERT INTO articles VALUES (?,?,?,?,?,?,?) "
                       "ON CONFLICT(source, article_id, symbol) DO NOTHING", rows)
        changed = db.total_changes
        db.commit()
        # Lock through the atomic export so concurrent collectors cannot publish an older snapshot.
        db.execute("BEGIN IMMEDIATE")
        placeholders = ",".join("?" for _ in pairs)
        records = db.execute(f"SELECT article_id,symbol,headline,available_at FROM articles "
                             f"WHERE source=? AND symbol IN ({placeholders}) "
                             "ORDER BY available_at,article_id,symbol", (args.source, *pairs))
        exposures = []
        for key, symbol, headline, available in records:
            event_id = args.source + "-" + hashlib.sha256(key.encode()).hexdigest()
            exposures.append((event_id, symbol, "news", available, headline_score(headline)))
        frame = pd.DataFrame(exposures, columns=["event_id", "symbol", "kind", "available_at", "value"])
        write_csv(frame, out)
    print(f"Archived {changed} new article-symbol pairs. Exported {len(frame)} news signals "
          f"({int(frame.value.ne(0).sum())} directional) to {out}")
