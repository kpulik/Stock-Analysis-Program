"""Daily market forecasts, historical replay and prospective scoring."""
import argparse
from contextlib import closing
from datetime import timedelta
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

import numpy as np
import pandas as pd

from forecast import (VARIANTS, ZONE, cutoff, examples, fit, read_bars,
                      read_events, replay, session_predictions, universe)


def write_csv(frame, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False, mode="w") as file:
        try:
            frame.to_csv(file, index=False)
            file.close()
            os.replace(file.name, destination)
        finally:
            Path(file.name).unlink(missing_ok=True)


def completed_through(now):
    """Allow today's daily bar one hour after the exchange's scheduled close."""
    import exchange_calendars as xcals

    local = now.tz_convert(ZONE)
    day = local.date()
    calendar = xcals.get_calendar("XNYS")
    if calendar.is_session(str(day)):
        if local >= calendar.session_close(str(day)) + pd.Timedelta(hours=1):
            return day
    return day - timedelta(days=1)


def download(args, pairs):
    import yfinance as yf

    # The provider's end date is exclusive.
    latest_end = completed_through(pd.Timestamp.now(tz=ZONE)) + timedelta(days=1)
    end = pd.Timestamp(args.end).date() if args.end else latest_end
    if end > latest_end or pd.Timestamp(args.start).date() >= end:
        raise ValueError("Invalid date range or incomplete daily bars requested")
    symbols = sorted(set(pairs) | set(pairs.values()))
    quotes = yf.download(symbols, start=args.start, end=end.isoformat(), interval="1d",
                         auto_adjust=False, actions=False, prepost=False, progress=False,
                         group_by="ticker", multi_level_index=True, threads=False)
    frames = []
    for symbol in symbols:
        if symbol not in quotes.columns.get_level_values(0):
            raise ValueError(f"No response for {symbol}")
        bars = quotes[symbol][["Open", "Close"]].rename(columns=str.lower)
        bars = bars.dropna(how="all")
        if bars.empty or bars.isna().any().any() or (bars <= 0).any().any():
            raise ValueError(f"Missing or incomplete daily bars for {symbol}")
        bars["date"] = bars.index.strftime("%Y-%m-%d")
        bars["symbol"] = symbol
        frames.append(bars[["date", "symbol", "open", "close"]])
    result = pd.concat(frames).sort_values(["date", "symbol"])
    write_csv(result, args.out)
    print(f"Wrote {len(result)} daily bars to {args.out}")


def ledger(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE IF NOT EXISTS forecasts (
        day TEXT, symbol TEXT, model TEXT, benchmark TEXT NOT NULL,
        version TEXT NOT NULL, issued_at TEXT NOT NULL, probability REAL NOT NULL,
        features TEXT NOT NULL, outcome INTEGER, scored_at TEXT,
        PRIMARY KEY(day,symbol,model))""")
    return db


def live(args, pairs, bars, now, clock=None):
    import exchange_calendars as xcals

    day = now.tz_convert(ZONE).tz_localize(None).normalize()
    cal = xcals.get_calendar("XNYS")
    if not cal.is_session(day) or now > cutoff(day):
        raise ValueError("Live issuance requires a trading day at or before 09:25 New York time")
    previous = pd.Timestamp(cal.previous_session(day)).date()
    available = bars[bars.date.dt.date == previous]
    missing = (set(pairs) | set(pairs.values())) - set(available.symbol)
    if missing:
        raise ValueError(f"Missing previous session prices for {', '.join(sorted(missing))}")
    events = read_events(args.events, pairs)
    frame = examples(bars, pairs, events, live_time=now)
    today = frame[frame.date == day]
    if len(today) != len(pairs):
        raise ValueError("Insufficient recent bars for the requested universe")
    labeled = frame.dropna(subset=["target"])
    history = labeled[labeled.date < day.to_period("W-SUN").start_time]
    variants = selected(events, getattr(args, "tree", False))
    if getattr(args, "news_healthy", False) and "news" not in variants:
        variants.append("news")
    weekly = {name: fit(history, name) for name in variants}
    if any(model is None for model in weekly.values()):
        raise ValueError("Insufficient training history before this week")
    with closing(ledger(args.db)) as db, db:
        outputs = list(session_predictions(labeled, day, today, weekly, daily=True))
        operational = getattr(args, "operational", False)
        if operational:
            control = today.assign(model="constant_50", version="constant-0.5", probability=0.5)
            outputs.append(("prices", control))
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT 1 FROM forecasts WHERE day=? LIMIT 1",
                              (day.date().isoformat(),)).fetchone()
        issued = pd.Timestamp.now(tz=ZONE) if clock is None else clock()
        for name, output in outputs:
            if operational and existing:
                break
            for index, row in output.iterrows():
                values = today.loc[index, VARIANTS[name]].astype(float).to_json(double_precision=15)
                db.execute("""INSERT INTO forecasts(day,symbol,model,benchmark,version,issued_at,
                            probability,features) VALUES(?,?,?,?,?,?,?,?)
                            ON CONFLICT(day,symbol,model) DO NOTHING""",
                           (day.date().isoformat(), row.symbol, row.model, row.benchmark,
                            row.version, issued.isoformat(), row.probability, values))
        # Reject any run crossing the cutoff before committing new predictions.
        finished = pd.Timestamp.now(tz=ZONE) if clock is None else clock()
        if finished > cutoff(day):
            raise ValueError("Issuance crossed the cutoff. No new predictions were committed.")
        saved = pd.read_sql_query("SELECT symbol,model,benchmark,probability,issued_at FROM forecasts "
                                  "WHERE day=? ORDER BY model,symbol", db, params=[str(day.date())])
    print(paper_calls(saved).to_string(index=False))


def score(args, bars, now):
    past = bars[bars.date.dt.date <= completed_through(now)]
    returns = {(str(row.date.date()), row.symbol): row.close / row.open - 1
               for row in past.itertuples()}
    with closing(ledger(args.db)) as db, db:
        for day, symbol, benchmark, model in db.execute(
                "SELECT day,symbol,benchmark,model FROM forecasts WHERE outcome IS NULL").fetchall():
            stock = returns.get((day, symbol))
            reference = returns.get((day, benchmark))
            if stock is not None and reference is not None:
                db.execute("UPDATE forecasts SET outcome=?,scored_at=? WHERE day=? AND "
                           "symbol=? AND model=? AND outcome IS NULL",
                           (int(stock > reference), now.isoformat(), day, symbol, model))
        resolved = pd.read_sql_query("SELECT model,probability,outcome AS target FROM forecasts "
                                     "WHERE outcome IS NOT NULL", db)
        pending = db.execute("SELECT count(*) FROM forecasts WHERE outcome IS NULL").fetchone()[0]
    print(f"Pending forecasts: {pending}")
    if not resolved.empty:
        summary(resolved)


def paper_calls(rows):
    probability = rows.probability
    calls = np.select([probability >= 0.55, probability <= 0.45],
                      ["outperform", "lag_or_tie"], default="abstain")
    return rows.assign(paper_call=calls)


def summary(rows):
    rows = rows.assign(brier=(rows.probability - rows.target) ** 2)
    print(rows.groupby("model").brier.agg(["count", "mean"]).rename(
        columns={"mean": "brier"}).to_string())


def selected(events, tree=False):
    kinds = {kind for _, kind in events}
    controls = ["prices", "price_tree"] if tree else ["prices"]
    return controls + [variant for variant, kind in
                         (("news", "news"), ("llm", "llm"), ("politics", "congress"), ("weather", "weather"))
                         if kind in kinds]


def demo(args):
    rng = np.random.default_rng(17)
    days = pd.bdate_range("2023-01-02", periods=120)
    pairs = {"DEMO1": "BENCH", "DEMO2": "BENCH"}
    bars = [(day, symbol, 100.0, 100 + rng.normal())
            for day in days for symbol in [*pairs, "BENCH"]]
    frame = pd.DataFrame(bars, columns=["date", "symbol", "open", "close"])
    result = replay(examples(frame, pairs, {}), ["prices"], days[60])
    print("Synthetic demonstration. These scores do not measure market performance.")
    summary(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo")
    news = commands.add_parser("news", help="Collect company headlines and export timestamped events")
    news.add_argument("--source", choices=["yahoo", "alpaca"], default="yahoo")
    news.add_argument("--universe", default="examples/universe.csv")
    news.add_argument("--start")
    news.add_argument("--end")
    news.add_argument("--db")
    news.add_argument("--out")
    for name in ["download", "replay", "predict", "score"]:
        command = commands.add_parser(name)
        if name != "score":
            command.add_argument("--universe", default="examples/universe.csv")
        if name != "download":
            command.add_argument("--bars", default="data/bars.csv")
        if name in {"replay", "predict"}:
            command.add_argument("--events")
            command.add_argument("--tree", action="store_true", help="Compare a shallow decision tree")
        if name in {"predict", "score"}:
            command.add_argument("--db", default="data/forecasts.sqlite")
        if name in {"download", "replay"}:
            command.add_argument("--start", required=True)
        if name == "download":
            command.add_argument("--end")
            command.add_argument("--out", default="data/bars.csv")
        if name == "replay":
            command.add_argument("--calls", action="store_true", help="Report selective paper calls")
            command.add_argument("--daily", action="store_true", help="Include daily refits beside weekly controls")
            command.add_argument("--out", default="data/replay.csv")
    args = parser.parse_args()
    if args.command == "demo":
        return demo(args)
    now = pd.Timestamp.now(tz=ZONE)
    if args.command == "score":
        return score(args, read_bars(args.bars), now)
    pairs = universe(args.universe)
    if args.command == "news":
        from news import collect
        return collect(args, pairs, write_csv)
    if args.command == "download":
        return download(args, pairs)
    bars = read_bars(args.bars)
    if args.command == "predict":
        return live(args, pairs, bars, now)
    events = read_events(args.events, pairs)
    frame = examples(bars, pairs, events)
    result = replay(frame, selected(events, args.tree), pd.Timestamp(args.start), daily=args.daily)
    evaluated = result[result.model == "prices"][["date", "symbol"]]
    for signal in ("news", "llm"):
        if signal in selected(events):
            column = signal + "_count"
            coverage = evaluated.merge(frame[["date", "symbol", column]],
                                       on=["date", "symbol"], validate="one_to_one")
            matches = int(coverage[column].gt(0).sum())
            print(f"{signal} available on {matches} of {len(coverage)} evaluated instrument-days.")
            if not matches:
                print(f"This replay cannot measure the value of {signal} on these dates.")
    if args.calls or args.tree or "llm" in selected(events):
        result = paper_calls(result)
        called = result.paper_call != "abstain"
        hit = (result.paper_call == "outperform") == result.target
        print(result.assign(called=called, hit=hit.where(called)).groupby("model")
              .agg(calls=("called", "sum"), coverage=("called", "mean"), hit_rate=("hit", "mean")))
    write_csv(result, args.out)
    summary(result)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        sys.exit(f"Error: {exc}")
