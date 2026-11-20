"""Prospective feed capture, issuance and settlement on one private SQLite ledger."""
import argparse
from contextlib import closing
import hashlib
from pathlib import Path
import tempfile

import exchange_calendars as xcals
import pandas as pd

from forecast import ZONE, cutoff, read_bars, universe
from main import download, ledger, live, score, write_csv
from news import headline_score, yahoo


def initialize(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS news_checks(
            symbol TEXT, observed_at TEXT, status TEXT, eligible INTEGER, error TEXT,
            PRIMARY KEY(symbol, observed_at));
        CREATE TABLE IF NOT EXISTS news_signals(
            event_id TEXT, symbol TEXT, available_at TEXT, value INTEGER,
            PRIMARY KEY(event_id, symbol));
        CREATE TABLE IF NOT EXISTS missed_sessions(day TEXT PRIMARY KEY, observed_at TEXT);
    """)


def capture(db, pairs, fetch=yahoo):
    complete = True
    for symbol in pairs:
        try:
            rows = fetch([symbol])
            events = [("yahoo-" + hashlib.sha256(key.encode()).hexdigest(), ticker, available,
                       headline_score(title)) for _, key, ticker, title, _, _, available in rows]
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            complete = False
            status, count, error = "error", None, type(exc).__name__
            events = []
        else:
            status, count, error = "ok", len(events), None
        observed = pd.Timestamp.now(tz="UTC").isoformat()
        with db:
            db.executemany("INSERT OR IGNORE INTO news_signals VALUES(?,?,?,?)", events)
            db.execute("INSERT INTO news_checks VALUES(?,?,?,?,?)", (symbol, observed, status, count, error))
        print(symbol, status, count if count is not None else error)
    return complete


def audit(db, now):
    local = now.tz_convert(ZONE)
    day = local.tz_localize(None).normalize()
    issued, settled = db.execute("SELECT count(*),count(outcome) FROM forecasts WHERE day=?",
                                 (str(day.date()),)).fetchone()
    session = xcals.get_calendar("XNYS").is_session(day)
    if session and local > cutoff(day) and not issued:
        with db:
            db.execute("INSERT OR IGNORE INTO missed_sessions VALUES(?,?)",
                       (str(day.date()), now.isoformat()))
    print(f"{day.date()} session={session} forecasts={issued} settled={settled}")
    return session, day


def run(args):
    pairs = universe(args.universe)
    with closing(ledger(args.db)) as db, tempfile.TemporaryDirectory() as scratch:
        initialize(db)
        now = pd.Timestamp.now(tz=ZONE)
        session, day = audit(db, now)
        if args.action == "audit":
            return
        if args.action == "capture":
            return capture(db, pairs)
        if args.action == "issue" and (not session or now > cutoff(day)):
            raise ValueError("Outside premarket issuance window")
        path = Path(scratch) / "bars.csv"
        download(argparse.Namespace(start=args.start, end=None, out=path), pairs)
        bars = read_bars(path)
        if args.action == "settle":
            score(args, bars, pd.Timestamp.now(tz=ZONE))
            return
        healthy = capture(db, pairs)
        args.operational = True
        args.news_healthy = healthy
        args.events = None
        if healthy:
            signals = pd.read_sql_query("SELECT event_id,symbol,'news' AS kind,available_at,value "
                                        "FROM news_signals ORDER BY available_at,event_id,symbol", db)
            args.events = Path(scratch) / "events.csv"
            write_csv(signals[signals.symbol.isin(pairs)], args.events)
        # Recheck after network I/O. A run that crosses the cutoff is missed.
        try:
            live(args, pairs, bars, pd.Timestamp.now(tz=ZONE))
        finally:
            audit(db, pd.Timestamp.now(tz=ZONE))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "capture", "issue", "settle"])
    parser.add_argument("--db", default="data/market-forecast-operations.sqlite")
    parser.add_argument("--universe", default="examples/universe.csv")
    parser.add_argument("--start", default="2024-01-01")
    run(parser.parse_args())
