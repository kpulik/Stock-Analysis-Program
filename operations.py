"""Prospective feed capture, issuance and settlement on one private SQLite ledger."""
import argparse
from contextlib import closing
import hashlib
from pathlib import Path
import tempfile

import exchange_calendars as xcals
import pandas as pd

from forecast import ZONE, cutoff, read_bars, universe
from main import completed_through, download, ledger, live, score, write_csv
from news import headline_score, yahoo


def initialize(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS news_checks(
            symbol TEXT, observed_at TEXT, status TEXT, eligible INTEGER, error TEXT,
            PRIMARY KEY(symbol, observed_at));
        CREATE TABLE IF NOT EXISTS news_signals(
            event_id TEXT, symbol TEXT, available_at TEXT, value INTEGER,
            PRIMARY KEY(event_id, symbol));
        CREATE TABLE IF NOT EXISTS audit_cursor(id INTEGER PRIMARY KEY CHECK(id=1), day TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS price_checks(
            observed_at TEXT PRIMARY KEY, action TEXT NOT NULL, status TEXT NOT NULL, error TEXT);
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
    calendar = xcals.get_calendar("XNYS")
    session = calendar.is_session(day)
    with db:
        db.execute("BEGIN IMMEDIATE")
        # Legacy ledgers start at the earliest documented operational session.
        db.execute("""INSERT OR IGNORE INTO audit_cursor
                      SELECT 1, min(day) FROM (SELECT min(day) AS day FROM forecasts
                      UNION ALL SELECT min(day) FROM missed_sessions UNION ALL SELECT ?)""",
                   (str(day.date()),))
        start = pd.Timestamp(db.execute("SELECT day FROM audit_cursor").fetchone()[0])
        end = day if local >= cutoff(day) else day - pd.Timedelta(days=1)
        if start <= end:
            for date in calendar.sessions_in_range(start, end):
                key = str(date.date())
                db.execute("""INSERT OR IGNORE INTO missed_sessions SELECT ?,?
                              WHERE NOT EXISTS(SELECT 1 FROM forecasts WHERE day=?)""",
                           (key, now.isoformat(), key))
            db.execute("UPDATE audit_cursor SET day=?", (str((end + pd.Timedelta(days=1)).date()),))
    print(f"{day.date()} session={session} forecasts={issued} settled={settled}")
    return session, day


def timed_action(db, session, day, now):
    """Choose one due action when invoked by a private runner."""
    deadline = cutoff(day)
    if session and deadline - pd.Timedelta(minutes=25) <= now < deadline:
        if not db.execute("SELECT 1 FROM forecasts WHERE day=? LIMIT 1", (str(day.date()),)).fetchone():
            return "issue"
    if db.execute("SELECT 1 FROM forecasts WHERE outcome IS NULL AND day<=? LIMIT 1",
                  (completed_through(now).isoformat(),)).fetchone():
        return "settle"
    return "idle"


def run(args, clock=None):
    if clock is None:
        clock = lambda: pd.Timestamp.now(tz=ZONE)
    if args.action == "timed" and not Path(args.db).is_file():
        raise ValueError("Restore the private ledger before timed operation")
    pairs = universe(args.universe)
    with closing(ledger(args.db)) as db, tempfile.TemporaryDirectory() as scratch:
        if args.action == "timed" and not db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='news_checks'").fetchone():
            raise ValueError("Timed operation requires an initialized private ledger")
        initialize(db)
        now = clock()
        session, day = audit(db, now)
        action = timed_action(db, session, day, now) if args.action == "timed" else args.action
        if action == "idle":
            print("No forecast or settlement due")
            return
        if action == "audit":
            return
        if action == "capture":
            return capture(db, pairs)
        if action == "issue" and (not session or now >= cutoff(day)):
            raise ValueError("Outside premarket issuance window")
        path = Path(scratch) / "bars.csv"
        try:
            download(argparse.Namespace(start=args.start, end=None, out=path), pairs)
            bars = read_bars(path)
        except Exception as exc:
            with db:
                db.execute("INSERT INTO price_checks VALUES(?,?,?,?)",
                           (pd.Timestamp.now(tz="UTC").isoformat(), action, "error", type(exc).__name__))
            audit(db, clock())
            raise
        with db:
            db.execute("INSERT INTO price_checks VALUES(?,?,?,NULL)",
                       (pd.Timestamp.now(tz="UTC").isoformat(), action, "ok"))
        if action == "settle":
            score(args, bars, clock())
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
            ready = clock()
            if ready.tz_convert(ZONE).date() != day.date():
                raise ValueError("Target session changed during data refresh")
            live(args, pairs, bars, ready, clock)
        finally:
            audit(db, clock())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "capture", "issue", "settle", "timed"])
    parser.add_argument("--db", default="data/market-forecast-operations.sqlite")
    parser.add_argument("--universe", default="examples/universe.csv")
    parser.add_argument("--start", default="2024-01-01")
    run(parser.parse_args())
