"""Archive the two-airport TAF pilot in the private operations ledger."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import re
from urllib.parse import urlencode
import csv

from news import iso, request_bytes


def archive(db, stations, bulletins, observed):
    """Preserve first observation and separate revisions, even at the same issue time."""
    db.execute("""CREATE TABLE IF NOT EXISTS airport_tafs(
        station TEXT, issued TEXT, digest TEXT, first_observed TEXT NOT NULL,
        valid_from INTEGER NOT NULL, valid_to INTEGER NOT NULL, groups_json TEXT NOT NULL,
        PRIMARY KEY(station, issued, digest))""")
    found = set()
    with db:
        for item in bulletins:
            station = item["icaoId"]
            if station not in stations:
                raise ValueError("Unexpected station")
            issued = iso(datetime.fromisoformat(item["issueTime"]))
            raw = item["rawTAF"]
            if not raw or item["validTimeFrom"] >= item["validTimeTo"]:
                raise ValueError("Invalid bulletin")
            groups = [{key: group[key] for key in
                       ("timeFrom", "timeTo", "timeBec", "fcstChange", "probability", "wxString", "notDecoded")}
                      for group in item["fcsts"]]
            digest = hashlib.sha256(raw.encode()).hexdigest()
            db.execute("INSERT OR IGNORE INTO airport_tafs VALUES(?,?,?,?,?,?,?)",
                       (station, issued, digest, iso(observed), item["validTimeFrom"],
                        item["validTimeTo"], json.dumps(groups)))
            found.add(station)
    return stations - found


def storm_for_session(db, station, before, opens, closes):
    """Reconstruct one station input from the latest bulletin known before issuance."""
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='airport_tafs'").fetchone():
        return None, "no eligible bulletin", None
    stamp = iso(before)
    rows = db.execute("""SELECT digest,issued,first_observed,valid_from,valid_to,groups_json
                         FROM airport_tafs WHERE station=? AND issued<? AND first_observed<?
                         ORDER BY issued DESC,first_observed DESC LIMIT 2""",
                      (station, stamp, stamp)).fetchall()
    if not rows:
        return None, "no eligible bulletin", None
    digest, issued, seen, valid_from, valid_to, payload = rows[0]
    if len(rows) == 2 and rows[1][1:3] == (issued, seen):
        return None, "ambiguous revision", None
    start, end = opens.timestamp(), closes.timestamp()
    if valid_from > start or valid_to < end:
        return None, "incomplete validity", digest

    intervals = []
    storm = False
    for group in json.loads(payload):
        first, last = group["timeFrom"], group["timeTo"]
        if not isinstance(first, int) or not isinstance(last, int) or first >= last:
            return None, "invalid group interval", digest
        if last <= start or first >= end:
            continue
        change = group["fcstChange"]
        if group["probability"] or (change or "").startswith("PROB"):
            continue
        if change not in (None, "FM", "TEMPO") or group["notDecoded"]:
            return None, "unsupported group", digest
        if change != "TEMPO":
            intervals.append((first, last))
        for token in (group["wxString"] or "").split():
            if re.fullmatch(r"[+-]?(?:VC)?TS(?:RA|SN|PL|GR|GS|DZ|SG|IC|UP)*", token):
                storm = True
    covered_to = start
    for first, last in sorted(intervals):
        if first > covered_to:
            break
        covered_to = max(covered_to, last)
    if covered_to < end:
        return None, "incomplete prevailing groups", digest
    return int(storm), "covered", digest


def pilot_exposures():
    mapping = Path(__file__).parent / "research/weather-airports-v1.csv"
    with mapping.open() as file:
        return list(csv.DictReader(file))


def record_snapshot(db, pairs, day, before, opens, closes):
    """Called inside the first operational issue transaction, using its input time."""
    db.execute("""CREATE TABLE IF NOT EXISTS airport_snapshots(
        day TEXT, symbol TEXT, station TEXT, weight REAL NOT NULL, snapshot_at TEXT NOT NULL,
        storm INTEGER, status TEXT NOT NULL, bulletin_digest TEXT,
        PRIMARY KEY(day,symbol,station))""")
    for exposure in pilot_exposures():
        symbol, station = exposure["symbol"], exposure["station"]
        if symbol in pairs:
            storm, status, digest = storm_for_session(db, station, before, opens, closes)
            db.execute("INSERT INTO airport_snapshots VALUES(?,?,?,?,?,?,?,?)",
                       (str(day.date()), symbol, station, float(exposure["weight"]),
                        iso(before), storm, status, digest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--session", help="Inspect archived inputs at the session cutoff (YYYY-MM-DD)")
    args = parser.parse_args()
    exposures = pilot_exposures()
    stations = {row["station"] for row in exposures}
    if args.session:
        import exchange_calendars as xcals
        import pandas as pd
        from forecast import cutoff

        day = pd.Timestamp(datetime.strptime(args.session, "%Y-%m-%d").date())
        cal = xcals.get_calendar("XNYS")
        opens, closes = cal.session_open(day), cal.session_close(day)
        if cutoff(day) > datetime.now(timezone.utc):
            parser.error("Session cutoff has not elapsed")
        with closing(sqlite3.connect(Path(args.db).resolve().as_uri() + "?mode=ro", uri=True)) as db:
            for row in exposures:
                value, status, digest = storm_for_session(db, row["station"], cutoff(day), opens, closes)
                print(row["symbol"], row["station"], status, value, digest or "")
        return
    params = urlencode({"ids": ",".join(sorted(stations)), "format": "json"})
    data = request_bytes("https://aviationweather.gov/api/data/taf?" + params)
    observed = datetime.now(timezone.utc)
    # An HTTP 204 can have an empty body.
    bulletins = json.loads(data) if data else []
    with closing(sqlite3.connect(args.db)) as db:
        missing = archive(db, stations, bulletins, observed)
    print(f"Observed {len(stations - missing)}/{len(stations)} stations at {iso(observed)}")
    if missing:
        raise SystemExit("Missing TAF for " + ", ".join(sorted(missing)))


if __name__ == "__main__":
    main()
