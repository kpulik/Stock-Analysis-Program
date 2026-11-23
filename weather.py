"""Archive the two-airport TAF pilot in the private operations ledger."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    mapping = Path(__file__).parent / "research/weather-airports-v1.csv"
    with mapping.open() as file:
        stations = {row["station"] for row in csv.DictReader(file)}
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
