from contextlib import closing
from datetime import datetime, timedelta, timezone
import sqlite3
import unittest

from weather import archive


class WeatherArchiveTests(unittest.TestCase):
    def test_retries_preserve_observation_and_distinct_revisions(self):
        seen = datetime(2026, 11, 23, 17, tzinfo=timezone.utc)
        group = dict(timeFrom=100, timeTo=200, timeBec=None, fcstChange=None,
                     probability=None, wxString=None, notDecoded=None)
        bulletin = dict(icaoId="KATL", issueTime="2026-11-23T16:04:00Z", rawTAF="synthetic",
                        validTimeFrom=100, validTimeTo=200, fcsts=[group])
        with closing(sqlite3.connect(":memory:")) as db:
            stations = {"KATL", "KORD"}
            self.assertEqual(archive(db, stations, [bulletin], seen), {"KORD"})
            later = seen + timedelta(minutes=5)
            archive(db, stations, [bulletin, dict(bulletin, rawTAF="synthetic amendment")], later)
            dates = db.execute("SELECT first_observed FROM airport_tafs ORDER BY first_observed").fetchall()
            self.assertEqual(dates, [(t.isoformat(timespec="microseconds"),) for t in (seen, later)])
