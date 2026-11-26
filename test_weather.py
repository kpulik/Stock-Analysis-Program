from contextlib import closing
from datetime import datetime, timedelta, timezone
import sqlite3
import unittest

from weather import archive, storm_for_session


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

    def test_session_groups_and_revision_cutoff(self):
        import exchange_calendars as xcals

        calendar = xcals.get_calendar("XNYS")
        opening = calendar.session_open("2026-11-27")
        closing_time = calendar.session_close("2026-11-27")
        before = opening - timedelta(minutes=5)
        start, end = int(opening.timestamp()), int(closing_time.timestamp())
        base = dict(timeFrom=start, timeTo=end, timeBec=None, fcstChange=None,
                    probability=None, wxString=None, notDecoded=None)
        bulletin = dict(icaoId="KORD", issueTime=(before - timedelta(hours=1)).isoformat(),
                        validTimeFrom=start, validTimeTo=end, rawTAF="base", fcsts=[base])
        cases = [
            ([base], 0),
            ([base, dict(base, fcstChange="TEMPO", wxString="+TSRAGR")], 1),
            ([dict(base, wxString="VCTS")], 1),
            ([base, dict(base, probability=30, fcstChange="TEMPO", wxString="TSRA")], 0),
            ([base, dict(base, timeFrom=end, timeTo=end+600, fcstChange="FM", wxString="TS")], 0),
            ([dict(base, timeTo=end-60), dict(base, fcstChange="TEMPO", wxString="TS")], None),
            ([dict(base, timeTo=start+10), dict(base, timeFrom=start+11)], None),
            ([dict(base, notDecoded="unknown")], None),
        ]
        for groups, expected in cases:
            with self.subTest(groups=groups), closing(sqlite3.connect(":memory:")) as db:
                archive(db, {"KORD"}, [dict(bulletin, fcsts=groups)], before - timedelta(minutes=2))
                archive(db, {"KORD"}, [dict(bulletin, rawTAF="late amendment")], before)
                value, _, digest = storm_for_session(db, "KORD", before, opening, closing_time)
                self.assertEqual(value, expected)
                self.assertTrue(digest)
                # The early-close bulletin cannot cover a longer session.
                self.assertIsNone(storm_for_session(db, "KORD", before, opening,
                                                    closing_time + timedelta(hours=3))[0])

    def test_no_stale_fallback_or_arbitrary_revision(self):
        seen = datetime(2026, 11, 24, 12, tzinfo=timezone.utc)
        before = seen + timedelta(minutes=30)
        start = before + timedelta(minutes=5)
        end = start + timedelta(hours=6)
        a, b = int(start.timestamp()), int(end.timestamp())
        group = dict(timeFrom=a, timeTo=b, timeBec=None, fcstChange=None, probability=None,
                     wxString=None, notDecoded=None)
        item = dict(icaoId="KATL", issueTime=seen.isoformat(), validTimeFrom=a, validTimeTo=b,
                    fcsts=[group], rawTAF="original")
        with closing(sqlite3.connect(":memory:")) as db:
            archive(db, {"KATL"}, [item], seen)
            revised = dict(item, validTimeTo=b-1, rawTAF="shorter")
            archive(db, {"KATL"}, [revised], seen + timedelta(minutes=1))
            self.assertEqual(storm_for_session(db,"KATL",before,start,end)[1], "incomplete validity")
            archive(db, {"KATL"}, [dict(revised, rawTAF="second")], seen + timedelta(minutes=1))
            self.assertEqual(storm_for_session(db,"KATL",before,start,end)[1], "ambiguous revision")

    def test_snapshot_uses_input_time_even_when_amendment_precedes_cutoff(self):
        import exchange_calendars as xcals
        import pandas as pd
        from weather import record_snapshot

        day = pd.Timestamp("2026-11-27")
        cal = xcals.get_calendar("XNYS")
        opens, closes = cal.session_open(day), cal.session_close(day)
        before = opens - timedelta(minutes=20)
        interval = dict(timeFrom=int(opens.timestamp()), timeTo=int(closes.timestamp()),
                        timeBec=None, fcstChange=None, probability=None, notDecoded=None)
        bulletin = dict(icaoId="KATL", issueTime=(before - timedelta(hours=1)).isoformat(),
                        validTimeFrom=interval["timeFrom"], validTimeTo=interval["timeTo"],
                        fcsts=[dict(interval, wxString="TSRA")], rawTAF="original")
        with closing(sqlite3.connect(":memory:")) as db:
            archive(db, {"KATL"}, [bulletin], before - timedelta(minutes=1))
            archive(db, {"KATL"}, [dict(bulletin, fcsts=[dict(interval, wxString=None)],
                                        rawTAF="amended")], before + timedelta(minutes=1))
            db.execute("BEGIN")
            record_snapshot(db, {"DAL":"IYT"}, day, before, opens, closes)
            row = db.execute("SELECT storm, status, bulletin_digest FROM airport_snapshots").fetchone()
            self.assertEqual(row[:2], (1, "covered"))
            self.assertEqual(row[2], db.execute("SELECT digest FROM airport_tafs ORDER BY first_observed").fetchone()[0])
            db.rollback()
            self.assertIsNone(db.execute("SELECT 1 FROM sqlite_master WHERE name='airport_snapshots'").fetchone())
