import argparse
from contextlib import closing, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from forecast import PRICES, ZONE, cutoff, examples, read_events, replay
from main import completed_through, ledger, live, score


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.pairs = {"A": "ETF", "B": "ETF"}
        days = pd.bdate_range(end="2025-06-13", periods=90)
        rng = np.random.default_rng(42)
        self.bars = pd.DataFrame([(day, symbol, 100., 100 + rng.normal())
                                  for day in days for symbol in ["A", "B", "ETF"]],
                                 columns=["date", "symbol", "open", "close"])
        self.day = days[-1]
        self.now = cutoff(self.day) - pd.Timedelta(minutes=1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = argparse.Namespace(events=None, db=str(Path(self.temp.name) / "log.sqlite"))

    def test_target_day_prices_do_not_change_features_or_forecast(self):
        original = examples(self.bars, self.pairs, {})
        self.bars.loc[self.bars.date == self.day, "close"] *= 3
        revised = examples(self.bars, self.pairs, {})
        for frame in (original, revised):
            self.assertEqual(len(frame[frame.date == self.day]), 2)
        pd.testing.assert_frame_equal(original.loc[original.date == self.day, PRICES],
                                      revised.loc[revised.date == self.day, PRICES])
        p1 = replay(original, ["prices"], self.day, daily=True)
        p2 = replay(revised, ["prices"], self.day, daily=True)
        pd.testing.assert_series_equal(p1.probability, p2.probability)

    def test_event_cutoff_and_timezone(self):
        path = Path(self.temp.name) / "events.csv"
        path.write_text("event_id,symbol,kind,available_at,value\n"
                        "a,A,weather,2025-06-13T13:25:00Z,0.2\n"
                        "b,A,weather,2025-06-13T09:25:01-04:00,0.7\n"
                        "c,A,weather,2025-06-12T13:25:00Z,0.9\n")
        events = read_events(path, self.pairs)
        rows = examples(self.bars, self.pairs, events)
        self.assertAlmostEqual(rows[(rows.date == self.day) & (rows.symbol == "A")].weather.item(), .2)
        path.write_text(path.read_text().replace("13:25:00Z", "13:25:00"))
        with self.assertRaisesRegex(ValueError, "timezone"):
            read_events(path, self.pairs)

    def test_frozen_control_is_fixed(self):
        rows = replay(examples(self.bars, self.pairs, {}), ["prices"],
                      self.day - pd.Timedelta(days=30))
        self.assertEqual(rows[rows.model == "prices_frozen"].version.nunique(), 1)
        self.assertGreater(rows[rows.model == "prices"].version.nunique(), 1)

    def test_daily_candidate_changes_with_each_completed_session(self):
        frame = examples(self.bars, self.pairs, {})
        reports = replay(frame, ["prices"], self.day - pd.Timedelta(days=4), daily=True)
        versions = reports.groupby(["date", "model"]).version.first().unstack()
        self.assertEqual(versions.prices.nunique(), 1)
        self.assertEqual(versions.prices_daily.nunique(), 5)
        self.assertEqual(versions.prices.iloc[0], versions.prices_daily.iloc[0])
        with redirect_stdout(io.StringIO()):
            live(self.args, self.pairs, self.bars, self.now, lambda: self.now)
        with closing(ledger(self.args.db)) as db:
            stored = dict(db.execute("SELECT model,version FROM forecasts WHERE symbol='A'"))
        self.assertEqual(stored["prices_daily"], versions.prices_daily.iloc[-1])

    def test_live_rerun_and_score(self):
        with redirect_stdout(io.StringIO()):
            live(self.args, self.pairs, self.bars, self.now, lambda: self.now)
            later = self.now + pd.Timedelta(seconds=30)
            live(self.args, self.pairs, self.bars, later, lambda: later)
            score(self.args, self.bars, self.now)
            with closing(ledger(self.args.db)) as db:
                self.assertEqual(db.execute("SELECT count(*),count(outcome),min(issued_at) "
                                            "FROM forecasts").fetchone(),
                                 (4, 0, self.now.isoformat()))
            after_close = self.now + pd.Timedelta(hours=9)
            score(self.args, self.bars, after_close)
            score(self.args, self.bars, after_close + pd.Timedelta(hours=1))
        with closing(ledger(self.args.db)) as db:
            self.assertEqual(db.execute("SELECT count(outcome),max(scored_at) FROM forecasts").fetchone(),
                             (4, after_close.isoformat()))

    def test_early_close_holiday_and_late_run(self):
        when = pd.Timestamp("2025-11-28 14:05", tz=ZONE)
        self.assertEqual(completed_through(when), when.date())
        for late in [self.now + pd.Timedelta(hours=1), pd.Timestamp("2025-07-04 08:00", tz=ZONE)]:
            with self.assertRaisesRegex(ValueError, "trading day"):
                live(self.args, self.pairs, self.bars, late, lambda: late)
        with self.assertRaisesRegex(ValueError, "crossed"):
            live(self.args, self.pairs, self.bars, self.now, lambda: self.now + pd.Timedelta(hours=1))
        with closing(ledger(self.args.db)) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM forecasts").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
