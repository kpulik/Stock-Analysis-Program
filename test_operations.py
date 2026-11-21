from argparse import Namespace
from contextlib import closing, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from forecast import examples, fit, predict
from main import ledger, live, score
from operations import audit, capture, initialize


class OperationsTests(unittest.TestCase):
    def test_outage_is_distinct_from_quiet_feed(self):
        with closing(ledger(":memory:")) as db:
            initialize(db)
            def fetch(symbols):
                if symbols == ["B"]:
                    raise TimeoutError()
                return []
            with redirect_stdout(io.StringIO()):
                self.assertFalse(capture(db, ["A", "B"], fetch))
            self.assertEqual(db.execute("SELECT status,eligible FROM news_checks ORDER BY symbol").fetchall(),
                             [("ok", 0), ("error", None)])

    def test_restart_settles_exact_issued_cohort(self):
        days = pd.bdate_range(end="2025-06-13", periods=90)
        bars = pd.DataFrame([(day, symbol, 100, 100 + (i + j) % 3)
                             for i, day in enumerate(days) for j, symbol in enumerate(("A","B","ETF"))],
                            columns=["date","symbol","open","close"])
        morning = pd.Timestamp("2025-06-13 09:19", tz="America/New_York")
        evening = morning + pd.Timedelta(hours=9)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            args = Namespace(db=Path(folder) / "ops.sqlite", events=None, operational=True, news_healthy=False)
            pairs = {"A":"ETF", "B":"ETF"}
            live(args, pairs, bars, morning, lambda: morning)
            args.news_healthy = True
            live(args, pairs, bars, morning, lambda: morning + pd.Timedelta(minutes=1))
            score(args, bars, evening)
            with closing(ledger(args.db)) as db:
                initialize(db)
                audit(db, evening)
                self.assertEqual(db.execute("SELECT model,count(outcome) FROM forecasts GROUP BY model").fetchall(),
                                 [("constant_50",2), ("prices",2), ("prices_daily",2), ("prices_frozen",2)])
                self.assertEqual(db.execute("SELECT count(*) FROM missed_sessions").fetchone(), (0,))
                for _ in range(2):
                    audit(db, evening + pd.Timedelta(days=3))
                self.assertEqual(db.execute("SELECT day FROM missed_sessions").fetchall(), [("2025-06-16",)])

    def test_frozen_fit_survives_revised_history_and_restart(self):
        days = pd.bdate_range(end="2025-06-16", periods=90)
        dist = np.random.default_rng(815)
        bars = pd.DataFrame([(d, s, 100., 100 + dist.normal()) for d in days
                             for s in ("A", "B", "ETF")], columns=["date","symbol","open","close"])
        pairs = {"A": "ETF", "B": "ETF"}
        first = pd.Timestamp("2025-06-13 09:15", tz="America/New_York")
        second = first + pd.Timedelta(days=3)
        frame = examples(bars, pairs, {})
        original = fit(frame[frame.date < "2025-06-09"], "prices")
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            args = Namespace(db=Path(folder) / "ledger.sqlite", events=None, operational=True)
            late = first.replace(minute=25)
            with self.assertRaisesRegex(ValueError, "crossed"):
                live(args, pairs, bars, first, lambda: late)
            with closing(ledger(args.db)) as db:
                self.assertFalse(db.execute("SELECT 1 FROM forecasts").fetchone())
                self.assertFalse(db.execute("SELECT 1 FROM sqlite_master WHERE name='frozen_prices'").fetchone())
            live(args, pairs, bars, first, lambda: first)
            bars.loc[(bars.date < "2025-05-01") & (bars.symbol == "A"), "close"] = 120.
            live(args, pairs, bars, second, lambda: second)
            revised = examples(bars, pairs, {}, live_time=second)
            expected = predict(original, revised[revised.date == days[-1]], "prices").sort_values("symbol")
            with closing(ledger(args.db)) as db:
                rows = db.execute("SELECT probability FROM forecasts WHERE model='prices_frozen' "
                                  "AND day='2025-06-16' ORDER BY symbol").fetchall()
                np.testing.assert_allclose([r[0] for r in rows], expected.probability, rtol=1e-12)
                self.assertEqual(db.execute("SELECT first_session,version FROM frozen_prices").fetchall(),
                                 [("2025-06-13", original[1])])
                self.assertEqual(db.execute("SELECT count(DISTINCT version) FROM forecasts "
                                            "WHERE model='prices'").fetchone(), (2,))

    def test_legacy_batch_cannot_seed_frozen_prior_on_retry(self):
        days = pd.bdate_range(end="2025-06-16", periods=90)
        bars = pd.DataFrame([(d, s, 100., 101. if s == "A" else 100.)
                             for d in days for s in ("A", "ETF")],
                            columns=["date","symbol","open","close"])
        first = pd.Timestamp("2025-06-13 09:15", tz="America/New_York")
        second = first + pd.Timedelta(days=3)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            args = Namespace(db=Path(folder) / "ledger.sqlite", events=None)
            live(args, {"A":"ETF"}, bars, first, lambda: first)
            args.operational = True
            live(args, {"A":"ETF"}, bars, first, lambda: first)
            live(args, {"A":"ETF"}, bars, second, lambda: second)
            with closing(ledger(args.db)) as db:
                frozen = db.execute("SELECT day,probability FROM forecasts WHERE model='prices_frozen'").fetchall()
                weekly = db.execute("SELECT day,probability FROM forecasts WHERE model='prices' "
                                    "AND day='2025-06-16'").fetchall()
                self.assertEqual(frozen, weekly)
                self.assertEqual(len(frozen), 1)
                self.assertGreater(frozen[0][1], 0.5)
