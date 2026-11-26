from argparse import Namespace
from contextlib import closing, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecast import examples, fit, predict
from main import ledger, live, score
from prospective import report
from operations import audit, capture, initialize, run, timed_action


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

    def test_timed_recovery_after_holiday_and_price_outage(self):
        pairs = {"DAL": "ETF", "UAL": "ETF"}
        days = pd.bdate_range(end="2025-11-28", periods=90)
        days = days.drop(pd.Timestamp("2025-11-27"))
        bars = pd.DataFrame([(d, s, 100., 100 + (i + j) % 3) for i, d in enumerate(days)
                             for j, s in enumerate([*pairs, "ETF"])],
                            columns=["date", "symbol", "open", "close"])
        friday = pd.Timestamp("2025-11-28 09:05", tz="America/New_York")
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            args = Namespace(action="timed", db=Path(folder) / "ops.sqlite",
                             universe=Path(folder) / "universe.csv", start="2025-01-01")
            pd.DataFrame(pairs.items(), columns=["symbol", "benchmark"]).to_csv(args.universe, index=False)
            with self.assertRaisesRegex(ValueError, "Restore"):
                run(args)
            with closing(ledger(args.db)) as db:
                initialize(db)
                audit(db, friday - pd.Timedelta(days=2))
            with patch("operations.download", side_effect=TimeoutError):
                with self.assertRaises(TimeoutError):
                    run(args, lambda: friday)
            def quotes(request, universe):
                bars.to_csv(request.out, index=False)
            with patch("operations.download", side_effect=quotes) as downloads, patch("operations.capture", return_value=True):
                run(args, lambda: friday)
                run(args, lambda: friday + pd.Timedelta(minutes=1))
                run(args, lambda: friday.replace(hour=13, minute=59))
                self.assertEqual(downloads.call_count, 1)
                run(args, lambda: friday.replace(hour=14, minute=0))
                run(args, lambda: friday + pd.Timedelta(days=2))
                self.assertEqual(downloads.call_count, 2)
            with closing(ledger(args.db)) as db:
                self.assertEqual(db.execute("SELECT day FROM missed_sessions").fetchall(), [("2025-11-26",)])
                self.assertEqual(db.execute("SELECT count(*),count(outcome) FROM forecasts").fetchone(), (12, 12))
                snapshots = db.execute("SELECT symbol,station,storm,status FROM airport_snapshots ORDER BY symbol")
                self.assertEqual(snapshots.fetchall(), [("DAL","KATL",None,"no eligible bulletin"),
                                                        ("UAL","KORD",None,"no eligible bulletin")])
                matched = report(db)
                self.assertEqual(matched["settled_sessions"], 1)
                self.assertTrue(all(pair["instrument_days"] == 2 for pair in matched["comparisons"].values()))
                self.assertEqual(db.execute("SELECT action,status,error FROM price_checks ORDER BY observed_at").fetchall(),
                                 [("issue","error","TimeoutError"), ("issue","ok",None), ("settle","ok",None)])

    def test_issue_window_tracks_exchange_clock_across_dst(self):
        with closing(ledger(":memory:")) as db:
            for day, hour in [("2025-10-31", 13), ("2025-11-03", 14)]:
                start = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=hour)
                for seconds, action in [(-1, "idle"), (0, "issue"), (1499, "issue"), (1500, "idle")]:
                    now = start + pd.Timedelta(seconds=seconds)
                    self.assertEqual(timed_action(db, True, pd.Timestamp(day), now), action)

    def test_probe_rejects_stale_benchmark_without_forecasting(self):
        now = pd.Timestamp("2026-11-23 09:03", tz="America/New_York")
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            args = Namespace(action="probe", db=Path(folder) / "ops.sqlite",
                             universe=Path(folder) / "pairs.csv", start="2026-11-19")
            args.universe.write_text("symbol,benchmark\nA,ETF\n")
            def stale(request, pairs):
                request.out.write_text("date,symbol,open,close\n2026-11-20,A,2,2\n2026-11-19,ETF,1,1\n")
            with patch("operations.download", side_effect=stale):
                with self.assertRaisesRegex(ValueError, "ETF"):
                    run(args, lambda: now)
            with closing(ledger(args.db)) as db:
                self.assertEqual(db.execute("SELECT status FROM price_checks").fetchone(), ("error",))
                self.assertEqual(db.execute("SELECT count(*) FROM forecasts").fetchone(), (0,))
