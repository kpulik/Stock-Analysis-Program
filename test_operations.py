from argparse import Namespace
from contextlib import closing, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import pandas as pd

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
                                 [("constant_50",2), ("prices",2), ("prices_daily",2)])
                self.assertEqual(db.execute("SELECT count(*) FROM missed_sessions").fetchone(), (0,))
                for _ in range(2):
                    audit(db, evening + pd.Timedelta(days=3))
                self.assertEqual(db.execute("SELECT day FROM missed_sessions").fetchall(), [("2025-06-16",)])
