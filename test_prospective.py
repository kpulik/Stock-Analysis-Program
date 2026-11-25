from contextlib import closing
import unittest

from main import ledger
from prospective import report


class ProspectiveTests(unittest.TestCase):
    def test_outage_pending_and_revised_target(self):
        controls = ("constant_50", "prices_frozen", "prices", "prices_daily")
        with closing(ledger(":memory:")) as db:
            for day, models, target in [("2026-11-20", controls, 0),
                                        ("2026-11-23", controls + ("news", "news_daily"), 1),
                                        ("2026-11-24", controls, None)]:
                for model in models:
                    db.execute("INSERT INTO forecasts VALUES(?,?,?,?,?,?,?,?,?,?)",
                               (day, "A", model, "ETF", "v", day + "T09:08:00-05:00",
                                0.7 if model == "prices_daily" else 0.5, "{}", target, None))
            result = report(db)
            self.assertEqual((result["settled_sessions"], result["pending_sessions"]), (2, 1))
            pairs = result["comparisons"]
            self.assertEqual(pairs["prices_daily vs prices"]["sessions"], 2)
            self.assertEqual(pairs["news_daily vs prices_daily"]["sessions"], 1)
            self.assertAlmostEqual(pairs["news_daily vs prices_daily"]["session_mean_brier"]["prices_daily"], 0.09)
            db.execute("UPDATE forecasts SET outcome=0 WHERE model='news'")
            with self.assertRaisesRegex(ValueError, "Inconsistent"):
                report(db)
