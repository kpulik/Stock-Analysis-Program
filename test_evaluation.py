from datetime import date
import unittest

import numpy as np
import pandas as pd

from archive_news import following_midnight, import_rows
from evaluate import comparison, paired, paired_interval


class ArchiveTests(unittest.TestCase):
    def test_following_date_across_dst(self):
        self.assertEqual(following_midnight(date(2019, 3, 10)), "2019-03-11T00:00:00-04:00")
        self.assertEqual(following_midnight(date(2019, 11, 3)), "2019-11-04T00:00:00-05:00")

    def test_duplicate_later_version_and_date_filter(self):
        row = dict(stock="A", date="2019-01-02", headline="A beats estimates", url="https://example.org/1")
        rows = [row, dict(row, date="2019-01-03", headline="A cuts guidance"),
                dict(row, stock="B"), dict(row, date="2020-01-01")]
        frame, count = import_rows(rows, {"A"}, date(2019, 1, 1), date(2020, 1, 1))
        self.assertEqual((count, len(frame)), (2, 1))
        self.assertEqual(frame.iloc[0].available_at, "2019-01-04T00:00:00-05:00")
        self.assertEqual(frame.iloc[0].value, -1)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame([(d, "A", "ETF", model, .5, int(d.day % 2))
                                  for d in pd.date_range("2019-01-01", periods=12)
                                  for model in ["news", "prices"]],
                                 columns=["date", "symbol", "benchmark", "model", "probability", "target"])

    def test_bad_pairs_rejected(self):
        mismatch = self.rows.copy()
        mismatch.loc[0, "target"] = 1 - mismatch.loc[0, "target"]
        for bad in (mismatch, self.rows.iloc[1:], pd.concat([self.rows, self.rows.iloc[:1]])):
            with self.assertRaises(ValueError):
                paired(bad)

    def test_constant_and_deterministic_bootstrap(self):
        result = comparison(paired(self.rows), "news", "prices")
        self.assertEqual(result["paired"], dict(mean_delta=0, interval_95=[0, 0]))
        self.assertEqual(result["sessions"], 12)
        deltas = np.arange(12) / 100
        self.assertEqual(paired_interval(deltas), paired_interval(deltas))

    def test_equal_session_weight_with_unequal_symbol_counts(self):
        other = self.rows.iloc[:2].assign(symbol="B", probability=1)
        result = comparison(paired(pd.concat([self.rows, other])), "news", "prices")
        self.assertNotEqual(result["session_mean_brier"]["news"], result["pooled_brier"]["news"])
