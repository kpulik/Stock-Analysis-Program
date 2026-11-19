from argparse import Namespace
from contextlib import closing, redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pandas as pd

from forecast import event_window, read_events
from main import selected, write_csv
from news import collect, headline_score, historical, rss_rows


class NewsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = Namespace(source="yahoo", start=None, end=None,
                              db=str(Path(self.temp.name) / "archive.sqlite"),
                              out=str(Path(self.temp.name) / "events.csv"))
        self.pairs = {"A": "ETF"}
        self.now = datetime.fromisoformat("2025-03-04T14:15:00+00:00")

    def test_rss_uses_observation_and_rejects_old_or_future_articles(self):
        items = "".join(f"<item><pubDate>{date}</pubDate><guid>{i}</guid>"
                        "<title>A raises guidance</title></item>" for i, date in enumerate([
                            "Tue, 4 Mar 2025 14:00:00 GMT", "Mon, 3 Mar 2025 14:15:00 GMT",
                            "Tue, 4 Mar 2025 14:16:00 GMT"]))
        rows = list(rss_rows(("<rss><channel>" + items + "</channel></rss>").encode(), "A", self.now))
        self.assertEqual(len(rows), 1)
        self.assertEqual(datetime.fromisoformat(rows[0][-1]), self.now)
        self.assertLess(rows[0][-2], rows[0][-1])

    def test_repeat_collection_keeps_first_headline_and_timestamp(self):
        row = ("yahoo", "1", "A", "A raises guidance", "https://example.org", "2025-03-04T14:00:00Z",
               "2025-03-04T14:15:00Z")
        revision = (*row[:3], "A cuts guidance", *row[4:6], "2025-03-04T15:15:00Z")
        with patch("news.yahoo", side_effect=[[row], [revision]]), redirect_stdout(io.StringIO()):
            collect(self.args, self.pairs, write_csv)
            first = Path(self.args.out).read_bytes()
            collect(self.args, self.pairs, write_csv)
            self.assertEqual(Path(self.args.out).read_bytes(), first)
        events = read_events(self.args.out, self.pairs)
        self.assertEqual(selected(events), ["prices", "news"])
        self.assertEqual(events[("A", "news")].iloc[0], 1)
        with closing(sqlite3.connect(self.args.db)) as db:
            self.assertEqual(db.execute("SELECT headline FROM articles").fetchone()[0], row[3])

    def article(self, updated):
        return {"id": 33, "headline": "A cuts outlook", "symbols": ["A", "UNKNOWN", "A"],
                "created_at": "2025-03-03T10:00:00Z", "updated_at": updated}

    @patch.dict("os.environ", APCA_API_KEY_ID="fixture-id", APCA_API_SECRET_KEY="fixture-secret")
    def test_history_paginates_and_never_backdates_revision(self):
        first = {"news": [self.article("2025-03-04T15:00:00Z")], "next_page_token": "page2"}
        second = {"news": [self.article("2025-03-05T00:00:00Z")]}
        with patch("news.request_bytes", side_effect=[json.dumps(first), json.dumps(second)]) as fetch:
            rows = list(historical(self.pairs, "2025-03-03", "2025-03-05"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], "A")
        self.assertEqual(datetime.fromisoformat(rows[0][-1]),
                         datetime(2025, 3, 4, 15, tzinfo=timezone.utc))
        self.assertEqual(parse_qs(urlparse(fetch.call_args.args[0]).query)["page_token"], ["page2"])

    @patch.dict("os.environ", APCA_API_KEY_ID="fixture-id", APCA_API_SECRET_KEY="fixture-secret")
    def test_failed_page_preserves_archive_and_export(self):
        self.args.source, self.args.start, self.args.end = "alpaca", "2025-03-03", "2025-03-05"
        Path(self.args.out).write_text("prior export")
        page = {"news": [self.article("2025-03-04T15:00:00Z")], "next_page_token": "again"}
        with patch("news.request_bytes", side_effect=[json.dumps(page), OSError("timeout")]):
            with self.assertRaisesRegex(OSError, "timeout"):
                collect(self.args, self.pairs, write_csv)
        self.assertEqual(Path(self.args.out).read_text(), "prior export")
        with closing(sqlite3.connect(self.args.db)) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM articles").fetchone()[0], 0)

    def test_neutral_headline_counts_only_when_available(self):
        times = pd.to_datetime(["2025-01-01T00:00Z", "2025-01-02T00:00Z", "2025-01-02T00:01Z"])
        total, count = event_window(pd.Series([1, 0, -1], index=times), [times[1]])
        self.assertEqual((float(total[0]), int(count[0])), (0, 1))

    def test_neutral_and_conflicting_signals(self):
        self.assertEqual(headline_score("A schedules its earnings call"), 0)
        self.assertEqual(headline_score("A beats estimates but cuts outlook"), 0)
        self.assertEqual(headline_score("A misses expectations"), -1)
