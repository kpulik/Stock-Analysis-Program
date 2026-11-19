from argparse import Namespace
from contextlib import closing, redirect_stdout
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from forecast import VARIANTS, read_events, universe
from llm_news import classify, run
from main import paper_calls, selected
from news import iso


class ModelTests(unittest.TestCase):
    def test_abstention_boundaries(self):
        sample = pd.DataFrame({"probability": [.44, .45, .50, .549, .55, .9]})
        self.assertEqual(paper_calls(sample).paper_call.tolist(),
                         ["lag_or_tie", "lag_or_tie", "abstain", "abstain", "outperform", "outperform"])
        self.assertEqual(VARIANTS["price_tree"], VARIANTS["prices"])

    def test_local_classification_validates_reply(self):
        for response in (b'{"model":"wrong","choices":[]}',
                         b'{"model":"pinned","choices":[{"message":{"content":"{}"}}]}'):
            with patch("llm_news.urlopen", return_value=io.BytesIO(response)), self.assertRaises(ValueError):
                classify("http://localhost:1234/v1", "pinned", "A", "A raises guidance")
        with self.assertRaises(ValueError):
            classify("https://example.org", "pinned", "A", "A raises guidance")

    def test_asof_export_and_retry(self):
        with tempfile.TemporaryDirectory() as root:
            args = Namespace(db=root + "/cache.sqlite", out=root + "/events.csv", source="yahoo",
                             universe=root + "/universe.csv", limit=8, model="pinned", base_url="unused")
            Path(args.universe).write_text("symbol,benchmark\nA,ETF\n")
            earlier = iso(datetime.now(timezone.utc) - timedelta(minutes=30))
            with closing(sqlite3.connect(args.db)) as db, db:
                db.execute("CREATE TABLE articles (source,article_id,symbol,headline,published_at,available_at)")
                db.execute("INSERT INTO articles VALUES ('yahoo','1','A','A cuts guidance',?,?)",
                           (earlier, earlier))
            with patch("llm_news.classify", return_value=-1) as infer, redirect_stdout(io.StringIO()):
                run(args)
                first = Path(args.out).read_bytes()
                run(args)
                self.assertEqual(infer.call_count, 1)
                self.assertEqual(first, Path(args.out).read_bytes())
            events = read_events(args.out, universe(args.universe))
            self.assertGreater(events["A","llm"].index[0], events["A","news"].index[0])
            self.assertEqual(selected(events, True), ["prices", "price_tree", "news", "llm"])
