from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

from checkpoint import copy_ledger
from main import ledger
from operations import initialize


class CheckpointTests(unittest.TestCase):
    def test_live_wal_snapshot_and_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            source, snapshot, restored = [Path(folder) / name for name in ("live", "backup", "restored")]
            with closing(ledger(source)) as db:
                initialize(db)
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("INSERT INTO missed_sessions VALUES('2026-11-20','a')")
                db.commit()
                db.execute("INSERT INTO missed_sessions VALUES('2026-11-23','b')")
                digest = copy_ledger(source, snapshot)
                db.commit()
            self.assertEqual(digest, hashlib.sha256(snapshot.read_bytes()).hexdigest())
            self.assertEqual(snapshot.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                copy_ledger(source, snapshot)
            copy_ledger(snapshot, restored)
            with closing(sqlite3.connect(restored)) as db:
                self.assertEqual(db.execute("SELECT day FROM missed_sessions").fetchall(), [("2026-11-20",)])

    def test_invalid_input_leaves_destination_absent(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source", Path(folder) / "target"
            with self.assertRaises(sqlite3.OperationalError):
                copy_ledger(source, target)
            self.assertFalse(source.exists())
            with closing(sqlite3.connect(source)) as db:
                db.execute("CREATE TABLE unrelated(value)")
            with self.assertRaisesRegex(ValueError, "ledger table"):
                copy_ledger(source, target)
            self.assertFalse(target.exists())
