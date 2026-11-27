"""Create a verified, private SQLite backup or restore it to a new ledger path."""
import argparse
from contextlib import closing
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile


def copy_ledger(source, destination):
    """Use SQLite's online backup API and publish only a complete snapshot."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    source_uri = source.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(source_uri, uri=True)) as reader:
        with tempfile.TemporaryDirectory(dir=destination.parent) as private:
            staged = Path(private) / "snapshot.sqlite"
            staged.touch(mode=0o600)
            with closing(sqlite3.connect(staged)) as writer:
                reader.backup(writer)
                writer.execute("PRAGMA journal_mode=DELETE")
                if writer.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                    raise ValueError("Checkpoint failed SQLite integrity check")
                for table in ("forecasts", "news_checks", "missed_sessions"):
                    if not writer.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                                          (table,)).fetchone():
                        raise ValueError(f"Missing operations ledger table: {table}")
            with staged.open("rb") as file:
                checksum = hashlib.file_digest(file, "sha256").hexdigest()
                os.fsync(file.fileno())
            os.link(staged, destination)
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    return checksum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(copy_ledger(args.source, args.destination))
