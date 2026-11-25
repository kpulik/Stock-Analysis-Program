"""Report registered paper comparisons on complete, simultaneously issued cohorts."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pandas as pd

from evaluate import comparison, paired
from forecast import cutoff, ZONE


def report(db):
    rows = pd.read_sql_query("SELECT day AS date,symbol,benchmark,model,issued_at,probability,"
                             "outcome AS target FROM forecasts", db)
    complete = []
    pending = 0
    controls = {"constant_50", "prices_frozen", "prices", "prices_daily"}
    for day, batch in rows.groupby("date"):
        models = set(batch.model)
        if "prices_frozen" not in models:
            continue  # Earlier batches cannot enter the registered prospective experiment.
        if not controls <= models or (("news" in models) != ("news_daily" in models)):
            raise ValueError(f"Incomplete model cohort on {day}")
        issued = pd.Timestamp(batch.issued_at.iloc[0])
        if (batch.issued_at.nunique() != 1 or issued.tzinfo is None or
                str(issued.tz_convert(ZONE).date()) != day or issued >= cutoff(pd.Timestamp(day))):
            raise ValueError(f"Invalid issue time on {day}")
        if batch.target.isna().any():
            pending += 1
            continue
        complete.append(paired(batch))
    settled = pd.concat(complete) if complete else rows.iloc[:0]
    output = {}
    for candidate, control in [("prices_daily", "prices"), ("prices_daily", "prices_frozen"),
                               ("prices_daily", "constant_50"), ("news_daily", "news"),
                               ("news_daily", "prices_daily")]:
        days = settled.loc[settled.model == candidate, "date"]
        subset = settled[settled.date.isin(days) & settled.model.isin([candidate, control])]
        result = None
        if not subset.empty:
            result = comparison(subset, candidate, control)
            errors = (subset.probability - subset.target).groupby(subset.model).mean().abs()
            result["absolute_mean_calibration_error"] = errors.to_dict()
        output[f"{candidate} vs {control}"] = result
    return dict(settled_sessions=len(complete), pending_sessions=pending, comparisons=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    args = parser.parse_args()
    with closing(sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        print(json.dumps(report(db), indent=2, allow_nan=False))
