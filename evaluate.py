"""Audit a walk-forward replay using exactly paired instrument-days."""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import examples, read_bars, read_events, universe


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paired(rows):
    keys = ["date", "symbol"]
    required = keys + ["model", "benchmark", "probability", "target"]
    if rows.empty or rows[required].isna().any().any() or rows.duplicated(keys + ["model"]).any():
        raise ValueError("Empty replay, missing fields or duplicate model forecasts")
    if not rows.probability.between(0, 1).all() or not rows.target.isin([0, 1]).all():
        raise ValueError("Invalid probability or outcome")
    groups = rows.groupby(keys)
    if (groups.model.size() != rows.model.nunique()).any():
        raise ValueError("Model forecasts cover different instrument-days")
    if groups.benchmark.nunique().ne(1).any() or groups.target.nunique().ne(1).any():
        raise ValueError("Inconsistent benchmarks or outcomes")
    return rows.assign(loss=(rows.probability - rows.target) ** 2)


def paired_interval(deltas, block=5, draws=5000):
    values = np.asarray(deltas, dtype=float)
    n = len(values)
    if not 1 <= block <= n or draws < 2:
        raise ValueError("Insufficient sessions or bootstrap draws")
    rng = np.random.default_rng(20261119)
    starts = rng.integers(n, size=(draws, (n + block - 1) // block))
    ix = (starts[:, :, None] + np.arange(block)) % n
    means = values[ix.reshape(draws, -1)[:, :n]].mean(axis=1)
    return {"mean_delta": float(values.mean()),
            "interval_95": np.quantile(means, [.025, .975]).tolist()}


def comparison(frame, candidate, control):
    daily = frame.groupby(["date", "model"]).loss.mean().unstack()
    if candidate not in daily or control not in daily:
        raise ValueError("Requested comparison is missing from replay")
    delta = daily[candidate] - daily[control]
    return {"sessions": len(daily), "instrument_days": int((frame.model == control).sum()),
            "session_mean_brier": daily.mean().to_dict(),
            "pooled_brier": frame.groupby("model").loss.mean().to_dict(),
            "paired": paired_interval(delta) if len(delta) >= 5 else None}


def calibrate(rows):
    bins = pd.cut(rows.probability, np.linspace(0, 1, 11), include_lowest=True)
    buckets = rows.groupby(bins, observed=True).agg(n=("target", "size"),
            mean_forecast=("probability", "mean"), observed_rate=("target", "mean"))
    return [dict(bin=str(interval), n=int(r.n), mean_forecast=r.mean_forecast,
                 observed_rate=r.observed_rate) for interval, r in buckets.iterrows()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", required=True)
    p.add_argument("--bars", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--universe", default="examples/universe.csv")
    p.add_argument("--candidate", default="news")
    p.add_argument("--control", default="prices")
    p.add_argument("--archive", help="Optional source archive to fingerprint")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = paired(pd.read_csv(args.replay, parse_dates=["date"]))
    pairs = universe(args.universe)
    features = examples(read_bars(args.bars), pairs, read_events(args.events, pairs))
    keys = ["date", "symbol", "benchmark", "target"]
    merged = rows.merge(features[keys + ["news_count"]], on=keys, how="left", validate="many_to_one")
    if merged.news_count.isna().any():
        raise ValueError("Replay outcomes or symbols differ from supplied price history")
    target = merged[merged.model == args.candidate]
    files = {key: digest(getattr(args, key)) for key in ("bars", "events", "replay", "universe", "archive")
             if getattr(args, key) is not None}
    report = {
        "candidate": args.candidate, "control": args.control,
        "evaluation_dates": [str(target.date.min().date()), str(target.date.max().date())],
        "sha256": files, "runtime": {"python": platform.python_version(),
                                   **{k: version(k) for k in ("pandas", "numpy", "scikit-learn")}},
        "bootstrap": {"unit": "session", "scheme": "circular moving blocks", "block": 5,
                      "draws": 5000, "seed": 20261119},
        "overall": comparison(merged, args.candidate, args.control),
        "calibration": {name: calibrate(group) for name, group in merged.groupby("model")},
        "by_benchmark": {}, "news_coverage": {}, "covered_only": None,
    }
    for name, group in merged.groupby("benchmark"):
        report["by_benchmark"][name] = comparison(group, args.candidate, args.control)
        observed = group[group.model == args.candidate]
        report["news_coverage"][name] = {"instrument_days": len(observed),
                                         "with_headlines": int(observed.news_count.gt(0).sum())}
    covered = merged[merged.news_count > 0]
    if not covered.empty:
        report["covered_only"] = comparison(covered, args.candidate, args.control)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
