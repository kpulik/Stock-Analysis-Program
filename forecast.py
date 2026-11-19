"""Point-in-time features and expanding-window probability models."""
from datetime import time
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ZONE = "America/New_York"
KINDS = ("news", "congress", "weather")
PRICES = ["relative_last", "relative_week", "relative_volatility", "benchmark_week"]
NEWS = ["news", "news_count"]
VARIANTS = {
    "prices": PRICES,
    "news": PRICES + NEWS,
    "politics": PRICES + NEWS + ["congress"],
    "weather": PRICES + NEWS + ["congress", "weather"],
}
MIN_ROWS = 40
MIN_DAYS = 20


def universe(path):
    frame = pd.read_csv(path, dtype=str)
    if frame.empty or frame[["symbol", "benchmark"]].isna().any().any():
        raise ValueError("Universe must contain symbol and benchmark pairs")
    if frame.symbol.duplicated().any() or (frame.symbol == frame.benchmark).any():
        raise ValueError("Each symbol needs exactly one distinct benchmark")
    return dict(zip(frame.symbol, frame.benchmark))


def read_bars(path):
    bars = pd.read_csv(path, usecols=["date", "symbol", "open", "close"])
    bars["date"] = pd.to_datetime(bars.date, format="%Y-%m-%d", errors="raise")
    values = bars[["open", "close"]].to_numpy(dtype=float)
    if (bars.empty or bars.isna().any().any() or not np.isfinite(values).all()
            or (values <= 0).any() or bars.duplicated(["date", "symbol"]).any()):
        raise ValueError("Bars must have unique dates per symbol and finite positive prices")
    return bars


def read_events(path, pairs):
    if path is None:
        return {}
    events = pd.read_csv(path, usecols=["event_id", "symbol", "kind", "available_at", "value"])
    if events.empty:
        return {}
    if events.isna().any().any() or events.duplicated(["event_id", "symbol", "kind"]).any():
        raise ValueError("Events need complete fields and unique IDs per exposure")
    if set(events.kind) - set(KINDS) or set(events.symbol) - set(pairs):
        raise ValueError("Event kind or exposed symbol is outside the configured universe")
    if not events.available_at.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$").all():
        raise ValueError("Event availability must include a timezone offset")
    events["available_at"] = pd.to_datetime(events.available_at, format="ISO8601", utc=True)
    events["value"] = pd.to_numeric(events.value)
    if not events.value.between(-1, 1).all():
        raise ValueError("Each event exposure must be between -1 and 1")
    return {(symbol, kind): group.sort_values("available_at").set_index("available_at").value
            for (symbol, kind), group in events.groupby(["symbol", "kind"])}


def cutoff(day):
    return pd.Timestamp.combine(day.date(), time(9, 25)).tz_localize(ZONE)


def event_window(series, cutoffs):
    # Window is (forecast time - 24 hours, forecast time]. All times are UTC.
    times = series.index.as_unit("ns").asi8
    ends = pd.DatetimeIndex(cutoffs).as_unit("ns").asi8
    sums = np.r_[0.0, series.cumsum().to_numpy()]
    left = times.searchsorted(ends - pd.Timedelta(days=1).value, side="right")
    right = times.searchsorted(ends, side="right")
    return np.clip(sums[right] - sums[left], -1, 1), right - left


def examples(bars, pairs, events, live_time=None):
    """A live target row is constructed from past bars even without a target-day quote."""
    parts = []
    if live_time is not None:
        live_day = live_time.tz_convert(ZONE).tz_localize(None).normalize()
        bars = bars[bars.date < live_day]
    returns = bars.assign(r=bars.close / bars.open - 1).pivot(index="date", columns="symbol", values="r")
    for symbol, benchmark in pairs.items():
        if symbol not in returns or benchmark not in returns:
            raise ValueError(f"Missing price history for {symbol} or {benchmark}")
        # Benchmark sessions supply the index. A missing stock bar stays missing.
        benchmark_r = returns[benchmark].dropna()
        if live_time is not None:
            benchmark_r = benchmark_r.reindex(benchmark_r.index.union([live_day]))
        stock_r = returns[symbol].reindex(benchmark_r.index)
        excess = stock_r - benchmark_r
        past = excess.shift(1)
        frame = pd.DataFrame({
            "symbol": symbol, "benchmark": benchmark, "relative_last": past,
            "relative_week": past.rolling(5).mean(),
            "relative_volatility": past.rolling(5).std(),
            "benchmark_week": benchmark_r.shift(1).rolling(5).mean(),
            "target": (excess > 0).astype(float).where(excess.notna()),
        }, index=benchmark_r.index)
        frame.index.name = "date"
        cutoffs = [cutoff(day) for day in frame.index]
        if live_time is not None:
            cutoffs[-1] = live_time
        for kind in KINDS:
            series = events.get((symbol, kind))
            total, count = (0.0, 0) if series is None else event_window(series, cutoffs)
            frame[kind] = total
            if kind == "news":
                frame["news_count"] = count
        parts.append(frame.dropna(subset=PRICES).reset_index())
    return pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"])


def fit(rows, variant):
    if len(rows) < MIN_ROWS or rows.date.nunique() < MIN_DAYS:
        return None
    fields = VARIANTS[variant]
    y = rows.target.astype(int)
    if y.nunique() < 2:
        model = (int(y.sum()) + 1) / (len(y) + 2)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        model.fit(rows[fields], y)
    training_hash = hashlib.sha256(
        pd.util.hash_pandas_object(rows[["date", "symbol", "benchmark", *fields, "target"]],
                                   index=False).values.tobytes()
    ).hexdigest()
    version = hashlib.sha256(json.dumps(["logit-v1", variant, training_hash]).encode()).hexdigest()[:16]
    return model, version


def predict(fitted, rows, variant):
    model, version = fitted
    results = rows[["date", "symbol", "benchmark", "target"]].copy()
    results["probability"] = (model if isinstance(model, float) else
                              model.predict_proba(rows[VARIANTS[variant]])[:, 1])
    results["model"] = variant
    results["version"] = version
    return results


def replay(frame, variants, start):
    labeled = frame.dropna(subset=["target"])
    predictions = []
    frozen = None
    previous_week = None
    fitted = {}
    for day, rows in labeled[labeled.date >= start].groupby("date", sort=True):
        week = day.to_period("W-SUN")
        if week != previous_week:
            history = labeled[labeled.date < week.start_time]
            fitted = {name: fit(history, name) for name in variants}
            previous_week = week
        if frozen is None and fitted.get("prices") is not None:
            frozen = fitted["prices"]
        if frozen is not None:
            chance = predict((0.5, "constant-0.5"), rows, "prices")
            chance["model"] = "constant_50"
            predictions.append(chance)
            control = predict(frozen, rows, "prices")
            control["model"] = "prices_frozen"
            predictions.append(control)
        for name, model in fitted.items():
            if model is not None:
                predictions.append(predict(model, rows, name))
    if not predictions:
        raise ValueError("No eligible predictions. Supply more history or an earlier start date.")
    return pd.concat(predictions, ignore_index=True)
