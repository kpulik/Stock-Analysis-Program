# Stock Market Forecast Lab

Estimate the probability that each instrument beats its benchmark from the open to the close of the next decision session. Run a historical replay or record a forecast before the market opens. All outputs are research predictions.

The example universe covers technology, oil and energy, transportation, the Nasdaq Composite and the Nasdaq-100. Stocks use industry ETFs as benchmarks. The two indexes use SPY. Edit `examples/universe.csv` to select additional instruments.

## Quick start

Use Python 3.11 or later. From the repository root on macOS or Linux run

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py demo
python -m unittest -v
```

The demo generates synthetic daily prices offline. To replay real daily bars using the starter universe run

```sh
python main.py download --start 2025-01-01 --end 2025-09-01
python main.py replay --start 2025-05-01
```

The download uses Yahoo Finance via yfinance. Each successful download replaces the specified CSV with the entire requested range. The end date is exclusive. Include the full training history when refreshing. For a repeatable experiment keep a dated copy of the input, universe, events and output. You can also supply a CSV from another data source with `--bars`. Required columns are `date,symbol,open,close`, with one regular-session daily bar per instrument, including benchmarks.

## Historical replay

For each symbol, the model uses the prior session's excess return, its five-session average and volatility, and the benchmark's five-session average. Features use intraday returns, so a split between two sessions does not change the comparison. The target is one when that day's close/open return exceeds the benchmark's, and zero otherwise, including ties.

The logistic model pools the universe and fits on labeled sessions strictly before the forecast week's Monday. Training starts after 40 examples across 20 dates. The replay saves probabilities, realized outcomes, benchmark and model versions to `data/replay.csv`. It compares weekly refits with the initial frozen price model and a constant 50% forecast. The Brier score is the mean squared difference between forecast probability and outcome. Lower scores indicate better probability forecasts for the evaluated instrument-days.

### Historical news, congressional disclosures and weather

Supply normalized events with a known availability timestamp. Each row associates a source event with one exposed instrument. For example, this fictional input represents a weather forecast revision affecting two businesses differently.

```csv
event_id,symbol,kind,available_at,value
demo-storm-001,DAL,weather,2025-05-05T12:10:00Z,-0.6
demo-storm-001,XOM,weather,2025-05-05T12:10:00Z,0.2
```

Run `python main.py replay --start 2025-05-01 --events data/events.csv`. The replay evaluates prices alone, then prices plus news, then congressional disclosures, then weather. Valid kinds are `news`, `congress` and `weather`. The input value is a signed exposure estimate from -1 to 1. The model sums values in the preceding 24 hours for each kind and clips each sum to this range.

Use the earliest time a signal and its exact content were available. For a congressional trade, use public release time. For weather, use the release time of the forecast run. For news, use the available article version. Encode revisions as new events when released. Preserve the mapping from event IDs to source records. This release accepts user-prepared signals. Source ingestion, text scoring, politician track records and weather-to-company mapping are future work.

## Prospective forecasts

On a U.S. trading morning, refresh prices and issue forecasts by 09:25 New York time. Use the same event file in replay and live runs. Omit the events argument for price-only forecasts.

```sh
python main.py download --start 2024-01-01
python main.py predict --db data/forecasts.sqlite
```

The forecast uses information available through invocation. It requires prior-session bars for every instrument and benchmark. SQLite retains the first forecast for each day, symbol and model, together with its issue time, input features, training version and benchmark. Use separate databases for different research universes or experiments.

After the scheduled close plus one hour, refresh the daily bars and score the outstanding forecasts.

```sh
python main.py download --start 2024-01-01
python main.py score --db data/forecasts.sqlite
```

Scoring handles early closes and leaves missing outcomes pending. The app runs on demand. Continuous feeds, scheduled runs and automatic model promotion are outside this release.

## Interpreting results

The starter universe is a fixed illustrative basket. A study of an entire historical sector requires dated membership, delisted securities, and checks on missing bars. Provider history can be revised. Freeze the files used for each evaluation. A price-only replay serves as a baseline for testing other inputs. A prospective comparison provides stronger evidence before adopting a new model.
