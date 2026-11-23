# Stock Market Forecast Lab

Estimate the probability that each instrument beats its benchmark from the open to the close of the next decision session. Run a historical replay or record a forecast before the market opens. All outputs are research predictions.

[Development roadmap](ROADMAP.md) | [Historical experiment results](research/README.md)

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

The weekly control pools the universe and fits on labeled sessions strictly before the forecast week's Monday. Training starts after 40 examples across 20 dates. The replay saves probabilities, realized outcomes, benchmark and model versions to `data/replay.csv`. It compares weekly refits with the initial frozen price model and a constant 50% forecast. The Brier score is the mean squared difference between forecast probability and outcome. Lower scores indicate better probability forecasts for the evaluated instrument-days.

Prospective predictions include a daily challenger for each eligible feature set. Each challenger refits through the preceding completed session and gets its own ledger entry. Add `--daily` to a replay to compare these challengers with weekly controls on matched dates. The original experiment command reproduces the weekly baseline.

### Company news

Collect the currently visible Yahoo Finance RSS headlines for each instrument and pass the exported signals to the forecasting command.

```sh
python main.py news
python main.py predict --events data/yahoo-news.csv
```

The collector archives the headline, source ID, publisher timestamp, URL, associated ticker and first observation time in SQLite. It accepts items published in the preceding 24 hours and preserves the first captured version across retries. RSS history builds from the first collection onward. Run the collector before the decision cutoff to use its latest snapshot. A sector or multi-company headline can appear in more than one ticker feed.

For historical research, set `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` in your environment. The following commands use the Alpaca archive. Dates are UTC and the end date is exclusive.

```sh
python main.py news --source alpaca --start 2025-01-01 --end 2025-09-01
python main.py replay --start 2025-05-01 --events data/alpaca-news.csv
```

The importer paginates through articles associated with the supplied universe. It dates an archived version at the later of its publication and update timestamps. This approximates public availability. Subscription delays, unavailable original versions and changes to vendor ticker tags limit historical reconstruction. Repeated imports preserve the first captured version for each article and ticker. Keep source-specific archives and compare models on dates where the chosen source has coverage. [Alpaca News API](https://docs.alpaca.markets/us/reference/news-3)

The news model uses two features over the preceding 24 hours: article count and a signed headline score. Simple phrase rules capture events such as a guidance cut or an earnings beat. Unmatched and conflicting headlines score zero and still count toward coverage. These rules are a baseline for out-of-sample evaluation.

### Custom news, congressional disclosures and weather

Supply normalized events with a known availability timestamp. Each row associates a source event with one exposed instrument. For example, this fictional input represents a weather forecast revision affecting two businesses differently.

```csv
event_id,symbol,kind,available_at,value
demo-storm-001,DAL,weather,2025-05-05T12:10:00Z,-0.6
demo-storm-001,XOM,weather,2025-05-05T12:10:00Z,0.2
```

Run `python main.py replay --start 2025-05-01 --events data/events.csv`. The replay evaluates prices alone and adds variants for the event kinds present in the file. Valid kinds are `news`, `congress` and `weather`. The input value is a signed exposure estimate from -1 to 1. The model sums values in the preceding 24 hours for each kind and clips each sum to this range.

Use the earliest time a signal and its exact content were available. For a congressional trade, use public release time. For weather, use the release time of the forecast run. For news, use the available article version. Encode revisions as new events when released. Preserve the mapping from event IDs to source records. Congressional disclosures and weather exposures currently require user-prepared inputs.

## Prospective forecasts

On a U.S. trading morning, refresh prices and issue forecasts before 09:25 New York time. Use the same event file in replay and live runs. Omit the events argument for price-only forecasts.

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

## Experimental language and decision models

The local LLM labels a headline's stated business implications for the tagged company as positive, neutral or negative. The forecast model can learn from that label and the number of completed classifications alongside the existing price, headline count and phrase-rule features. Keep the same model identifier tied to the same weights for the entire experiment. Use separate forecast databases and export paths when comparing two different extractors, since both use the forecast variant name `llm`. The collector stores the first valid score, prompt fingerprint and inference completion time. It processes recent articles only and supports bounded, resumable batches. Compare a prospective LLM cohort to the news model on identical sessions. Monitor batching and missing coverage. Until resolved training sessions include LLM scores, the new inputs have no learned predictive weight.

Run a local LM Studio or llama.cpp server that supports structured output and a fixed model ID. Set `MODEL_ID` to its served model name. From the repo root run

```sh
python main.py news
python llm_news.py --model "$MODEL_ID" --base-url http://127.0.0.1:1234/v1
python main.py predict --events data/llm-events.csv --tree
```

The optional decision tree uses the same price inputs and weekly or daily training cutoffs as logistic regression. The first experiment fixes depth at 3 and requires 100 training examples per leaf. The simulated decision rule calls outperformance at 55% or higher, underperformance or a tie at 45% or lower, and abstains in between. The cutoffs are experimental settings. The replay prints call count, coverage, hit rate among calls and Brier loss across all forecasts. These measure directional decisions and probability quality. Trading returns would require a separate execution and cost model.

```sh
python main.py replay --start 2025-05-01 --tree --bars data/bars.csv
```

### Operating the private prospective ledger

Restore the latest private `market-forecast-operations.sqlite` checkpoint before each run. Use a single writer and version-checked checkpoint replacement. Preserve successful captures and audit records even when the forecast window was missed. Keep the database outside git. A fresh installation starts with `python operations.py capture`.

| Command | Effect |
| --- | --- |
| `python operations.py capture` | Poll each ticker, record the time, eligible article count or error, and retain hashed event IDs and derived scores. A reachable feed with zero eligible stories has an explicit zero. |
| `python operations.py issue` | Refresh daily prices and news, then commit a single premarket forecast batch with a shared issue time. Include daily and weekly price fits, the frozen first weekly price fit and the 50% control. The frozen parameters are saved in the same transaction as the first eligible batch. Include both news models when every ticker poll succeeds. |
| `python operations.py settle` | Fetch bars and resolve previously issued rows using the exchange calendar and close-plus-one-hour rule. |
| `python operations.py audit` | Report issued and settled counts and record an empty trading session after the cutoff as missed. |

A failed news poll suppresses the news comparison for the whole session. Retries retain the first complete forecast batch. Poll counts measure eligible items observed at polling time. They cannot establish continuous provider coverage between polls. The operations commands run on invocation.

For a private timer, invoke `python operations.py timed --db PATH` after restoring the checkpoint. Each invocation audits elapsed sessions and performs at most one due action. It prioritizes an unissued trading session from 09:00 until just before 09:25 New York time. Otherwise it settles eligible pending rows, including older sessions after downtime. Idle runs skip network access. The caller must retain ledger updates, including missed sessions and failed price checks, on error as well as success. This command runs once per invocation.

Use `python operations.py probe --db PATH` for an engineering check of live price retrieval. It verifies every configured symbol and benchmark has a bar for the latest eligible exchange session, records the attempt and issues zero forecasts.

Use `python weather.py --db PATH` with the private ledger to archive the registered pilot stations' decoded TAF groups, UTC issue and first-observed times, and bulletin fingerprints. A missing station produces a nonzero exit after retaining any stations received. This research command runs independently of forecast issuance.
