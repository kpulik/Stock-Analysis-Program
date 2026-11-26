# Market Forecast Lab development plan

## Objective

Produce a probability before 09:25 New York time that a stock or index will outperform its benchmark from today's open through close. Keep a durable ledger of forecasts and evaluate forecast quality after the close. The initial universe spans tech, oil, energy, transportation, the Nasdaq Composite and the Nasdaq-100. Production research decisions use prospective data. The system has no order execution path.

## Milestones

| Stage | Deliverable and method | Acceptance evidence |
| --- | --- | --- |
| 1. Reproducible evaluation | Record input hashes, universe, dates, versions, coverage and paired model losses. Bootstrap blocks of entire sessions to preserve cross-stock dependence. | Re-running the same inputs reproduces the result. Reject duplicate, missing or mismatched target pairs. Publish a real benchmark with uncertainty. |
| 2. Historical news | Import dated, ticker-linked headlines and compare the unchanged price model with price plus headline count and direction. Preserve source IDs and conservative availability. | Complete a covered historical replay, report coverage by sector and test a later untouched period or live cohort before adopting a signal. |
| 3. Prospective operations | Persist licensed inputs or permitted derived records in durable private storage. Automate premarket snapshots, one forecast per session and post-close settlement. Expose feed freshness and late or missed runs. | Verify restart recovery, exchange holidays, early closes, DST and immutable issue times. Demonstrate five real sessions with durable evidence. |
| 4. Political disclosures | Pilot reported House and Senate transactions using public release timestamps, amendments, ownership and transaction direction. Shrink member-level track records toward a pooled average when samples are small. | Trace sampled rows to public filings. Use only previously resolved outcomes. Compare against the same dates with prices and news. |
| 5. Weather exposure | Start with archived temperature forecast revisions for energy demand and storm forecasts at airline hubs. Version each company's location weights and forecast issue time. | Reproduce two historical releases. Quantify coverage and isolated model impact by industry on matched sessions. |
| 6. Model selection | Compare regularized linear models, a tree model and a frozen incumbent. Register each candidate, training cutoff and hypothesis before evaluation. | Publish Brier, calibration, coverage, sector breakdown and uncertainty. Retain the incumbent when evidence is inconclusive. |
| 7. Controlled adaptation | Refit candidates each trading session on previously resolved outcomes, serve in shadow alongside the incumbent, log versions and support rollback. Expand the universe with dated membership, delistings and ticker changes before making sector-wide claims. | Candidate and incumbent have at least 60 common prospective sessions. Predeclared paired loss and calibration gates pass. Reproduce rollback from the ledger. |

## Decision rules

- Primary metric is the mean Brier loss across symbols within each session, then across sessions. Compare the exact same symbol-date outcomes. Also show the pooled Brier score, reliability by probability bin, and coverage to diagnose the result.
- Use the fixed 50% forecast, the weekly price model and its frozen version as controls. Compare each proposed feature set to its immediate predecessor. Record negative results and how many variants were tried.
- For the first pilot use a five-session moving-block bootstrap, 5,000 draws and a fixed seed. Report a paired interval for candidate minus control. Negative values favor the candidate. The interval describes sampling variation under this scheme. It does not correct for repeated model selection.
- The initial promotion proposal requires the entire 95% paired interval to favor the candidate, lower mean Brier than the constant control, and no deterioration in absolute mean calibration error greater than 0.02. Lock these rules before the prospective cohort. Treat borderline outcomes as inconclusive. Re-examine sector-level failure modes and longer block lengths before expanding scope.
- Start research in the fixed starter basket. Track missing data separately from neutral headlines. Do not claim full-universe performance from surviving tickers. Keep vendors' raw news and credentials out of the public repository.

## First registered experiment

Train from January 2017. Evaluate all available 2019 sessions in the starter basket, with no model or phrase-list tuning after scoring. Compare prices plus news to prices alone and report both control baselines. Use Miguel Aenlle's historical headlines and make the recorded calendar date eligible at midnight New York time the following day. This sacrifices intraday precision and does not reconstruct revisions or actual delivery latency. Reserve 2020 and prospective dates for separate decisions. If a source is unavailable, record the failure and use a clearly identified alternate dataset or window.

## Engineering cadence

Advance engineering every calendar day. Use weekends and exchange holidays for research, reliability and testing. Work in tested, reviewable commits. Update the milestone evidence and record any unresolved data or operational dependency in this file. Continue from the first open acceptance criterion. Keep research data collection and model release decisions separate from trading.

## Milestone evidence and next work

- Stage 1 delivered the paired evaluator, reproducibility manifest, block bootstrap and calibration output. [Experiment 001](research/README.md) records the first application.
- Stage 2 delivered a keyless historical archive importer and the first covered replay. Prices plus news failed the promotion criteria. Preserve this result. The next confirmation requires prospective dates or an independently fixed later cohort.
- **Next engineering milestone is Stage 3.** Establish durable persistence for forward forecasts and source health, then run the price incumbent, a constant control and the news challenger on identical sessions. Persist an explicit source-coverage flag so outages cannot silently appear as neutral news. Operational acceptance requires real elapsed sessions.
- In parallel with collecting those sessions, take Stage 4 through a small verified disclosure dataset, then Stage 5 through an archived forecast pair. The first evaluation for each feed must include sample-level provenance and a matched no-feed comparison.

### Experimental challengers

- The prospective LLM candidate adds a ticker-specific direction and completed-headline count to the rule-based news model. Keep the endpoint, model weights, prompt, universe and scoring policy fixed within a cohort. Test label fidelity on human-reviewed examples and collect fresh outcomes before considering promotion. Current LLM features start at inference completion.
- The shallow price tree and fixed 45%/55% abstention rule are available for shadow runs. On the previously inspected May through August 2025 development window, the tree's Brier score was 0.252312 on 1,176 forecasts, compared with 0.250682 for logistic regression and 0.250000 for the constant control. Keep the tree experimental. This comparison does not change the daily-operations priority.

The Qwen2.5-0.5B-Instruct Q4 smoke run processed 26 real article-symbol pairs. In a separate four-headline diagnostic it classified a guidance cut, a call scheduling notice and an unrelated story as positive. The model is unsuitable for deployment based on this check. Require a blinded, ticker-specific label audit before promoting any text extractor.

A second local smoke test used Qwen2.5-3B-Instruct Q4_K_M, weight revision `7dabda4d13d513e3e842b20f0d435c732f172cbe`, with the same prompt. It matched the expected directions on the same four examples and four additional synthetic headlines covering contract gains, losses, a meeting notice and falling profits. This small diagnostic checks basic behavior only. Compare ticker relevance, class balance, coverage and real prospective Brier scores before release.

**2026-11-20 prospective operations.** Shipped private-ledger feed capture, first-batch issuance with an explicit 50% control, and settlement through the existing exchange-calendar rules. Captured 143 distinct article-symbol signals from 14 successful ticker polls. Zero forecasts were issued for November 20. A late issue attempt was rejected and the session was recorded as missed. The private database checkpoint passed SQLite integrity checking. A restart test issued and settled synthetic rows and verified that a retry after an outage cannot add a news challenger to an earlier price-only batch. The 2019 benchmark is unchanged.

**November 20 handoff.** Connect the timed issuer and settler to the versioned private checkpoint and verify restoration before counting the first real session. Continue recording all misses. Compare daily and weekly price and news forecasts only on common sessions with successful source checks; keep the existing frozen historical control. Add a persisted frozen prospective model before evaluating the promotion gate. A successful zero-count feed response measures a poll, not complete news coverage. Automatic timing and five real settled sessions remain outstanding.

### November 21, 2026. Prospective controls and checkpoint recovery

- Restored the private SQLite checkpoint, verified the November 20 miss, and saved version 1 after 14 successful Saturday polls. They returned 130 eligible article-symbol observations, including two polls with zero items. SQLite integrity passed. The real-clock issuer rejected Saturday. The ledger contains zero real forecasts and zero settled sessions.
- The first successful operational batch now pins the weekly price model parameters and training version in the forecast transaction. Starting at 09:25:00 New York time, issuance is closed. A missed deadline rolls back the forecasts and initial fit together. An older issued batch stays intact on retry. Later sessions use the saved scaler and logistic coefficients or the original single-class probability.
- Verification: 24 tests passed. A restart and revised historical bars reproduced the original frozen fit's predictions within relative tolerance 1e-12. A separate synthetic end-to-end check issued 12 forecasts over two instruments and six models, then settled all 12 in a fresh process. These observations are excluded from prospective evidence.
- Before the first real batch, lock daily price versus weekly price as the refit comparison, with frozen prices and 50% alongside. Compare daily news versus weekly news on the intersection of healthy, settled instrument-days and daily news versus daily prices on those same rows. Keep the 60-session promotion gate and the 2019 benchmark.

### Weather pilot source check

The first two sampled NHC Helene forecast advisories each supply a nine-hour-ahead wind forecast. Times below are UTC. On September 25, 2024 the research cutoff was 13:25 UTC.

| Advisory | Bulletin issue time | Forecast valid time | Maximum wind forecast | Before that day's cutoff by bulletin time? |
| --- | --- | --- | ---: | --- |
| [8](https://www.nhc.noaa.gov/archive/2024/al09/al092024.fstadv.008.shtml) | Sep 25 09:00 | Sep 25 18:00 | 70 kt | Yes |
| [9](https://www.nhc.noaa.gov/archive/2024/al09/al092024.fstadv.009.shtml) | Sep 25 15:00 | Sep 26 00:00 | 85 kt | No |

These have different valid times. A revision comparison requires the same valid time and location. The archive supplies bulletin times. Live use also requires first-observed timestamps. Before scoring weather, fix airline hub exposures and compare the immediate predecessor with and without the weather feature on the exact same settled airline-days. No weather outcomes have been scored.

**Next action.** Connect a timed premarket issuer and post-close settler to the private ledger with version-checked writes. Verify an unattended restore, collect five real settled sessions, and keep auditing misses. The weekend restore verifies storage only. A real feed outage and live model comparisons remain unobserved.

### November 22, 2026. Timed dispatch and recovery after downtime

- Added a single-invocation timed command for a restored private ledger. It selects premarket issuance, eligible post-close settlement, or idle using the exchange clock. It records price retrieval failures and catches missed sessions across restarts using the actual observation timestamp. A closed day can settle older eligible forecasts.
- Verified 26 tests, including the UTC window on both sides of the fall DST transition. A synthetic holiday and early-close run recovered from a price timeout, recorded one missed session, issued 12 forecasts once and settled all 12 an hour after the early close. The real Sunday command was idle. All 14 news polls succeeded with 80 eligible article-symbol observations and six empty results. SQLite integrity passed and private checkpoint version 2 retained the Friday miss and zero real forecasts.
- Predeclared the [two-airport pilot](research/weather-airports-v1.csv). For each airline use a single equally weighted station proxy. Encode one feature for an explicit thunderstorm forecast in any applicable prevailing or temporary TAF group overlapping that trading session. Compare daily prices plus news with and without this binary feature on identical settled airline-days. An absent bulletin or incomplete session coverage excludes the pair. Require original issue and first-observed times before issuance, and retain amendments as separate versions. These weights are experimental choices. The next weather step is to verify archived groups and effective times before producing model inputs. [National Weather Service station reference](https://www.weather.gov/media/aviation/NWS%20TAF%20Sites.pdf)
- Next operational step is to connect the command to an always-available private runner with guarded checkpoint replacement, then verify a real issue and settlement. The timed command itself does not provision a scheduler. Five settled sessions remain the operational acceptance milestone.

### November 23, 2026. Live price readiness and first airport captures

- The restored ledger recorded Monday as missed at 12:20 New York time, after the 09:25 cutoff. It now contains two missed sessions, zero issued forecasts and zero outcomes. There was no eligible prospective daily-refit comparison.
- The new engineering price probe completed a real download of 13,068 daily bars and verified the November 20 open and close for all 18 required instruments. Operational issuance now checks required-session coverage before recording price retrieval as healthy. All 14 news polls succeeded with 154 eligible article-symbol observations, including one empty poll. Private checkpoint version 3 passed SQLite integrity checking.
- Archived the first two airport TAFs with bulletin fingerprints, original issue times, first observation and effective group intervals. Each had four decoded groups. Atlanta was issued at 16:04 UTC and Chicago at 14:54 UTC. Both were observed at 17:20 UTC, after the 14:25 UTC cutoff. Neither qualified for Monday. Keep the registered matched airline comparison. Before fitting it, verify group overlap against regular and early closes, amendments, probability groups and missing coverage. [Aviation Weather Center API](https://aviationweather.gov/data/api/)
- All 28 tests passed. Regression checks reject a stale benchmark even when its historical download succeeds and retain distinct bulletin revisions across repeated captures. End-to-end price, news, weather and clock-based audit commands ran against the restored private checkpoint.
- Next operational action is a private runner with access to the guarded checkpoint during the 09:00 to 09:25 New York issue window and after close plus one hour. A real unattended invocation is outstanding. Track the first five settled sessions before accepting operations and retain the 60-session promotion gate.

### November 24, 2026. Weather eligibility at the premarket cutoff

- Recorded Tuesday as the third missed session. Zero real forecasts or outcomes are available. The price probe retrieved 13,086 daily bars and verified all 18 required instruments for Monday. Fourteen successful news polls yielded 157 eligible article-symbol observations. Private checkpoint version 4 passed SQLite integrity checking.
- Shipped a read-only reconstruction of the registered airline weather input. It uses the latest eligible observed revision, its digest and the exchange session's actual open and close. Gaps in prevailing coverage, unsupported groups or ambiguous revisions exclude the row. Temporary thunderstorm groups cannot fill missing prevailing coverage. Probability-only groups are excluded, and explicit vicinity thunderstorms count. This fixes the coding rules before measuring weather outcomes. [AWC TAF product guidance](https://www.connect.aviationweather.gov/help/data/)
- Real end-to-end check: Monday had zero eligible archived bulletins at the cutoff. On Tuesday both earlier bulletins expired at 18:00 UTC, before the 21:00 UTC close. Both Tuesday replacements were first observed after the cutoff. This yields zero fully covered airline-days across four candidates, with no weather value imputed and no change to the historical scores. All 30 tests passed, including early close, gaps, amendments, probability groups and cutoff equality.
- Next operational action: give one persistent private runner the current checkpoint and sole writer responsibility. Invoke the timed command early in the 09:00 to 09:25 New York window and again after close plus one hour. Capture the two airport stations before the issue-time snapshot. Verify durable readback of the first issued batch before counting a settled session. Keep diagnostics and late engineering checks outside the prospective sample.

### November 25, 2026. Matched post-close comparisons

- Operational settlement now reports the five registered comparisons on complete, identically issued batches. The standalone reader can reproduce them without modifying the ledger. Sessions with pending outcomes wait until complete. Price-only sessions contribute to price comparisons. News comparisons require both news variants. Report paired counts, session-weighted and pooled Brier loss, absolute mean calibration error and the registered bootstrap interval once five sessions are available. No automatic promotion.
- The synthetic timeout, retry and early-close settlement check produced 12 settled rows and five comparisons over the same two symbols. A separate mixed-coverage test verified that adding a price-only session does not change the news comparison. All 31 tests passed.
- Wednesday was the fourth recorded miss. Restoring and auditing the private checkpoint returned zero real forecasts, zero pending sessions and zero settled sessions. All 14 news polls succeeded with 149 eligible observations. The price probe verified November 24 bars for all 18 instruments. Two fresh airport bulletins were archived. At the issuance cutoff neither station had full-session coverage. SQLite integrity passed and guarded checkpoint replacement produced version 5.
- Next operational acceptance step is a private always-on writer able to restore and persist the checkpoint in the premarket window and at close plus one hour. Verify a real issue and settlement there before counting the first of five sessions. Continue collecting airport coverage independently. The 2019 benchmark and 60-session prospective gate remain in force.

### November 26, 2026. Weather evidence at actual issuance

- Each first operational forecast batch now commits the registered airline inputs in the same SQLite transaction as its forecasts. Preserve the input timestamp, station weight, bulletin fingerprint, storm flag and coverage status. Retries keep the original evidence. Deadline rollback discards both. The research snapshot does not change model inputs or predictions.
- Thanksgiving audit returned idle with four prior misses, zero real forecasts and zero settled sessions. All 14 news polls succeeded with 135 eligible observations. Prices covered November 25 for all 18 instruments. Checkpoint version 6 passed SQLite integrity checking.
- Both airport bulletins captured Thursday covered Friday's scheduled trading hours in a diagnostic and yielded a zero storm flag. This checks coverage only. Friday's eligibility must be determined again at its actual input time. [NYSE calendar](https://www.nyse.com/trade/hours-calendars)
- All 32 tests passed. An amendment observed after a synthetic snapshot but before the deadline was excluded. Rollback removed the snapshot. The end-to-end outage and early-close check issued and settled 12 synthetic forecasts, preserved two explicit missing weather inputs and produced all five matched comparisons.
- Next: connect the single private ledger writer for a genuine premarket issuance, confirm the durable snapshot and score after close plus one hour. Friday's early close makes 14:00 Eastern the earliest eligible settlement. Keep the existing 60-session promotion criteria, matched airline comparison and 2019 benchmark.
