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
| 7. Controlled adaptation | Train candidates weekly, serve in shadow alongside the incumbent, log versions and support rollback. Expand the universe with dated membership, delistings and ticker changes before making sector-wide claims. | Candidate and incumbent have at least 60 common prospective sessions. Predeclared paired loss and calibration gates pass. Reproduce rollback from the ledger. |

## Decision rules

- Primary metric is the mean Brier loss across symbols within each session, then across sessions. Compare the exact same symbol-date outcomes. Also show the pooled Brier score, reliability by probability bin, and coverage to diagnose the result.
- Use the fixed 50% forecast, the weekly price model and its frozen version as controls. Compare each proposed feature set to its immediate predecessor. Record negative results and how many variants were tried.
- For the first pilot use a five-session moving-block bootstrap, 5,000 draws and a fixed seed. Report a paired interval for candidate minus control. Negative values favor the candidate. The interval describes sampling variation under this scheme. It does not correct for repeated model selection.
- The initial promotion proposal requires the entire 95% paired interval to favor the candidate, lower mean Brier than the constant control, and no deterioration in absolute mean calibration error greater than 0.02. Lock these rules before the prospective cohort. Treat borderline outcomes as inconclusive. Re-examine sector-level failure modes and longer block lengths before expanding scope.
- Start research in the fixed starter basket. Track missing data separately from neutral headlines. Do not claim full-universe performance from surviving tickers. Keep vendors' raw news and credentials out of the public repository.

## First registered experiment

Train from January 2017. Evaluate all available 2019 sessions in the starter basket, with no model or phrase-list tuning after scoring. Compare prices plus news to prices alone and report both control baselines. Use Miguel Aenlle's historical headlines and make the recorded calendar date eligible at midnight New York time the following day. This sacrifices intraday precision and does not reconstruct revisions or actual delivery latency. Reserve 2020 and prospective dates for separate decisions. If a source is unavailable, record the failure and use a clearly identified alternate dataset or window.

## Engineering cadence

Work in tested, reviewable commits. Update the milestone evidence and record any unresolved data or operational dependency in this file. Continue from the first open acceptance criterion. Keep research data collection and model release decisions separate from trading.

## Milestone evidence and next work

- Stage 1 delivered the paired evaluator, reproducibility manifest, block bootstrap and calibration output. [Experiment 001](research/README.md) records the first application.
- Stage 2 delivered a keyless historical archive importer and the first covered replay. Prices plus news failed the promotion criteria. Preserve this result. The next confirmation requires prospective dates or an independently fixed later cohort.
- **Next engineering milestone is Stage 3.** Establish durable persistence for forward forecasts and source health, then run the price incumbent, a constant control and the news challenger on identical sessions. Persist an explicit source-coverage flag so outages cannot silently appear as neutral news. Operational acceptance requires real elapsed sessions.
- In parallel with collecting those sessions, take Stage 4 through a small verified disclosure dataset, then Stage 5 through an archived forecast pair. The first evaluation for each feed must include sample-level provenance and a matched no-feed comparison.
