# Experiment 001. Can historical headlines improve same-day forecasts?

**Decision.** Keep the news model in research. On 252 sessions in 2019 it had a Brier loss of 0.250340. Prices alone scored 0.250248 and a constant probability of 0.5 scored 0.250000. The matched news-minus-price difference was +0.000092 with a 95% five-session block bootstrap interval of [-0.000092, +0.000282]. A positive difference means a worse forecast.

**Inputs.** The starter universe comprised 12 companies and two indexes. The price pull covered January 2017 through December 2019. The archive importer found 4,019 ticker-linked headlines for seven companies over that period. Existing phrase rules scored 195 as directional. Headline count contributes for neutral items as well. The date-only replay assigned the next calendar midnight in New York as the earliest eligible time. Preprocessing used ticker, publisher date, URL and headline only. The source publisher cautions that the raw timestamps have mixed precision. [Source archive](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests)

**Coverage.** Headlines were available on 627 of 3,528 evaluated instrument-days (17.8%). The energy group had coverage on 298/1,008 observations, transport on 178/1,008, tech on 151/1,008, and the two indexes on 0/504. The covered subset also favored the price-only model numerically. Cross-stock dependence, sparse coverage, unknown revisions, a fixed present-day basket, and the 24-hour lookback limit the interpretation. Market holidays and weekends can push older stories outside the window. This experiment cannot establish how a richer text model or broader news archive would perform.

**Method.** The first weekly fit used historical data from before the evaluation week. Weekly refits used labels resolved before each week's Monday. Each model used the same target, instrument-days, dates and benchmarks. All reported portfolio-wide averages give each session equal weight. Full scores, calibration bins, subgroup intervals, package versions and input hashes are in [the machine-readable report](news-2019.json). Subgroup intervals are exploratory.

## Reproduction

From the repository root with the project requirements installed, run the following. The download command uses Yahoo historical daily bars. Archive and price vendors may revise downloads. Check the input hashes when comparing results.

```sh
mkdir -p data
curl -fLsS 'https://www.kaggle.com/api/v1/datasets/download/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests' -o data/headlines.zip
python archive_news.py data/headlines.zip --start 2017-01-01 --end 2020-01-01
python main.py download --start 2017-01-01 --end 2020-01-01 --out data/bars-2017-2019.csv
python main.py replay --bars data/bars-2017-2019.csv --events data/archive-news.csv --start 2019-01-01 --out data/replay-2019.csv
python evaluate.py --bars data/bars-2017-2019.csv --events data/archive-news.csv --replay data/replay-2019.csv --archive data/headlines.zip --out data/news-2019.json
```

The 2019 results are now development evidence. Reserve later dates for experiments specified before viewing their outcomes.
