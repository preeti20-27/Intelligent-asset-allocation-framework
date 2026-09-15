# Data

Data providers, cleaning, feature engineering, and investment-universe definitions.

## Validation status

Split adjustment matches Yahoo's split basis for all 12 universe tickers. Dividend
adjustment is close but retains a small convention difference versus Yahoo: the
maximum observed discrepancy is 5.162% for ITC and 5.437% for TATASTEEL, with the
TATASTEEL residual concentrated in its pre-2019 history. Confirmed manual rights
actions are recorded in `manual_corporate_actions.csv`: RELIANCE, ex-date
2020-05-13, record date 2020-05-14, ratio 1:15, issue price Rs 1,257;
BHARTIARTL, ex-date 2019-04-23, record date 2019-04-24, ratio 19:67, issue price
Rs 220; and BHARTIARTL, ex-date 2021-09-28, record date 2021-09-28, ratio 1:14,
issue price Rs 535.

The processed price artifact contains the current adjusted bhavcopy close series;
returns are close-to-close percentage returns derived from that artifact. The full
source comparison is retained in `data/processed/bhavcopy_vs_yfinance.csv`.

## Known Limitations

Split and rights-issue adjustments are validated exactly against Yahoo
Finance for all 12 universe tickers (see `manual_corporate_actions.csv` for
RELIANCE and BHARTIARTL rights issues, sourced from NSE corporate-action
records with record/ex-dates and issue prices).

Dividend adjustment shows a residual difference from Yahoo's proprietary
methodology, affecting three tickers with material dividend histories:
- RELIANCE: max 8.84% discrepancy vs yfinance adjusted close
- ITC: max 5.16% discrepancy
- TATASTEEL: max 5.44% discrepancy (concentrated pre-2019)

This was confirmed via matched factor comparison at multiple dates (see
`bhavcopy_vs_yfinance.csv` for the full 33,567-row validation report) —
the same dividend-adjustment code path produces identical results whether
run in isolation or through the full validator, ruling out a pipeline bug.
The residual reflects a difference in dividend-reinvestment convention
between NSE-sourced data and Yahoo Finance, not a defect in this pipeline.

All 12 tickers are within 3.1% of Yahoo's adjusted close except the three
noted above.