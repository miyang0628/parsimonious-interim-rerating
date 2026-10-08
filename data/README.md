# Data

The raw datasets are **not redistributed** in this repository (see each provider's license).
Download them and place the files exactly as below; the notebooks read from these paths.

## Study 1 — Polish companies bankruptcy
- Source: UCI Machine Learning Repository, dataset #365 ("Polish companies bankruptcy data"),
  Zieba, Tomczak & Tomczak (2016). Licensed CC BY 4.0.
- Files: five ARFF files for forecast years 1–5.
- Place as:
  ```
  data/polish/1year.arff
  data/polish/2year.arff
  data/polish/3year.arff
  data/polish/4year.arff
  data/polish/5year.arff
  ```
- Each file: 64 financial ratios (`Attr1`…`Attr64`) + a binary bankruptcy `class`.
  Study 1 uses `Attr1` (net profit / total assets, profitability) and `Attr2`
  (total liabilities / total assets, leverage) as the two proxy indicators.

## Study 2 — Corporate credit ratings
- Source: OpenML dataset #43344 ("Corporate-Credit-Rating"); original compilation from
  FinancialModelingPrep / public filings. 2,029 firm-ratings, 2005–2016, multiple agencies.
- Place as:
  ```
  data/corporate_rating.csv
  ```
- Required columns: `Rating` (letter grade AAA…D), `Rating Agency Name`, `Sector`, and 25 financial
  ratios including `debtRatio`, `returnOnAssets`, `currentRatio`, `assetTurnover`.
- Note: the `Rating` column must be populated (some mirrors ship a feature-only copy with an empty
  `Rating` column — that version will not work).

## Reproducibility note
No raw data is required to read the committed `results/` outputs; data is only needed to re-run the
notebooks.
