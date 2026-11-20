# Data

This repository **includes** the datasets used in the analysis, under the paths below, so the notebooks
and the review-round scripts run without any manual download. Each dataset remains under its original
provider's license (listed per dataset); redistribution here is for reproducibility. If you plan to
publish this repository, confirm that each provider's license permits redistribution, or replace the
files with download instructions.

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

## External cross-sections (review-round replication) — `data/external/`

These firm-independent datasets are used by `analysis_review/multi_dataset_robustness.py` to show the
identity-over-count pattern is not a pooling artifact of the Polish panel. Each row is an independent
record (no repeated-firm panel).

| File | Rows | Target | Two-indicator pair tested | Source / license |
|---|---|---|---|---|
| `credit_risk_dataset.csv` | 32,581 | `loan_status` | `loan_percent_income`, `person_income` | Public consumer credit-risk dataset (Kaggle), CC0 |
| `gmsc_raw.parquet` | 150,000 | `DEFAULT` | `DEBT_RATIO`, `INCOME` | "Give Me Some Credit" (Kaggle competition), competition terms |
| `taiwan_raw.parquet` | 30,000 | `DEFAULT` | `LIMIT_BAL`, `PAY_0` | UCI "Default of Credit Card Clients" (Yeh & Lien 2009), CC BY 4.0 |
| `adult_raw.parquet` | 48,842 | `HIGH_INCOME` | `education-num`, `hours-per-week` | UCI "Adult" / Census Income — **non-credit control**, CC BY 4.0 |

The Adult set is a deliberate negative control: it is not a credit dataset, so it checks that the test
does not manufacture a spurious increment on arbitrary tabular data. Any synthetic leakage column that
some mirrors ship (e.g. a `VIP`-style field correlated with the target) is dropped explicitly inside the
script (`drop_cols=[...]`).

## Reproducibility note
The committed `results/` and `results/review/` outputs can be read without re-running anything. To
regenerate the review-round JSONs from the shipped data, run `bash run_review_analyses.sh` from the repo
root.
