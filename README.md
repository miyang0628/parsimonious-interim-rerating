# parsimonious-interim-rerating

Replication code for the manuscript **"The Information Value of Parsimonious Interim Re-rating"**
(anonymous author(s), manuscript under review).

All analyses use **public datasets** and are fully reproducible. The study asks whether a
*parsimonious* interim re-rating — a credit grade update that uses only **two financial indicators**
laid over an annual grade built from the full financial statement — carries information, and how that
value varies across the rating spectrum.

Conceptually, the annual full-statement grade is a **through-the-cycle (TTC)** assessment and the
two-indicator interim update is a low-cost **point-in-time (PIT)** overlay. The question is whether the
cheap PIT overlay adds default-relevant information beyond the TTC grade, and for which firms.

## Two studies

| | Study 1 | Study 2 |
|---|---|---|
| Data | Polish companies bankruptcy (default outcome) | Corporate credit ratings (letter grades) |
| Question | Does a 2-indicator signal add **default-prediction** power beyond a base grade? | How much rating information do 2 indicators **recover**, and where is the loss largest? |
| Outcome | bankruptcy label | letter grade (AAA…CCC) |

The two studies use different data with complementary strengths (one has a default outcome but no
ratings; the other has ratings but no default outcome). They share no firms and differ in country,
period, and outcome, so they provide complementary, corroborating evidence rather than two estimates
of one quantity.

## Hypotheses

- **H1 / H2** — a 2-indicator signal adds statistically significant default information beyond a base
  grade built from the *other* features (incremental AUC; DeLong and likelihood-ratio tests).
- **H3** — the incremental information may be **asymmetric** (deterioration more informative than
  improvement). *Revised status:* under the correct equal-slopes contrast this is **not** statistically
  established and is reported as suggestive only (see [Review-round additions](#review-round-additions)).
- **H4** — the incremental value is **largest in the middle band** (B/BB analog) and small at the extremes.
- **H5** — in rating recovery, the information lost by using only two indicators is **larger at the
  extremes** than in the middle, which makes excluding the extremes from interim re-rating defensible.

> **What the final manuscript claims.** The central result is sharpened to **identity, not count**:
> *which* two indicators are used — their informativeness — is the first-order source of specification
> risk, while a permutation test finds the broad accounting category (solvency vs liquidity/activity)
> does **not** predict a pair's rank. See the review-round section below.

## Repository structure

```
parsimonious-interim-rerating/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md              # data provenance + licenses
│   ├── polish/                # 1year.arff … 5year.arff (Polish bankruptcy, Study 1)
│   ├── corporate_rating.csv   # labeled corporate-rating CSV (Study 2)
│   └── external/              # firm-independent cross-sections for the replication check
│       ├── credit_risk_dataset.csv
│       ├── gmsc_raw.parquet
│       ├── taiwan_raw.parquet
│       └── adult_raw.parquet  # non-credit control
├── notebooks/                 # original analysis (run in order 01 → 05)
│   ├── 01_eda.ipynb
│   ├── 02_study1_polish.ipynb
│   ├── 03_study2_ratings.ipynb
│   ├── 04_robustness.ipynb
│   └── 05_additional_experiments.ipynb
├── analysis_review/           # review-round (rebuttal) scripts — added after peer review
│   ├── multi_dataset_robustness.py   # identity-vs-count across 5 datasets; DeLong; placebo
│   ├── h3_asymmetry_robustness.py    # H3 under the CORRECT equal-slopes null (fixed)
│   └── referee_round2_extra.py       # continuous base PD; 66-pair FDR; Study 2 band bootstrap CI
├── run_review_analyses.sh     # one command to reproduce all review-round results from shipped data
├── manuscript/                # final submitted LaTeX source + compiled PDF
│   ├── main.tex
│   ├── references.bib
│   └── main.pdf
└── results/
    ├── figures/               # png + pdf, grayscale, 600 dpi
    ├── tables/                # csv
    └── review/                # review-round JSON outputs (written by run_review_analyses.sh)
        ├── multidata_results.json
        ├── round2_extra.json
        ├── h3_robustness.json
        ├── h3_correct_null.json      # corrected asymmetry test (p = 0.109, not significant)
        └── category_permutation.json # solvency vs liquidity/activity (p = 0.80)
```

## Reproduce

1. Python ≥ 3.11. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Obtain the two public datasets and place them as described in [`data/README.md`](data/README.md).
3. Run the notebooks in order (`01` → `05`). Each notebook is a **single self-contained cell**; run it
   top-to-bottom. From the command line:
   ```
   cd notebooks
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 *.ipynb
   ```
   Outputs are written to `results/figures/` (png + pdf, grayscale, 600 dpi) and `results/tables/` (csv).

Random seeds are fixed (`random_state=0`, 5-fold stratified CV), so results are deterministic.

### Review-round analyses

The rebuttal analyses run directly from the datasets shipped in this repository — no manual paths
needed:

```
bash run_review_analyses.sh
```

This writes `results/review/*.json`. The three scripts can also be run individually; each one's argument
order is documented in its header docstring. (`POL` defaults to `data/polish`; override with
`POL=/path/to/polish` if your Polish files live elsewhere.)

## Review-round additions

These were added after peer review and are the basis for the final manuscript's sharpened claims.

- **External replication across firm-independent cross-sections** (`multi_dataset_robustness.py`). The
  Polish panel pools repeated-firm observations with no firm identifier, so the increment could in
  principle be a clustering artifact. Re-running the identity-vs-count test on four row-independent
  datasets (a consumer-loan set, "Give Me Some Credit", a Taiwan card set, and a non-credit Adult
  control) shows the informative pair keeps adding discrimination while random "placebo" pairs do not —
  the pattern is not a pooling artifact.
- **Correct asymmetry test** (`h3_asymmetry_robustness.py`, `results/review/h3_correct_null.json`). An
  earlier version tested the Wald null `γ⁺ + γ⁻ = 0`, which merely restates "the signal predicts on
  average" and is **not** a test of asymmetry. The correct equal-slopes contrast `γ⁺ − γ⁻ = 0` is **not
  rejected** (*p* = 0.109); the downside slope partly reflects the larger dispersion of `d⁺`
  (SD 0.052 vs 0.031). The manuscript now reports the downside pattern as suggestive only. The wrong
  contrast is kept in the code as a commented line so the correction is transparent.
- **Informativeness, not category** (`results/review/category_permutation.json`). A permutation test on
  the broad accounting labels does **not** find the solvency category to outrank liquidity/activity
  (diff = −2.8, *p* = 0.80); the top pairs are driven by one high-information working-capital ratio
  (Attr3), and a pure-leverage pair ranks near the bottom. The headline is therefore "identity
  (informativeness), not count", not "solvency beats liquidity".
- **Multiple-testing control and continuous-base-PD benchmark** (`referee_round2_extra.py`). The 66-pair
  ordering survives Bonferroni and Benjamini–Hochberg FDR; the pooled increment is not robust to a
  continuous (non-discretized) base PD, which the manuscript states plainly.

## Figures

All figures are grayscale, 600 dpi, exported in both PNG and PDF, with **no captions or titles**
(captions are provided in the manuscript). Axis labels and legends are embedded.

## Method notes

- Study 1 builds the base grade from the 62 features **excluding** the two proxy indicators, so the
  proxy signal tested for increment is genuinely new information.
- Incremental discrimination is tested with the **DeLong** test on paired out-of-fold ROC AUCs and a
  **likelihood-ratio** test on nested logits.
- Study 2 measures rating recovery with a nonparametric random forest and tests the two proxies with a
  low-dimensional ordered logit (agency fixed effects). The full-feature ordinal model with full fixed
  effects is estimated with an **L2-penalized proportional-odds** model (hand-rolled) to ensure
  convergence; conclusions are unchanged across estimators.
- "Mid band" = BBB/BB/B (and the quantile-grade analog in Study 1); "extremes" = the rest.
- `05_additional_experiments.ipynb` adds supplementary analyses supporting the manuscript: a
  leakage-free nested re-estimation (grade bucket edges from training folds only), economic-significance
  metrics (IDI, net reclassification, decision-value cost reduction), an all-pairs indicator analysis
  (66 pairs across four ratio categories), a formal Wald test of the H3 asymmetry, class-imbalance
  robustness, and bootstrap confidence intervals for the rating-recovery kappa.

## License

Code is released under the MIT License (see `LICENSE`). The datasets are distributed under their own
licenses by their original providers; see `data/README.md`.

## Citation

```
@unpublished{anon_interim_rerating,
  title  = {The Information Value of Parsimonious Interim Re-rating},
  author = {Anonymous},
  note   = {Manuscript under review},
  year   = {2026}
}
```
