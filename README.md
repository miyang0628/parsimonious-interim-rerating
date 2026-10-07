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
ratings; the other has ratings but no default outcome), so together they form an internal / external
validity pair.

## Hypotheses

- **H1 / H2** — a 2-indicator signal adds statistically significant default information beyond a base
  grade built from the *other* features (incremental AUC; DeLong and likelihood-ratio tests).
- **H3** — the incremental information is **asymmetric**: deterioration predicts default more strongly
  than improvement protects against it.
- **H4** — the incremental value is **largest in the middle band** (B/BB analog) and small at the extremes.
- **H5** — in rating recovery, the information lost by using only two indicators is **larger at the
  extremes** than in the middle, which makes excluding the extremes from interim re-rating defensible.

## Repository structure

```
parsimonious-interim-rerating/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md              # data provenance + download instructions (raw data not redistributed)
│   ├── polish/                # place 1year.arff … 5year.arff here
│   └── corporate_rating.csv   # place the labeled corporate-rating CSV here
├── notebooks/
│   ├── 01_eda.ipynb           # distributions, missingness, band structure, signal checks
│   ├── 02_study1_polish.ipynb # H1–H4 + pseudo-time gradient (incremental default prediction)
│   ├── 03_study2_ratings.ipynb# rating recovery, ordered-logit inference, migration matrix
│   └── 04_robustness.ipynb    # bucket sensitivity, regularized ordinal, indicator-choice swap
└── results/
    ├── figures/               # png + pdf, grayscale, 600 dpi
    └── tables/                # csv
```

## Reproduce

1. Python ≥ 3.11. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Obtain the two public datasets and place them as described in [`data/README.md`](data/README.md).
3. Run the notebooks in order (`01` → `04`). Each notebook is a **single self-contained cell**; run it
   top-to-bottom. From the command line:
   ```
   cd notebooks
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 *.ipynb
   ```
   Outputs are written to `results/figures/` (png + pdf, grayscale, 600 dpi) and `results/tables/` (csv).

Random seeds are fixed (`random_state=0`, 5-fold stratified CV), so results are deterministic.

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
