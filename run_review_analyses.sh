#!/usr/bin/env bash
# Reproduce the review-round (rebuttal) analyses from the datasets shipped in this repo.
# All paths are relative to the repository root. Run from the repo root:
#     bash run_review_analyses.sh
#
# Outputs are written into results/review/ (JSON), overwriting the checked-in copies.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p results/review

echo "[1/3] Multi-dataset identity-vs-count robustness ..."
python analysis_review/multi_dataset_robustness.py \
    data/external/credit_risk_dataset.csv \
    data/external/gmsc_raw.parquet \
    data/external/taiwan_raw.parquet \
    data/external/adult_raw.parquet \
    data/polish/1year.arff \
    > results/review/multidata_results.json
echo "    -> results/review/multidata_results.json"

echo "[2/3] Downside-asymmetry (H3) robustness [CORRECT equal-slopes null] ..."
# Reads the Polish .arff files from data/polish (override with POL=... if elsewhere).
python analysis_review/h3_asymmetry_robustness.py \
    > results/review/h3_robustness.json
echo "    -> results/review/h3_robustness.json"

echo "[3/3] Referee round-2 extras (continuous base PD, 66-pair FDR, Study 2 band bootstrap) ..."
# Polish .arff files come from POL (default data/polish); the one positional arg is the ratings CSV.
python analysis_review/referee_round2_extra.py \
    data/corporate_rating.csv \
    > results/review/round2_extra.json
echo "    -> results/review/round2_extra.json"

echo "Done. See results/review/ for JSON outputs."
