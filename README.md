# CMS Tg Pipeline v5 (XGB + SVR + alpha sweep + similarity diagnostics + SHAP)

## Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

## Configure
Edit configs/tg.yaml and set data.data_csv.

## Run
python scripts/run_all.py --config configs/tg.yaml

## Outputs
runs/<run_tag>/
  data/raw.csv, data/processed.csv
  splits/*.json
  metrics/results.csv
  metrics/summary_frac1_alpha.csv
  metrics/similarity_test_to_train.csv
  metrics/shap_top_features_xgb.csv (if enabled)
  figs/coverage_width_curve_frac1_xgb.png
  figs/coverage_width_curve_frac1_svr.png
  figs/similarity_violin.png
  figs/similarity_ecdf.png
  figs/shap_summary_xgb.png (if enabled)


