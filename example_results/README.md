# Example benchmark outputs (`tg_cms_v4`)

This directory contains a **frozen snapshot** of the main benchmark run used to generate Supporting Information tables and figures without rerunning the full cross-validation job.

## Contents

```
tg_cms_v4/
  data/processed.csv, dataset_meta.json, raw.csv
  metrics/results.csv
  metrics/summary_frac1_alpha.csv
  metrics/similarity_test_to_train.csv
  metrics/shap_top_features_xgb.csv
  figs/shap_summary_xgb.png   (when copied from a local run)
  run_meta.json               (when available)
```

## Usage

Supporting Information scripts read from here when `CMS_USE_EXAMPLE_RESULTS=1` (default in `supporting_information/run_all.sh`).

To regenerate this snapshot after a new benchmark:

```bash
python scripts/run_all.py --config configs/tg.yaml
cp -R runs/tg_cms_v4/* example_results/tg_cms_v4/
```

Then set `CMS_USE_EXAMPLE_RESULTS=0` if you want scripts to read directly from `runs/`.
