"""Repository-relative paths for Supporting Information analysis scripts."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT
ROOT = REPO_ROOT

RUN_TAG = os.environ.get("CMS_RUN_TAG", "tg_cms_v4")
_use_example = os.environ.get("CMS_USE_EXAMPLE_RESULTS", "").lower() in ("1", "true", "yes")
RUN = (
    REPO_ROOT / "example_results" / RUN_TAG
    if _use_example
    else REPO_ROOT / "runs" / RUN_TAG
)

OUT = REPO_ROOT / "supporting_information" / "outputs"
FINAL = OUT / "figures_final"

CONFIG_YAML = BENCHMARK / "configs" / "tg.yaml"
PROCESSED_CSV = RUN / "data" / "processed.csv"
DATASET_META_JSON = RUN / "data" / "dataset_meta.json"
RESULTS_CSV = RUN / "metrics" / "results.csv"
SUMMARY_CSV = RUN / "metrics" / "summary_frac1_alpha.csv"
SIMILARITY_CSV = RUN / "metrics" / "similarity_test_to_train.csv"
XGB_SHAP_CSV = RUN / "metrics" / "shap_top_features_xgb.csv"
XGB_SHAP_FIG = RUN / "figs" / "shap_summary_xgb.png"
INTERVALS_CSV = OUT / "TableS8_per_sample_conformal_intervals_frac1_alpha010.csv"
S8_MANIFEST = OUT / "TableS8_FigureS5_analysis_manifest.json"
