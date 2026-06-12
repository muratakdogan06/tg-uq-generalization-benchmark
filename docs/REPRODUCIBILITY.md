# Reproducibility guide

This document summarizes how manuscript figures and tables map to repository artifacts.

## Main-text analyses (benchmark pipeline)

Run:

```bash
python scripts/run_all.py --config configs/tg.yaml
```

| Analysis | Source module / output |
|----------|------------------------|
| Learning-curve point metrics | `src/cms_tg/eval.py` → `metrics/results.csv` |
| Split-conformal coverage & width | `src/cms_tg/uncertainty.py` → `results.csv` (α columns) |
| S<sub>max</sub> distributions | `src/cms_tg/similarity.py` → `metrics/similarity_test_to_train.csv` |
| XGBoost SHAP (Figure 7A) | `src/cms_tg/shap_analysis.py` → `metrics/shap_top_features_xgb.csv` |
| Stratified / scaffold / cluster splits | `src/cms_tg/splits.py` |

Default run tag: `tg_cms_v4` (`configs/tg.yaml`).

## Supporting Information

```bash
bash supporting_information/run_all.sh
```

| Script | Outputs |
|--------|---------|
| `generate_table_s5_scaffold_characterization.py` | Table S5 |
| `generate_table_s6_figure_s2_cluster_characterization.py` | Table S6, Figure S2 |
| `generate_figure_s3_fingerprint_sensitivity.py` | Figure S3 |
| `generate_table_s7_analysis.py` | Table S7 (paired SVR vs XGBoost; α=0.10 for width) |
| `generate_table_s8_figure_s5.py` | Table S8, Figure S5 (subgroup coverage; α=0.10) |
| `generate_figure7_table_s11_svr_permutation.py` | Figure 7B, Table S11 |
| `generate_figure_s8_tables_s9_s10_ad_diagnostics.py` | Figure S8, Tables S9–S10 |
| `generate_tables_s12_s13_figures_s10_s11.py` | Tables S12–S13, Figures S10–S11 (triage α=0.10) |

## Environment

- Python ≥ 3.9
- RDKit (conda-forge recommended)
- Dependencies: `requirements.txt`

Recorded benchmark environment snapshot: `example_results/tg_cms_v4/run_meta.json` (when present).

## Archival pin

For the submitted manuscript version, cite GitHub release **v1.0.0**.
