# tg-uq-generalization-benchmark

A reproducible benchmarking pipeline for **trustworthy polymer glass-transition temperature (T<sub>g</sub>) prediction** under **chemical novelty**, combining novelty-aware validation, **split conformal prediction** (distribution-free uncertainty quantification), applicability-domain diagnostics (train–test similarity), and model-specific feature-importance analyses.

This repository accompanies the manuscript:
**“Trust Beyond Accuracy: Conformal Uncertainty Quantification Reveals the Generalization Gap in Polymer T<sub>g</sub> Prediction.”**

**Repository:** [https://github.com/muratakdogan06/tg-uq-generalization-benchmark](https://github.com/muratakdogan06/tg-uq-generalization-benchmark)  
**Archival release cited in the manuscript:** `v1.0.0`

---

## What this repository contains

| Component | Location | Description |
|-----------|----------|-------------|
| Core Python package | `src/cms_tg/` | Data loading, featurization, splits, models, conformal UQ, S<sub>max</sub>, plots, SHAP |
| Benchmark driver | `scripts/run_all.py` | End-to-end cross-validation benchmark |
| Configuration | `configs/tg.yaml` | Seeds, regimes, learning fractions, α levels, models |
| Dataset | `data/MD_properties.csv` | Curated repeat-unit SMILES and MD properties (410 polymers) |
| Example outputs | `example_results/tg_cms_v4/` | Snapshot of `results.csv`, similarity, SHAP, processed data |
| SI post-processing | `supporting_information/scripts/` | Tables S5–S13, Figures S2–S3, S5, S8, S10–S11, Figure 7B, Table S11 |
| Manuscript availability text | `docs/CODE_AVAILABILITY.md`, `docs/DATA_AVAILABILITY.md` | Paste-ready statements |

### Pipeline capabilities

- **Models:** XGBoost (Optuna-tuned) and SVR-RBF (Optuna-tuned)
- **Features:** 217 RDKit 2D descriptors + 5 polymer-proxy features (`features.py`)
- **Splits:** stratified T<sub>g</sub> bins, Murcko scaffolds, Morgan cluster leader clustering (cutoffs 0.20 / 0.30 / 0.40)
- **Cross-validation:** 5 folds × 3 seeds × 3 regimes (+ cluster cutoffs)
- **Learning curves:** training fractions {0.2, 0.4, 0.6, 0.8, 1.0}
- **Uncertainty:** split-conformal intervals, α ∈ {0.05, 0.10, 0.20}
- **Diagnostics:** maximum test-to-train Morgan–Tanimoto similarity (S<sub>max</sub>)
- **Interpretability:** XGBoost mean \|SHAP\|; SVR permutation importance (SI scripts)

---

## Installation

```bash
git clone https://github.com/muratakdogan06/tg-uq-generalization-benchmark.git
cd tg-uq-generalization-benchmark
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### 1) Run the full benchmark

```bash
python scripts/run_all.py --config configs/tg.yaml
```

Writes to `runs/tg_cms_v4/`:

```
runs/tg_cms_v4/
  data/raw.csv, processed.csv, dataset_meta.json
  metrics/results.csv
  metrics/summary_frac1_alpha.csv
  metrics/similarity_test_to_train.csv
  metrics/shap_top_features_xgb.csv
  figs/coverage_width_curve_frac1_*.png
  figs/similarity_violin.png, similarity_ecdf.png
  figs/shap_summary_xgb.png
```

### 2) Regenerate Supporting Information outputs

Uses bundled `example_results/` by default:

```bash
bash supporting_information/run_all.sh
```

After a fresh benchmark run:

```bash
export CMS_USE_EXAMPLE_RESULTS=0
bash supporting_information/run_all.sh
```

SI tables/figures are written to `supporting_information/outputs/`.

---

## Repository layout

```
├── configs/tg.yaml
├── data/MD_properties.csv, data/README.md
├── docs/ (CODE_AVAILABILITY, DATA_AVAILABILITY, REPRODUCIBILITY)
├── example_results/tg_cms_v4/
├── scripts/run_all.py, run_benchmark.py
├── src/cms_tg/
└── supporting_information/scripts/, run_all.sh
```

See `docs/REPRODUCIBILITY.md` for a table-level map to manuscript outputs.

---

## Data provenance

The underlying molecular property data originate from **Project Elwood** (Materials Data Facility):

Schneider, L.; S. M.; Mysona, J.; Liang, H.; Han, M.; Rauscher, P.; Ting, J.; Venkatram, S.; Ross, R.; Schmidt, K.; Blaiszik, B.; Foster, I.; de Pablo, J.  
**Project Elwood: MD Simulated Monomer Properties.** *Materials Data Facility*, 2022.  
DOI: [10.18126/8p6m-e135](https://doi.org/10.18126/8p6m-e135)

See `data/README.md`.

---

## License

MIT License — see `LICENSE`.

---

## Citation

If you use this repository, please cite:

**Manuscript:** Akdoğan, M. *Trust Beyond Accuracy: Conformal Uncertainty Quantification Reveals the Generalization Gap in Polymer T<sub>g</sub> Prediction.* ACS Omega (submitted, 2026).

**Code (archival pin):** Akdoğan, M. tg-uq-generalization-benchmark (v1.0.0). https://github.com/muratakdogan06/tg-uq-generalization-benchmark

**Dataset:** Project Elwood, DOI 10.18126/8p6m-e135.
