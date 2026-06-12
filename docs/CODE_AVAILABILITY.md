# Code Availability (manuscript text)

Use the paragraph below in the **Code Availability** section of the manuscript. It matches the repository layout at submission time.

---

**Code Availability.** All scripts required to reproduce the main-text and Supporting Information results are available on GitHub at [https://github.com/muratakdogan06/tg-uq-generalization-benchmark](https://github.com/muratakdogan06/tg-uq-generalization-benchmark). The repository contains: (i) the complete cross-validation benchmarking pipeline for support vector regression (SVR) and gradient-boosted trees (XGBoost); (ii) RDKit 2D descriptor generation plus five polymer-proxy features (`src/cms_tg/features.py`); (iii) validation-split generation for stratified, Murcko-scaffold, and Morgan-fingerprint cluster regimes (`src/cms_tg/splits.py`); (iv) split-conformal prediction at nominal miscoverage levels α ∈ {0.05, 0.10, 0.20}; (v) maximum test-to-train Morgan–Tanimoto similarity diagnostics (S<sub>max</sub>); (vi) XGBoost SHAP importance; and (vii) Supporting Information post-processing scripts under `supporting_information/scripts/` for paired model comparisons (Table S7), split characterization (Tables S5–S6; Figures S2–S3), expanded applicability-domain metrics (Figure S8; Tables S9–S10), subgroup conformal coverage (Table S8; Figure S5), feature-importance comparison (Figure 7; Table S11), and ranking plus interval-aware triage analyses (Tables S12–S13; Figures S10–S11). The polymer dataset is bundled as `data/MD_properties.csv` with provenance notes in `data/README.md`. Example benchmark outputs used by the Supporting Information scripts are provided under `example_results/tg_cms_v4/` so tables and figures can be regenerated without rerunning the full benchmark when desired. For archival reproducibility, cite GitHub release **v1.0.0** corresponding to the submitted manuscript version.

---

## Quick reproduction commands

```bash
git clone https://github.com/muratakdogan06/tg-uq-generalization-benchmark.git
cd tg-uq-generalization-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Full benchmark (main-text learning curves, conformal metrics, Smax, SHAP)
python scripts/run_all.py --config configs/tg.yaml

# Supporting Information tables/figures (uses example_results/ by default)
bash supporting_information/run_all.sh
```

To force Supporting Information scripts to read freshly generated outputs instead of the bundled snapshot:

```bash
export CMS_USE_EXAMPLE_RESULTS=0
bash supporting_information/run_all.sh
```
