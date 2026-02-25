# tg-uq-generalization-benchmark

A reproducible benchmarking pipeline for **trustworthy polymer glass transition temperature (Tg) prediction** under **chemical novelty**, combining novelty-aware validation, **split conformal prediction** (distribution-free uncertainty quantification), applicability-domain diagnostics (train–test similarity), and SHAP interpretability.

This repository accompanies the JCIM manuscript:
**“Trust Beyond Accuracy: Conformal Uncertainty Quantification Reveals the Generalization Gap in Polymer Tg Prediction.”**

---

## Data provenance

The underlying molecular property data originate from Project Elwood (Materials Data Facility):

Schneider, L.; S. M.; Mysona, J.; Liang, H.; Han, M.; Rauscher, P.; Ting, J.; Venkatram, S.; Ross, R.; Schmidt, K.; Blaiszik, B.; Foster, I.; de Pablo, J.  
**Project Elwood: MD Simulated Monomer Properties.** *Materials Data Facility*, 2022.  
DOI: **10.18126/8p6m-e135**

See `data/README.md` for details on how `MD_properties.csv` is used in this repository.

---

## Installation

### Create and activate a virtual environment, install dependencies and run the full pipeline
```bash
python3 -m venv .venv
source .venv/bin/activate
 
pip install -U pip
pip install -r requirements.txt
pip install -e .

python scripts/run_all.py --config configs/tg.yaml
```
--- 

### License

See LICENSE.

---

### Citation

If you use this repository, please cite:

The associated manuscript (update with the final bibliographic information when available):
Akdoğan, M. Trust Beyond Accuracy: Conformal Uncertainty Quantification Reveals the Generalization Gap in Polymer Tg Prediction. JCIM (submitted, 2026).

The dataset source (Project Elwood):
Schneider, L.; S. M.; Mysona, J.; Liang, H.; Han, M.; Rauscher, P.; Ting, J.; Venkatram, S.; Ross, R.; Schmidt, K.; Blaiszik, B.; Foster, I.; de Pablo, J.
Project Elwood: MD Simulated Monomer Properties. Materials Data Facility, 2022. DOI: 10.18126/8p6m-e135

---
