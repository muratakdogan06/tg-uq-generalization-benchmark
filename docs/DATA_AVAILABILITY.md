# Data Availability (manuscript text)

---

**Data Availability.** The polymer dataset used in this study is provided with the code repository as `data/MD_properties.csv`, together with provenance information in `data/README.md`. The underlying source record is Project Elwood hosted by the Materials Data Facility (DOI: [10.18126/8p6m-e135](https://doi.org/10.18126/8p6m-e135)). All run-level benchmarking outputs supporting the figures and tables are provided in the Supporting Information (e.g., full results table and column dictionary) and as an illustrative snapshot in `example_results/tg_cms_v4/metrics/`, enabling exact reconstruction of summary statistics reported in the main text.

---

## Dataset summary

| Item | Location |
|------|----------|
| Curated repeat-unit SMILES and MD-derived properties | `data/MD_properties.csv` |
| Provenance and citation | `data/README.md` |
| Processed/frozen copy created by the pipeline | `runs/<run_tag>/data/processed.csv` |
| Full cross-validation result table | `runs/<run_tag>/metrics/results.csv` |
| Bundled snapshot for SI regeneration | `example_results/tg_cms_v4/metrics/results.csv` |
