# Supporting Information analyses

Scripts in `supporting_information/scripts/` regenerate the manuscript **Supporting Information** tables and figures from benchmark outputs.

## Prerequisites

1. Install the package (see root `README.md`).
2. Provide benchmark outputs either by:
   - **Default:** bundled snapshot `example_results/tg_cms_v4/` (no full re-run required), or
   - **Full reproduction:** run `python scripts/run_all.py --config configs/tg.yaml`, then `export CMS_USE_EXAMPLE_RESULTS=0`.

## One-command regeneration

```bash
bash supporting_information/run_all.sh
```

Outputs are written to `supporting_information/outputs/`.

## Script map

| Script | Manuscript outputs |
|--------|-------------------|
| `generate_table_s5_scaffold_characterization.py` | Table S5 |
| `generate_table_s6_figure_s2_cluster_characterization.py` | Table S6, Figure S2 |
| `generate_figure_s3_fingerprint_sensitivity.py` | Figure S3 |
| `generate_table_s7_analysis.py` | Table S7 |
| `generate_table_s8_figure_s5.py` | Table S8, Figure S5 |
| `generate_figure7_table_s11_svr_permutation.py` | Figure 7B, Table S11 |
| `generate_figure_s8_tables_s9_s10_ad_diagnostics.py` | Figure S8, Tables S9–S10 |
| `generate_tables_s12_s13_figures_s10_s11.py` | Tables S12–S13, Figures S10–S11 |
| `generate_table_sx_polymer_proxy_features_only.py` | Table Sx (polymer proxy features) |

## Path configuration

Shared paths live in `supporting_information/_paths.py`. Override with:

| Variable | Default | Meaning |
|----------|---------|---------|
| `CMS_USE_EXAMPLE_RESULTS` | `1` | Use `example_results/` instead of `runs/` |
| `CMS_RUN_TAG` | `tg_cms_v4` | Run directory name |
