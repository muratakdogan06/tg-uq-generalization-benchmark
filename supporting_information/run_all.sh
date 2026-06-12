#!/usr/bin/env bash
# Regenerate Supporting Information tables and figures.
# By default uses bundled snapshot: example_results/tg_cms_v4/
# Set CMS_USE_EXAMPLE_RESULTS=0 after running the full benchmark to use runs/tg_cms_v4/.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export CMS_USE_EXAMPLE_RESULTS="${CMS_USE_EXAMPLE_RESULTS:-1}"
export CMS_RUN_TAG="${CMS_RUN_TAG:-tg_cms_v4}"
SCRIPTS=(
  supporting_information/scripts/generate_table_s5_scaffold_characterization.py
  supporting_information/scripts/generate_table_s6_figure_s2_cluster_characterization.py
  supporting_information/scripts/generate_figure_s3_fingerprint_sensitivity.py
  supporting_information/scripts/generate_table_s7_analysis.py
  supporting_information/scripts/generate_table_s8_figure_s5.py
  supporting_information/scripts/generate_figure7_table_s11_svr_permutation.py
  supporting_information/scripts/generate_figure_s8_tables_s9_s10_ad_diagnostics.py
  supporting_information/scripts/generate_tables_s12_s13_figures_s10_s11.py
  supporting_information/scripts/generate_table_sx_polymer_proxy_features_only.py
)
mkdir -p supporting_information/outputs
for script in "${SCRIPTS[@]}"; do
  echo ">>> $script"
  python "$script"
done
echo "Outputs written to supporting_information/outputs/"
