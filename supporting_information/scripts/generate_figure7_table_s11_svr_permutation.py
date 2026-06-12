from __future__ import annotations

from pathlib import Path
import json
import sys
import zipfile
from html import escape

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


import sys
from pathlib import Path

_SI_ROOT = Path(__file__).resolve().parents[1]
if str(_SI_ROOT) not in sys.path:
    sys.path.insert(0, str(_SI_ROOT))
from _paths import (  # noqa: E402
    BENCHMARK,
    CONFIG_YAML,
    DATASET_META_JSON,
    FINAL,
    INTERVALS_CSV,
    OUT,
    PROCESSED_CSV,
    RESULTS_CSV,
    ROOT,
    RUN,
    S8_MANIFEST,
    SIMILARITY_CSV,
    SUMMARY_CSV,
    XGB_SHAP_CSV,
    XGB_SHAP_FIG,
)


CONFIG = BENCHMARK / "configs" / "tg.yaml"
PROCESSED = RUN / "data" / "processed.csv"
DATASET_META = RUN / "data" / "dataset_meta.json"
XGB_SHAP_CSV = RUN / "metrics" / "shap_top_features_xgb.csv"
XGB_SHAP_FIG = RUN / "figs" / "shap_summary_xgb.png"

SVR_PERM_CSV = OUT / "Figure7B_SVR_permutation_importance_raw.csv"
TABLE_S11_RAW = OUT / "TableS11_XGB_SHAP_SVR_Permutation_Comparison_raw.csv"
TABLE_S11_MANUSCRIPT = OUT / "TableS11_XGB_SHAP_SVR_Permutation_Comparison_manuscript.csv"
TABLE_S11_MD = OUT / "TableS11_XGB_SHAP_SVR_Permutation_Comparison_manuscript.md"
TABLE_S11_TEX = OUT / "TableS11_XGB_SHAP_SVR_Permutation_Comparison_manuscript.tex"
TABLE_S11_XLSX = OUT / "TableS11_XGB_SHAP_SVR_Permutation_Comparison.xlsx"
TABLE_S11_DOCX = OUT / "TableS11_XGB_SHAP_SVR_Permutation_Comparison.docx"
MANIFEST = OUT / "Figure7_TableS11_analysis_manifest.json"
NOTES = OUT / "Figure7_TableS11_methods_and_file_notes.md"

FIG_BASE = OUT / "Figure7_XGB_SHAP_SVR_permutation_importance"
FINAL_FIG_BASE = FINAL / "Figure7_XGB_SHAP_SVR_permutation_importance"

TOP_N_FIG = 20
TOP_N_TABLE = 25
PERM_REPEATS = 30

sys.path.insert(0, str(BENCHMARK / "src"))

from cms_tg.config import load_config  # noqa: E402
from cms_tg.features import featurize  # noqa: E402
from cms_tg.models.svr_model import fit_svr, tune_svr  # noqa: E402


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def dataframe_to_latex(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\hline",
        " & ".join(latex_escape(col) for col in cols) + r" \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(row[col]) for col in cols) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def excel_col_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def worksheet_xml(df: pd.DataFrame) -> str:
    rows = [list(df.columns)] + df.astype(str).values.tolist()
    xml_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            ref = f"{excel_col_name(c_idx)}{r_idx}"
            cells.append(
                f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'
            )
        xml_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        "</worksheet>"
    )


def write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    names = list(sheets.keys())
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(1, len(names) + 1)
            )
            + "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            + "".join(
                f'<sheet name="{escape(name[:31])}" sheetId="{i}" r:id="rId{i}"/>'
                for i, name in enumerate(names, start=1)
            )
            + "</sheets></workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{i}.xml"/>'
                for i in range(1, len(names) + 1)
            )
            + "</Relationships>",
        )
        for i, name in enumerate(names, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", worksheet_xml(sheets[name]))


def write_docx(path: Path, table: pd.DataFrame) -> None:
    rows = [list(table.columns)] + table.astype(str).values.tolist()
    table_rows = []
    for row in rows:
        cells = "".join(
            "<w:tc><w:p><w:r><w:t>"
            + escape(str(value))
            + "</w:t></w:r></w:p></w:tc>"
            for value in row
        )
        table_rows.append(f"<w:tr>{cells}</w:tr>")
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        "<w:p><w:r><w:t>Table S11. Comparison of XGBoost SHAP and SVR permutation-importance rankings</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>XGBoost values are mean absolute SHAP values. SVR values are permutation importance scores expressed as RMSE increase after feature permutation.</w:t></w:r></w:p>"
        + f"<w:tbl>{''.join(table_rows)}</w:tbl>"
        + '<w:sectPr><w:pgSz w:w="15840" w:h="12240" w:orient="landscape"/></w:sectPr>'
        + "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>",
        )
        zf.writestr("word/document.xml", document_xml)


def sample_rows(n_rows: int, n_sample: int, seed: int) -> np.ndarray:
    if n_rows <= n_sample:
        return np.arange(n_rows)
    rng = np.random.RandomState(int(seed))
    return rng.choice(np.arange(n_rows), size=n_sample, replace=False)


def load_features():
    cfg = load_config(str(CONFIG))
    df = pd.read_csv(PROCESSED)
    X_df, valid_idx, _, _ = featurize(cfg, df[cfg.data.smiles_col].values)
    df = df.iloc[valid_idx].reset_index(drop=True)
    y = df[cfg.data.target_col].values.astype(float)
    return cfg, df, X_df, y


def compute_svr_permutation() -> tuple[pd.DataFrame, dict]:
    if SVR_PERM_CSV.exists():
        perm = pd.read_csv(SVR_PERM_CSV)
        meta_path = OUT / "Figure7B_SVR_permutation_importance_model_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return perm, meta

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg, _, X_df, y = load_features()
    X = X_df.values.astype(float)
    seed = int(cfg.shap.random_seed)
    idx = sample_rows(X.shape[0], int(cfg.shap.sample_size), seed)
    X_sample = X[idx]
    y_sample = y[idx]

    params = tune_svr(cfg, X, y, seed=seed)
    model = fit_svr(cfg, X, y, seed=seed, params=params)
    pred_sample = model.predict(X_sample)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_sample, pred_sample)))
    baseline_r2 = float(r2_score(y_sample, pred_sample))

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="neg_root_mean_squared_error",
        n_repeats=PERM_REPEATS,
        random_state=seed,
        n_jobs=1,
    )
    perm = pd.DataFrame(
        {
            "feature": list(X_df.columns),
            "rmse_increase_mean": result.importances_mean,
            "rmse_increase_std": result.importances_std,
        }
    )
    perm["rmse_increase_mean"] = perm["rmse_increase_mean"].clip(lower=0)
    max_importance = float(perm["rmse_increase_mean"].max())
    perm["normalized_importance"] = (
        perm["rmse_increase_mean"] / max_importance if max_importance > 0 else 0.0
    )
    perm = perm.sort_values("rmse_increase_mean", ascending=False).reset_index(drop=True)
    perm["svr_permutation_rank"] = np.arange(1, len(perm) + 1)
    perm.to_csv(SVR_PERM_CSV, index=False)

    meta = {
        "model": "SVR",
        "importance": "permutation importance",
        "scoring": "negative root mean squared error",
        "reported_importance": "mean RMSE increase after permutation",
        "n_repeats": PERM_REPEATS,
        "sample_size": int(len(idx)),
        "random_seed": seed,
        "baseline_sample_rmse": baseline_rmse,
        "baseline_sample_r2": baseline_r2,
        "optuna_best_rmse_innercv": float(params.get("_optuna_best_value_rmse", np.nan)),
        "best_params": {k: v for k, v in params.items() if not str(k).startswith("_")},
    }
    (OUT / "Figure7B_SVR_permutation_importance_model_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return perm, meta


def load_xgb_shap() -> pd.DataFrame:
    shap_df = pd.read_csv(XGB_SHAP_CSV)
    shap_df = shap_df.rename(columns={"mean_abs_shap": "xgb_mean_abs_shap"})
    shap_df["xgb_shap_rank"] = np.arange(1, len(shap_df) + 1)
    max_shap = float(shap_df["xgb_mean_abs_shap"].max())
    shap_df["xgb_normalized_importance"] = shap_df["xgb_mean_abs_shap"] / max_shap
    return shap_df


def make_table_s11(shap_df: pd.DataFrame, perm_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = pd.merge(shap_df, perm_df, on="feature", how="outer")
    merged["in_xgb_top25"] = merged["xgb_shap_rank"] <= TOP_N_TABLE
    merged["in_svr_top25"] = merged["svr_permutation_rank"] <= TOP_N_TABLE
    merged["rank_difference_svr_minus_xgb"] = (
        merged["svr_permutation_rank"] - merged["xgb_shap_rank"]
    )
    merged["_best_rank"] = merged[["xgb_shap_rank", "svr_permutation_rank"]].min(axis=1)
    merged = merged.sort_values(["_best_rank", "feature"]).drop(columns=["_best_rank"])
    table = merged[
        merged["in_xgb_top25"].fillna(False) | merged["in_svr_top25"].fillna(False)
    ].copy()
    table.to_csv(TABLE_S11_RAW, index=False)

    view = table.copy()
    for col in ["xgb_mean_abs_shap", "xgb_normalized_importance", "rmse_increase_mean", "normalized_importance"]:
        view[col] = view[col].map(lambda x: "NA" if pd.isna(x) else f"{x:.4f}")
    view["rmse_increase_std"] = view["rmse_increase_std"].map(lambda x: "NA" if pd.isna(x) else f"{x:.4f}")
    for col in ["xgb_shap_rank", "svr_permutation_rank", "rank_difference_svr_minus_xgb"]:
        view[col] = view[col].map(lambda x: "NA" if pd.isna(x) else f"{int(x)}")
    view["Shared top-25"] = np.where(view["in_xgb_top25"] & view["in_svr_top25"], "yes", "no")
    view = view[
        [
            "feature",
            "xgb_shap_rank",
            "xgb_mean_abs_shap",
            "svr_permutation_rank",
            "rmse_increase_mean",
            "rmse_increase_std",
            "rank_difference_svr_minus_xgb",
            "Shared top-25",
        ]
    ].rename(
        columns={
            "feature": "Feature",
            "xgb_shap_rank": "XGBoost SHAP rank",
            "xgb_mean_abs_shap": "XGBoost mean |SHAP|",
            "svr_permutation_rank": "SVR permutation rank",
            "rmse_increase_mean": "SVR RMSE increase",
            "rmse_increase_std": "SVR RMSE increase SD",
            "rank_difference_svr_minus_xgb": "Rank difference",
        }
    )
    view.to_csv(TABLE_S11_MANUSCRIPT, index=False)
    TABLE_S11_MD.write_text(dataframe_to_markdown(view), encoding="utf-8")
    TABLE_S11_TEX.write_text(dataframe_to_latex(view), encoding="utf-8")
    write_xlsx(TABLE_S11_XLSX, {"Table S11 manuscript": view, "raw comparison": table})
    write_docx(TABLE_S11_DOCX, view)
    return table, view


def plot_figure7(shap_df: pd.DataFrame, perm_df: pd.DataFrame) -> list[str]:
    xgb_top = shap_df.head(TOP_N_FIG).iloc[::-1].copy()
    svr_top = perm_df.head(TOP_N_FIG).iloc[::-1].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), constrained_layout=True)

    bar_color = "#1565c0"
    axes[0].barh(xgb_top["feature"], xgb_top["xgb_mean_abs_shap"], color=bar_color)
    axes[0].set_title("(A)")
    axes[0].set_xlabel("Mean absolute SHAP value")
    axes[0].grid(axis="x", color="0.9", lw=0.8)

    axes[1].barh(svr_top["feature"], svr_top["rmse_increase_mean"], color=bar_color)
    axes[1].set_title("(B)")
    axes[1].set_xlabel("RMSE increase after permutation (K)")
    axes[1].grid(axis="x", color="0.9", lw=0.8)
    paths = []
    for ext in ("png", "svg", "pdf"):
        path = FIG_BASE.with_suffix(f".{ext}")
        fig.savefig(path, dpi=600 if ext == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    tiff_path = FIG_BASE.with_suffix(".tiff")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    paths.append(str(tiff_path))

    FINAL.mkdir(parents=True, exist_ok=True)
    final_png = FINAL_FIG_BASE.with_suffix(".png")
    final_tiff = FINAL_FIG_BASE.with_suffix(".tiff")
    fig.savefig(final_png, dpi=600, bbox_inches="tight")
    fig.savefig(final_tiff, dpi=600, bbox_inches="tight")
    paths.extend([str(final_png), str(final_tiff)])
    plt.close(fig)
    return paths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shap_df = load_xgb_shap()
    perm_df, svr_meta = compute_svr_permutation()
    table_raw, table_view = make_table_s11(shap_df, perm_df)
    fig_paths = plot_figure7(shap_df, perm_df)

    with open(DATASET_META) as f:
        dataset_meta = json.load(f)

    common_top25 = int(
        np.sum(table_raw["in_xgb_top25"].fillna(False) & table_raw["in_svr_top25"].fillna(False))
    )
    manifest = {
        "analysis": "Main Figure 7B SVR permutation importance and Table S11 XGBoost/SVR importance comparison",
        "input_files": {
            "xgb_shap_csv": str(XGB_SHAP_CSV),
            "xgb_shap_figure": str(XGB_SHAP_FIG),
            "processed_data": str(PROCESSED),
            "dataset_metadata": str(DATASET_META),
            "config": str(CONFIG),
        },
        "dataset": dataset_meta,
        "settings": {
            "xgb_importance": "mean absolute SHAP values from existing benchmark output",
            "svr_importance": "permutation importance using RMSE increase",
            "sample_size": int(svr_meta.get("sample_size", 0)),
            "random_seed": int(svr_meta.get("random_seed", 123)),
            "permutation_repeats": PERM_REPEATS,
            "top_features_in_figure": TOP_N_FIG,
            "top_features_in_table_union": TOP_N_TABLE,
        },
        "svr_model": svr_meta,
        "summary": {
            "xgb_top_feature": str(shap_df.iloc[0]["feature"]),
            "svr_top_feature": str(perm_df.iloc[0]["feature"]),
            "shared_top25_features": common_top25,
        },
        "outputs": {
            "svr_permutation_raw": str(SVR_PERM_CSV),
            "table_s11_raw": str(TABLE_S11_RAW),
            "table_s11_manuscript_csv": str(TABLE_S11_MANUSCRIPT),
            "table_s11_manuscript_md": str(TABLE_S11_MD),
            "table_s11_manuscript_tex": str(TABLE_S11_TEX),
            "table_s11_xlsx": str(TABLE_S11_XLSX),
            "table_s11_docx": str(TABLE_S11_DOCX),
            "figures": fig_paths,
            "notes": str(NOTES),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    NOTES.write_text(
        "\n".join(
            [
                "# Figure 7 / Table S11 Analysis Notes",
                "",
                "## Input files used",
                f"- `{XGB_SHAP_CSV}`: existing XGBoost SHAP feature ranking used for Figure 7A and Table S11.",
                f"- `{XGB_SHAP_FIG}`: existing XGBoost SHAP bar plot checked as the source figure style.",
                f"- `{PROCESSED}`: processed SMILES/Tg data used to reconstruct descriptor matrix and train SVR.",
                f"- `{CONFIG}`: benchmark settings, including SHAP sample size and random seed.",
                f"- `{DATASET_META}`: dataset identity, row count, target column, and hashes.",
                "",
                "## Method",
                "- Figure 7A keeps XGBoost mean absolute SHAP importance from the benchmark output.",
                "- Figure 7B adds model-agnostic SVR permutation importance.",
                "- SVR was tuned and fit on the full processed descriptor matrix using the benchmark SVR search space.",
                "- Permutation importance was evaluated on the same 250-row random sample used by the SHAP routine.",
                f"- Importance is reported as mean RMSE increase across {PERM_REPEATS} permutations per feature.",
                "- Table S11 compares the union of XGBoost top-25 SHAP features and SVR top-25 permutation features.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
