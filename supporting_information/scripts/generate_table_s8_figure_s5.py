from __future__ import annotations

from pathlib import Path
import json
import sys
import zipfile
from html import escape

import numpy as np
import pandas as pd
from scipy.stats import binomtest

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


sys.path.insert(0, str(BENCHMARK / "src"))

from cms_tg.config import load_config  # noqa: E402
from cms_tg.features import featurize, morgan_fingerprints  # noqa: E402
from cms_tg.models.svr_model import fit_svr, tune_svr  # noqa: E402
from cms_tg.models.xgb_model import fit_xgb, tune_xgb  # noqa: E402
from cms_tg.similarity import max_tanimoto_test_to_train  # noqa: E402
from cms_tg.splits import get_splits  # noqa: E402
from cms_tg.uncertainty import conformal_quantiles  # noqa: E402


CONFIG = BENCHMARK / "configs" / "tg.yaml"
PROCESSED = RUN / "data" / "processed.csv"
DATASET_META = RUN / "data" / "dataset_meta.json"
RESULTS_CSV = RUN / "metrics" / "results.csv"
SIMILARITY_CSV = RUN / "metrics" / "similarity_test_to_train.csv"

MAIN_FRAC = 1.0
MAIN_ALPHA = 0.10
NOMINAL = 1.0 - MAIN_ALPHA
MODELS = ["svr", "xgb"]

INTERVAL_CSV = OUT / "TableS8_per_sample_conformal_intervals_frac1_alpha010.csv"
TABLE_RAW_CSV = OUT / "TableS8_Subgroup_Coverage_raw.csv"
TABLE_MANUSCRIPT_CSV = OUT / "TableS8_Subgroup_Coverage_manuscript.csv"
TABLE_MANUSCRIPT_MD = OUT / "TableS8_Subgroup_Coverage_manuscript.md"
TABLE_MANUSCRIPT_TEX = OUT / "TableS8_Subgroup_Coverage_manuscript.tex"
TABLE_XLSX = OUT / "TableS8_Subgroup_Coverage.xlsx"
TABLE_DOCX = OUT / "TableS8_Subgroup_Coverage.docx"
MANIFEST_JSON = OUT / "TableS8_FigureS5_analysis_manifest.json"
NOTES_MD = OUT / "TableS8_FigureS5_methods_and_file_notes.md"


def p_format(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


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


def write_docx(path: Path, manuscript: pd.DataFrame) -> None:
    rows = [list(manuscript.columns)] + manuscript.astype(str).values.tolist()
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
        "<w:p><w:r><w:t>Table S8. Subgroup-level empirical coverage and interval width</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Coverage is calculated from per-sample split-conformal intervals at alpha = 0.10. "
        "Rows report empirical coverage, mean interval width, and coverage gap relative to 90% nominal coverage.</w:t></w:r></w:p>"
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


def binomial_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return float(ci.low), float(ci.high)


def chemistry_metadata(mols) -> pd.DataFrame:
    rows = []
    hetero_fracs = []
    for i, mol in enumerate(mols):
        heavy = mol.GetNumHeavyAtoms()
        denom = heavy if heavy > 0 else 1
        arom_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        hetero_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
        hetero_frac = hetero_atoms / denom
        hetero_fracs.append(hetero_frac)
        rows.append(
            {
                "sample_index": i,
                "aromatic_class": "aromatic" if arom_atoms > 0 else "non-aromatic",
                "ring_class": "ring-containing" if mol.GetRingInfo().NumRings() > 0 else "acyclic",
                "heteroatom_fraction": hetero_frac,
                "heavy_atoms": heavy,
            }
        )
    meta = pd.DataFrame(rows)
    threshold = float(np.median(hetero_fracs))
    meta["heteroatom_class"] = np.where(
        meta["heteroatom_fraction"] >= threshold,
        "heteroatom-rich",
        "heteroatom-poor",
    )
    return meta, threshold


def train_predict(model_name: str, cfg, X_fit, y_fit, X_cal, y_cal, X_te, seed: int):
    if model_name == "xgb":
        params = tune_xgb(cfg, X_fit, y_fit, seed=seed)
        model = fit_xgb(cfg, X_fit, y_fit, seed=seed, params=params)
    elif model_name == "svr":
        params = tune_svr(cfg, X_fit, y_fit, seed=seed)
        model = fit_svr(cfg, X_fit, y_fit, seed=seed, params=params)
    else:
        raise ValueError(model_name)
    yhat_te = model.predict(X_te)
    yhat_cal = model.predict(X_cal)
    qhat = conformal_quantiles(np.abs(y_cal - yhat_cal), [MAIN_ALPHA])[MAIN_ALPHA]
    return yhat_te, float(qhat), float(params.get("_optuna_best_value_rmse", np.nan))


def generate_intervals() -> pd.DataFrame:
    if INTERVAL_CSV.exists():
        return pd.read_csv(INTERVAL_CSV)

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(CONFIG))
    df = pd.read_csv(PROCESSED)
    with open(DATASET_META) as f:
        dataset_id = json.load(f)["dataset_id"]

    X_df, valid_idx, mols_valid, _ = featurize(cfg, df[cfg.data.smiles_col].values)
    df = df.iloc[valid_idx].reset_index(drop=True)
    y = df[cfg.data.target_col].values.astype(float)
    fps = morgan_fingerprints(cfg, mols_valid)
    chem, hetero_threshold = chemistry_metadata(mols_valid)

    rows = []
    for base_regime in cfg.eval.regimes:
        cutoffs = [None] if base_regime != "cluster" else list(cfg.eval.cluster_cutoffs)
        for cutoff in cutoffs:
            regime_label = base_regime if cutoff is None else f"cluster_c{cutoff:.2f}"
            for seed in cfg.eval.seeds:
                splits = get_splits(cfg, y, mols_valid, fps, dataset_id, base_regime, seed, cutoff=cutoff)
                for fold, (tr_idx, te_idx) in enumerate(splits):
                    n_tr = len(tr_idx)
                    n_sub = max(10, int(np.floor(MAIN_FRAC * n_tr)))
                    rng = np.random.RandomState(int(seed) + 1000 * fold + int(MAIN_FRAC * 100))
                    sub_local = rng.choice(np.arange(n_tr), size=n_sub, replace=False)
                    sub_idx = tr_idx[sub_local]
                    n_cal = max(10, int(np.floor(0.2 * len(sub_idx))))
                    perm = rng.permutation(len(sub_idx))
                    cal_idx = sub_idx[perm[:n_cal]]
                    fit_idx = sub_idx[perm[n_cal:]]

                    X_fit = X_df.values[fit_idx]
                    y_fit = y[fit_idx]
                    X_cal = X_df.values[cal_idx]
                    y_cal = y[cal_idx]
                    X_te = X_df.values[te_idx]
                    y_te = y[te_idx]
                    smax = max_tanimoto_test_to_train(fps, tr_idx, te_idx)

                    for model_name in MODELS:
                        yhat_te, qhat, inner_rmse = train_predict(
                            model_name,
                            cfg,
                            X_fit,
                            y_fit,
                            X_cal,
                            y_cal,
                            X_te,
                            int(seed),
                        )
                        lo = yhat_te - qhat
                        hi = yhat_te + qhat
                        covered = (y_te >= lo) & (y_te <= hi)
                        for j, sample_idx in enumerate(te_idx):
                            c = chem.iloc[int(sample_idx)]
                            rows.append(
                                {
                                    "model": model_name,
                                    "regime": regime_label,
                                    "base_regime": base_regime,
                                    "cutoff": np.nan if cutoff is None else float(cutoff),
                                    "seed": int(seed),
                                    "fold": int(fold),
                                    "frac": MAIN_FRAC,
                                    "alpha": MAIN_ALPHA,
                                    "nominal_coverage": NOMINAL,
                                    "sample_index": int(sample_idx),
                                    "SMILES": df.iloc[int(sample_idx)][cfg.data.smiles_col],
                                    "Tg": float(y_te[j]),
                                    "y_pred": float(yhat_te[j]),
                                    "lower": float(lo[j]),
                                    "upper": float(hi[j]),
                                    "qhat": qhat,
                                    "interval_width": float(hi[j] - lo[j]),
                                    "covered": bool(covered[j]),
                                    "Smax": float(smax[j]),
                                    "aromatic_class": c["aromatic_class"],
                                    "ring_class": c["ring_class"],
                                    "heteroatom_fraction": float(c["heteroatom_fraction"]),
                                    "heteroatom_class": c["heteroatom_class"],
                                    "heavy_atoms": int(c["heavy_atoms"]),
                                    "heteroatom_rich_threshold": hetero_threshold,
                                    "optuna_best_rmse_innercv": inner_rmse,
                                }
                            )
                        print(
                            f"done {model_name} {regime_label} seed={seed} fold={fold}",
                            flush=True,
                        )

    intervals = pd.DataFrame(rows)
    intervals.to_csv(INTERVAL_CSV, index=False)
    return intervals


def assign_bins(intervals: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    df = intervals.copy()
    smax_edges = np.quantile(df["Smax"].dropna(), [0, 1 / 3, 2 / 3, 1])
    tg_edges = np.quantile(df["Tg"].dropna(), [0, 1 / 3, 2 / 3, 1])
    smax_edges[0] = -np.inf
    smax_edges[-1] = np.inf
    tg_edges[0] = -np.inf
    tg_edges[-1] = np.inf
    labels = ["low", "medium", "high"]
    df["Smax_bin"] = pd.cut(df["Smax"], bins=smax_edges, labels=labels, include_lowest=True)
    df["Tg_range"] = pd.cut(df["Tg"], bins=tg_edges, labels=labels, include_lowest=True)
    return df, {
        "Smax_tertile_edges_observed": [float(x) for x in np.quantile(intervals["Smax"].dropna(), [0, 1 / 3, 2 / 3, 1])],
        "Tg_tertile_edges_observed": [float(x) for x in np.quantile(intervals["Tg"].dropna(), [0, 1 / 3, 2 / 3, 1])],
    }


def subgroup_rows(df: pd.DataFrame, group_col: str, family: str) -> list[dict]:
    rows = []
    for model in MODELS:
        model_df = df[df["model"] == model]
        for subgroup, g in model_df.groupby(group_col, observed=True, sort=False):
            n = int(g.shape[0])
            hits = int(g["covered"].sum())
            ci_low, ci_high = binomial_ci(hits, n)
            rows.append(
                {
                    "model": model,
                    "subgroup_family": family,
                    "subgroup": str(subgroup),
                    "n_intervals": n,
                    "covered_n": hits,
                    "empirical_coverage": hits / n if n else np.nan,
                    "coverage_95CI_low": ci_low,
                    "coverage_95CI_high": ci_high,
                    "coverage_gap_vs_0.90": hits / n - NOMINAL if n else np.nan,
                    "mean_interval_width": float(g["interval_width"].mean()),
                    "median_interval_width": float(g["interval_width"].median()),
                    "mean_Smax": float(g["Smax"].mean()),
                    "mean_Tg": float(g["Tg"].mean()),
                }
            )
    return rows


def make_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.extend(subgroup_rows(df, "Smax_bin", "Smax bins"))
    rows.extend(subgroup_rows(df, "Tg_range", "Tg range"))
    rows.extend(subgroup_rows(df, "aromatic_class", "Chemistry: aromaticity"))
    rows.extend(subgroup_rows(df, "ring_class", "Chemistry: rings"))
    rows.extend(subgroup_rows(df, "heteroatom_class", "Chemistry: heteroatom content"))
    table = pd.DataFrame(rows)
    family_order = {
        "Smax bins": 0,
        "Tg range": 1,
        "Chemistry: aromaticity": 2,
        "Chemistry: rings": 3,
        "Chemistry: heteroatom content": 4,
    }
    subgroup_order = {
        "low": 0,
        "medium": 1,
        "high": 2,
        "aromatic": 0,
        "non-aromatic": 1,
        "ring-containing": 0,
        "acyclic": 1,
        "heteroatom-rich": 0,
        "heteroatom-poor": 1,
    }
    table["_family_order"] = table["subgroup_family"].map(family_order)
    table["_subgroup_order"] = table["subgroup"].map(subgroup_order).fillna(99)
    return table.sort_values(["model", "_family_order", "_subgroup_order"]).drop(
        columns=["_family_order", "_subgroup_order"]
    )


def manuscript_view(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["Empirical coverage"] = out["empirical_coverage"].map(lambda x: f"{x:.3f}")
    out["95% binomial CI"] = [
        f"[{lo:.3f}, {hi:.3f}]"
        for lo, hi in zip(out["coverage_95CI_low"], out["coverage_95CI_high"])
    ]
    out["Coverage gap"] = out["coverage_gap_vs_0.90"].map(lambda x: f"{x:+.3f}")
    out["Mean interval width (K)"] = out["mean_interval_width"].map(lambda x: f"{x:.2f}")
    out["Median interval width (K)"] = out["median_interval_width"].map(lambda x: f"{x:.2f}")
    return out[
        [
            "model",
            "subgroup_family",
            "subgroup",
            "n_intervals",
            "covered_n",
            "Empirical coverage",
            "95% binomial CI",
            "Coverage gap",
            "Mean interval width (K)",
            "Median interval width (K)",
        ]
    ].rename(
        columns={
            "model": "Model",
            "subgroup_family": "Subgroup family",
            "subgroup": "Subgroup",
            "n_intervals": "Intervals",
            "covered_n": "Covered",
        }
    )


def plot_figure_s5(table: pd.DataFrame) -> list[str]:
    paths = []
    family_order = {
        "Smax bins": 0,
        "Tg range": 1,
        "Chemistry: aromaticity": 2,
        "Chemistry: rings": 3,
        "Chemistry: heteroatom content": 4,
    }
    subgroup_order = {
        "low": 0,
        "medium": 1,
        "high": 2,
        "aromatic": 0,
        "non-aromatic": 1,
        "ring-containing": 0,
        "acyclic": 1,
        "heteroatom-rich": 0,
        "heteroatom-poor": 1,
    }
    plot_df = table.copy()
    plot_df["_family_order"] = plot_df["subgroup_family"].map(family_order)
    plot_df["_subgroup_order"] = plot_df["subgroup"].map(subgroup_order).fillna(99)
    plot_df["_model_order"] = plot_df["model"].map({"svr": 0, "xgb": 1})
    plot_df = plot_df.sort_values(["_family_order", "_subgroup_order", "_model_order"]).reset_index(drop=True)
    def short_label(row: pd.Series) -> str:
        model = str(row["model"]).upper()
        subgroup = str(row["subgroup"])
        family = str(row["subgroup_family"])
        subgroup_name = {
            "low": "Low",
            "medium": "Medium",
            "high": "High",
            "aromatic": "Aromatic",
            "non-aromatic": "Non-aromatic",
            "ring-containing": "Ring-containing",
            "acyclic": "Acyclic",
            "heteroatom-rich": "Heteroatom-rich",
            "heteroatom-poor": "Heteroatom-poor",
        }.get(subgroup, subgroup)
        if family == "Smax bins":
            return f"{subgroup_name} S$_{{max}}$ ({model})"
        if family == "Tg range":
            return f"{subgroup_name} T$_g$ ({model})"
        return f"{subgroup_name} ({model})"

    plot_df["label"] = plot_df.apply(short_label, axis=1)
    y = np.arange(len(plot_df))
    colors = {"svr": "#1f77b4", "xgb": "#ff7f0e"}

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 9.0),
        sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1.05, 1.0]},
        constrained_layout=True,
    )

    ax = axes[0]
    x = plot_df["empirical_coverage"].to_numpy(float)
    lo = plot_df["coverage_95CI_low"].to_numpy(float)
    hi = plot_df["coverage_95CI_high"].to_numpy(float)
    for i, row in plot_df.iterrows():
        ax.errorbar(
            x[i],
            y[i],
            xerr=[[x[i] - lo[i]], [hi[i] - x[i]]],
            fmt="o",
            color=colors[row["model"]],
            ecolor=colors[row["model"]],
            capsize=3,
            markersize=4.5,
        )
    ax.axvline(NOMINAL, color="0.35", ls="--", lw=1)
    ax.set_xlim(0.80, 0.955)
    ax.set_yticks(y, plot_df["label"])
    ax.invert_yaxis()
    ax.set_title("A) Empirical coverage")
    ax.set_xlabel("Coverage with Wilson 95% CI")
    ax.grid(axis="x", color="0.9", lw=0.8)

    ax = axes[1]
    gaps = plot_df["coverage_gap_vs_0.90"].to_numpy(float)
    bar_colors = [colors[m] for m in plot_df["model"]]
    ax.barh(y, gaps, color=bar_colors)
    ax.axvline(0, color="0.35", lw=1)
    ax.set_xlim(-0.085, 0.055)
    ax.set_title("B) Coverage gap")
    ax.set_xlabel("Coverage - 0.90")
    ax.grid(axis="x", color="0.9", lw=0.8)
    for i, gap in enumerate(gaps):
        ha = "right" if gap < 0 else "left"
        offset = -0.003 if gap < 0 else 0.003
        ax.text(gap + offset, y[i], f"{gap:+.3f}", va="center", ha=ha, fontsize=7)

    ax = axes[2]
    widths = plot_df["mean_interval_width"].to_numpy(float)
    ax.barh(y, widths, color=[colors[m] for m in plot_df["model"]])
    ax.set_xlim(84, 91)
    ax.set_title("C) Mean interval width")
    ax.set_xlabel("Width (K)")
    ax.grid(axis="x", color="0.9", lw=0.8)

    for ext in ("png", "svg", "pdf"):
        path = OUT / f"FigureS5_subgroup_conformal_coverage.{ext}"
        fig.savefig(path, dpi=600 if ext == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_width(table: pd.DataFrame) -> list[str]:
    paths = []
    fig, ax = plt.subplots(figsize=(11, 6.8), constrained_layout=True)
    plot_df = table.copy()
    plot_df["label"] = plot_df["subgroup_family"] + ": " + plot_df["subgroup"] + " (" + plot_df["model"].str.upper() + ")"
    y = np.arange(len(plot_df))
    colors = ["black" if m == "svr" else "0.45" for m in plot_df["model"]]
    ax.barh(y, plot_df["mean_interval_width"], color=colors)
    ax.set_yticks(y, plot_df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean interval width (K)")
    ax.set_title("Subgroup-level conformal interval width")
    ax.grid(axis="x", color="0.9", lw=0.8)
    for ext in ("png", "svg", "pdf"):
        path = OUT / f"FigureS5_subgroup_interval_width.{ext}"
        fig.savefig(path, dpi=600 if ext == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    intervals = generate_intervals()
    intervals, bin_edges = assign_bins(intervals)
    intervals.to_csv(INTERVAL_CSV, index=False)

    table = make_table(intervals)
    manuscript = manuscript_view(table)
    table.to_csv(TABLE_RAW_CSV, index=False)
    manuscript.to_csv(TABLE_MANUSCRIPT_CSV, index=False)
    TABLE_MANUSCRIPT_MD.write_text(dataframe_to_markdown(manuscript), encoding="utf-8")
    TABLE_MANUSCRIPT_TEX.write_text(dataframe_to_latex(manuscript), encoding="utf-8")
    write_xlsx(TABLE_XLSX, {"Table S8 manuscript": manuscript, "raw statistics": table})
    write_docx(TABLE_DOCX, manuscript)

    fig_paths = plot_figure_s5(table)
    fig_paths.extend(plot_width(table))

    with open(DATASET_META) as f:
        dataset_meta = json.load(f)

    manifest = {
        "analysis": "Table S8 and Figure S5 subgroup conditional coverage analysis",
        "input_files": {
            "processed_data": str(PROCESSED),
            "dataset_metadata": str(DATASET_META),
            "benchmark_results_checked": str(RESULTS_CSV),
            "similarity_diagnostics_checked": str(SIMILARITY_CSV),
            "config": str(CONFIG),
        },
        "dataset": dataset_meta,
        "settings": {
            "learning_fraction": MAIN_FRAC,
            "alpha": MAIN_ALPHA,
            "nominal_coverage": NOMINAL,
            "models": MODELS,
            "subgroups": [
                "Smax tertiles",
                "Tg tertiles",
                "aromatic vs non-aromatic",
                "ring-containing vs acyclic",
                "heteroatom-rich vs heteroatom-poor",
            ],
            "heteroatom_rich_definition": "heteroatom fraction >= dataset median",
            "heteroatom_fraction_threshold": float(intervals["heteroatom_rich_threshold"].iloc[0]),
            **bin_edges,
        },
        "outputs": {
            "per_sample_intervals": str(INTERVAL_CSV),
            "raw_csv": str(TABLE_RAW_CSV),
            "manuscript_csv": str(TABLE_MANUSCRIPT_CSV),
            "manuscript_md": str(TABLE_MANUSCRIPT_MD),
            "manuscript_tex": str(TABLE_MANUSCRIPT_TEX),
            "xlsx": str(TABLE_XLSX),
            "docx": str(TABLE_DOCX),
            "figures": fig_paths,
            "primary_figure_s5": [
                str(OUT / "FigureS5_subgroup_conformal_coverage.png"),
                str(OUT / "FigureS5_subgroup_conformal_coverage.svg"),
                str(OUT / "FigureS5_subgroup_conformal_coverage.pdf"),
            ],
            "legacy_width_only_figure": [
                str(OUT / "FigureS5_subgroup_interval_width.png"),
                str(OUT / "FigureS5_subgroup_interval_width.svg"),
                str(OUT / "FigureS5_subgroup_interval_width.pdf"),
            ],
            "notes": str(NOTES_MD),
        },
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    NOTES_MD.write_text(
        "\n".join(
            [
                "# Table S8 / Figure S5 Analysis Notes",
                "",
                "## Input files used",
                f"- `{PROCESSED}`: processed SMILES/Tg data used to reconstruct model inputs and Tg subgroups.",
                f"- `{DATASET_META}`: dataset identity, target column, row count, and hashes.",
                f"- `{CONFIG}`: benchmark settings, seeds, regimes, folds, models, and conformal alpha values.",
                f"- Saved split JSON files under `{RUN / 'splits'}`: ensure subgroup intervals use the exact benchmark seed/fold assignments.",
                f"- `{RESULTS_CSV}`: checked to align the analysis with `frac = 1.0` and `alpha = 0.10` convention used for reviewer statistics.",
                f"- `{SIMILARITY_CSV}`: checked as the benchmark's existing Smax diagnostic; Smax was recomputed per test row for intervals.",
                "",
                "## Subgroup definitions",
                "- Smax bins: low, medium, high tertiles of observed test-to-train maximum Tanimoto similarity across interval records.",
                "- Tg range: low, medium, high tertiles of observed Tg values across interval records.",
                "- Aromaticity: aromatic if at least one atom is aromatic; otherwise non-aromatic.",
                "- Rings: ring-containing if RDKit reports at least one ring; otherwise acyclic.",
                "- Heteroatom content: heteroatom-rich if heteroatom fraction is at or above the dataset median; otherwise heteroatom-poor.",
                "",
                "## Statistical reporting",
                "- Empirical coverage is the fraction of test points where `Tg` lies inside `[prediction - qhat, prediction + qhat]`.",
                "- Mean interval width is averaged over per-sample intervals in each subgroup.",
                "- Coverage confidence intervals are Wilson 95% binomial intervals.",
                "- The main analysis uses full-training evaluations (`frac = 1.0`) and `alpha = 0.10` (90% nominal coverage).",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
