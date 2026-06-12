from __future__ import annotations

from pathlib import Path
import json
import zipfile
from html import escape

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


TARGETS = [300.0, 350.0, 400.0]
MAIN_TARGET = 350.0
TOPK_FRACTIONS = [0.05, 0.10, 0.20]
MAIN_TOPK = 0.10
MODELS = ["svr", "xgb"]
NOMINAL_ALPHA = 0.10


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


def write_docx(path: Path, title: str, note: str, table: pd.DataFrame) -> None:
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
        f"<w:p><w:r><w:t>{escape(title)}</w:t></w:r></w:p>"
        f"<w:p><w:r><w:t>{escape(note)}</w:t></w:r></w:p>"
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


def concordance_index(y_true: np.ndarray, y_score: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0
    n = len(y_true)
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            comparable += 1
            true_order = np.sign(y_true[i] - y_true[j])
            score_order = np.sign(y_score[i] - y_score[j])
            if score_order == 0:
                concordant += 0.5
            elif true_order == score_order:
                concordant += 1.0
    return concordant / comparable if comparable else np.nan


def topk_metrics(y_true: np.ndarray, y_score: np.ndarray, target: float, k_fraction: float) -> dict:
    positives = y_true >= target
    n = len(y_true)
    k = max(1, int(np.ceil(k_fraction * n)))
    selected_idx = np.argsort(y_score)[::-1][:k]
    selected = np.zeros(n, dtype=bool)
    selected[selected_idx] = True
    tp = int(np.sum(selected & positives))
    fp = int(np.sum(selected & ~positives))
    pos = int(np.sum(positives))
    neg = int(np.sum(~positives))
    precision = tp / k
    recall = tp / pos if pos else np.nan
    prevalence = pos / n if n else np.nan
    ef = precision / prevalence if prevalence and prevalence > 0 else np.nan
    fpr = fp / neg if neg else np.nan
    return {
        "target_Tg": target,
        "topk_fraction": k_fraction,
        "n": n,
        "k": k,
        "positives": pos,
        "prevalence": prevalence,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "enrichment_factor": ef,
        "false_positive_rate": fpr,
    }


def ranking_by_unit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    topk_rows = []
    group_cols = ["model", "regime", "seed", "fold"]
    for keys, g in df.groupby(group_cols, sort=True):
        model, regime, seed, fold = keys
        y = g["Tg"].to_numpy(float)
        pred = g["y_pred"].to_numpy(float)
        rho = spearmanr(y, pred).statistic
        cidx = concordance_index(y, pred)
        base = {
            "model": model,
            "regime": regime,
            "seed": int(seed),
            "fold": int(fold),
            "n": int(len(g)),
            "spearman_rho": float(rho),
            "concordance_index": float(cidx),
        }
        rows.append(base)
        for target in TARGETS:
            for k_fraction in TOPK_FRACTIONS:
                topk_rows.append({**base, **topk_metrics(y, pred, target, k_fraction)})
    return pd.DataFrame(rows), pd.DataFrame(topk_rows)


def summarize_ranking(unit: pd.DataFrame, topk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = [("Overall", unit)]
    groups.extend((regime, g) for regime, g in unit.groupby("regime", sort=True))
    rows = []
    for label, g in groups:
        for model, gm in g.groupby("model", sort=True):
            top = topk[
                (topk["model"] == model)
                & (topk["target_Tg"] == MAIN_TARGET)
                & (topk["topk_fraction"] == MAIN_TOPK)
            ]
            if label != "Overall":
                top = top[top["regime"] == label]
            rows.append(
                {
                    "comparison_set": label,
                    "model": model,
                    "n_units": int(gm.shape[0]),
                    "spearman_mean": float(gm["spearman_rho"].mean()),
                    "spearman_std": float(gm["spearman_rho"].std()),
                    "concordance_index_mean": float(gm["concordance_index"].mean()),
                    "concordance_index_std": float(gm["concordance_index"].std()),
                    "precision_at_10pct_mean": float(top["precision_at_k"].mean()),
                    "precision_at_10pct_std": float(top["precision_at_k"].std()),
                    "recall_at_10pct_mean": float(top["recall_at_k"].mean()),
                    "recall_at_10pct_std": float(top["recall_at_k"].std()),
                    "enrichment_factor_at_10pct_mean": float(top["enrichment_factor"].mean()),
                    "false_positive_rate_at_10pct_mean": float(top["false_positive_rate"].mean()),
                    "target_Tg_for_topk": MAIN_TARGET,
                    "topk_fraction": MAIN_TOPK,
                }
            )
    raw = pd.DataFrame(rows)
    view = raw.copy()
    for col in [
        "spearman_mean",
        "spearman_std",
        "concordance_index_mean",
        "concordance_index_std",
        "precision_at_10pct_mean",
        "precision_at_10pct_std",
        "recall_at_10pct_mean",
        "recall_at_10pct_std",
        "enrichment_factor_at_10pct_mean",
        "false_positive_rate_at_10pct_mean",
    ]:
        view[col] = view[col].map(lambda x: f"{x:.3f}")
    view["Target Tg for top-k"] = view["target_Tg_for_topk"].map(lambda x: f"{x:.0f} K")
    view["Top-k"] = view["topk_fraction"].map(lambda x: f"{int(x * 100)}%")
    view = view[
        [
            "comparison_set",
            "model",
            "n_units",
            "spearman_mean",
            "spearman_std",
            "concordance_index_mean",
            "concordance_index_std",
            "precision_at_10pct_mean",
            "recall_at_10pct_mean",
            "enrichment_factor_at_10pct_mean",
            "false_positive_rate_at_10pct_mean",
            "Target Tg for top-k",
            "Top-k",
        ]
    ].rename(
        columns={
            "comparison_set": "Comparison set",
            "model": "Model",
            "n_units": "Seed/fold units",
            "spearman_mean": "Spearman mean",
            "spearman_std": "Spearman SD",
            "concordance_index_mean": "C-index mean",
            "concordance_index_std": "C-index SD",
            "precision_at_10pct_mean": "Precision@10%",
            "recall_at_10pct_mean": "Recall@10%",
            "enrichment_factor_at_10pct_mean": "EF@10%",
            "false_positive_rate_at_10pct_mean": "FPR@10%",
        }
    )
    return raw, view


def triage_for_group(g: pd.DataFrame, target: float, rule: str) -> dict:
    y = g["Tg"].to_numpy(float)
    pred = g["y_pred"].to_numpy(float)
    lower = g["lower"].to_numpy(float)
    upper = g["upper"].to_numpy(float)
    positives = y >= target
    inconclusive = np.zeros(len(g), dtype=bool)
    if rule == "point_prediction":
        selected = pred >= target
    elif rule == "conservative_lower_bound":
        selected = lower >= target
    elif rule == "uncertainty_flagged":
        selected = lower >= target
        inconclusive = (lower < target) & (upper >= target)
    else:
        raise ValueError(rule)
    tp = int(np.sum(selected & positives))
    fp = int(np.sum(selected & ~positives))
    fn = int(np.sum(~selected & ~inconclusive & positives))
    tn = int(np.sum(~selected & ~inconclusive & ~positives))
    selected_n = int(np.sum(selected))
    inconclusive_n = int(np.sum(inconclusive))
    pos = int(np.sum(positives))
    neg = int(np.sum(~positives))
    precision = tp / selected_n if selected_n else np.nan
    recall = tp / pos if pos else np.nan
    fpr = fp / neg if neg else np.nan
    return {
        "n": int(len(g)),
        "target_Tg": target,
        "rule": rule,
        "positives": pos,
        "selected_n": selected_n,
        "selected_fraction": selected_n / len(g) if len(g) else np.nan,
        "inconclusive_n": inconclusive_n,
        "inconclusive_fraction": inconclusive_n / len(g) if len(g) else np.nan,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "mean_interval_width_selected": float(g.loc[selected, "interval_width"].mean()) if selected_n else np.nan,
    }


def triage_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for model, gm in df.groupby("model", sort=True):
        for target in TARGETS:
            for rule in ["point_prediction", "conservative_lower_bound", "uncertainty_flagged"]:
                rows.append({"comparison_set": "Overall", "model": model, **triage_for_group(gm, target, rule)})
        for regime, gr in gm.groupby("regime", sort=True):
            for target in TARGETS:
                for rule in ["point_prediction", "conservative_lower_bound", "uncertainty_flagged"]:
                    rows.append({"comparison_set": regime, "model": model, **triage_for_group(gr, target, rule)})
    raw = pd.DataFrame(rows)
    manuscript_raw = raw[raw["comparison_set"].eq("Overall")].copy()
    view = manuscript_raw.copy()
    rule_labels = {
        "point_prediction": "Point prediction",
        "conservative_lower_bound": "Conservative lower bound",
        "uncertainty_flagged": "Uncertainty-flagged triage",
    }
    view["Rule"] = view["rule"].map(rule_labels)
    view["Target Tg"] = view["target_Tg"].map(lambda x: f"{x:.0f} K")
    for col in ["selected_fraction", "inconclusive_fraction", "precision", "recall", "false_positive_rate"]:
        view[col] = view[col].map(lambda x: "NA" if pd.isna(x) else f"{x:.3f}")
    view["Mean selected width (K)"] = view["mean_interval_width_selected"].map(
        lambda x: "NA" if pd.isna(x) else f"{x:.2f}"
    )
    view = view[
        [
            "model",
            "Target Tg",
            "Rule",
            "n",
            "positives",
            "selected_n",
            "selected_fraction",
            "inconclusive_n",
            "inconclusive_fraction",
            "precision",
            "recall",
            "false_positive_rate",
            "Mean selected width (K)",
        ]
    ].rename(
        columns={
            "model": "Model",
            "n": "Intervals",
            "positives": "True positives available",
            "selected_n": "Selected",
            "selected_fraction": "Selected fraction",
            "inconclusive_n": "Inconclusive",
            "inconclusive_fraction": "Inconclusive fraction",
            "precision": "Precision",
            "recall": "Recall",
            "false_positive_rate": "FPR",
        }
    )
    return raw, view


def save_table_bundle(stem: str, raw: pd.DataFrame, manuscript: pd.DataFrame, title: str, note: str) -> dict:
    paths = {}
    raw_csv = OUT / f"{stem}_raw.csv"
    csv = OUT / f"{stem}_manuscript.csv"
    md = OUT / f"{stem}_manuscript.md"
    tex = OUT / f"{stem}_manuscript.tex"
    xlsx = OUT / f"{stem}.xlsx"
    docx = OUT / f"{stem}.docx"
    raw.to_csv(raw_csv, index=False)
    manuscript.to_csv(csv, index=False)
    md.write_text(dataframe_to_markdown(manuscript), encoding="utf-8")
    tex.write_text(dataframe_to_latex(manuscript), encoding="utf-8")
    write_xlsx(xlsx, {"manuscript": manuscript, "raw": raw})
    write_docx(docx, title, note, manuscript)
    paths.update(
        {
            "raw_csv": str(raw_csv),
            "manuscript_csv": str(csv),
            "manuscript_md": str(md),
            "manuscript_tex": str(tex),
            "xlsx": str(xlsx),
            "docx": str(docx),
        }
    )
    return paths


def plot_s10(ranking_raw: pd.DataFrame) -> list[str]:
    paths = []
    regimes = ["Overall", "cluster_c0.20", "cluster_c0.30", "cluster_c0.40", "scaffold", "stratified"]
    regime_labels = ["Overall", "Cluster (c=0.20)", "Cluster (c=0.30)", "Cluster (c=0.40)", "Scaffold", "Stratified"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for ax, metric, title, ylabel in [
        (axes[0], "spearman_mean", "A) Spearman rank correlation", r"Spearman $\rho$"),
        (axes[1], "concordance_index_mean", "B) Concordance index", "Concordance index"),
    ]:
        width = 0.35
        x = np.arange(len(regimes))
        for offset, model in [(-width / 2, "svr"), (width / 2, "xgb")]:
            sub = ranking_raw[ranking_raw["model"].eq(model)].set_index("comparison_set").reindex(regimes)
            ax.bar(x + offset, sub[metric], width=width, label=model.upper())
        ax.set_xticks(x, regime_labels, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", color="0.9")
    axes[0].legend(frameon=False)
    for ext in ("png", "svg", "pdf"):
        path = OUT / f"FigureS10_ranking_fidelity.{ext}"
        fig.savefig(path, dpi=600 if ext == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_s11(triage_raw: pd.DataFrame) -> list[str]:
    paths = []
    overall = triage_raw[triage_raw["comparison_set"].eq("Overall")].copy()
    rule_order = ["point_prediction", "conservative_lower_bound", "uncertainty_flagged"]
    rule_labels = ["Point", "Lower-bound", "Triage"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    for ax, metric, title, ylim in [
        (axes[0, 0], "precision", "A) Precision", (0, 1.02)),
        (axes[0, 1], "recall", "B) Recall", (0, 1.02)),
        (axes[1, 0], "false_positive_rate", "C) False-positive rate", (0, 1.02)),
        (axes[1, 1], "inconclusive_fraction", "D) Inconclusive fraction", (0, 1.02)),
    ]:
        x = np.arange(len(TARGETS))
        width = 0.12
        for model_i, model in enumerate(MODELS):
            for rule_i, rule in enumerate(rule_order):
                sub = overall[(overall["model"].eq(model)) & (overall["rule"].eq(rule))].set_index(
                    "target_Tg"
                ).reindex(TARGETS)
                offset = (model_i * len(rule_order) + rule_i - 2.5) * width
                label = f"{model.upper()} {rule_labels[rule_i]}"
                ax.bar(x + offset, sub[metric], width=width, label=label)
        ax.set_xticks(x, [f"{int(t)} K" for t in TARGETS])
        ax.set_xlabel(r"$T_g$ (K)")
        ax.set_ylim(*ylim)
        ax.set_title(title)
    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        leg_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    for ext, dpi in (("png", 600), ("svg", None), ("pdf", None), ("tiff", 600)):
        path = OUT / f"FigureS11_threshold_conformal_triage.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    intervals = pd.read_csv(INTERVALS_CSV)
    intervals = intervals[intervals["alpha"].eq(NOMINAL_ALPHA) & intervals["frac"].eq(1.0)].copy()
    unit, topk = ranking_by_unit(intervals)
    ranking_raw, ranking_view = summarize_ranking(unit, topk)
    triage_raw, triage_view = triage_tables(intervals)

    topk.to_csv(OUT / "TableS12_topk_all_thresholds_raw.csv", index=False)
    unit.to_csv(OUT / "TableS12_rank_metrics_by_seed_fold_raw.csv", index=False)

    s12_paths = save_table_bundle(
        "TableS12_Ranking_TopK_Performance",
        ranking_raw,
        ranking_view,
        "Table S12. Ranking and top-k selection performance",
        "Top-k columns use Tg >= 350 K and top 10% selection; raw audit files include 300/350/400 K and top 5/10/20%.",
    )
    s13_paths = save_table_bundle(
        "TableS13_Interval_Aware_Triage",
        triage_raw,
        triage_view,
        "Table S13. Interval-aware triage analysis under Tg-targeted screening windows",
        "Rules compare point prediction, conservative lower-bound selection, and uncertainty-flagged triage across 300/350/400 K Tg targets.",
    )
    fig_s10 = plot_s10(ranking_raw)
    fig_s11 = plot_s11(triage_raw)

    s8_manifest = json.loads(S8_MANIFEST.read_text(encoding="utf-8")) if S8_MANIFEST.exists() else {}
    manifest = {
        "analysis": "Tables S12-S13 and Figures S10-S11 ranking plus interval-aware triage",
        "input_files": {
            "per_sample_intervals": str(INTERVALS_CSV),
            "table_s8_manifest": str(S8_MANIFEST),
        },
        "settings": {
            "alpha": NOMINAL_ALPHA,
            "targets_Tg_K": TARGETS,
            "main_topk_target_Tg_K": MAIN_TARGET,
            "topk_fractions": TOPK_FRACTIONS,
            "main_topk_fraction": MAIN_TOPK,
            "ranking_unit": "model x validation regime x seed x fold",
            "triage_rules": {
                "point_prediction": "select if y_pred >= Tg target",
                "conservative_lower_bound": "select if lower conformal bound >= Tg target",
                "uncertainty_flagged": "select if lower >= target; inconclusive if lower < target <= upper",
            },
        },
        "source_dataset": s8_manifest.get("dataset", {}),
        "outputs": {
            "table_s12": s12_paths,
            "table_s12_topk_raw": str(OUT / "TableS12_topk_all_thresholds_raw.csv"),
            "table_s12_rank_unit_raw": str(OUT / "TableS12_rank_metrics_by_seed_fold_raw.csv"),
            "table_s13": s13_paths,
            "figures": fig_s10 + fig_s11,
            "notes": str(OUT / "TablesS12_S13_FiguresS10_S11_methods_and_file_notes.md"),
        },
    }
    (OUT / "TablesS12_S13_FiguresS10_S11_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (OUT / "TablesS12_S13_FiguresS10_S11_methods_and_file_notes.md").write_text(
        "\n".join(
            [
                "# Tables S12-S13 / Figures S10-S11 Analysis Notes",
                "",
                "## Input files used",
                f"- `{INTERVALS_CSV}`: per-sample predictions, conformal bounds, true Tg values, regimes, seeds, and folds. This is the primary input for both ranking and triage.",
                f"- `{S8_MANIFEST}`: provenance for how the per-sample conformal interval file was reconstructed from benchmark splits and model settings.",
                "",
                "## Ranking metrics",
                "- Spearman rank correlation and concordance index are computed within each matched validation unit (`model`, `regime`, `seed`, `fold`) and summarized by regime.",
                "- Precision@k, Recall@k, enrichment factor, and false-positive rate are calculated after sorting candidates by point prediction.",
                f"- Manuscript Table S12 reports top 10% selection for Tg >= {MAIN_TARGET:.0f} K; raw files include Tg targets 300, 350, and 400 K and top 5%, 10%, and 20%.",
                "",
                "## Triage rules",
                "- Point-prediction selection: select if `y_pred >= Tg_target`.",
                "- Conservative lower-bound selection: select only if the conformal lower bound is above the target.",
                "- Uncertainty-flagged triage: select if lower bound is above target; mark inconclusive if the interval crosses the target.",
                "- Table S13 reports overall results across Tg targets 300, 350, and 400 K. The raw table also includes validation-regime-specific rows.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
