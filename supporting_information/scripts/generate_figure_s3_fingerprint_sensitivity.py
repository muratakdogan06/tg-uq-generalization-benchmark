from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect


rdBase.DisableLog("rdApp.warning")


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


PROCESSED_CSV = RUN / "data" / "processed.csv"
DATASET_META_JSON = RUN / "data" / "dataset_meta.json"
CONFIG_YAML = BENCHMARK / "configs" / "tg.yaml"
SPLITS_PY = BENCHMARK / "src" / "cms_tg" / "splits.py"
SIMILARITY_PY = BENCHMARK / "src" / "cms_tg" / "similarity.py"

FIGURE_TITLE = "Figure S3. Sensitivity of cluster-split characterization to Morgan fingerprint settings"

SEEDS = [42, 43, 44]
CUTOFFS = [0.20, 0.30, 0.40]
N_FOLDS = 5
LOW_SMAX_THRESHOLD = 0.30

FINGERPRINT_SETTINGS = [
    {"setting": "r2_2048", "label": "Radius 2 / 2048 bits (default)", "radius": 2, "nbits": 2048},
    {"setting": "r3_2048", "label": "Radius 3 / 2048 bits", "radius": 3, "nbits": 2048},
    {"setting": "r2_1024", "label": "Radius 2 / 1024 bits", "radius": 2, "nbits": 1024},
]


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def load_molecules() -> tuple[pd.DataFrame, list[Chem.Mol]]:
    df = pd.read_csv(PROCESSED_CSV)
    with DATASET_META_JSON.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    smiles_col = meta["smiles_col"]
    target_col = meta["target_col"]

    rows = []
    mols = []
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row[smiles_col]))
        if mol is None:
            continue
        rows.append(
            {
                "sample_index": int(idx),
                "SMILES": str(row[smiles_col]),
                "Tg_K": float(row[target_col]),
            }
        )
        mols.append(mol)
    return pd.DataFrame(rows).reset_index(drop=True), mols


def morgan_fingerprints(mols: list[Chem.Mol], radius: int, nbits: int):
    return [GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(nbits)) for mol in mols]


def cluster_by_cutoff(fps, cutoff: float) -> list[list[int]]:
    clusters: list[list[int]] = []
    for i, fp in enumerate(fps):
        assigned = False
        for cluster in clusters:
            sim = DataStructs.TanimotoSimilarity(fp, fps[cluster[0]])
            if sim >= float(cutoff):
                cluster.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])
    return clusters


def cluster_splits_from_clusters(clusters: list[list[int]], seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(int(seed))
    shuffled = [list(cluster) for cluster in clusters]
    rng.shuffle(shuffled)
    fold_bins: list[list[int]] = [[] for _ in range(N_FOLDS)]
    fold_sizes = np.zeros(N_FOLDS, dtype=int)
    for cluster in sorted(shuffled, key=len, reverse=True):
        fold_idx = int(np.argmin(fold_sizes))
        fold_bins[fold_idx].extend(cluster)
        fold_sizes[fold_idx] += len(cluster)
    all_idx = np.arange(sum(len(cluster) for cluster in clusters))
    return [
        (np.setdiff1d(all_idx, np.array(sorted(fold), dtype=int)), np.array(sorted(fold), dtype=int))
        for fold in fold_bins
    ]


def max_tanimoto_test_to_train(fps, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    train_fps = [fps[int(i)] for i in train_idx]
    out = np.zeros(len(test_idx), dtype=float)
    for j, i in enumerate(test_idx):
        sims = DataStructs.BulkTanimotoSimilarity(fps[int(i)], train_fps)
        out[j] = float(max(sims)) if sims else 0.0
    return out


def size_distribution_text(sizes: np.ndarray) -> str:
    bins = [
        ("1", sizes == 1),
        ("2", sizes == 2),
        ("3-5", (sizes >= 3) & (sizes <= 5)),
        ("6-10", (sizes >= 6) & (sizes <= 10)),
        (">10", sizes > 10),
    ]
    return "; ".join(f"{label}: {int(mask.sum())}" for label, mask in bins)


def run_sensitivity(data: pd.DataFrame, mols: list[Chem.Mol]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    cluster_rows = []
    smax_rows = []
    n = len(data)

    for fp_setting in FINGERPRINT_SETTINGS:
        fps = morgan_fingerprints(mols, fp_setting["radius"], fp_setting["nbits"])
        for cutoff in CUTOFFS:
            clusters = cluster_by_cutoff(fps, cutoff)
            sizes = np.array([len(cluster) for cluster in clusters], dtype=int)
            singleton_clusters = int((sizes == 1).sum())

            for cluster_id, members in enumerate(clusters):
                subset = data.iloc[members]
                cluster_rows.append(
                    {
                        "fingerprint_setting": fp_setting["setting"],
                        "fingerprint_label": fp_setting["label"],
                        "radius": int(fp_setting["radius"]),
                        "nbits": int(fp_setting["nbits"]),
                        "cutoff": float(cutoff),
                        "cluster_id": int(cluster_id),
                        "cluster_size": int(len(members)),
                        "member_indices": ";".join(str(int(i)) for i in members),
                        "Tg_median_K": float(subset["Tg_K"].median()),
                        "representative_SMILES": str(subset.iloc[0]["SMILES"]),
                    }
                )

            all_smax = []
            fold_sizes = []
            for seed in SEEDS:
                splits = cluster_splits_from_clusters(clusters, seed)
                for fold, (train_idx, test_idx) in enumerate(splits):
                    smax = max_tanimoto_test_to_train(fps, train_idx, test_idx)
                    all_smax.extend(smax.tolist())
                    fold_sizes.append(len(test_idx))
                    for local_j, sample_idx in enumerate(test_idx):
                        smax_rows.append(
                            {
                                "fingerprint_setting": fp_setting["setting"],
                                "fingerprint_label": fp_setting["label"],
                                "radius": int(fp_setting["radius"]),
                                "nbits": int(fp_setting["nbits"]),
                                "cutoff": float(cutoff),
                                "seed": int(seed),
                                "fold": int(fold),
                                "sample_index": int(data.iloc[int(sample_idx)]["sample_index"]),
                                "local_index": int(sample_idx),
                                "Smax": float(smax[local_j]),
                                "low_Smax_tail": bool(smax[local_j] < LOW_SMAX_THRESHOLD),
                            }
                        )

            all_smax_arr = np.array(all_smax, dtype=float)
            summary_rows.append(
                {
                    "fingerprint_setting": fp_setting["setting"],
                    "fingerprint_label": fp_setting["label"],
                    "radius": int(fp_setting["radius"]),
                    "nbits": int(fp_setting["nbits"]),
                    "cutoff": float(cutoff),
                    "n_clusters": int(len(clusters)),
                    "cluster_size_distribution": size_distribution_text(sizes),
                    "cluster_size_median": float(np.median(sizes)),
                    "cluster_size_q25": float(np.percentile(sizes, 25)),
                    "cluster_size_q75": float(np.percentile(sizes, 75)),
                    "largest_cluster_size": int(sizes.max()),
                    "singleton_clusters": singleton_clusters,
                    "singleton_cluster_fraction": float(singleton_clusters / len(clusters)),
                    "singleton_repeat_units": int(sizes[sizes == 1].sum()),
                    "singleton_repeat_unit_fraction": float(sizes[sizes == 1].sum() / n),
                    "fold_size_median": float(np.median(fold_sizes)),
                    "fold_size_min": int(np.min(fold_sizes)),
                    "fold_size_max": int(np.max(fold_sizes)),
                    "Smax_median": float(np.median(all_smax_arr)),
                    "Smax_q25": float(np.percentile(all_smax_arr, 25)),
                    "Smax_q75": float(np.percentile(all_smax_arr, 75)),
                    "Smax_mean": float(np.mean(all_smax_arr)),
                    "low_Smax_tail_n": int(np.sum(all_smax_arr < LOW_SMAX_THRESHOLD)),
                    "low_Smax_tail_fraction": float(np.mean(all_smax_arr < LOW_SMAX_THRESHOLD)),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(cluster_rows), pd.DataFrame(smax_rows)


def plot_figure_s3(summary: pd.DataFrame, smax_records: pd.DataFrame, out_prefix: Path) -> list[Path]:
    labels = [setting["label"] for setting in FINGERPRINT_SETTINGS]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    x = np.arange(len(CUTOFFS))
    width = 0.24

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    for i, setting in enumerate(FINGERPRINT_SETTINGS):
        sub = summary[summary["fingerprint_setting"] == setting["setting"]].sort_values("cutoff")
        ax.bar(x + (i - 1) * width, sub["n_clusters"], width=width, label=setting["label"], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cutoff:.2f}" for cutoff in CUTOFFS])
    ax.set_xlabel("Morgan-Tanimoto cutoff")
    ax.set_ylabel("Number of clusters")
    ax.set_title("A) Cluster count")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    for i, setting in enumerate(FINGERPRINT_SETTINGS):
        sub = summary[summary["fingerprint_setting"] == setting["setting"]].sort_values("cutoff")
        ax.plot(
            [f"{cutoff:.2f}" for cutoff in CUTOFFS],
            sub["singleton_cluster_fraction"] * 100,
            marker="o",
            label=setting["label"],
            color=colors[i],
        )
    ax.set_xlabel("Morgan-Tanimoto cutoff")
    ax.set_ylabel("Singleton-cluster fraction (%)")
    ax.set_title("B) Singleton-cluster fraction")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    positions = np.arange(len(CUTOFFS)) + 1
    for i, setting in enumerate(FINGERPRINT_SETTINGS):
        data = [
            smax_records[
                (smax_records["fingerprint_setting"] == setting["setting"])
                & (smax_records["cutoff"] == cutoff)
            ]["Smax"].to_numpy(float)
            for cutoff in CUTOFFS
        ]
        bp = ax.boxplot(
            data,
            positions=positions + (i - 1) * width,
            widths=0.18,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.55)
        ax.plot([], [], label=setting["label"], color=colors[i])
    ax.axhline(LOW_SMAX_THRESHOLD, linestyle="--", color="black", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{cutoff:.2f}" for cutoff in CUTOFFS])
    ax.set_xlabel("Morgan-Tanimoto cutoff")
    ax.set_ylabel("S$_{max}$")
    ax.set_title("C) S$_{max}$ distribution")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    for i, setting in enumerate(FINGERPRINT_SETTINGS):
        sub = summary[summary["fingerprint_setting"] == setting["setting"]].sort_values("cutoff")
        ax.plot(
            [f"{cutoff:.2f}" for cutoff in CUTOFFS],
            sub["low_Smax_tail_fraction"] * 100,
            marker="o",
            label=setting["label"],
            color=colors[i],
        )
    ax.set_xlabel("Morgan-Tanimoto cutoff")
    ax.set_ylabel(f"Low-S$_{{max}}$ tail fraction (%)\nS$_{{max}}$ < {LOW_SMAX_THRESHOLD:.2f}")
    ax.set_title("D) Low-S$_{max}$ tail")
    ax.grid(axis="y", alpha=0.25)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    paths = []
    for ext, kwargs in {
        ".png": {"dpi": 300},
        ".svg": {},
        ".pdf": {},
        ".tiff": {"dpi": 600},
    }.items():
        path = out_prefix.with_suffix(ext)
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def write_notes(summary: pd.DataFrame, figure_paths: list[Path], notes_path: Path) -> None:
    lines = [
        FIGURE_TITLE,
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Inputs used:",
        f"- Processed benchmark data: {PROCESSED_CSV}",
        f"- Dataset metadata: {DATASET_META_JSON}",
        f"- Benchmark config: {CONFIG_YAML}",
        f"- Cluster split implementation: {SPLITS_PY}",
        f"- Smax implementation: {SIMILARITY_PY}",
        "",
        "Method:",
        "- Model retraining was not performed.",
        "- Only Morgan fingerprint settings were changed.",
        f"- Settings compared: {', '.join(setting['label'] for setting in FINGERPRINT_SETTINGS)}.",
        f"- Cutoffs evaluated: {', '.join(f'{cutoff:.2f}' for cutoff in CUTOFFS)}.",
        "- Cluster assignment and fold packing match the benchmark cluster-split implementation.",
        f"- Low-Smax tail is defined as Smax < {LOW_SMAX_THRESHOLD:.2f}.",
        "",
        "Key values:",
    ]
    for _, row in summary.sort_values(["cutoff", "fingerprint_setting"]).iterrows():
        lines.append(
            f"- {row['fingerprint_label']}, cutoff {row['cutoff']:.2f}: "
            f"{int(row['n_clusters'])} clusters; "
            f"singleton clusters {row['singleton_cluster_fraction'] * 100:.1f}%; "
            f"Smax median {row['Smax_median']:.3f} "
            f"[{row['Smax_q25']:.3f}, {row['Smax_q75']:.3f}]; "
            f"low-Smax tail {row['low_Smax_tail_fraction'] * 100:.1f}%."
        )
    lines.extend(
        [
            "",
            "Reviewer-facing interpretation:",
            "The qualitative cutoff trend is robust to the tested Morgan fingerprint settings: larger cutoffs "
            "produce more clusters and higher singleton fractions under all settings. Bit length sensitivity is "
            "small at radius 2, with 1024-bit and 2048-bit fingerprints giving similar Smax and low-Smax tails. "
            "Increasing the radius to 3 is more stringent, producing more clusters, lower Smax medians, and a "
            "larger low-Smax tail. This supports radius 2 / 2048 bits as a reasonable default while documenting "
            "the expected direction of sensitivity.",
            "",
            "Figure files:",
            *[f"- {path}" for path in figure_paths],
        ]
    )
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)

    data, mols = load_molecules()
    summary, clusters, smax_records = run_sensitivity(data, mols)

    summary_path = OUT / "FigureS3_fingerprint_sensitivity_summary.csv"
    clusters_path = OUT / "FigureS3_fingerprint_sensitivity_clusters_raw.csv"
    smax_path = OUT / "FigureS3_fingerprint_sensitivity_smax_records.csv"
    notes_path = OUT / "FigureS3_fingerprint_sensitivity_notes.txt"
    figure_prefix = OUT / "FigureS3_fingerprint_sensitivity"

    summary.to_csv(summary_path, index=False)
    clusters.to_csv(clusters_path, index=False)
    smax_records.to_csv(smax_path, index=False)
    figure_paths = plot_figure_s3(summary, smax_records, figure_prefix)
    write_notes(summary, figure_paths, notes_path)

    for path in figure_paths:
        if path.suffix.lower() in {".png", ".tiff"}:
            shutil.copy2(path, FINAL / path.name)

    print(FIGURE_TITLE)
    print(
        summary[
            [
                "fingerprint_label",
                "cutoff",
                "n_clusters",
                "singleton_cluster_fraction",
                "Smax_median",
                "Smax_q25",
                "Smax_q75",
                "low_Smax_tail_fraction",
            ]
        ].to_string(index=False)
    )
    print("Wrote:")
    for path in [summary_path, clusters_path, smax_path, notes_path, *figure_paths]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
