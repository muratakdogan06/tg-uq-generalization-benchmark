from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import ensure_dir
import os

def savefig(path: str):
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_similarity_violin(sim_df: pd.DataFrame, out_path: str):
    regimes = list(sim_df["regime_label"].unique())
    data = [sim_df.loc[sim_df["regime_label"] == r, "max_tanimoto_to_train"].values for r in regimes]
    plt.figure()
    plt.violinplot(data, showmeans=True, showextrema=True)
    plt.xticks(np.arange(1, len(regimes)+1), regimes, rotation=30, ha="right")
    plt.ylabel("Max Tanimoto(test, train)")
    plt.title("Train–test similarity by split regime")
    savefig(out_path)

def plot_similarity_ecdf(sim_df: pd.DataFrame, out_path: str):
    plt.figure()
    for r, g in sim_df.groupby("regime_label"):
        x = np.sort(g["max_tanimoto_to_train"].values)
        y = np.arange(1, len(x)+1) / len(x)
        plt.plot(x, y, label=r)
    plt.xlabel("Max Tanimoto(test, train)")
    plt.ylabel("ECDF")
    plt.title("Similarity shift across regimes")
    plt.legend()
    savefig(out_path)

def plot_cov_width_curve(summary_df: pd.DataFrame, out_path: str):
    plt.figure()
    for r, g in summary_df.groupby("regime_label"):
        g = g.sort_values("alpha")
        plt.plot(g["width_mean"].values, g["cov_mean"].values, marker="o", label=r)
        for _, row in g.iterrows():
            plt.text(row["width_mean"], row["cov_mean"], f"α={row['alpha']:.2f}", fontsize=8)
    plt.xlabel("Mean interval width (K)")
    plt.ylabel("Empirical coverage")
    plt.title("Coverage–width trade-off (conformal alpha sweep)")
    plt.legend()
    savefig(out_path)
