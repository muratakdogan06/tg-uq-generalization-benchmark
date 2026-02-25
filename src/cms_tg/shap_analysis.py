from __future__ import annotations
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from .utils import ensure_dir
from .models.xgb_model import tune_xgb, fit_xgb

def _sample_rows(X: np.ndarray, n: int, seed: int):
    n = int(n)
    if X.shape[0] <= n:
        return np.arange(X.shape[0])
    rng = np.random.RandomState(int(seed))
    return rng.choice(np.arange(X.shape[0]), size=n, replace=False)

def run_shap(cfg, X_df: pd.DataFrame, y: np.ndarray):
    if not cfg.shap.enabled:
        return

    ensure_dir(cfg.figs_dir)
    ensure_dir(cfg.metrics_dir)

    X = X_df.values.astype(float)
    feature_names = list(X_df.columns)
    idx = _sample_rows(X, cfg.shap.sample_size, cfg.shap.random_seed)
    X_sample = X[idx]

    for model_name in cfg.shap.models:
        model_name = str(model_name).lower().strip()
        if model_name == "xgb":
            params = tune_xgb(cfg, X, y, seed=int(cfg.shap.random_seed))
            model = fit_xgb(cfg, X, y, seed=int(cfg.shap.random_seed), params=params)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            continue

        plt.figure()
        shap.summary_plot(
            shap_values,
            features=X_sample,
            feature_names=feature_names,
            show=False,
            plot_type="bar",
            max_display=25
        )
        out_fig = os.path.join(cfg.figs_dir, f"shap_summary_{model_name}.png")
        plt.tight_layout()
        plt.savefig(out_fig, dpi=300)
        plt.close()

        sv = np.array(shap_values)
        if sv.ndim == 3:
            sv = sv[0]
        mean_abs = np.mean(np.abs(sv), axis=0)
        top = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        top = top.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        top.to_csv(os.path.join(cfg.metrics_dir, f"shap_top_features_{model_name}.csv"), index=False)
