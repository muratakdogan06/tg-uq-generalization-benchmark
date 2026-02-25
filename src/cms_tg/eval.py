from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .data import load_and_freeze_data
from .features import featurize, morgan_fingerprints
from .splits import get_splits
from .metrics import rmse, mae, r2
from .uncertainty import conformal_quantiles, interval_metrics
from .similarity import max_tanimoto_test_to_train, similarity_records
from .plots import plot_cov_width_curve, plot_similarity_violin, plot_similarity_ecdf
from .utils import ensure_dir, write_json, env_snapshot
from .shap_analysis import run_shap

from .models.xgb_model import tune_xgb, fit_xgb
from .models.svr_model import tune_svr, fit_svr

def _train_predict(cfg, model_name: str, X_fit, y_fit, X_cal, y_cal, X_te, seed: int):
    model_name = model_name.lower().strip()
    if model_name == "xgb":
        params = tune_xgb(cfg, X_fit, y_fit, seed=seed)
        model = fit_xgb(cfg, X_fit, y_fit, seed=seed, params=params)
    elif model_name == "svr":
        params = tune_svr(cfg, X_fit, y_fit, seed=seed)
        model = fit_svr(cfg, X_fit, y_fit, seed=seed, params=params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    yhat_te = model.predict(X_te)
    yhat_cal = model.predict(X_cal) if (cfg.uncertainty.enabled and X_cal is not None) else None
    return yhat_te, yhat_cal, params

def run_benchmark(cfg):
    ensure_dir(cfg.run_dir)
    ensure_dir(cfg.metrics_dir)
    ensure_dir(cfg.figs_dir)

    df, dataset_id = load_and_freeze_data(cfg)
    X_df, valid_idx, mols_valid, _ = featurize(cfg, df[cfg.data.smiles_col].values)
    df = df.iloc[valid_idx].reset_index(drop=True)
    y = df[cfg.data.target_col].values.astype(float)
    fps = morgan_fingerprints(cfg, mols_valid)

    # SHAP on descriptor space
    run_shap(cfg, X_df, y)

    rows, sim_rows = [], []
    alphas = list(cfg.uncertainty.alphas) if cfg.uncertainty.enabled else [None]
    model_list = [m.lower().strip() for m in cfg.models.enabled]

    for base_regime in cfg.eval.regimes:
        cutoffs = [None] if base_regime != "cluster" else list(cfg.eval.cluster_cutoffs)
        for cutoff in cutoffs:
            regime_label = base_regime if cutoff is None else f"cluster_c{cutoff:.2f}"
            for seed in cfg.eval.seeds:
                splits = get_splits(cfg, y, mols_valid, fps, dataset_id, base_regime, seed, cutoff=cutoff)

                for fold, (tr_idx, te_idx) in enumerate(splits):
                    X_te = X_df.values[te_idx]
                    y_te = y[te_idx]

                    # Similarity diagnostic
                    max_sims = max_tanimoto_test_to_train(fps, tr_idx, te_idx)
                    sim_rows.extend(similarity_records(base_regime, seed, fold, cutoff, te_idx, max_sims))

                    n_tr = len(tr_idx)
                    for frac in cfg.eval.learning_fractions:
                        frac = float(frac)
                        n_sub = max(10, int(np.floor(frac * n_tr)))
                        rng = np.random.RandomState(int(seed) + 1000 * fold + int(frac * 100))
                        sub_local = rng.choice(np.arange(n_tr), size=n_sub, replace=False)
                        sub_idx = tr_idx[sub_local]

                        # Conformal split inside sub-train
                        if cfg.uncertainty.enabled:
                            n_cal = max(10, int(np.floor(0.2 * len(sub_idx))))
                            perm = rng.permutation(len(sub_idx))
                            cal_local = perm[:n_cal]
                            fit_local = perm[n_cal:]
                            fit_idx = sub_idx[fit_local]
                            cal_idx = sub_idx[cal_local]
                            X_fit = X_df.values[fit_idx]; y_fit = y[fit_idx]
                            X_cal = X_df.values[cal_idx]; y_cal = y[cal_idx]
                        else:
                            X_fit = X_df.values[sub_idx]; y_fit = y[sub_idx]
                            X_cal = y_cal = None

                        for model_name in model_list:
                            yhat_te, yhat_cal, params = _train_predict(
                                cfg, model_name, X_fit, y_fit, X_cal, y_cal, X_te, seed=int(seed)
                            )

                            base_metrics = {
                                "regime": regime_label,
                                "base_regime": base_regime,
                                "cutoff": None if cutoff is None else float(cutoff),
                                "model": model_name,
                                "seed": int(seed),
                                "fold": int(fold),
                                "frac": float(frac),
                                "RMSE": rmse(y_te, yhat_te),
                                "MAE": mae(y_te, yhat_te),
                                "R2": r2(y_te, yhat_te),
                                "optuna_best_rmse_innercv": float(params.get("_optuna_best_value_rmse", np.nan)),
                            }

                            if cfg.uncertainty.enabled:
                                resid = np.abs(y_cal - yhat_cal)
                                qhats = conformal_quantiles(resid, alphas)
                                for a, q in qhats.items():
                                    cov, width = interval_metrics(y_te, yhat_te, q)
                                    row = dict(base_metrics)
                                    row.update({
                                        "alpha": float(a),
                                        "nominal_coverage": float(1.0 - float(a)),
                                        "cov": float(cov),
                                        "width": float(width),
                                        "qhat": float(q),
                                    })
                                    rows.append(row)
                            else:
                                row = dict(base_metrics)
                                row.update({"alpha": np.nan, "nominal_coverage": np.nan, "cov": np.nan, "width": np.nan, "qhat": np.nan})
                                rows.append(row)

    results = pd.DataFrame(rows)
    results_path = os.path.join(cfg.metrics_dir, "results.csv")
    results.to_csv(results_path, index=False)

    summ = (results[results["frac"] == 1.0]
            .groupby(["regime", "model", "alpha"], dropna=False)
            .agg(RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                 MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                 R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                 cov_mean=("cov", "mean"), cov_std=("cov", "std"),
                 width_mean=("width", "mean"), width_std=("width", "std"),
                 n=("RMSE", "count"))
            .reset_index())
    summ_path = os.path.join(cfg.metrics_dir, "summary_frac1_alpha.csv")
    summ.to_csv(summ_path, index=False)

    sim_df = pd.DataFrame(sim_rows)
    sim_df["regime_label"] = sim_df.apply(
        lambda r: r["regime"] if r["cutoff"] is None else f"cluster_c{float(r['cutoff']):.2f}", axis=1
    )
    sim_path = os.path.join(cfg.metrics_dir, "similarity_test_to_train.csv")
    sim_df.to_csv(sim_path, index=False)

    # Coverage-width curve per model
    for model_name in sorted(results["model"].unique()):
        summ_m = summ[summ["model"] == model_name].copy()
        if len(summ_m) == 0:
            continue
        plot_cov_width_curve(
            summ_m.assign(regime_label=summ_m["regime"]),
            os.path.join(cfg.figs_dir, f"coverage_width_curve_frac1_{model_name}.png")
        )

    plot_similarity_violin(
        sim_df[["regime_label", "max_tanimoto_to_train"]],
        os.path.join(cfg.figs_dir, "similarity_violin.png")
    )
    plot_similarity_ecdf(
        sim_df[["regime_label", "max_tanimoto_to_train"]],
        os.path.join(cfg.figs_dir, "similarity_ecdf.png")
    )

    write_json(os.path.join(cfg.run_dir, "run_meta.json"), {
        "dataset_id": dataset_id,
        "env": env_snapshot(),
        "enabled_models": model_list,
        "outputs": {
            "results": results_path,
            "summary_frac1_alpha": summ_path,
            "similarity_csv": sim_path,
            "figs_dir": cfg.figs_dir,
        }
    })

    return results, summ, sim_df
