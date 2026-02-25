from __future__ import annotations
import numpy as np
import optuna
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from typing import Dict, Any

def tune_xgb(cfg, X, y, seed: int) -> Dict[str, Any]:
    trials = int(cfg.models.xgb.optuna_trials)
    inner_folds = int(cfg.models.xgb.optuna_inner_folds)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-6, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        }
        kf = KFold(n_splits=inner_folds, shuffle=True, random_state=int(seed))
        rmses = []
        for tr, va in kf.split(X):
            m = XGBRegressor(
                objective="reg:squarederror",
                random_state=int(seed),
                n_jobs=int(cfg.models.xgb.n_jobs),
                **params,
            )
            m.fit(X[tr], y[tr])
            pred = m.predict(X[va])
            rmses.append(float(np.sqrt(np.mean((y[va] - pred) ** 2))))
        return float(np.mean(rmses))

    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best = study.best_params
    best["_optuna_best_value_rmse"] = float(study.best_value)
    return best

def fit_xgb(cfg, X, y, seed: int, params: Dict[str, Any]) -> XGBRegressor:
    p = dict(params)
    p.pop("_optuna_best_value_rmse", None)
    p.pop("objective", None)
    p.pop("n_jobs", None)
    p.pop("random_state", None)
    p.pop("seed", None)

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=int(seed),
        n_jobs=int(cfg.models.xgb.n_jobs),
        **p,
    )
    model.fit(X, y)
    return model
