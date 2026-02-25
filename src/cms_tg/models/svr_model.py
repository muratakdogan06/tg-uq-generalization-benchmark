from __future__ import annotations
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from typing import Dict, Any

def tune_svr(cfg, X, y, seed: int) -> Dict[str, Any]:
    trials = int(cfg.models.svr.optuna_trials)
    inner_folds = int(cfg.models.svr.optuna_inner_folds)

    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 1e-1, 1e3, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1e1, log=True),
            "gamma": trial.suggest_float("gamma", 1e-5, 1e0, log=True),
        }
        kf = KFold(n_splits=inner_folds, shuffle=True, random_state=int(seed))
        rmses = []
        for tr, va in kf.split(X):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf", cache_size=2000, **params)),
            ])
            model.fit(X[tr], y[tr])
            pred = model.predict(X[va])
            rmses.append(float(np.sqrt(np.mean((y[va] - pred) ** 2))))
        return float(np.mean(rmses))

    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best = study.best_params
    best["_optuna_best_value_rmse"] = float(study.best_value)
    return best

def fit_svr(cfg, X, y, seed: int, params: Dict[str, Any]):
    p = dict(params)
    p.pop("_optuna_best_value_rmse", None)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", cache_size=2000, **p)),
    ])
    model.fit(X, y)
    return model
