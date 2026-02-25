from __future__ import annotations
import numpy as np

def rmse(y, yhat) -> float:
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def mae(y, yhat) -> float:
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat)))

def r2(y, yhat) -> float:
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
