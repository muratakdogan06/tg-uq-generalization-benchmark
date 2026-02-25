from __future__ import annotations
import numpy as np

def conformal_quantiles(residuals: np.ndarray, alphas):
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[~np.isnan(residuals)]
    if residuals.size == 0:
        raise ValueError("No residuals for conformal calibration.")
    qs = {}
    for a in alphas:
        a = float(a)
        n = residuals.size
        k = int(np.ceil((n + 1) * (1.0 - a)))
        k = min(max(k, 1), n)
        q = float(np.partition(residuals, k - 1)[k - 1])
        qs[a] = q
    return qs

def interval_metrics(y_true, y_pred, qhat):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lo = y_pred - float(qhat)
    hi = y_pred + float(qhat)
    cov = float(np.mean((y_true >= lo) & (y_true <= hi)))
    width = float(np.mean(hi - lo))
    return cov, width
