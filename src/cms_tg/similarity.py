from __future__ import annotations
import numpy as np
from rdkit import DataStructs

def max_tanimoto_test_to_train(fps, train_idx, test_idx):
    train_fps = [fps[i] for i in train_idx]
    out = np.zeros(len(test_idx), dtype=float)
    for j, i in enumerate(test_idx):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)
        out[j] = float(max(sims)) if len(sims) else 0.0
    return out

def similarity_records(regime: str, seed: int, fold: int, cutoff, test_idx, max_sims):
    rows = []
    for idx, s in zip(test_idx, max_sims):
        rows.append({
            "regime": regime,
            "seed": int(seed),
            "fold": int(fold),
            "cutoff": None if cutoff is None else float(cutoff),
            "test_local_index": int(idx),
            "max_tanimoto_to_train": float(s),
        })
    return rows
