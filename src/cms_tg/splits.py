from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from sklearn.model_selection import StratifiedKFold, KFold
from rdkit.Chem.Scaffolds import MurckoScaffold
from .utils import ensure_dir

IndexSplit = List[Tuple[np.ndarray, np.ndarray]]

def stratified_splits(cfg, y, seed: int) -> IndexSplit:
    y = pd.Series(y).copy()
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.isna().any():
        raise ValueError("Target has NaNs after numeric coercion.")

    n_folds = int(cfg.eval.n_folds)
    strat_bins = int(cfg.eval.strat_bins)

    try:
        bins = pd.qcut(y_num, q=strat_bins, labels=False, duplicates="drop")
        bins = np.asarray(bins, dtype=np.int64)
    except Exception:
        bins = pd.cut(y_num, bins=strat_bins, labels=False, duplicates="drop")
        bins = np.asarray(bins, dtype=np.int64)

    if np.any(pd.isna(bins)) or len(np.unique(bins)) < 2:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
        return [(tr, te) for tr, te in kf.split(np.zeros(len(y_num)))]

    counts = np.bincount(bins[bins >= 0])
    if len(counts) == 0 or counts.min() < n_folds:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
        return [(tr, te) for tr, te in kf.split(np.zeros(len(y_num)))]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    return [(tr, te) for tr, te in skf.split(np.zeros(len(y_num)), bins)]

def scaffold_splits(cfg, mols_valid, seed: int) -> IndexSplit:
    n_folds = int(cfg.eval.n_folds)
    rng = np.random.RandomState(int(seed))

    scaff = {}
    for i, m in enumerate(mols_valid):
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        scaff.setdefault(s, []).append(i)

    groups = list(scaff.values())
    rng.shuffle(groups)

    fold_bins = [[] for _ in range(n_folds)]
    fold_sizes = np.zeros(n_folds, dtype=int)
    for g in sorted(groups, key=len, reverse=True):
        j = int(np.argmin(fold_sizes))
        fold_bins[j].extend(g)
        fold_sizes[j] += len(g)

    splits = []
    all_idx = np.arange(len(mols_valid))
    for k in range(n_folds):
        te = np.array(sorted(fold_bins[k]), dtype=int)
        tr = np.setdiff1d(all_idx, te)
        splits.append((tr, te))
    return splits

def _cluster_by_cutoff(fps, cutoff: float):
    from rdkit import DataStructs
    clusters = []
    for i in range(len(fps)):
        assigned = False
        for c in clusters:
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[c[0]])
            if sim >= cutoff:
                c.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])
    return clusters

def cluster_splits(cfg, fps, seed: int, cutoff: float) -> IndexSplit:
    n_folds = int(cfg.eval.n_folds)
    rng = np.random.RandomState(int(seed))

    clusters = _cluster_by_cutoff(fps, float(cutoff))
    rng.shuffle(clusters)

    fold_bins = [[] for _ in range(n_folds)]
    fold_sizes = np.zeros(n_folds, dtype=int)
    for c in sorted(clusters, key=len, reverse=True):
        j = int(np.argmin(fold_sizes))
        fold_bins[j].extend(c)
        fold_sizes[j] += len(c)

    splits = []
    all_idx = np.arange(len(fps))
    for k in range(n_folds):
        te = np.array(sorted(fold_bins[k]), dtype=int)
        tr = np.setdiff1d(all_idx, te)
        splits.append((tr, te))
    return splits

def get_splits(cfg, y, mols_valid, fps, dataset_id: str, regime: str, seed: int, cutoff: Optional[float] = None) -> IndexSplit:
    ensure_dir(cfg.splits_dir)

    if regime == "cluster":
        if cutoff is None:
            raise ValueError("cluster regime requires cutoff")
        key = f"{dataset_id}_{regime}_c{cutoff:.2f}_seed{seed}_k{cfg.eval.n_folds}"
    else:
        key = f"{dataset_id}_{regime}_seed{seed}_k{cfg.eval.n_folds}"

    path = os.path.join(cfg.splits_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return [(np.array(s["train"], dtype=int), np.array(s["test"], dtype=int)) for s in obj["splits"]]

    if regime == "stratified":
        splits = stratified_splits(cfg, y, seed)
    elif regime == "scaffold":
        splits = scaffold_splits(cfg, mols_valid, seed)
    elif regime == "cluster":
        splits = cluster_splits(cfg, fps, seed, float(cutoff))
    else:
        raise ValueError(f"Unknown regime: {regime}")

    obj = {"key": key, "regime": regime, "seed": int(seed), "cutoff": None if cutoff is None else float(cutoff),
           "n_folds": int(cfg.eval.n_folds),
           "splits": [{"train": tr.tolist(), "test": te.tolist()} for tr, te in splits]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return splits
