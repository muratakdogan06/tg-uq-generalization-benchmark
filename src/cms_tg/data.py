from __future__ import annotations
import os
import pandas as pd
from .utils import ensure_dir, write_json, sha256_file

def load_and_freeze_data(cfg):
    ensure_dir(cfg.data_dir)
    raw_path = os.path.join(cfg.data_dir, "raw.csv")
    proc_path = os.path.join(cfg.data_dir, "processed.csv")

    df = pd.read_csv(cfg.data.data_csv)
    df.to_csv(raw_path, index=False)

    df = df[[cfg.data.smiles_col, cfg.data.target_col]].copy()
    df = df.dropna(subset=[cfg.data.smiles_col, cfg.data.target_col])
    df[cfg.data.target_col] = pd.to_numeric(df[cfg.data.target_col], errors="coerce")
    df = df.dropna(subset=[cfg.data.target_col]).reset_index(drop=True)

    df.to_csv(proc_path, index=False)

    dataset_id = sha256_file(proc_path)[:16]
    write_json(os.path.join(cfg.data_dir, "dataset_meta.json"), {
        "dataset_id": dataset_id,
        "raw_sha256": sha256_file(raw_path),
        "processed_sha256": sha256_file(proc_path),
        "n_rows_processed": int(len(df)),
        "smiles_col": cfg.data.smiles_col,
        "target_col": cfg.data.target_col,
        "data_csv": cfg.data.data_csv,
    })
    return df, dataset_id
