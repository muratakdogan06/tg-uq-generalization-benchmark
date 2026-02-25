
import os
import argparse
import numpy as np
import pandas as pd
from cms_tg.config import load_config
from cms_tg.data import load_and_freeze_data
from cms_tg.features import featurize, morgan_fingerprints
from cms_tg.splits import get_splits
from cms_tg.models.xgb_model import fit_xgb, tune_xgb
from cms_tg.models.svr_model import fit_svr, tune_svr # Added SVR
from cms_tg.similarity import max_tanimoto_test_to_train
from cms_tg.utils import ensure_dir

# Hardcoded settings for reproduction
SEEDS = [42, 43, 44]
REGIMES = [
    ("stratified", None),
    ("cluster", 0.20)
]


def reproduce_oof(model_name="xgb", target_seed=None, target_fold=None):
    print(f"Loading config for {model_name}...")
    cfg = load_config("configs/tg.yaml")
    
    # Adjust output path based on model and targeting
    suffix = f"_{model_name}"
    if target_seed is not None:
        suffix += "_best"
    
    output_file = f"/Users/muratakdogan/Desktop/murat-makale-2026/tables/metadata_oof_parity{suffix}.csv"
    
    print("Loading data...")
    df, dataset_id = load_and_freeze_data(cfg)
    X_df, valid_idx, mols_valid, _ = featurize(cfg, df[cfg.data.smiles_col].values)
    df = df.iloc[valid_idx].reset_index(drop=True)
    y = df[cfg.data.target_col].values.astype(float)
    fps = morgan_fingerprints(cfg, mols_valid)
    
    oof_rows = []
    
    print(f"Starting OOF generation for {model_name} on {REGIMES}...")
    if target_seed is not None:
        print(f"  Targeting Seed: {target_seed}, Fold: {target_fold}")
    
    for base_regime, cutoff in REGIMES:
        regime_label = base_regime if cutoff is None else f"cluster_c{cutoff:.2f}"
        print(f"Processing Regime: {regime_label}")
        
        for seed in SEEDS:
            if target_seed is not None and seed != target_seed:
                continue
                
            print(f"  Seed {seed}...")
            splits = get_splits(cfg, y, mols_valid, fps, dataset_id, base_regime, seed, cutoff=cutoff)
            
            for fold, (tr_idx, te_idx) in enumerate(splits):
                if target_fold is not None and fold != target_fold:
                    continue
                    
                print(f"    Fold {fold}: Tuning & Fitting {model_name.upper()}...")
                
                # Data
                X_tr, y_tr = X_df.values[tr_idx], y[tr_idx]
                X_te, y_te = X_df.values[te_idx], y[te_idx]
                
                max_sims = max_tanimoto_test_to_train(fps, tr_idx, te_idx)
                
                try:
                    if model_name == "xgb":
                        best_params = tune_xgb(cfg, X_tr, y_tr, seed=seed)
                        model = fit_xgb(cfg, X_tr, y_tr, seed=seed, params=best_params)
                    elif model_name == "svr":
                        best_params = tune_svr(cfg, X_tr, y_tr, seed=seed)
                        model = fit_svr(cfg, X_tr, y_tr, seed=seed, params=best_params)
                    else:
                        raise ValueError(f"Unknown model: {model_name}")
                except Exception as e:
                    print(f"      Tuning failed ({e}). Using defaults/simpler approach not implemented for fallback.")
                    raise e
                
                y_pred = model.predict(X_te)
                
                smiles_te = df.iloc[te_idx][cfg.data.smiles_col].values
                
                for i in range(len(te_idx)):
                    oof_rows.append({
                        "Regime": regime_label,
                        "Seed": seed,
                        "Fold": fold,
                        "SMILES": smiles_te[i],
                        "y_true": y_te[i],
                        "y_pred": y_pred[i],
                        "Smax": max_sims[i]
                    })
                    
    ensure_dir(os.path.dirname(output_file))
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(output_file, index=False)
    print(f"OOF predictions saved to {output_file}")

if __name__ == "__main__":
    import sys
    model = "xgb"
    seed = None
    fold = None
    
    if len(sys.argv) > 1:
        model = sys.argv[1]
    if len(sys.argv) > 3:
        seed = int(sys.argv[2])
        fold = int(sys.argv[3])
        
    reproduce_oof(model, seed, fold)
