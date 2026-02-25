from __future__ import annotations
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.impute import SimpleImputer

def _rdkit_2d_descriptor_matrix(mols):
    desc_names = [d[0] for d in Descriptors._descList]
    funcs = [d[1] for d in Descriptors._descList]
    rows = []
    for m in mols:
        vals = []
        for f in funcs:
            try:
                vals.append(float(f(m)))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
    return pd.DataFrame(rows, columns=desc_names)

def _polymer_proxy_features(mols):
    rows = []
    for m in mols:
        heavy = m.GetNumHeavyAtoms()
        rings = int(m.GetRingInfo().NumRings())
        arom = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
        hetero = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (1, 6))
        hal = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))
        denom = heavy if heavy > 0 else 1
        rows.append({
            "proxy_heavy_atoms": heavy,
            "proxy_rings": rings,
            "proxy_aromatic_frac": arom / denom,
            "proxy_hetero_frac": hetero / denom,
            "proxy_halogen_frac": hal / denom,
        })
    return pd.DataFrame(rows)

def featurize(cfg, smiles):
    mols, ok = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            ok.append(False); mols.append(None)
        else:
            ok.append(True); mols.append(m)

    valid_idx = np.where(ok)[0]
    mols_valid = [mols[i] for i in valid_idx]

    X = _rdkit_2d_descriptor_matrix(mols_valid)
    if cfg.features.add_polymer_proxy_features:
        X = pd.concat([X, _polymer_proxy_features(mols_valid)], axis=1)

    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    return X_imp, valid_idx, mols_valid, imp

def morgan_fingerprints(cfg, mols_valid):
    fps = []
    for m in mols_valid:
        fp = GetMorganFingerprintAsBitVect(m, int(cfg.eval.morgan_radius), nBits=int(cfg.eval.morgan_nbits))
        fps.append(fp)
    return fps
