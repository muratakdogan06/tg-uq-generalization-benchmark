from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml

@dataclass
class DataConfig:
    data_csv: str
    smiles_col: str = "SMILES"
    target_col: str = "T_g (K)"

@dataclass
class EvalConfig:
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    regimes: List[str] = field(default_factory=lambda: ["stratified", "scaffold", "cluster"])
    n_folds: int = 5
    strat_bins: int = 10
    learning_fractions: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])
    cluster_cutoffs: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4])
    morgan_radius: int = 2
    morgan_nbits: int = 2048

@dataclass
class FeatureConfig:
    use_3d: bool = False
    var_thresh: float = 0.0
    add_polymer_proxy_features: bool = True

@dataclass
class XGBConfig:
    optuna_trials: int = 25
    optuna_inner_folds: int = 3
    n_jobs: int = -1

@dataclass
class SVRConfig:
    optuna_trials: int = 25
    optuna_inner_folds: int = 3

@dataclass
class ModelConfig:
    enabled: List[str] = field(default_factory=lambda: ["xgb", "svr"])
    xgb: XGBConfig = field(default_factory=XGBConfig)
    svr: SVRConfig = field(default_factory=SVRConfig)

@dataclass
class UncertaintyConfig:

    enabled: bool = True
    alphas: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])

@dataclass
class ShapConfig:
    enabled: bool = True
    models: List[str] = field(default_factory=lambda: ["xgb"])
    sample_size: int = 250
    random_seed: int = 123

@dataclass
class Config:
    run_tag: str = "tg_cms_v4"
    out_dir: str = "runs"
    data: DataConfig = field(default_factory=lambda: DataConfig(data_csv=""))
    eval: EvalConfig = field(default_factory=EvalConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    shap: ShapConfig = field(default_factory=ShapConfig)

    @property
    def run_dir(self) -> str:
        return f"{self.out_dir}/{self.run_tag}"

    @property
    def figs_dir(self) -> str:
        return f"{self.run_dir}/figs"

    @property
    def models_dir(self) -> str:
        return f"{self.run_dir}/models"

    @property
    def splits_dir(self) -> str:
        return f"{self.run_dir}/splits"

    @property
    def data_dir(self) -> str:
        return f"{self.run_dir}/data"

    @property
    def metrics_dir(self) -> str:
        return f"{self.run_dir}/metrics"

def _merge_dataclass(dc, d: Dict[str, Any]):
    for k, v in d.items():
        if hasattr(dc, k):
            cur = getattr(dc, k)
            if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
                _merge_dataclass(cur, v)
            else:
                setattr(dc, k, v)

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    cfg = Config()
    for k in ["run_tag", "out_dir"]:
        if k in d:
            setattr(cfg, k, d[k])

    if "data" in d: _merge_dataclass(cfg.data, d["data"])
    if "eval" in d: _merge_dataclass(cfg.eval, d["eval"])
    if "features" in d: _merge_dataclass(cfg.features, d["features"])
    if "models" in d: _merge_dataclass(cfg.models, d["models"])
    if "uncertainty" in d: _merge_dataclass(cfg.uncertainty, d["uncertainty"])
    if "shap" in d: _merge_dataclass(cfg.shap, d["shap"])

    if not cfg.data.data_csv:
        raise ValueError("Config.data.data_csv is required.")

    # Backward compatibility: allow uncertainty.alpha (float)
    if isinstance(getattr(cfg.uncertainty, "alphas", None), (int, float)):
        cfg.uncertainty.alphas = [float(cfg.uncertainty.alphas)]

    for a in cfg.uncertainty.alphas:
        if not (0.0 < float(a) < 1.0):
            raise ValueError(f"uncertainty.alphas must be in (0,1). Got {a}")

    return cfg
