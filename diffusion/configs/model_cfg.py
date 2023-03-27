from dataclasses import dataclass

from .dynamic_cfg import SDECfg

@dataclass
class EMACfg:
    decay: float

@dataclass
class EncNormalizerCfg:
    ema_mean_path: str
    ema_std_path: str


@dataclass
class LightningModelCfg:
    _target_: str
    sde: SDECfg
    enc_normalizer: EncNormalizerCfg
    ema_partial: EMACfg
    ce_coef: float