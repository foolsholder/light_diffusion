from dataclasses import dataclass

from .dataset_cfg import DataModuleCfg
from .model_cfg import LightningModelCfg

@dataclass
class Config:
    lightning_wrapper: LightningModelCfg
    datamodule: DataModuleCfg

    max_steps: int
    grad_clip_norm: float
    project: str
    exp_name: str
    seed: int
    every_n_train_steps: int
    pretrained_path: str
