from dataclasses import dataclass

@dataclass
class DatasetCfg:
    _target_: str
    max_length: int


@dataclass
class DataLoaderCfg:
    _target_: str
    batch_size: int
    num_workers: int
    shuffle: bool
    drop_last: bool


@dataclass
class DataModuleCfg:
    train_dataset_cfg: DatasetCfg
    valid_dataset_cfg: DatasetCfg
    train_dataloader_cfg: DataLoaderCfg
    valid_dataloader_cfg: DataLoaderCfg
