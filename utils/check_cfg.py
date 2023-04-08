import hydra
import os
import os.path as osp

from torch.utils.data import DataLoader

from json import load
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from typing import Dict, List, Optional

from diffusion import Config
import diffusion


@hydra.main(version_base=None, config_path='../configs', config_name='launch')
def main(cfg: Config):
    yaml_cfg = OmegaConf.to_yaml(cfg)
    print(yaml_cfg)

    obj = instantiate(cfg.lightning_wrapper, _recursive_=False)
    print(type(obj), obj.label_mask_pos)
    cfg.datamodule.train_dataloader_cfg.num_workers = 1
    cfg.datamodule.valid_dataloader_cfg.num_workers = 1
    data: diffusion.GlueDataModule = instantiate(cfg.datamodule, _recursive_=False)
    print(data.train_dataset_cfg)
    data.setup("fit")
    loader = data.train_dataloader()
    iter_loader = iter(loader)
    next(iter_loader)

if __name__ == '__main__':
    main()