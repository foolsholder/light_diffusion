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


@hydra.main(version_base=None, config_path='../configs', config_name='first_voc2_sst2')
def main(cfg: Config):
    yaml_cfg = OmegaConf.to_yaml(cfg)
    print(yaml_cfg)

    obj = instantiate(cfg.lightning_wrapper)
    print(type(obj))
    data: diffusion.GlueDataModule = instantiate(cfg.datamodule, _recursive_=False)
    print(data.train_dataset_cfg)
    data.setup("fit")
    loader = data.train_dataloader()
    iter_loader = iter(loader)
    next(iter_loader)

if __name__ == '__main__':
    main()