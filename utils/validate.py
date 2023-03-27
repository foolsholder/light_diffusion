import hydra
import os
import os.path as osp

import torch

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from typing import Dict, List, Optional, Union, Tuple

from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from glob import glob

from diffusion import Config


def main(exp_folder: str, ckpt_num: int):
    seed_everything(1337, workers=True)

    cfg = OmegaConf.load(osp.join(exp_folder, 'config.yaml'))
    yaml_cfg = OmegaConf.to_yaml(cfg)
    print(yaml_cfg)
    print(osp.abspath('.'))

    wrapped_model = instantiate(cfg.lightning_wrapper, _recursive_=False)
    ckpt_path = osp.join(exp_folder, f'step={ckpt_num}.ckpt')
    print(f'ckpt_path={ckpt_path}')
    wrapped_model.load_state_dict(
        torch.load(
            ckpt_path,
            map_location='cpu'
        ),
        strict=False
    )
    wrapped_model.eval()

    trainer = Trainer(
        accelerator='auto',
        precision='16-mixed'
    )

    cfg.datamodule.valid_dataloader_cfg.batch_size = 4
    trainer.test(
        wrapped_model,
        datamodule=instantiate(cfg.datamodule, _recursive_=False)
    )

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_folder', type=str)
    parser.add_argument('ckpt_num', type=int)
    return parser.parse_args()

if __name__ == '__main__':
    os.environ['EXP_PATH'] = osp.abspath('experiments/')
    args = parse_args()
    main(args.exp_folder, args.ckpt_num)
