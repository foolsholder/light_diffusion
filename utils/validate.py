import hydra
import os
import os.path as osp

import torch
from pathlib import Path
import json

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from typing import Dict, List, Optional, Union, Tuple

from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from glob import glob
from torch_ema import ExponentialMovingAverage

from diffusion import Config
import diffusion


def main(exp_folder: str, ckpt_name: str, use_ema: bool = False):
    seed_everything(1337, workers=True)

    cfg: Config = OmegaConf.load(osp.join(exp_folder, 'config.yaml'))
    yaml_cfg = OmegaConf.to_yaml(cfg)
    print(yaml_cfg)
    print(osp.abspath('.'))

    wrapped_model = instantiate(cfg.lightning_wrapper, _recursive_=False)
    ckpt_path = osp.join(exp_folder, ckpt_name)
    print(f'ckpt_path={ckpt_path}')
    ckpt = torch.load(
        ckpt_path,
        map_location='cpu'
    )
    wrapped_model.load_state_dict(
        ckpt['state_dict'],
        strict=True
    )
    prefix_folder = 'ema_' if use_ema else ''
    if use_ema:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(wrapped_model.parameters(), 0)
        ema.load_state_dict(
            ckpt['callbacks']['EMACallback']
        )
        ema.copy_to(wrapped_model.parameters())
    #wrapped_model.score_estimator.load_state_dict(
    #    torch.load('score_estimator.pth', map_location='cpu')
    #)
    wrapped_model.eval()
    #ema = ExponentialMovingAverage(wrapped_model.parameters(), 0)
    #ema.load_state_dict(ckpt['callbacks']['EMACallback'])
    #ema.copy_to(wrapped_model.parameters())

    trainer = Trainer(
        accelerator='auto',
        precision='32'
    )

    cfg.datamodule.valid_dataloader_cfg.batch_size = 16
    metrics = trainer.test(
        wrapped_model,
        datamodule=instantiate(cfg.datamodule, _recursive_=False)
    )[0]
    exp_name = Path(exp_name).name
    with open(osp.join(os.environ['BASE_PATH'], 'metrics', exp_name, ckpt_name), 'w') as fout:
        json.dump(metrics, fout, indent=4)

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('--ema', default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()

if __name__ == '__main__':
    os.environ['EXP_PATH'] = osp.abspath('experiments/')
    os.environ['BASE_PATH'] = osp.abspath('./')
    args = parse_args()
    path = args.ckpt_path
    exp_folder = str(Path(path).parent)
    ckpt_name = Path(path).name
    main(exp_folder, ckpt_name, args.ema)
