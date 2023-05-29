import hydra
import os
import os.path as osp

import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from typing import Dict, List, Optional

from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy

from diffusion import Config
from diffusion.callbacks import EMACallback


def main(exp_folder: str, ckpt_name: str, seed: Optional[int]):
    cfg = OmegaConf.load(osp.join(exp_folder, 'config.yaml'))
    if seed is None:
        seed = cfg.seed
    seed_everything(seed)

    wrapped_model = instantiate(cfg.lightning_wrapper, _recursive_=False)

    wrapped_model.load_state_dict(
        torch.load(osp.join(exp_folder, ckpt_name), map_location='cpu')['state_dict'], strict=True
    )
    from torch_ema import ExponentialMovingAverage
    ema = ExponentialMovingAverage(wrapped_model.parameters(), 0)
    ema.load_state_dict(
        torch.load(osp.join(exp_folder, ckpt_name), map_location='cpu')['callbacks']['EMACallback']
    )
    #tmp = torch.load(osp.join(exp_folder, ckpt_name), map_location='cpu')['callbacks']['EMACallback']
    ema.copy_to(wrapped_model.parameters())
    torch.save(
        {
            "state_dict": wrapped_model.state_dict()
        },
        osp.join(exp_folder, 'ema-' + ckpt_name)
    )

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_folder', type=str)
    parser.add_argument('--ckpt_name', type=str, default='last.ckpt')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    os.environ['DATA_PATH'] = osp.abspath('data/')
    os.environ['BASE_PATH'] = './'
    os.environ['EXP_PATH'] = osp.abspath('experiments/')
    args = parse_args()
    main(args.exp_folder, args.ckpt_name, args.seed)