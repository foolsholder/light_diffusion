import hydra
import os
import os.path as osp

import torch

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from typing import Dict, List, Optional, Union, Tuple
from torch.utils.data import DataLoader
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from glob import glob
from torch_ema import ExponentialMovingAverage
from pathlib import Path
from transformers import BertTokenizerFast
from diffusion.utils import dict_to_device
from diffusion import Config
import diffusion


def main(exp_folder: str, ckpt_name: str, use_ema: bool = False):
    seed_everything(1337, workers=True)

    cfg = OmegaConf.load(osp.join(exp_folder, 'config.yaml'))
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
    wrapped_model.eval()
    datamodule: diffusion.SimpleDataModule = instantiate(cfg.datamodule, _recursive_=False)
    wrapped_model: diffusion.lightning_wrappers.contextual_denoising.ContextualDenoising

    save_folder = osp.join('generated_texts', prefix_folder + osp.basename(exp_folder))
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
    datamodule.setup()
    train_loader: DataLoader = datamodule.train_dataloader()
    device = 'cuda:0'
    batch = next(iter(train_loader))
    batch = dict_to_device(batch, device)
    wrapped_model.to(device)
    generated_ids = wrapped_model.generate_text(batch)
    dataset: diffusion.dataset.wiki_dataset.WikiDataset = train_loader.dataset
    tokenizer = dataset.noisy_tokenizer
    generated_text = []
    for sent, attn_mask in zip(generated_ids, batch['noisy_attention_mask']):
        sent = sent[:sum(attn_mask)]
        generated_text += [tokenizer.decode(sent)]

    with open(osp.join(save_folder, Path(ckpt_name).stem + '.txt'), 'w') as fout:
        condition = dataset.clean_tokenizer.batch_decode(
            batch['clean_input_ids'], skip_special_tokens=True
        )
        for fst, snd in zip(condition, generated_text):
            print("CONDITION:", fst, file=fout)
            print("GENERATED:", snd, file=fout)
            print("-" * 100, file=fout)

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_ckpt', type=str)
    parser.add_argument('--ema', default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()

if __name__ == '__main__':
    os.environ['EXP_PATH'] = osp.abspath('experiments/')
    os.environ['BASE_PATH'] = osp.abspath('./')
    args = parse_args()
    path = Path(args.path_to_ckpt)
    main(path.parent, path.name, args.ema)
