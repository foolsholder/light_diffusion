import hydra
import os
import os.path as osp
import json
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
from tqdm.auto import trange
from diffusion import Config
import diffusion

from torchmetrics import MeanSquaredError


def main(exp_folder: str, ckpt_name: str, use_ema: bool = False,
         count: int = 64, batch_size: int = 64,
         N: int = 200, empty: bool = False):
    seed_everything(1337, workers=True)

    cfg = OmegaConf.load(osp.join(exp_folder, 'config.yaml'))
    cfg.lightning_wrapper.sde_cfg.N = N
    cfg.lightning_wrapper.sde_cfg.ode_sampling = True

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

    cfg: diffusion.Config
    cfg.datamodule.train_dataloader_cfg.batch_size = batch_size

    datamodule: diffusion.SimpleDataModule = instantiate(cfg.datamodule, _recursive_=False)
    wrapped_model: diffusion.lightning_wrappers.contextual_denoising.ContextualDenoising
    wrapped_model.noisy_part_encoder.restore_decoder()

    save_folder = osp.join('generated_texts', prefix_folder + osp.basename(exp_folder))
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
    datamodule.setup()

    if empty:
        datamodule.valid_dataset.setup_empty_cond(True)

    loader: DataLoader = datamodule.val_dataloader()[0]
    device = 'cuda:0'
    iter_loader = iter(loader)
    wrapped_model.to(device)

    generated_text = []
    gt_text = []
    conditions = []
    
    mse_metric = MeanSquaredError().to(device)
    
    for _ in trange(0, count, batch_size):
        batch = next(iter_loader)
        batch = dict_to_device(batch, device)

        latents, true_normed_x0 = wrapped_model.ode_forward_dynamic(batch)
        generated_ids, gen_normed_x0 = wrapped_model.generate_text(batch, init_x=latents)
        mse_metric.update(gen_normed_x0, true_normed_x0)

        dataset: diffusion.dataset.wiki_dataset.WikiDataset = loader.dataset
        tokenizer = dataset.noisy_tokenizer
        for gt, sent, attn_mask in zip(batch['noisy_input_ids'],
                                    generated_ids,
                                    batch['noisy_attention_mask']):
            # sent = sent[:sum(attn_mask)]
            generated_text += [tokenizer.decode(sent, skip_special_tokens=True)]
            gt_text += [tokenizer.decode(gt, skip_special_tokens=True)]
        condition = dataset.clean_tokenizer.batch_decode(
            batch['clean_input_ids'], skip_special_tokens=True
        )
        conditions += condition

    assert len(conditions) == len(gt_text)
    assert len(gt_text) == len(generated_text)
    assert len(conditions) >= count

    to_json_format: List[Dict[str, str]] = []
    with open(osp.join(save_folder, Path(ckpt_name).stem + '.txt'), 'w') as fout:
        for fst, snd, gt in zip(conditions, generated_text, gt_text):
            print("CONDITION:", fst, file=fout)
            print("GENERATED:", snd, file=fout)
            print("GT:", gt, file=fout)
            print("-" * 100, file=fout)
            to_json_format += [
                {
                    'CONDITION': fst,
                    'GENERATED': snd,
                    'GT': gt
                }
            ]
    suffix = f'_restore_ode_{N}.json'
    if empty:
        suffix = '_empty' + suffix

    with open(osp.join(save_folder, Path(ckpt_name).stem + suffix), 'w') as fout:
        json.dump(to_json_format, fout, indent=4)
    if False:
        with open(osp.join(save_folder, Path(ckpt_name).stem + suffix), 'w') as fout:
            json.dump({"mse": float(mse_metric.compute().detach().cpu().item()), "texts": to_json_format}, fout, indent=4)
    print(f"MSE {N}:",  mse_metric.compute())


import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_ckpt', type=str)
    parser.add_argument('--ema', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--count', default=64, type=int)
    parser.add_argument('--N', default=200, type=int)
    parser.add_argument('--empty', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--batch_size', default=64, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    os.environ['EXP_PATH'] = osp.abspath('experiments/')
    os.environ['BASE_PATH'] = osp.abspath('./')
    args = parse_args()
    path = Path(args.path_to_ckpt)
    main(path.parent, path.name, args.ema, args.count, args.batch_size, args.N, args.empty)
