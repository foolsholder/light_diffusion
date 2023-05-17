import hydra
import os
import os.path as osp

import torch

from tqdm.auto import tqdm, trange
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from typing import Dict, List, Optional, Union, Tuple

from torchmetrics.classification import BinaryAccuracy
from lightning import seed_everything, Trainer
from glob import glob
from torch_ema import ExponentialMovingAverage


from diffusion import Config
from diffusion.utils import dict_to_device
import diffusion


@torch.no_grad()
def half_time_restore_label(
    wrapper: diffusion.FirstVoc2, val_loader: DataLoader,
    binary_label: int, device: str = 'cpu', step_init: int = 500
) -> Union[float, float]:
    accuracy = BinaryAccuracy()
    true_accuracy = BinaryAccuracy()
    true_accuracy.to(device)
    accuracy.to(device)
    wrapper.to(device)

    label_30k = binary_label * 2748 + (1 - binary_label) * 2053

    for batch in tqdm(val_loader):
        batch = dict_to_device(batch, device)

        input_ids = batch['input_ids']
        batch_true_labels = batch['labels'].view(-1)
        batch_size = len(input_ids)

        input_ids[:, wrapper.label_mask_pos] = label_30k
        attn_mask = batch['attention_mask']
        encodings = wrapper.encoder.bert(input_ids=input_ids, attention_mask=attn_mask)[0]
        clean_x = wrapper.enc_normalizer.normalize(encodings)

        ddrm_mask = torch.ones_like(attn_mask)
        ddrm_mask[:, wrapper.label_mask_pos] = 0
        ddrm_mask = ddrm_mask.bool()

        tiled_ddrm_input = wrapper.tile_input({
            'clean_x': clean_x,
            'attn_mask': attn_mask,
            'ddrm_mask': ddrm_mask
        })
        clean_x = tiled_ddrm_input.pop('clean_x')
        attn_mask = tiled_ddrm_input.pop('attn_mask')
        ddrm_mask = tiled_ddrm_input.pop('ddrm_mask')

        aug_batch_size = len(clean_x)
        start_time = wrapper.sde.T * (step_init / float(wrapper.sde.N))
        last_time = wrapper.sde.T / wrapper.sde.N
        start_time_tensor = torch.ones(aug_batch_size, device=clean_x.device) * start_time
        x_t = wrapper.sde.marginal_forward(clean_x, start_time_tensor)['x_t']
        timesteps = torch.linspace(start_time, last_time, step_init, device=x_t.device)

        x_t = wrapper.ddrm_cycle(clean_x, ddrm_mask, attn_mask, x_t, timesteps, gamma=1)

        pred_encodings = wrapper.enc_normalizer.denormalize(x_t)
        logits = wrapper.encoder.forward(pred_encodings=pred_encodings).logits

        logits_binary = logits[:, wrapper.label_mask_pos, [2053, 2748]]
        pred_label = torch.argmax(logits_binary, dim=-1).float().reshape(wrapper.test_count, batch_size).mean(dim=0)
        accuracy.update(pred_label, torch.ones_like(pred_label).long() * binary_label)
        true_accuracy.update(pred_label.view(-1), batch_true_labels)
        print(f'synthetic={accuracy.compute()}, true_accuracy={true_accuracy.compute()}', binary_label, step_init)
    return accuracy.compute(), true_accuracy.compute()



def main(exp_folder: str, ckpt_num: int, use_ema: bool = False, step_init: int = 500):
    seed_everything(1337, workers=True)

    cfg = OmegaConf.load(osp.join(exp_folder, 'config.yaml'))
    yaml_cfg = OmegaConf.to_yaml(cfg)
    print(yaml_cfg)
    print(osp.abspath('.'))

    wrapped_model = instantiate(cfg.lightning_wrapper, _recursive_=False)
    ckpt_path = osp.join(exp_folder, f'step={ckpt_num}.ckpt')
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
        ema = ExponentialMovingAverage(wrapped_model.parameters(), 0)
        ema.load_state_dict(
            ckpt['callbacks']['EMACallback']
        )
        ema.copy_to(wrapped_model.parameters())

    wrapped_model.eval()

    cfg.datamodule.valid_dataloader_cfg.batch_size = 2
    cfg.datamodule.valid_dataloader_cfg.num_workers = 4

    datamodule: diffusion.SimpleDataModule = instantiate(cfg.datamodule, _recursive_=False)
    datamodule.setup("just_stage")
    val_loader = datamodule.val_dataloader()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'


    wrapped_model: diffusion.lightning_wrappers.ZeroVoc2

    save_folder = osp.join('stats', prefix_folder + osp.basename(exp_folder))
    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    for binary_label in [0, 1]:
        s_acc, t_acc = half_time_restore_label(wrapped_model, val_loader, binary_label, device, step_init)

        with open(osp.join(save_folder, f'{binary_label}_{step_init}_step_{ckpt_num}.txt'), 'w') as fout:
            print(f'synt:{s_acc}', file=fout)
            print(f'true:{t_acc}', file=fout)

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_folder', type=str)
    parser.add_argument('ckpt_num', type=int)
    parser.add_argument('--ema', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--step_init', type=int, default=500)
    return parser.parse_args()

if __name__ == '__main__':
    os.environ['EXP_PATH'] = osp.abspath('experiments/')
    args = parse_args()
    main(args.exp_folder, args.ckpt_num, args.ema, args.step_init)
