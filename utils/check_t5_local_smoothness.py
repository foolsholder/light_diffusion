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
from transformers import BertTokenizerFast, T5TokenizerFast
from transformers import T5Config, BertConfig
from diffusion.utils import dict_to_device
from tqdm.auto import trange
from diffusion import Config
import diffusion
from datasets import load_dataset
from diffusion.models.contextual_denoising.modeling_noisy_encoder import BertEncoderPlusSlavaHead
from diffusion.models.contextual_denoising.modeling_clean_encoder import T5EncoderPlusSlavaHead
from torchmetrics import BLEUScore


def main(count: int = 64, batch_size: int = 64, peshechka: float = 0.3):
    seed_everything(1337, workers=True)

    bert_tok: BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased')
    t5_tok: T5TokenizerFast = T5TokenizerFast.from_pretrained('t5-base')

    dataset = load_dataset("Graphcore/wikipedia-bert-128", split='train')
    dataset.remove_columns(["token_type_ids", "labels", "next_sentence_label"])
    dt = dataset.train_test_split(test_size=0.001, seed=0)
    dataset = dt["test"]
    device = 'cuda:0'

    config = T5Config.from_pretrained('t5-base')
    model = T5EncoderPlusSlavaHead.from_pretrained('t5-base')
    model.load_head()
    for param in model.parameters():
        param.requires_grad = False
    model.eval().to(device)

    save_folder = osp.join('generated_texts', "local_smoothness")
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
    for peshechka in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        metric = BLEUScore().to(device)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False
        )
        iterator = iter(dataloader)
        bar = trange(0, count, batch_size)
        for _ in bar:
            batch = next(iterator)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            true_str = bert_tok.batch_decode(input_ids, skip_special_tokens=True)

            batch = t5_tok(true_str, padding=True, max_length=128, return_tensors="pt")
            batch = dict_to_device(batch, device)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            logits = model.forward(input_ids=input_ids, attention_mask=attention_mask)
            # [BS; SEQ_LEN; |Vocab|]
            restored_ids = logits.argmax(dim=-1)

            random_ids = torch.randint_like(input_ids, high=len(bert_tok.vocab))
            mask = torch.rand_like(input_ids.float()) < peshechka
            input_ids_2 = torch.where(mask, random_ids, input_ids)
            logits_2 = model.forward(input_ids=input_ids_2, attention_mask=attention_mask)
            restored_ids_2 = logits_2.argmax(dim=-1)

            restored_str = t5_tok.batch_decode(restored_ids, skip_special_tokens=True)
            restored_str_2 = t5_tok.batch_decode(restored_ids_2, skip_special_tokens=True)
            restored_str_2 = [[x] for x in restored_str_2]
            metric.update(restored_str, restored_str_2)
            bar.set_description(f'bleu_metric: {metric.compute().item():.5f}, p: {peshechka}')
        print(f'bleu_metric: {metric.compute().item():.5f}, p: {peshechka}\n', flush=True)



import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--count', default=8192, type=int)
    parser.add_argument('--p', default=0.3, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    os.environ['EXP_PATH'] = osp.abspath('experiments/')
    os.environ['BASE_PATH'] = osp.abspath('./')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parse_args()
    main(args.count, args.batch_size, args.p)
