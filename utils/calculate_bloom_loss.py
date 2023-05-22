import hydra
import os
import os.path as osp
import json
import torch

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from lightning import seed_everything
from typing import Dict, List, Optional, Union, Tuple
from glob import glob
from torch_ema import ExponentialMovingAverage
from pathlib import Path
from transformers import BertTokenizerFast, BloomForCausalLM, BloomTokenizerFast
from diffusion.utils import dict_to_device
from tqdm.auto import trange, tqdm


class BloomMetric:
    def __init__(self, device="cpu"):
        self.name = "bigscience/bloom-7b1"
        self.model = BloomForCausalLM.from_pretrained(self.name).eval().to(device)
        self.tokenizer = BloomTokenizerFast.from_pretrained(self.name)
        self.device = device

    @torch.no_grad()
    def __call__(self, text, reduce="mean"):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = dict_to_device(inputs, self.device)

        loss = self.model(**inputs, labels=inputs["input_ids"]).loss.detach().cpu()
        num_tokens = torch.sum(inputs["attention_mask"]).item() - 1
        if reduce == "sum":
            return loss.item() * num_tokens, num_tokens
        return loss.item(), num_tokens


def main(generated_text_folder_name: str, ckpt_name: str):
    seed_everything(1337, workers=True)

    save_folder = osp.join('metrics', generated_text_folder_name)
    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    with open(osp.join(
            'generated_texts',
            generated_text_folder_name, Path(ckpt_name).stem + '.json'
        ), 'r') as fin:
        texts: List[Dict[str, str]] = json.load(fin)

    device = 'cuda:0'
    metric = BloomMetric(device)

    total_loss: float = 0
    total_count: int = 0
    bar = tqdm(texts)
    for sent in bar:
        sum_loss, num_toks = metric(sent["CONDITION"] + " " + sent["GENERATED"], reduce='sum')
        total_loss += sum_loss
        total_count += num_toks
        bar.set_description(f'bloom_loss: {total_loss / total_count:.4f}')

    with open(osp.join(save_folder, Path(ckpt_name).stem + '.json'), 'w') as fout:
        json.dump({"condition+generated": total_loss / total_count}, fout, indent=4)

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_json', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    os.environ['EXP_PATH'] = osp.abspath('experiments/')
    os.environ['BASE_PATH'] = osp.abspath('./')
    args = parse_args()
    path = Path(args.path_to_json)
    main(path.parent.stem, path.name)
