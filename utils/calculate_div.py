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
from torch import Tensor
from torch_ema import ExponentialMovingAverage
from pathlib import Path
from transformers import BertTokenizerFast, BloomForCausalLM, BloomTokenizerFast
from diffusion.utils import dict_to_device
from tqdm.auto import trange, tqdm

import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPTNeoForCausalLM
from torch.nn.functional import cross_entropy
from typing import List
from torchmetrics import MeanMetric


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

    gen_texts: List[str] = [sent["GENERATED"] for sent in texts]
    from diffusion.metrics.diversity import NGramStats
    div_metrics = NGramStats()
    results = div_metrics.compute(gen_texts)
    print(json.dumps(results, indent=4))
    with open(osp.join(save_folder, Path(ckpt_name).stem + '_div.json'), 'w') as fout:
        json.dump(results, fout, indent=4)

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
