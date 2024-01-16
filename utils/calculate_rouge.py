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
from rouge_score import rouge_scorer

import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPTNeoForCausalLM
from torch.nn.functional import cross_entropy
from typing import List
from torchmetrics import MeanMetric


def process_eval(gen_list: List[str], tgt_list: List[str]):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    avg_score = []
    for index, sentence in enumerate(tqdm(gen_list)):

        target = tgt_list[index]
        scores = scorer.score(sentence, target)

        rouge_1 = scores['rouge1'].fmeasure
        rouge_2 = scores['rouge2'].fmeasure
        rouge_l = scores['rougeL'].fmeasure

        avg_score.append({'rouge1': rouge_1, 'rouge2': rouge_2, 'rougeL': rouge_l})

    rouge_1 = 0
    rouge_2 = 0
    rouge_l = 0
    for score_dict in avg_score:
        rouge_1 += score_dict['rouge1']
        rouge_2 += score_dict['rouge2']
        rouge_l += score_dict['rougeL']
    avg_rouge_1 = rouge_1 / len(avg_score)
    avg_rouge_2 = rouge_2 / len(avg_score)
    avg_rouge_l = rouge_l / len(avg_score)


    scores = {'avg_rouge_1':avg_rouge_1, 'avg_rouge_2':avg_rouge_2, 'avg_rouge_l':avg_rouge_l }

    return scores

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

    gen_list = []
    tgt_list = []
    for sent in texts:
        gen_list += [sent["CONDITION"]]
        tgt_list += [sent["GENERATED"]]
    metrics = process_eval(gen_list, tgt_list)
    # print(metrics)
    with open(osp.join(save_folder, Path(ckpt_name).stem + '_rouge.json'), 'w') as fout:
        json.dump(metrics, fout, indent=4)

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
