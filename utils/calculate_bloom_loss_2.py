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


class BloomMetric:
    def __init__(self, device="cpu"):
        self.name = "bigscience/bloom-7b1"
        self.model = BloomForCausalLM.from_pretrained(self.name).eval().to(device)
        self.tokenizer = BloomTokenizerFast.from_pretrained(self.name)
        self.device = device

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def __call__(self, text, reduce="mean"):
        if not text:
            return 0, 0
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = dict_to_device(inputs, self.device)

        loss = self.model(**inputs, labels=inputs["input_ids"]).loss.detach().cpu()
        num_tokens = torch.sum(inputs["attention_mask"]).item() - 1
        if reduce == "sum":
            return loss.item() * num_tokens, num_tokens
        return loss.item(), num_tokens


class BloomMetricConditional:
    def __init__(self, device="cpu"):
        self.name = "bigscience/bloom-7b1"
        self.model = BloomForCausalLM.from_pretrained(self.name).eval().to(device)
        self.tokenizer = BloomTokenizerFast.from_pretrained(self.name)
        self.device = device

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def __call__(self, cond_text: str, gen_text: str, reduce="mean"):
        # the first word is necessary for tokens to start with an unnecessary word, because metric doeesn't count it
        inputs = self.tokenizer(f" {cond_text} {gen_text}", return_tensors="pt")
        inputs_gen = self.tokenizer(f"{gen_text}", return_tensors="pt")

        inputs = dict_to_device(inputs, self.device)
        outputs = self.model(**inputs, labels=inputs["input_ids"])

        losses = cross_entropy(
            input=outputs.logits.reshape(-1, outputs.logits.shape[-1])[:-1],
            target=inputs["input_ids"].reshape(-1)[1:],
            reduce=False,
        )
        losses = losses[-torch.sum(inputs_gen["attention_mask"]).item():]
        num_tokens = losses.shape[0]
        loss = torch.mean(losses)

        assert num_tokens == torch.sum(inputs_gen["attention_mask"]).item()

        if reduce == "sum":
            return loss.item() * num_tokens, num_tokens
        return loss.item(), num_tokens


class GPTMetric:
    def __init__(self, device="cpu"):
        self.name = "gpt2-large"
        self.model = GPT2LMHeadModel.from_pretrained(self.name).eval().to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.name)
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


class GPTNEOMetric:
    def __init__(self, device="cpu"):
        self.name = "EleutherAI/gpt-neo-2.7B"
        self.model = GPTNeoForCausalLM.from_pretrained(self.name).eval().to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.name)
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


class RobertaMetric:
    def __init__(self, device: str = "cpu"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.name = "textattack/roberta-base-CoLA"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.device = device

    @torch.no_grad()
    def __call__(self, texts: List[str], reduce="mean"):
        input = self.tokenizer(texts, padding=True)
        input = {k: torch.LongTensor(v) for k, v in input.items()}
        input = dict_to_device(input, self.device)
        output = self.model(**input)
        probs = torch.softmax(output.logits, -1)[:, 1]
        naturalness = torch.mean((probs.round() == 1) * 1.)
        return naturalness.item(), 1


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
    #bloom_metric = BloomMetric(device)
    bloom_cond_metric = BloomMetricConditional(device)
    roberta_metric = RobertaMetric(device)

    #bloom_mean = MeanMetric()
    bloom_cond_mean = MeanMetric()
    roberta_mean = MeanMetric()

    total_loss: float = 0
    total_count: int = 0
    bar = tqdm(texts)
    for sent in bar:
        #sum_loss, num_toks = bloom_metric(sent["CONDITION"] + " " + sent["GENERATED"], reduce='sum')
        #bloom_mean.update(sum_loss / num_toks, num_toks)

        #total_loss += sum_loss
        #total_count += num_toks
        #bar.set_description(f'bloom_loss: {total_loss / total_count:.4f}')

        sum_loss, num_toks = bloom_cond_metric(sent["CONDITION"], sent["GENERATED"], reduce='sum')
        bloom_cond_mean.update(sum_loss / num_toks, num_toks)

        sum_loss, num_toks = roberta_metric([sent["CONDITION"] + " " + sent["GENERATED"]], reduce='sum')
        roberta_mean.update(sum_loss / num_toks, num_toks)

    with open(osp.join(save_folder, Path(ckpt_name).stem + '.json'), 'w') as fout:
        json.dump({
            #"condition+generated": bloom_mean.compute().item(),
            "generated": bloom_cond_mean.compute().item(),
            "naturalness": roberta_mean.compute().item()
        }, fout, indent=4)

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
