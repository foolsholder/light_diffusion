import hydra
import os
import os.path as osp
import json
import torch

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from typing import Dict, List, Optional, Union, Tuple
from torch.utils.data import DataLoader, Dataset as TorchDataset
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
from tqdm.auto import trange, tqdm
from diffusion import Config
import diffusion
from datasets import load_dataset, Dataset
from diffusion.models.contextual_denoising.modeling_noisy_encoder import BertEncoderPlusSlavaHead
from diffusion.models.contextual_denoising.modeling_clean_encoder import T5EncoderPlusSlavaHead
from torchmetrics import BLEUScore, MeanMetric
from transformers import BloomForCausalLM, BloomTokenizerFast

bert_tok: BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased')
t5_tok: T5TokenizerFast = T5TokenizerFast.from_pretrained('t5-base')

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


class WikiDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        length: int = 128,
        end: int = 128
    ):
        super().__init__()
        self.dataset = [x for x in dataset if sum(x['attention_mask']) == length]
        self.end = end

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        elem = self.dataset[index]
        true_str = bert_tok.decode(elem['input_ids'], skip_special_tokens=True)
        batch = t5_tok(true_str, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
        return {k: v[0][:self.end] for k, v in batch.items()}


def main(count: int = 64, batch_size: int = 64, peshechka: float = 0.3):
    seed_everything(1337, workers=True)

    if not os.path.exists('data/wiki_test/data-00000-of-00001.arrow'):
        dataset = load_dataset("Graphcore/wikipedia-bert-128", split='train')
        dataset.remove_columns(["token_type_ids", "labels", "next_sentence_label"])
        dt = dataset.train_test_split(test_size=0.001, seed=0)
        dataset = dt["test"]
        dataset.save_to_disk('data/wiki_test')
    else:
        dataset = Dataset.from_file('data/wiki_test/data-00000-of-00001.arrow')

    #device = 'cpu'
    device = 'cuda:0'

    model = T5EncoderPlusSlavaHead.from_pretrained('t5-base')
    model.load_head()
    for param in model.parameters():
        param.requires_grad = False
    model.eval().to(device)

    bloom_comp = BloomMetric(device=device)

    wiki_dataset = WikiDataset(dataset)
    save_folder = osp.join('generated_texts', "local_smoothness")
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
    wiki_dataset.end = 128
    for mix_p in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        metric = MeanMetric()
        dataloader = DataLoader(
            wiki_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        iterator = iter(dataloader)
        bar = trange(0, count, batch_size)
        for index in bar:
            batch = next(iterator)
            batch = dict_to_device(batch, device)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            hidden_states = model.encoder.forward(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            hidden_states = mix_p * hidden_states + torch.roll(hidden_states, 1, 0) * (1 - mix_p)
            logits = model.cls.forward(hidden_states)
            restored_ids = logits.argmax(dim=-1)
            restored_ids = torch.where(attention_mask.bool(), restored_ids, input_ids)
            # [BS; SEQ_LEN; |Vocab|]
            restored_str = t5_tok.batch_decode(restored_ids, skip_special_tokens=True)
            for r_str in restored_str:
                loss, cnt = bloom_comp(r_str, reduce="mean")
                metric.update(loss, cnt)
            with open(f'data/{mix_p}_t5.json', 'w') as fout:
                json.dump(restored_str, fp=fout, indent=4)
            bar.set_description(f'bloom_metric: {metric.compute().item():.5f}, mix_p: {mix_p}')

        print(f'bloom_metric: {metric.compute().item():.5f}, mix_p: {mix_p}\n', flush=True)


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