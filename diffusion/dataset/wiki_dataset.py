import torch
import os
import numpy as np
import pandas as pd


from torch.utils.data import Dataset, DataLoader
from typing import Union, Dict, Any, Optional, List

from transformers import BertTokenizerFast, T5TokenizerFast
from datasets import load_dataset
from json import dump, load


class WikiDataset(Dataset):
    def __init__(
            self,
            train: bool = True,
            max_length: int = 96,
            pos_begin: float = 0.33,
            pos_end: float = 0.67
    ):
        super(WikiDataset, self).__init__()

        self.noisy_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.clean_tokenizer = T5TokenizerFast.from_pretrained('t5-base')
        self.max_length = max_length

        if train:
            self.dataset = load_dataset("Graphcore/wikipedia-bert-128", split='train')
        else:
            self.dataset = []

        self.pos_begin = pos_begin
        self.pos_end = pos_end

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
            self,
            index: int
    ):
        obj = self.dataset[index]

        input_ids = obj['input_ids']
        attention_mask = obj['attention_mask']
        elem_count = sum(attention_mask)
        input_ids = input_ids[:elem_count]

        delimeter_pos = int(
            (
                np.random.rand() * (self.pos_end - self.pos_begin) + self.pos_begin
            ) * elem_count
        )
        clean_part = input_ids[:delimeter_pos]
        # cause parts were tokenized by bertTokenizer
        clean_part_sentence = self.noisy_tokenizer.decode(clean_part, skip_special_tokens=True)

        noisy_part = input_ids[delimeter_pos:]
        noisy_part_sentence = self.noisy_tokenizer.decode(noisy_part, skip_special_tokens=True)

        result: Dict[str, List[int]] = dict()
        for sentence, prefix, tokenizer in zip(
            [clean_part_sentence, noisy_part_sentence],
            ['clean_', 'noisy_'],
            [self.clean_tokenizer, self.noisy_tokenizer]
        ):
            tokenized: Dict[str, List[int]] = tokenizer(
                sentence,
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
            for k, v in tokenized.items():
                result[prefix + k] = torch.LongTensor(v)

        return result
