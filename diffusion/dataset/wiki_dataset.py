import torch
import os
import numpy as np
import pandas as pd
from functools import partial


from torch.utils.data import Dataset, DataLoader
from typing import Union, Dict, Any, Optional, List

from transformers import BertTokenizerFast, T5TokenizerFast
from datasets import load_dataset
from json import dump, load
from datasets.utils.info_utils import VerificationMode


class WikiDataset(Dataset):
    def __init__(
            self,
            train: bool = True,
            max_length: int = 96,
            pos_begin: float = 0.33,
            pos_end: float = 0.67
    ):
        super(WikiDataset, self).__init__()

        self.noisy_tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.clean_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained('t5-base')
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


class ContextualSST2Dataset(Dataset):
    def __init__(
            self,
            train: bool = True,
            max_length: int = 96,
    ):
        super(ContextualSST2Dataset, self).__init__()

        self.noisy_tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.clean_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained('t5-base')
        self.max_length = max_length

        self.dataset = load_dataset("glue", "sst2", split='train' if train else 'validation')

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
            self,
            index: int
    ):
        obj = self.dataset[index]
        sentence = obj['sentence']
        label = int(obj['label'])
        # cause parts were tokenized by bertTokenizer
        clean_part_sentence = sentence

        result: Dict[str, List[int]] = dict()
        for sentence, prefix, tokenizer in zip(
            [clean_part_sentence, "yes" if label else "no"],
            ['clean_', 'noisy_'],
            [partial(self.clean_tokenizer, max_length=self.max_length),
             partial(self.noisy_tokenizer, max_length=3)]
        ):
            tokenized: Dict[str, List[int]] = tokenizer(
                sentence,
                truncation=True,
                padding="max_length"
            )
            for k, v in tokenized.items():
                result[prefix + k] = torch.LongTensor(v)
        result['labels'] = torch.LongTensor([label])[0]

        return result


class ContextualColaDataset(Dataset):
    def __init__(
            self,
            train: bool = True,
            max_length: int = 96,
    ):
        super(ContextualColaDataset, self).__init__()

        self.noisy_tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.clean_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained('t5-base')
        self.max_length = max_length

        self.dataset = load_dataset("glue", "cola", split='train' if train else 'validation')

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
            self,
            index: int
    ):
        obj = self.dataset[index]
        sentence = obj['sentence']
        label = int(obj['label'])
        # cause parts were tokenized by bertTokenizer
        clean_part_sentence = sentence

        result: Dict[str, List[int]] = dict()
        for sentence, prefix, tokenizer in zip(
            [clean_part_sentence, "yes" if label else "no"],
            ['clean_', 'noisy_'],
            [partial(self.clean_tokenizer, max_length=self.max_length),
             partial(self.noisy_tokenizer, max_length=3)]
        ):
            tokenized: Dict[str, List[int]] = tokenizer(
                sentence,
                truncation=True,
                padding="max_length"
            )
            for k, v in tokenized.items():
                result[prefix + k] = torch.LongTensor(v)
        result['labels'] = torch.LongTensor([label])[0]

        return result



class ContextualQQPDataset(Dataset):
    def __init__(
            self,
            train: bool = True,
            max_length: int = 96,
    ):
        super(ContextualQQPDataset, self).__init__()

        self.noisy_tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.clean_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained('t5-base')
        self.max_length = max_length

        self.dataset = load_dataset("glue", "qqp", split='train' if train else 'validation')

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
            self,
            index: int
    ):
        obj = self.dataset[index]
        sentence = obj['question1'] + " " + obj['question2']
        label = int(obj['label'])
        # cause parts were tokenized by bertTokenizer
        clean_part_sentence = sentence

        result: Dict[str, List[int]] = dict()
        for sentence, prefix, tokenizer in zip(
            [clean_part_sentence, "yes" if label else "no"],
            ['clean_', 'noisy_'],
            [partial(self.clean_tokenizer, max_length=self.max_length),
             partial(self.noisy_tokenizer, max_length=3)]
        ):
            tokenized: Dict[str, List[int]] = tokenizer(
                sentence,
                truncation=True,
                padding="max_length"
            )
            for k, v in tokenized.items():
                result[prefix + k] = torch.LongTensor(v)
        result['labels'] = torch.LongTensor([label])[0]

        return result


class ContextualRTEDataset(Dataset):
    def __init__(
            self,
            train: bool = True,
            max_length: int = 96,
    ):
        super(ContextualRTEDataset, self).__init__()

        self.noisy_tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.clean_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained('t5-base')
        self.max_length = max_length

        self.dataset = load_dataset("glue", "rte", split='train' if train else 'validation')

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
            self,
            index: int
    ):
        obj = self.dataset[index]
        sentence = obj['sentence1'] + " " + obj['sentence2']
        label = int(obj['label'])
        # cause parts were tokenized by bertTokenizer
        clean_part_sentence = sentence

        result: Dict[str, List[int]] = dict()
        for sentence, prefix, tokenizer in zip(
            [clean_part_sentence, "yes" if label else "no"],
            ['clean_', 'noisy_'],
            [partial(self.clean_tokenizer, max_length=self.max_length),
             partial(self.noisy_tokenizer, max_length=3)]
        ):
            tokenized: Dict[str, List[int]] = tokenizer(
                sentence,
                truncation=True,
                padding="max_length"
            )
            for k, v in tokenized.items():
                result[prefix + k] = torch.LongTensor(v)
        result['labels'] = torch.LongTensor([label])[0]

        return result
