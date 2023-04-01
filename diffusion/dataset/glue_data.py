import torch

from torch.utils.data import Dataset
from typing import Union, Dict, Any, Optional, List

from transformers import BertTokenizerFast
from datasets import load_dataset


class SST2Dataset(Dataset):
    def __init__(self, max_length: int = 71, train: bool = True) -> None:
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.dataset = load_dataset('glue', 'sst2', split='train' if train else 'validation')

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
            self,
            index: int
    ):
        obj = self.dataset[index]
        sent: str = obj['sentence']

        label = obj['label']

        dct = self.tokenizer(
           sent,
           truncation=True,
           padding="max_length",
           max_length=self.max_length
        )
        dct['labels'] = [label]
        dct['input_ids'][0:1] = [dct['input_ids'][0], label * 2748 + (1 - label) * 2053]
        dct['token_type_ids'][0:1] = [dct['token_type_ids'][0], dct['token_type_ids'][1]]
        dct['attention_mask'][0:1] = [1, 1]
        #print(dct['token_type_ids'])
        dct = {k: torch.LongTensor(v) for k, v in dct.items()}
        #print(dct['input_ids'][0], self.tokenizer.vocab['[CLS]'])
        #dec = []
        #for obj in dct['input_ids']:
        #    dec += [obj]
        #    if obj == self.tokenizer.vocab['[SEP]']:
        #        break
        #print(self.tokenizer.decode(dec), sent)
        # ['CLS']
        #dct['input_ids'][0] = 2053 * (1 - label) + label * 2748
        return dct
