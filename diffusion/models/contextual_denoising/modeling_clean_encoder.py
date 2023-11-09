import torch
from typing import List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from torch import FloatTensor, LongTensor, norm

from diffusion.dataset.enc_normalizer import EncNormalizer
from diffusion.configs.model_cfg import EncNormalizerCfg
from hydra.utils import instantiate

import os
import os.path as osp

from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertConfig
from transformers.models.t5.modeling_t5 import (
    BaseModelOutputWithPastAndCrossAttentions, \
    T5EncoderModel as HuggingFaceT5EncoderModel,
)
from transformers import T5TokenizerFast
from .typings import EncoderOutput


class T5EncoderModel(HuggingFaceT5EncoderModel):
    def __init__(self, config, enc_normalizer_cfg: EncNormalizerCfg):
        super().__init__(config)
        self.enc_normalizer: EncNormalizer = instantiate(enc_normalizer_cfg)

    def forward(
            self,
            *args, **kwargs
    ):
        outputs: BaseModelOutputWithPastAndCrossAttentions = super().forward(
            *args, **kwargs
        )

        sequence_output = outputs.last_hidden_state
        normed = self.enc_normalizer.normalize(sequence_output)

        return EncoderOutput(
            normed=normed,
            true=sequence_output
        )


class Decoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size=768, vocab_size=32100, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = torch.nn.Linear(input_size, hidden_size)
        self.act_fn = torch.nn.GELU()
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.decoder = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class T5EncoderPlusSlavaHead(HuggingFaceT5EncoderModel):
    def __init__(self, config):
        super().__init__(config)
        self.cls = Decoder(768)
        decoder_path = "data/new_slava_ckpt/decoder-t5_base-wikipedia-128.pth"
        self.cls.load_state_dict(
            torch.load(
                osp.join(
                    os.environ['BASE_PATH'],
                    decoder_path
                ), map_location='cpu'
            )["decoder"]
        )
        print("RESTORED SLAVYAN T5")

    def forward(
            self,
            *args, **kwargs
    ):
        outputs: BaseModelOutputWithPastAndCrossAttentions = super().forward(
            *args, **kwargs
        )
        sequence_output: FloatTensor = outputs.last_hidden_state
        return self.cls.forward(sequence_output)
