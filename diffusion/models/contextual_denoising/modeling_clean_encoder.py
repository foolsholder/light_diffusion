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


class T5EncoderPlusSlavaHead(HuggingFaceT5EncoderModel):
    def __init__(self, config):
        super().__init__(config)
        config = BertConfig.from_pretrained('bert-base-uncased')
        self.cls = BertOnlyMLMHead(config)
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
