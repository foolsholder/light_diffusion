import torch
from typing import List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from torch import FloatTensor, LongTensor, norm

from diffusion.dataset.enc_normalizer import EncNormalizer
from diffusion.configs.model_cfg import EncNormalizerCfg
from hydra.utils import instantiate

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