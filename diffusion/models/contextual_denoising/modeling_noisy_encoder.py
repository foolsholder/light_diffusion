import torch
from typing import List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from torch import FloatTensor, norm

from diffusion.dataset.enc_normalizer import EncNormalizer
from diffusion.configs.model_cfg import EncNormalizerCfg
from hydra.utils import instantiate

from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import (
    BertLMHeadModel as HuggingFaceBertLMHeadModel, \
    BaseModelOutputWithPoolingAndCrossAttentions
)
from .typings import EncoderOutput
import os
import os.path as osp


class BertLMHeadModel(HuggingFaceBertLMHeadModel):
    def __init__(self, config, enc_normalizer_cfg: EncNormalizerCfg):
        super().__init__(config)
        self.enc_normalizer: EncNormalizer = instantiate(enc_normalizer_cfg)
        #self.restore_decoder()

    def restore_decoder(self):
        decoder_path = "data/new_slava_ckpt/decoder-wikipedia-128.pth"
        self.cls.load_state_dict(
            torch.load(
                osp.join(
                    os.environ['BASE_PATH'],
                    decoder_path
                ), map_location='cpu'
            )["decoder"]
        )
        print("RESTORED SLAVYAN")

    def classify(
        self,
        *,
        denormed_or_true_encs: FloatTensor = None,
        normed: FloatTensor = None
    ) -> FloatTensor:
        assert (denormed_or_true_encs is not None) ^ (normed is not None)
        if normed is not None:
            denormed_or_true_encs = self.enc_normalizer.denormalize(normed)
        return self.cls(denormed_or_true_encs)

    def denorm_encodings(self, normed_encodings: FloatTensor) -> FloatTensor:
        return self.enc_normalizer.denormalize(normed_encodings)

    def forward(
            self,
            *args, **kwargs
    ):
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(
            *args, **kwargs
        )

        sequence_output = outputs.last_hidden_state
        normed = self.enc_normalizer.normalize(sequence_output)
        return EncoderOutput(
            normed=normed,
            true=sequence_output
        )


class BertEncoderPlusSlavaHead(HuggingFaceBertLMHeadModel):
    def __init__(self, config):
        super().__init__(config)

    def load_head(self):
        decoder_path = "data/new_slava_ckpt/decoder-wikipedia-128.pth"
        self.cls.load_state_dict(
            torch.load(
                osp.join(
                    os.environ['BASE_PATH'],
                    decoder_path
                ), map_location='cpu'
            )["decoder"]
        )
        print("RESTORED SLAVYAN BERT")

    def forward(
            self,
            *args, **kwargs
    ):
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(
            *args, **kwargs
        )
        sequence_output: FloatTensor = outputs.last_hidden_state
        return self.cls.forward(sequence_output)
