import torch

from torch import nn, Tensor
from torch.nn import functional as F

from typing import Any, Optional, Dict, List, Union, Callable, Any
from hydra.utils import instantiate

from transformers import BertTokenizerFast, BertConfig
from torchmetrics import Accuracy, MeanMetric
from torch.nn import functional as F

from collections import Counter
from copy import copy

import lightning as L

from torch_ema import ExponentialMovingAverage

STEP_OUTPUT = Dict[str, Tensor]
EPOCH_OUTPUT = List[STEP_OUTPUT]


from diffusion.utils import calc_model_grads_norm, calc_model_weights_norm, filter_losses
from diffusion.helper import LinearWarmupLR
from transformers.models.bert.modeling_bert import (
    BertLMHeadModel as BB, BertLMPredictionHead, BertModel
)
from diffusion.models.base_denoising.modeling_bert import BertModel as TBM

from torchmetrics.classification import BinaryAccuracy

from .base_cls import GLUEFreezedClassification


class SST2FreezedClassification(GLUEFreezedClassification):
    def __init__(
        self,
        optim_partial: Callable[[(...,)], torch.optim.Optimizer]
    ) -> None:
        super().__init__(optim_partial=optim_partial)
        bert_cfg_name = 'bert-base-uncased'
        bert_config = BertConfig(bert_cfg_name)
        bert_config.vocab_size = 1
        self.bert = TBM.from_pretrained(bert_cfg_name, label_mask_pos=0)
        self.cls = BertLMPredictionHead(bert_config)
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        outs = self.bert.forward(**batch)
        last_hidden_state: Tensor = outs.last_hidden_state
        hidden_state_cls = last_hidden_state[:, 0]
        cls_logits = self.cls(hidden_state_cls)
        return {
            "logits": cls_logits
        }