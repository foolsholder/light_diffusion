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
    BertLMHeadModel as BB, BertLMPredictionHead, BertModel, BertOnlyMLMHead
)
from diffusion.models.modeling_bert import BertModel as TBM

from torchmetrics.classification import BinaryAccuracy

from .base_cls import GLUEFreezedClassification


class PretrainedSST2FreezedClassification(GLUEFreezedClassification):
    def __init__(
        self,
        optim_partial: Callable[[(...,)], torch.optim.Optimizer]
    ) -> None:
        super().__init__(optim_partial=optim_partial)
        bert_cfg_name = 'bert-base-uncased'
        bert_config = BertConfig(bert_cfg_name)
        body: BB = BB.from_pretrained(bert_cfg_name)
        head: BertLMPredictionHead = body.cls.predictions

        bert_config.vocab_size = 2
        self.bert = TBM.from_pretrained(bert_cfg_name, label_mask_pos=0)
        self.cls = BertOnlyMLMHead(bert_config)

        self.cls.predictions.transform.load_state_dict(
            head.transform.state_dict()
        )

        self.cls.predictions.decoder.weight.data[:] = head.decoder.weight.data[[2053, 2748]]
        self.cls.predictions.decoder.bias.data[:] = head.decoder.bias.data[[2053, 2748]]
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        outs = self.bert.forward(**batch)
        last_hidden_state: Tensor = outs.last_hidden_state
        hidden_state_cls = last_hidden_state[:, 0]
        cls_logits = self.cls(hidden_state_cls)
        # [BS; 2]
        return {
            "logits": cls_logits
        }

    def step_logic(self, batch: Dict[str, Tensor], acc: BinaryAccuracy, loss: MeanMetric) -> STEP_OUTPUT:
        batch = copy(batch)
        labels = batch.pop('labels').view(-1).long()
        outputs = self.forward(batch)
        logits = outputs['logits']
        bce_loss = F.cross_entropy(logits, labels)
        probs = F.softmax(logits, dim=1)
        acc.update(probs[:, 1], labels)
        loss.update(bce_loss, len(labels))
        return {
            "loss": bce_loss
        }