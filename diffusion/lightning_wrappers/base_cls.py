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

from torchmetrics.classification import BinaryAccuracy


class GLUEFreezedClassification(L.LightningModule):
    def __init__(
        self,
        optim_partial: Callable[[(...,)], torch.optim.Optimizer]
    ) -> None:
        super().__init__()
        bert_cfg_name = 'bert-base-uncased'
        bert_config = BertConfig(bert_cfg_name)

        self.bert = BertModel.from_pretrained(bert_cfg_name).eval()

        for param in self.bert.parameters():
            param.requires_grad = False

        self.tokenizer = BertTokenizerFast.from_pretrained(bert_cfg_name)

        self._optim_partial = instantiate(optim_partial)

        self.train_loss: MeanMetric = MeanMetric()
        self.valid_loss: MeanMetric = MeanMetric()

        self.train_accuracy = BinaryAccuracy()
        self.valid_accuracy = BinaryAccuracy()

        self.cls = None # define it in ancestors

    def configure_optimizers(self) -> Any:
        optim = self._optim_partial(params=self.cls.parameters())
        return {
            "optimizer": optim
        }

    def train(self, mode: bool = True):
        self.bert.eval()
        self.cls.train(mode)
        return self

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise "Not impl yet"

    def step_logic(self, batch: Dict[str, Tensor], acc: BinaryAccuracy, loss: MeanMetric) -> STEP_OUTPUT:
        batch = copy(batch)
        labels = batch.pop('labels').view(-1).long()
        outputs = self.forward(batch)
        logits = outputs['logits'].view(-1)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        probs = F.sigmoid(logits)
        acc.update(probs, labels)
        loss.update(bce_loss, len(labels))
        return {
            "loss": bce_loss
        }

    def on_epoch_end(self, acc: BinaryAccuracy, loss: MeanMetric, is_train: bool = True):
        self.log_dict(
            {
                f'accuracy_epoch': acc.compute(),
                f'loss_epoch': loss.compute()
            },
            is_train=is_train,
            sync_dist=False
        )
        acc.reset()
        loss.reset()

    def on_train_epoch_end(self) -> None:
        self.on_epoch_end(self.train_accuracy, self.train_loss, is_train=True)
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self.on_epoch_end(self.valid_accuracy, self.valid_loss, is_train=False)
        return super().on_validation_epoch_end()

    def log_dict(self, losses: Dict[str, Tensor], is_train: bool = True, *args, **kwargs):
        suffix = 'train' if is_train else 'valid'
        losses = {key + f'/{suffix}': value for key, value in losses.items()}
        return super().log_dict(losses, *args, **kwargs)

    def training_step(self, batch: Dict[str, Tensor], *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        outputs = self.step_logic(batch, self.train_accuracy, self.train_loss)
        self.log_dict(outputs, is_train=True, sync_dist=True)
        return outputs

    def validation_step(self, batch: Dict[str, Tensor], *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.step_logic(batch, self.valid_accuracy, self.valid_loss)

    def on_before_optimizer_step(self, *args, **kwargs) -> None:
        self.logger.log_metrics({'model/grads_norm': calc_model_grads_norm(self.cls)})
        return super().on_before_optimizer_step(*args, **kwargs)

    def optimizer_step(self, *args, **kwargs) -> None:
        returns =  super().optimizer_step(*args, **kwargs)
        self.logger.log_metrics({'model/weights_norm': calc_model_weights_norm(self.cls)})
        return returns
