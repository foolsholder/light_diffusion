import torch

from torch import nn, Tensor
from torch.nn import functional as F

from typing import Any, Optional, Dict, List, OrderedDict, Union, Callable, Any
from hydra.utils import instantiate

from transformers import BertTokenizerFast, BertConfig
from torchmetrics import Accuracy

from collections import Counter

import lightning as L

from torch_ema import ExponentialMovingAverage

STEP_OUTPUT = Dict[str, Tensor]
EPOCH_OUTPUT = List[STEP_OUTPUT]


from diffusion.dynamics import SDE, RSDE, EulerSolver
from diffusion.utils import calc_model_grads_norm, calc_model_weights_norm, filter_losses
from diffusion.models import BertLMHeadModel as TBB, ScoreEstimator
from diffusion.helper import LinearWarmupLR
from diffusion.dataset import EncNormalizer
from transformers.models.bert.modeling_bert import BertLMHeadModel as BB, BertLMPredictionHead

from .base import ZeroVoc2


def filter_dict(st_dict: OrderedDict, key: str):
    res = type(st_dict)()
    for k, v in st_dict.items():
        if key in k:
            res[k[len(key):]] = v
    return res


class FinetunedZeroVoc2(ZeroVoc2):
    def __init__(
        self,
        cls_head_ckpt_path: str,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.label_mask_pos
        bert_config_name = 'bert-base-uncased'
        bert_config = BertConfig(bert_config_name)
        bert_config.vocab_size = 1
        self.encoder.cls = BertLMPredictionHead(bert_config)
        ckpt = torch.load(cls_head_ckpt_path, map_location='cpu')
        st_dict = filter_dict(ckpt['state_dict'], 'cls.')
        self.encoder.cls.load_state_dict(
            st_dict
        )
        for param in self.encoder.parameters():
            param.requires_grad = False

    def step_logic(self, batch: Dict[str, Tensor]) -> STEP_OUTPUT:
        outputs = self.forward(batch)
        input_ids = batch['input_ids']

        pred_x_0 = outputs['x_0']
        clean_x_0 = outputs['clean_x']

        # print(pred_x_0.shape, clean_x_0.shape, flush=True)

        x0_loss = torch.mean((pred_x_0[:, 0, :] - clean_x_0[:, 0, :])**2)

        pred_encodings = self.enc_normalizer.denormalize(pred_x_0)
        logits = self.encoder.forward(pred_encodings=pred_encodings).logits

        # print(logits.shape, flush=True)

        logits = logits[:, 0]
        labels_binary = batch['labels'].view(-1)

        bce_loss = torch.nn.functional.cross_entropy(logits.view(-1), labels_binary.view(-1).float())

        batch_size = len(logits)

        return {
            'batch_size': batch_size,
            'bce_loss': bce_loss,
            'x0_loss': x0_loss,
            'loss': x0_loss + self.ce_coef * bce_loss
        }

    def get_label_pred_label(self, logits):
        logits_binary = logits[:, self.label_mask_pos]
        probs_binary = F.sigmoid(logits_binary)
        pred_label = (probs_binary >= 0.5).float().reshape(self.test_count, -1).mean(dim=0)
        return pred_label