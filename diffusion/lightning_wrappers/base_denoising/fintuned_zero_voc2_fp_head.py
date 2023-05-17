import torch

from torch import nn, Tensor
from torch.nn import functional as F

from typing import Any, Optional, Dict, List, OrderedDict, Union, Callable, Any
from hydra.utils import instantiate

from transformers import BertTokenizerFast, BertConfig
from torchmetrics import Accuracy
import os
import os.path as osp
from collections import Counter

import lightning as L

from torch_ema import ExponentialMovingAverage

STEP_OUTPUT = Dict[str, Tensor]
EPOCH_OUTPUT = List[STEP_OUTPUT]


from diffusion.dynamics import SDE, RSDE, EulerSolver
from diffusion.utils import calc_model_grads_norm, calc_model_weights_norm, filter_losses
from diffusion.models.base_denoising import BertLMHeadModel as TBB, ScoreEstimator
from diffusion.helper import LinearWarmupLR
from diffusion.dataset import EncNormalizer
from transformers.models.bert.modeling_bert import BertLMHeadModel as BB, BertLMPredictionHead, BertOnlyMLMHead

from .base import ZeroVoc2


def filter_dict(st_dict: OrderedDict, key: str):
    res = type(st_dict)()
    for k, v in st_dict.items():
        if key in k:
            res[k[len(key):]] = v
    return res


class FinetunedZeroVoc2FPHead(ZeroVoc2):
    def __init__(
        self,
        cls_head_ckpt_path: str,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.label_mask_pos
        bert_config_name = 'bert-base-uncased'
        bert_config = BertConfig(bert_config_name)
        bert_config.vocab_size = 2
        self.encoder.cls = BertOnlyMLMHead(bert_config)
        cls_head_ckpt_path = osp.join(os.environ['BASE_PATH'], cls_head_ckpt_path)
        ckpt = torch.load(cls_head_ckpt_path, map_location='cpu')
        st_dict = filter_dict(ckpt['state_dict'], 'cls.')
        self.encoder.cls.load_state_dict(
            st_dict
        )
        self.encoder.load_state_dict(ckpt['state_dict'])
        for param in self.encoder.parameters():
            param.requires_grad = False

    def step_logic(self, batch: Dict[str, Tensor]) -> STEP_OUTPUT:
        input_ids = batch['input_ids']
        att_mask = batch['attention_mask']
        
        #encodings = self.encoder.bert(
        #    input_ids=input_ids, 
        #    attention_mask=att_mask, 
        #    token_type_ids=batch['token_type_ids']
        #).last_hidden_state
        #clean_x = self.enc_normalizer.normalize(encodings)
        
        outputs = self.forward(batch)

        pred_x_0 = outputs['x_0']
        clean_x_0 = outputs['clean_x']

        # print(pred_x_0.shape, clean_x_0.shape, flush=True)

        x0_loss = torch.mean((pred_x_0[:, 0, :] - clean_x_0[:, 0, :])**2)
        #pred_x_0 = clean_x
        pred_encodings = self.enc_normalizer.denormalize(pred_x_0)[:, 0]
        #identity_enc = self.enc_normalizer.denormalize(
        #        self.enc_normalizer.normalize(encodings)
        #    )
        #mse = torch.mean((identity_enc - encodings)**2)
        logits = self.encoder.cls(
            #identity_enc[:, 0]
            pred_encodings
        )

        # print(logits.shape, flush=True)

        labels_binary = batch['labels'].view(-1)

        bce_loss = torch.nn.functional.cross_entropy(
            logits, labels_binary.view(-1).long()
        )
        #print(bce_loss, mse)

        batch_size = len(logits)

        return {
            'batch_size': batch_size,
            'bce_loss': bce_loss,
            'x0_loss': x0_loss,
            'loss': x0_loss + self.ce_coef * bce_loss
        }

    def get_label_pred_label(self, pred_encodings):
        logits = self.encoder.cls(pred_encodings[:, self.label_mask_pos])
        probs_binary = F.softmax(logits)[:, 1]
        pred_label = (probs_binary >= 0.5).float().reshape(self.test_count, -1).mean(dim=0)
        return pred_label