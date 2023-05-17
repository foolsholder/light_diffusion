import torch

from torch import nn, Tensor
from torch.nn import functional as F

from typing import Any, Optional, Dict, List, Union, Callable, Any

STEP_OUTPUT = Dict[str, Tensor]
EPOCH_OUTPUT = List[STEP_OUTPUT]


from .base import ZeroVoc2

class ZeroVoc30k(ZeroVoc2):
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
        labels_30k = input_ids[:, 0]

        ce_loss = torch.nn.functional.cross_entropy(logits, labels_30k)
        bce_loss = torch.nn.functional.cross_entropy(logits[:, [2053, 2748]], labels_binary)

        batch_size = len(logits)

        return {
            'batch_size': batch_size,
            'ce30k_loss': ce_loss,
            'bce_loss': bce_loss,
            'x0_loss': x0_loss,
            'loss': x0_loss + self.ce_coef * ce_loss
        }