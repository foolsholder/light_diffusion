import torch

from torch import nn, Tensor
from torch.nn import functional as F

from typing import Any, Optional, Dict, List, Union, Callable, Any

STEP_OUTPUT = Dict[str, Tensor]
EPOCH_OUTPUT = List[STEP_OUTPUT]


from .base import ZeroVoc2

class Joint(ZeroVoc2):
    def step_logic(self, batch: Dict[str, Tensor]) -> STEP_OUTPUT:
        outputs = self.forward(batch)
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        pred_x_0 = outputs['x_0']
        clean_x_0 = outputs['clean_x']

        # print(pred_x_0.shape, clean_x_0.shape, flush=True)

        x0_loss = torch.mean((pred_x_0 - clean_x_0)**2, dim=-1)
        x0_loss = torch.sum(x0_loss * attention_mask) / torch.sum(attention_mask)

        pred_encodings = self.enc_normalizer.denormalize(pred_x_0)
        logits = self.encoder.forward(pred_encodings=pred_encodings).logits

        # print(logits.shape, flush=True)

        pred_ids_logits = logits
        
        ce_full = torch.nn.functional.cross_entropy(
            pred_ids_logits, input_ids, reduction='none'
        )
        ce_full = torch.sum(ce_full * attention_mask) / torch.sum(attention_mask)

        label_logits = logits[:, 0]
        labels_binary = batch['labels'].view(-1)
        labels_30k = input_ids[:, 0]

        ce_loss = torch.nn.functional.cross_entropy(
            label_logits, labels_30k
        )
        bce_loss = torch.nn.functional.cross_entropy(
            label_logits[:, [2053, 2748]], labels_binary
        )

        batch_size = len(logits)

        return {
            'batch_size': batch_size,
            'ce30k_loss': ce_loss,
            'bce_loss': bce_loss,
            'x0_loss': x0_loss,
            'ce_full': ce_full,
            'loss': x0_loss + self.ce_coef * ce_full
        }