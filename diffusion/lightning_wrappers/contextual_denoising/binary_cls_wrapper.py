from lightning.pytorch.utilities.types import STEP_OUTPUT
from sympy import Float
import torch

from functools import partial
from torch import FloatTensor, nn, Tensor
from torch.nn import functional as F

from typing import Any, Optional, Dict, List, Union, Callable, Any, Tuple
from hydra.utils import instantiate
import os
import os.path as osp
from transformers import BertTokenizerFast, BertConfig, T5TokenizerFast, T5Config
from torchmetrics.classification import BinaryAccuracy
import logging
from tqdm.auto import trange

from collections import Counter

import lightning as L

from torch_ema import ExponentialMovingAverage
from diffusion.configs.model_cfg import EncNormalizerCfg
from diffusion.lightning_wrappers.contextual_denoising.base_contextual import STEP_OUTPUT

STEP_OUTPUT = Dict[str, Tensor]
EPOCH_OUTPUT = List[STEP_OUTPUT]


from diffusion.dynamics import SDE, RSDE, EulerSolver
from diffusion.utils import calc_model_grads_norm, calc_model_weights_norm, filter_losses
from diffusion.models.contextual_denoising.modeling_clean_encoder import T5EncoderModel
from diffusion.models.contextual_denoising.modeling_noisy_encoder import BertLMHeadModel
from diffusion.models.contextual_denoising.score_estimator import ScoreEstimator
from diffusion.models.contextual_denoising.typings import EncoderOutput

from diffusion.helper import LinearWarmupLR
from diffusion.dataset import EncNormalizer, enc_normalizer
from .slava_contextual import SlavaContextualDenoising


class BinaryClassification(SlavaContextualDenoising):
    def __init__(
        self,
        noisy_enc_normalizer_cfg: EncNormalizerCfg,
        clean_enc_normalizer_cfg: EncNormalizerCfg,
        sde_cfg: SDE,
        optim_partial: Callable[[Any], torch.optim.Optimizer],
        sched_partial: Callable[[Any], LinearWarmupLR],
        ce_coef: float = 0,
        test_count: int = 11,
    ) -> None:
        super().__init__(
            noisy_enc_normalizer_cfg,
            clean_enc_normalizer_cfg,
            sde_cfg, optim_partial,
            sched_partial,
            ce_coef
        )
        noisy_tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased')
        encodings_yes: EncoderOutput = self.noisy_part_encoder.forward(
            **noisy_tokenizer("yes", return_tensors="pt")
        )
        encodings_no: EncoderOutput = self.noisy_part_encoder.forward(
            **noisy_tokenizer("no", return_tensors="pt")
        )
        assert encodings_yes.normed.shape == (1, 3, 768)
        assert encodings_no.normed.shape == (1, 3, 768)
        encodings_gt = torch.cat([encodings_no.normed, encodings_yes.normed], dim=0)
        encodings_gt_true = torch.cat([encodings_no.true, encodings_yes.true], dim=0)
        self.register_buffer(
            "gt_encodings_normed", encodings_gt
        )
        self.register_buffer(
            "gt_encodings_true", encodings_gt_true
        )
        self.valid_accuracy = BinaryAccuracy()
        self.accuracy_tiles = BinaryAccuracy()
        self.test_count = test_count

    def sample_encodings(self, batch: Dict[str, Tensor]) -> Dict[str, EncoderOutput]:
        to_clean_part, to_noise_part = self.split_batch(batch)

        clean_part: EncoderOutput = self.clean_part_encoder.forward(**to_clean_part)
        labels = batch['labels'].long().view(-1)
        normed_encs = self.gt_encodings_normed[labels]
        true_encs = self.gt_encodings_true[labels]
        return {
            "noisy_part": EncoderOutput(normed_encs, true_encs),
            "clean_part": clean_part
        }

    def validation_step(self, batch: Dict[str, Tensor], *args, **kwargs: Any) -> Dict[str, Tensor] | None:
        to_clean_part, to_noise_part = self.split_batch(batch)
        encodings = self.sample_encodings(batch)
        clean_part = encodings['clean_part']
        noisy_part_attention_mask = to_noise_part['attention_mask']
        gen_shape = noisy_part_attention_mask.shape[:2] + \
                    (clean_part.normed.shape[-1],)
        noisy_part_pred_encodings = self.generate_encodings(
            shape=gen_shape,
            cross_encodings=clean_part.normed,
            cross_attention_mask=to_clean_part['attention_mask'],
            attn_mask=noisy_part_attention_mask,
            verbose=True
        ) # normed
        # [2; 3; 768]
        # [BS; 3; 768]
        diff = self.gt_encodings_normed[None, :, 1] - noisy_part_pred_encodings[:, None, 1]
        # [BS; 2; 768]
        diff = diff.view(diff.shape[0], 2, -1)
        diff = torch.sum(diff**2, dim=2)
        # [BS; 2]
        labels = batch['labels'].long().view(-1)
        pred_labels = torch.argmin(diff, dim=1)
        self.valid_accuracy.update(preds=pred_labels, target=labels)

    def on_validation_epoch_start(self) -> None:
        self.valid_accuracy.reset()
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        self.log_dict({'accuracy/valid': self.valid_accuracy.compute()},
                      is_train=False, apply_suffix=False)
        return super().on_validation_epoch_end()

    def test_step(self, batch: Dict[str, Tensor], *args, **kwargs: Any) -> Dict[str, Tensor] | None:
        to_clean_part, to_noise_part = self.split_batch(batch)
        encodings = self.sample_encodings(batch)
        clean_part = encodings['clean_part']
        noisy_part_attention_mask = to_noise_part['attention_mask']
        gen_shape = (noisy_part_attention_mask.shape[0] * self.test_count,) + \
                    (noisy_part_attention_mask.shape[1],) + \
                    (clean_part.normed.shape[-1],)
        noisy_part_pred_encodings = self.generate_encodings(
            shape=gen_shape,
            cross_encodings=torch.tile(clean_part.normed, (self.test_count, 1, 1)),
            cross_attention_mask=torch.tile(
                to_clean_part['attention_mask'],
                (self.test_count, 1)
            ),
            attn_mask=torch.tile(
                noisy_part_attention_mask,
                (self.test_count, 1)
            ),
            verbose=True
        ) # normed
        # [2; 3; 768]
        # [BS; 3; 768]
        diff = self.gt_encodings_normed[None, :, 1] - noisy_part_pred_encodings[:, None, 1]
        # [BS; 2; 768]
        diff = diff.view(diff.shape[0], 2, -1)
        diff = torch.sum(diff**2, dim=2)
        # [BS; 2]
        labels = batch['labels'].long().view(-1)
        pred_labels = torch.argmin(diff, dim=1)
        self.valid_accuracy.update(preds=pred_labels[:len(labels)], target=labels)
        pred_labels = pred_labels.reshape(self.test_count, len(labels)).mean(dim=0)
        self.accuracy_tiles.update(preds=pred_labels, target=labels)

    def on_test_epoch_start(self) -> None:
        self.valid_accuracy.reset()
        self.accuracy_tiles.reset()
        return super().on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        self.log_dict({'accuracy/valid@1': self.valid_accuracy.compute()},
                      is_train=False, apply_suffix=False)
        self.log_dict({f'accuracy/valid@{self.test_count}': self.accuracy_tiles.compute()},
                      is_train=False, apply_suffix=False)
        return super().on_validation_epoch_end()