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
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryMatthewsCorrCoef
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
from .binary_cls_wrapper import BinaryClassification


class BinaryClassificationTrainableT5(BinaryClassification):
    def __init__(
        self,
        *args, **kwargs
    ) -> None:
        clean_enc_normalizer_cfg = kwargs.pop('clean_enc_normalizer_cfg')
        super().__init__(
            *args, **kwargs, clean_enc_normalizer_cfg=clean_enc_normalizer_cfg
        )
        t5_cfg_name = 't5-base'

        self.clean_part_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(
            t5_cfg_name, enc_normalizer_cfg=clean_enc_normalizer_cfg
        )

    def configure_optimizers(self) -> Any:
        optim = self._optim_partial(params=[
            {
                'params': self.score_estimator.parameters()
            },
            {
                'params': self.clean_part_encoder.parameters()
            },
        ])
        sched = self._sched_partial(optimizer=optim)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step"
            }
        }

    def train(self, mode: bool = True):
        self.score_estimator.train(mode)
        self.clean_part_encoder.train(mode)
        self.noisy_part_encoder.eval()
        return self