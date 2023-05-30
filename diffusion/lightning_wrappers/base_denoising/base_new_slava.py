import torch

from torch import nn, Tensor
from torch.nn import functional as F

from typing import Any, Optional, Dict, List, Union, Callable, Any
from hydra.utils import instantiate
import os
import os.path as osp
from transformers import BertTokenizerFast, BertConfig
from torchmetrics import Accuracy

from collections import Counter

import lightning as L

from torch_ema import ExponentialMovingAverage

from diffusion.models.base_denoising.modeling_bert import BertLMHeadModel

STEP_OUTPUT = Dict[str, Tensor]
EPOCH_OUTPUT = List[STEP_OUTPUT]


from diffusion.dynamics import SDE, RSDE, EulerSolver
from diffusion.utils import calc_model_grads_norm, calc_model_weights_norm, filter_losses
from diffusion.models.base_denoising import BertLMHeadModel as TBB, ScoreEstimator
from diffusion.helper import LinearWarmupLR
from diffusion.dataset import EncNormalizer
from transformers.models.bert.modeling_bert import BertLMHeadModel as BB
from .base import ZeroVoc2
from diffusion.models.base_denoising.new_slava import slava_bert_config, ScoreEstimatorEMB

class SecondZeroVoc2(ZeroVoc2):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.score_estimator = ScoreEstimatorEMB(slava_bert_config)

        encoder_ref: TBB = self.encoder
        encoder_ref.cls.load_state_dict(
            torch.load(
                osp.join(
                    os.environ['BASE_PATH'],
                    'data/new_slava_ckpt/decoder-wikipedia-128.pth'
                ),
                map_location='cpu'
            )['decoder']
        )
        self.score_estimator.load_state_dict(
            torch.load(
                osp.join(
                    os.environ['BASE_PATH'],
                    'data/new_slava_ckpt/ema-model.pth'
                ),
                map_location='cpu'
            )
        )