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
from transformers import RobertaModel
from torchmetrics import Accuracy
import logging
from tqdm.auto import trange

from collections import Counter

import lightning as L

from torch_ema import ExponentialMovingAverage
from diffusion.configs.model_cfg import EncNormalizerCfg

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

from .base_contextual import ContextualDenoising
from diffusion.models.contextual_denoising.slava_estimator import SlavaEstimator, bert_config_slava
from diffusion.utils import calc_group_grads_norm, calc_group_weights_norm


class SlavaContextualDenoising(ContextualDenoising):
    def __init__(self, roberta_pretrain: bool = False, *args, **kwargs) -> None:
        score_estimator = SlavaEstimator(bert_config_slava)
        if roberta_pretrain:
            roberta: RobertaModel = RobertaModel.from_pretrained('roberta-base')
            encoder = roberta.encoder
            roberta_encoder_state_dict = encoder.state_dict()
            new_state_dict = type(roberta_encoder_state_dict)()
            se_state_dict = score_estimator.encoder.state_dict()
            for k, v in roberta_encoder_state_dict.items():
                assert 'layer.' in k, k
                parts = k.split('.')
                num = int(parts[1])
                if num < 6:
                    prefix = 'input_blocks'
                elif num < 12:
                    prefix = 'output_blocks'
                    num -= 6
                else:
                    raise "???, dude, are you sure?"
                new_k = prefix + '.' + str(num) + '.' + ".".join(parts[2:])
                assert new_k in se_state_dict
                assert se_state_dict[new_k].shape == v.shape
                new_state_dict[new_k] = v
            missing_keys: List[str] = []
            for k in se_state_dict.keys():
                if k not in new_state_dict:
                    missing_keys += [k]
            print(f'Missing keys: {", ".join(missing_keys)}')
            score_estimator.encoder.load_state_dict(new_state_dict, strict=False)
            self.missing_keys = ['encoder.' + k for k in missing_keys]
        else:
            self.missing_keys: List[str] = []

        super().__init__(*args, **kwargs)
        self.score_estimator = score_estimator
        self.noisy_part_encoder.restore_decoder()

    def configure_optimizers(self) -> Any:
        if len(self.missing_keys) == 0:
            return super().configure_optimizers()

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.score_estimator.named_parameters() if n in self.missing_keys],
            },
            {
                "params": [p for n, p in self.score_estimator.named_parameters() if n not in self.missing_keys],
            },
        ]

        optim = self._optim_partial(params=optimizer_grouped_parameters)
        sched = self._sched_partial(optimizer=optim)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step"
            }
        }

    def on_before_optimizer_step(self, *args, **kwargs) -> None:
        if len(self.missing_keys) > 0:
            #print(self.missing_keys)
            missing_params = [p for n, p in self.score_estimator.named_parameters() if n in self.missing_keys]
            #print(missing_params)
            roberta_params = [p for n, p in self.score_estimator.named_parameters() if n not in self.missing_keys]
            self.logger.log_metrics({
                'grads_norm/missing': calc_group_grads_norm(missing_params),
                'grads_norm/roberta': calc_group_grads_norm(roberta_params),
            })
        return super().on_before_optimizer_step(*args, **kwargs)

    def optimizer_step(self, *args, **kwargs) -> None:
        returns =  super().optimizer_step(*args, **kwargs)
        if len(self.missing_keys) > 0:
            missing_params = [p for n, p in self.score_estimator.named_parameters() if n in self.missing_keys]
            roberta_params = [p for n, p in self.score_estimator.named_parameters() if n not in self.missing_keys]
            self.logger.log_metrics({
                'weights_norm/missing': calc_group_weights_norm(missing_params),
                'weights_norm/roberta': calc_group_weights_norm(roberta_params),
            })
        return returns


class SlavaContextualDenoisingT5T(SlavaContextualDenoising):
    def __init__(self, clean_enc_normalizer_cfg, *args, **kwargs) -> None:
        super().__init__(*args, clean_enc_normalizer_cfg=clean_enc_normalizer_cfg, **kwargs)
        t5_cfg_name = 't5-base'
        self.clean_part_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(
            t5_cfg_name, enc_normalizer_cfg=clean_enc_normalizer_cfg
        ).train()

    def train(self, mode: bool = True):
        self.score_estimator.train(mode)
        self.clean_part_encoder.train(mode)
        self.noisy_part_encoder.eval()

    def configure_optimizers(self) -> Any:
        optimizer_grouped_parameters = [
            {
                "params": self.score_estimator.parameters(),
            },
            {
                "params": self.clean_part_encoder.parameters()
            }
        ]

        optim = self._optim_partial(params=optimizer_grouped_parameters)
        sched = self._sched_partial(optimizer=optim)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step"
            }
        }

    def on_before_optimizer_step(self, *args, **kwargs) -> None:
        self.logger.log_metrics({
            'grads_norm/t5_encoder': calc_group_grads_norm(self.clean_part_encoder.parameters())
        })
        return super().on_before_optimizer_step(*args, **kwargs)

    def optimizer_step(self, *args, **kwargs) -> None:
        returns =  super().optimizer_step(*args, **kwargs)
        self.logger.log_metrics({
            'weights_norm/t5_encoder': calc_group_weights_norm(self.clean_part_encoder.parameters()),
        })
        return returns