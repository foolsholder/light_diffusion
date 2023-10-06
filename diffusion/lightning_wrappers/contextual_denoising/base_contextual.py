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
import numpy as np
from transformers import BertTokenizerFast, BertConfig, T5TokenizerFast, T5Config
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


class ContextualDenoising(L.LightningModule):
    def __init__(
        self,
        noisy_enc_normalizer_cfg: EncNormalizerCfg,
        clean_enc_normalizer_cfg: EncNormalizerCfg,
        sde_cfg: SDE,
        optim_partial: Callable[[(...,)], torch.optim.Optimizer],
        sched_partial: Callable[[(...,)], LinearWarmupLR],
        ce_coef: float = 0
    ) -> None:
        super().__init__()

        bert_cfg_name = 'bert-base-uncased'
        t5_cfg_name = 't5-base'

        bert_config = BertConfig(bert_cfg_name)
        self.noisy_part_encoder: BertLMHeadModel = BertLMHeadModel.from_pretrained(
            bert_cfg_name, enc_normalizer_cfg=noisy_enc_normalizer_cfg
        ).eval()

        self.clean_part_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(
            t5_cfg_name, enc_normalizer_cfg=clean_enc_normalizer_cfg
        ).eval()

        for encoder_model in [self.clean_part_encoder, self.noisy_part_encoder]:
            for param in encoder_model.parameters():
                param.requires_grad = False

        self.score_estimator = ScoreEstimator(bert_config)

        self.sde: SDE = instantiate(sde_cfg)
        rsde = RSDE(self.sde)
        self.solver = EulerSolver(rsde, self.sde.ode_sampling)

        self._optim_partial = instantiate(optim_partial)
        self._sched_partial = instantiate(sched_partial)

        self.ce_coef = ce_coef

        self.train_metric_to_log: Dict[str, Tensor] = Counter()
        self.valid_metric_to_log: Dict[str, Tensor] = Counter()

    def download_ema_ckpt(self, ema_path):
        ema = ExponentialMovingAverage(
            parameters=self.score_estimator.parameters(), decay=0
        )
        self.load_and_copy_ema(ema, ema_path)

    def load_and_copy_ema(self, ema, ema_path):
        ema_dct = torch.load(ema_path, map_location='cpu')['ema']
        ema_dct['collected_params'] = None
        ema.load_state_dict(ema_dct)
        ema.copy_to(self.score_estimator.parameters())

    def configure_optimizers(self) -> Any:
        optim = self._optim_partial(params=self.score_estimator.parameters())
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
        self.clean_part_encoder.eval()
        self.noisy_part_encoder.eval()
        return self

    def se_forward(
        self,
        x_t: FloatTensor,
        time_t: FloatTensor,
        attention_mask: FloatTensor,
        cross_encodings: FloatTensor,
        cross_attention_mask: FloatTensor
    ) -> Dict[str, Tensor]:
        score_call = partial(
            self.score_estimator.forward,
            cross_attention_mask=cross_attention_mask,
            cross_encodings=cross_encodings
        )
        return self.sde.calc_score(score_call, x_t, time_t, attention_mask)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch_size = len(batch['clean_attention_mask'])

        encodings: Dict[str, EncoderOutput] = self.sample_encodings(batch)
        cross_encodings: FloatTensor = encodings['clean_part'].normed
        normed_ddpm_target: FloatTensor = encodings['noisy_part'].normed
        # x0

        cross_attn_mask = batch['clean_attention_mask']
        attn_mask = torch.ones_like(
            batch['noisy_attention_mask']
        )

        time_t = torch.rand(batch_size, device=normed_ddpm_target.device)
        marg_forward = self.sde.marginal_forward(normed_ddpm_target, time_t)
        x_t = marg_forward['x_t']

        scores = self.se_forward(
            x_t=x_t,
            time_t=time_t,
            attention_mask=attn_mask,
            cross_encodings=cross_encodings,
            cross_attention_mask=cross_attn_mask
        )
        x_0 = scores['x_0']

        return {
            "x_0": x_0,
            "x_0_target": normed_ddpm_target
        }

    def split_batch(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        keys = [
            'input_ids',
            'attention_mask',
            'token_type_ids',
            'position_ids'
        ]
        to_clean_part: Dict[str, Tensor] = dict()
        to_noise_part: Dict[str, Tensor] = dict()

        for prefix, dct_part in zip(['clean_', 'noisy_'], [to_clean_part, to_noise_part]):
            for k in keys:
                key_name = prefix + k
                if key_name in batch:
                    dct_part[k] = batch[key_name]
        return to_clean_part, to_noise_part

    def sample_encodings(self, batch: Dict[str, Tensor]) -> Dict[str, EncoderOutput]:
        to_clean_part, to_noise_part = self.split_batch(batch)

        clean_part: EncoderOutput = self.clean_part_encoder.forward(**to_clean_part)
        noisy_part: EncoderOutput = self.noisy_part_encoder.forward(**to_noise_part)
        return {
            "noisy_part": noisy_part,
            "clean_part": clean_part
        }


    def step_logic(self, batch: Dict[str, Tensor]) -> STEP_OUTPUT:
        outputs = self.forward(batch)

        attn_mask = torch.ones_like(
            batch['noisy_attention_mask']
        )
        input_ids = batch['noisy_input_ids']

        pred_x_0 = outputs['x_0']
        gt_x_0 = outputs['x_0_target']

        x0_loss = torch.mean((pred_x_0 - gt_x_0)**2, dim=2)
        logits = self.noisy_part_encoder.classify(normed=pred_x_0).permute(0, 2, 1)
        # [BS; C; LEN]
        # [BS; LEN]
        ce_loss = torch.nn.functional.cross_entropy(logits, input_ids)

        def masked_loss(loss: FloatTensor, mask: FloatTensor):
            loss = loss * mask
            loss = torch.sum(loss, dim=1) / torch.sum(mask, dim=1)
            loss = torch.mean(loss)
            return loss

        x0_loss = masked_loss(x0_loss, attn_mask)
        ce_loss = masked_loss(ce_loss, attn_mask)

        batch_size = len(input_ids)

        return {
            'batch_size': batch_size,
            'ce_loss': ce_loss,
            'x0_loss': x0_loss,
            'loss': x0_loss + self.ce_coef * ce_loss
        }

    @torch.no_grad()
    def generate_text(self, batch: Dict[str, Tensor], init_x: Optional[Tensor] = None):
        self.eval()
        to_clean_part, to_noise_part = self.split_batch(batch)
        clean_part: EncoderOutput = self.clean_part_encoder.forward(**to_clean_part)
        noisy_part_attention_mask = torch.ones_like(
            batch['noisy_attention_mask']
        )
        noisy_part_pred_encodings = self.generate_encodings(
            shape=noisy_part_attention_mask.shape + (clean_part.normed.shape[-1],),
            cross_encodings=clean_part.normed,
            cross_attention_mask=to_clean_part['attention_mask'],
            attn_mask=noisy_part_attention_mask,
            verbose=True,
            init_x=init_x
        )
        logits = self.noisy_part_encoder.classify(normed=noisy_part_pred_encodings)
        text_labels = torch.argmax(logits, dim=-1)
        return text_labels, noisy_part_pred_encodings
    
    @torch.no_grad()
    def generate_text_cfg(self, batch: Dict[str, Tensor], init_x: Optional[Tensor] = None, w: float = 0):
        self.eval()
        to_clean_part, to_noise_part = self.split_batch(batch)
        clean_part: EncoderOutput = self.clean_part_encoder.forward(**to_clean_part)
        noisy_part_attention_mask = torch.ones_like(
            batch['noisy_attention_mask']
        )
        noisy_part_pred_encodings = self.generate_encodings_cfg(
            shape=noisy_part_attention_mask.shape + (clean_part.normed.shape[-1],),
            cross_encodings=clean_part.normed,
            cross_attention_mask=to_clean_part['attention_mask'],
            attn_mask=noisy_part_attention_mask,
            verbose=True,
            init_x=init_x,
            w=w
        )
        logits = self.noisy_part_encoder.classify(normed=noisy_part_pred_encodings)
        text_labels = torch.argmax(logits, dim=-1)
        return text_labels

    torch.no_grad()
    def ode_forward_dynamic(
        self,
        batch: Dict[str, Tensor],
    ):
        assert self.sde.ode_sampling is True
        to_clean_part, to_noise_part = self.split_batch(batch)

        clean_part: EncoderOutput = self.clean_part_encoder.forward(**to_clean_part)
        noisy_part: EncoderOutput = self.noisy_part_encoder.forward(**to_noise_part)
        
        noisy_part_attention_mask = torch.ones_like(
            batch['noisy_attention_mask']
        )
        
        shape = noisy_part_attention_mask.shape + (clean_part.normed.shape[-1],)
        cross_encodings = clean_part.normed
        device = cross_encodings.device
        
        target_encodings = noisy_part.normed
        
        cross_attention_mask = to_clean_part['attention_mask']
        attn_mask = noisy_part_attention_mask

        score_call = partial(
            self.score_estimator.forward,
            cross_attention_mask=cross_attention_mask,
            cross_encodings=cross_encodings
        )
        verbose = True
        
        prefix_time = 0.001
        batch_size = target_encodings.shape[0]
        input_t = torch.ones(batch_size, device=device) * prefix_time
        marg_forward = self.sde.marginal_forward(target_encodings, input_t)
        target_encodings = marg_forward['mean']
        prev_t = 0
        x_t = target_encodings

        timesteps = torch.linspace(
            0.001,
            self.sde.T,
            self.sde.N,
            device=device
        )
        rang = trange if verbose else range

        for idx in rang(self.sde.N):
            t = timesteps[idx]
            input_t = t * torch.ones(shape[0], device=device)
            
            dt = t - prev_t
            prev_t = t
            with torch.no_grad():
                rsde_params = self.solver.rsde.sde(score_call, x_t, input_t, attn_mask)
            
            drift = rsde_params['drift']
            # return rsde_params
            
            x_t = x_t + drift * dt
    
        return x_t, target_encodings

    def generate_encodings_cfg(
            self,
            shape: Tuple[int],
            cross_encodings: FloatTensor,
            cross_attention_mask: FloatTensor,
            attn_mask: Optional[torch.Tensor] = None,
            verbose: bool = False,
            init_x: Optional[torch.Tensor] = None,
            w: float = 0
    ) -> torch.Tensor:
        device = cross_encodings.device
        if attn_mask is None:
            attn_mask = torch.ones(shape[:-1], device=device)
        score_call = partial(
            self.score_estimator.forward,
            cross_attention_mask=cross_attention_mask,
            cross_encodings=cross_encodings
        )
        cross_encodings_uncond = torch.load(
            'data/cross_encodings_empty.pth', map_location=device
        )
        cross_encodings_uncond = torch.tile(cross_encodings_uncond, (shape[0], 1, 1))
        cross_attention_mask_uncond = torch.ones((shape[0], 1), device=device)
        score_uncond_call = partial(
            self.score_estimator.forward,
            cross_attention_mask=cross_attention_mask_uncond,
            cross_encodings=cross_encodings_uncond
        )
        with torch.no_grad():
            x_t = x_mean = self.sde.prior_sampling(shape).to(device) if init_x is None else init_x.to(device)
            timesteps = torch.linspace(self.sde.T, self.sde.T / self.sde.N, self.sde.N, device=device)
            rang = trange if verbose else range

            for idx in rang(self.sde.N):
                t = timesteps[idx]
                input_t = t * torch.ones(shape[0], device=device)
                
                dt = -1. / self.sde.N
                z = torch.randn_like(x_t)
                
                sde_params = self.sde.sde(x_t, input_t)
                drift_par, diffusion = sde_params['drift'], sde_params['diffusion']  # -1/2 * beta * x_t, sqrt(beta)

                scores_cond = self.sde.calc_score(score_call, x_t, input_t, attn_mask=attn_mask)
                score_cond = scores_cond['score']
                scores_uncond = self.sde.calc_score(score_uncond_call, x_t, input_t, attn_mask=attn_mask)
                score_uncond = scores_uncond['score']
                score = score_cond + w * (score_cond - score_uncond)
                
                drift = drift_par - diffusion[:, None, None] ** 2 * score * (0.5 if self.sde.ode_sampling else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.sde.ode_sampling else diffusion
                
                x_mean = x_t + drift * dt
                
                if not self.sde.ode_sampling:
                    x_t = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
                else:
                    x_t = x_mean
                
        return x_mean

    def generate_encodings(
            self,
            shape: Tuple[int],
            cross_encodings: FloatTensor,
            cross_attention_mask: FloatTensor,
            attn_mask: Optional[torch.Tensor] = None,
            verbose: bool = False,
            init_x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        device = cross_encodings.device
        if attn_mask is None:
            attn_mask = torch.ones(shape[:-1], device=device)
        score_call = partial(
            self.score_estimator.forward,
            cross_attention_mask=cross_attention_mask,
            cross_encodings=cross_encodings
        )
        with torch.no_grad():
            x_t = x_mean = self.sde.prior_sampling(shape).to(device) if init_x is None else init_x.to(device)
            timesteps = torch.linspace(self.sde.T, self.sde.T / self.sde.N, self.sde.N, device=device)
            rang = trange if verbose else range

            for idx in rang(self.sde.N):
                t = timesteps[idx]
                input_t = t * torch.ones(shape[0], device=device)
                new_state = self.solver.step(score_call, x_t, input_t, attn_mask)
                x_t, x_mean = new_state['x_t'], new_state['x_mean']

        return x_mean

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.compute_epoch_mean(self.train_metric_to_log), is_train=True, sync_dist=True)
        self.train_metric_to_log = Counter()
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.compute_epoch_mean(self.valid_metric_to_log), is_train=False, sync_dist=True)
        self.valid_metric_to_log = Counter()
        return super().on_validation_epoch_end()

    def compute_epoch_mean(self, dct: Dict[str, Tensor]):
        count = dct.pop('count')
        dct_to_log: Dict[str, Tensor] = {}
        for key, value in filter_losses(dct).items():
            dct_to_log[key + '_epoch'] = value / count
        return dct_to_log

    def add_to_dict(self, dct: Dict[str, Tensor], outputs: STEP_OUTPUT) -> None:
        batch_size = outputs['batch_size']
        dct['count'] += batch_size
        for key, value in filter_losses(outputs).items():
            dct[key] += value * batch_size

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.add_to_dict(self.train_metric_to_log, outputs)
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_validation_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.add_to_dict(self.valid_metric_to_log, outputs)
        return super().on_validation_batch_end(outputs, batch, batch_idx)

    def log_dict(self, losses: Dict[str, Tensor],
                 is_train: bool = True, apply_suffix: bool = True,
                 *args, **kwargs):
        if apply_suffix:
            suffix = 'train' if is_train else 'valid'
            losses = {key + f'/{suffix}': value for key, value in losses.items()}
        return super().log_dict(losses, *args, **kwargs)

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        outputs = self.step_logic(batch)
        return outputs

    def training_step(self, batch: Dict[str, Tensor], *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        outputs = self.step_logic(batch)
        self.log_dict(filter_losses(outputs), is_train=True, sync_dist=True)
        return outputs

    def on_before_optimizer_step(self, *args, **kwargs) -> None:
        self.logger.log_metrics({'grads_norm/score_estimator': calc_model_grads_norm(self.score_estimator)})
        return super().on_before_optimizer_step(*args, **kwargs)

    def optimizer_step(self, *args, **kwargs) -> None:
        returns =  super().optimizer_step(*args, **kwargs)
        self.logger.log_metrics({'weights_norm/score_estimator': calc_model_weights_norm(self.score_estimator)})
        return returns
