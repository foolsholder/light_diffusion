import torch

from torch import nn, Tensor
from torch.nn import functional as F

from typing import Any, Optional, Dict, List, Union, Callable, Any
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
from transformers.models.bert.modeling_bert import BertLMHeadModel as BB


class ZeroVoc2(L.LightningModule):
    def __init__(
        self,
        enc_normalizer_cfg: EncNormalizer,
        sde_cfg: SDE,
        optim_partial: Callable[[(...,)], torch.optim.Optimizer],
        sched_partial: Callable[[(...,)], LinearWarmupLR],
        label_mask_pos: int = 0,
        ce_coef: float = 0,
        test_count: int = 11,
    ) -> None:
        super().__init__()
        self.test_count = test_count
        self.enc_normalizer = instantiate(enc_normalizer_cfg)

        bert_cfg_name = 'bert-base-uncased'
        bert_config = BertConfig(bert_cfg_name)

        #self.encoder: TBB = TBB(bert_config, label_mask_pos=label_mask_pos).eval()
        #self.encoder.load_state_dict(
        #    BB.from_pretrained(bert_cfg_name).state_dict(),
        #)
        self.label_mask_pos = label_mask_pos
        self.encoder = TBB.from_pretrained(bert_cfg_name, label_mask_pos=label_mask_pos).eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.tokenizer = BertTokenizerFast.from_pretrained(bert_cfg_name)

        self.score_estimator = ScoreEstimator(bert_config)

        ema = ExponentialMovingAverage(
            parameters=self.score_estimator.parameters(), decay=0
        )
        self.load_and_copy_ema(ema, 'wiki_pret_old/slava_ckpt.pth')

        self.ema: Optional[ExponentialMovingAverage] = None

        self.sde = instantiate(sde_cfg)
        rsde = RSDE(self.sde)
        self.solver = EulerSolver(rsde, self.sde.ode_sampling)

        self._optim_partial = instantiate(optim_partial)
        self._sched_partial = instantiate(sched_partial)

        self.ce_coef = ce_coef

        self.train_metric_to_log: Dict[str, Tensor] = Counter()
        self.valid_metric_to_log: Dict[str, Tensor] = Counter()

        self.test_accuracy = Accuracy('binary')

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
        self.encoder.eval()
        return self

    def se_forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x_t, time_t, attn_mask = batch['x_t'], batch['time_t'], batch['attn_mask']
        return self.sde.calc_score(self.score_estimator, x_t, time_t, attn_mask)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch_size = len(batch['input_ids'])

        encodings = self.sample_encodings(batch)
        clean_x = self.enc_normalizer.normalize(encodings)

        attn_mask = batch['attention_mask']

        time_t = torch.rand(batch_size, device=clean_x.device)
        marg_forward = self.sde.marginal_forward(clean_x, time_t)
        x_t = marg_forward['x_t']

        scores = self.se_forward({
            'x_t': x_t,
            'time_t': time_t,
            'attn_mask': attn_mask
        })
        x_0 = scores['x_0']
        pred_x_0 = x_0
        clean_x_0 = clean_x
        #print(pred_x_0.shape,
        #      pred_x_0.requires_grad,
        #      torch.sqrt(torch.sum(encodings**2)),
        #      torch.sqrt(torch.sum(clean_x_0**2)),
        #      torch.sqrt(torch.sum(pred_x_0**2)),
        #      torch.mean((pred_x_0[:, 0] - clean_x_0[:, 0])**2),
        #      flush=True)

        return {
            "x_0": x_0,
            "clean_x": clean_x
        }

    def sample_encodings(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        keys = [
            'input_ids',
            'attention_mask',
            'token_type_ids',
            'position_ids'
        ]
        res = dict()
        for k in keys:
            if k in batch:
                res[k] = batch[k]
        last_hidden_state: Tensor = self.encoder.forward(**res, return_encoding=True)
        return last_hidden_state

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
            'bce_loss': bce_loss,
            'ce30k_loss': ce_loss,
            'x0_loss': x0_loss,
            'loss': x0_loss + self.ce_coef * bce_loss
        }

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.compute_epoch_mean(self.train_metric_to_log), is_train=True, sync_dist=True)
        self.train_metric_to_log = Counter()
        return super().on_train_epoch_end()

    def compute_epoch_mean(self, dct: Dict[str, Tensor]):
        count = dct.pop('count')
        dct_to_log: Dict[str, Tensor] = {}
        for key, value in filter_losses(dct).items():
            dct_to_log[key + '_epoch'] = value / count
        return dct_to_log

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.compute_epoch_mean(self.valid_metric_to_log), is_train=False, sync_dist=True)
        self.valid_metric_to_log = Counter()
        return super().on_validation_epoch_end()

    def add_to_dict(self, dct: Dict[str, Tensor], outputs: STEP_OUTPUT) -> None:
        batch_size = outputs['batch_size']
        dct['count'] += batch_size
        for key, value in filter_losses(outputs).items():
            dct[key] += value * batch_size

    def on_validation_batch_end(self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int) -> None:
        self.add_to_dict(self.valid_metric_to_log, outputs)
        return super().on_validation_batch_end(outputs, batch, batch_idx)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.add_to_dict(self.train_metric_to_log, outputs)
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def log_dict(self, losses: Dict[str, Tensor], is_train: bool = True, *args, **kwargs):
        suffix = 'train' if is_train else 'valid'
        losses = {key + f'/{suffix}': value for key, value in losses.items()}
        return super().log_dict(losses, *args, **kwargs)

    def training_step(self, batch: Dict[str, Tensor], *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        outputs = self.step_logic(batch)
        self.log_dict(filter_losses(outputs), is_train=True, sync_dist=True)
        return outputs

    def validation_step(self, batch: Dict[str, Tensor], *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.step_logic(batch)

    def on_before_optimizer_step(self, *args, **kwargs) -> None:
        self.logger.log_metrics({'model/grads_norm': calc_model_grads_norm(self.score_estimator)})
        return super().on_before_optimizer_step(*args, **kwargs)

    def optimizer_step(self, *args, **kwargs) -> None:
        returns =  super().optimizer_step(*args, **kwargs)
        self.logger.log_metrics({'model/weights_norm': calc_model_weights_norm(self.score_estimator)})
        return returns

    def ddrm_step(self, x_t, t, clean_x, ddrm_mask, attn_mask, gamma: float = 1):
        self.sde: SDE
        x_t_forward = self.sde.marginal_forward(clean_x, t)['x_t']

        x_t = torch.where(ddrm_mask[:, :, None], gamma * x_t_forward + (1 - gamma) * x_t, x_t)
        return self.solver.step(self.score_estimator, x_t, t, attn_mask)['x_t']

    def ddrm_glue_fst_pos(self, batch: Dict[str, Tensor]):
        encodings = self.encoder.bert(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])[0]
        labels_binary = batch['labels'].view(-1)

        batch_size = len(labels_binary)
        clean_x = self.enc_normalizer.normalize(encodings)
        attn_mask = batch['attention_mask']
        ddrm_mask = torch.ones_like(attn_mask)
        ddrm_mask[:, 0] = 0
        ddrm_mask = ddrm_mask.bool()

        clean_x = torch.tile(clean_x, (self.test_count, 1, 1))
        attn_mask = torch.tile(attn_mask, (self.test_count, 1))
        ddrm_mask = torch.tile(ddrm_mask, (self.test_count, 1))

        aug_batch_size = len(clean_x)

        x_t = self.sde.prior_sampling(clean_x.shape).to(clean_x.device)
        timesteps = torch.linspace(self.sde.T, self.sde.T / self.sde.N, self.sde.N, device=x_t.device)

        from tqdm.auto import trange
        #bar = trange
        bar = range
        for idx in bar(self.sde.N):
            time_tensor = torch.ones(aug_batch_size, device=clean_x.device) * timesteps[idx]
            x_t = self.ddrm_step(x_t, time_tensor, clean_x, ddrm_mask, attn_mask)
        #print(batch['input_ids'][:4, 0])
        pred_encodings = self.enc_normalizer.denormalize(x_t)
        logits = self.encoder.forward(pred_encodings=pred_encodings).logits
        #logits = self.encoder.cls(encodings)
        #logits = self.encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
        #tok = self.tokenizer.batch_decode(logits[:, 0].argmax(dim=-1))
        #full_tok = self.tokenizer.batch_decode(logits.argmax(dim=-1))
        #true_tok = self.tokenizer.batch_decode(batch['input_ids'][:, 0])
        #full_true_tok = self.tokenizer.batch_decode(batch['input_ids'])
        #print(tok, true_tok, logits[:, 0].argmax(dim=-1), batch['input_ids'][:, 0])
        #print(full_tok[0], '\n####\n', full_true_tok[0])
        logits_binary = logits[:, 0, [2053, 2748]]
        pred_label = torch.argmax(logits_binary, dim=-1).float().reshape(self.test_count, batch_size).mean(dim=0)
        self.test_accuracy.update(pred_label, labels_binary)
        print(self.test_accuracy.compute())

    def on_test_epoch_start(self):
        self.test_accuracy.reset()
        #print('test-start')
        #self.encoder = TBB.from_pretrained('bert-base-uncased').cuda().eval()
        # self.encoder.train()

    def reset_test_accuracy(self):
        self.test_accuracy.reset()

    def test_step(self, batch: Dict[str, Tensor], *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        self.ddrm_glue_fst_pos(batch)

    def on_test_epoch_end(self) -> None:
        self.log_dict({'accuracy': self.test_accuracy.compute()}, is_train=False)