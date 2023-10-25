import torch
import math

from torch import nn
from typing import Optional, List, Dict, Union

from diffusion.models.contextual_denoising.planner.modules import TransformerEncoder


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PlannerEstimator(nn.Module):
    def __init__(self, config):
        super(PlannerEstimator, self).__init__()
        self.config = config
        hidden_layer_dim = self.config.hidden_size
        self._hidden_layer_dim = hidden_layer_dim
        self.time_emb = torch.nn.Sequential(
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_layer_dim * 2, hidden_layer_dim)
        )

        self.encoder = TransformerEncoder(config)

        self._max_position_embeddings = self.config.max_position_embeddings
        self.register_buffer("position_ids", torch.arange(self._max_position_embeddings).expand((1, -1)))
        self.position_embeddings = torch.nn.Embedding(self._max_position_embeddings, self._hidden_layer_dim)

    def get_extended_attention_mask(self, attention_mask, dtype):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(
            self,
            x_t: torch.Tensor,
            time_t: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_encodings: Optional[torch.FloatTensor] = None,
            cross_attention_mask: Optional[torch.FloatTensor] = None
    ):
        assert time_t is not None

        emb_t = timestep_embedding(time_t, self._hidden_layer_dim)
        hidden_t = self.time_emb(emb_t)
        # hidden_t = hidden_t[:, None, :] UBRAL, HOCHU # [BS: H_DIM]

        seq_length = x_t.size(1)
        position_ids = self.position_ids[:, : seq_length]
        emb_pos = self.position_embeddings(position_ids)

        emb_x = x_t
        hidden_state = emb_x + emb_pos

        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(
                attention_mask=attention_mask,
                dtype=hidden_state.dtype
            )

        cond = cross_encodings
        cond_mask = cross_attention_mask
        assert cond_mask is not None

        if cond_mask is not None:
            cond_mask = self.get_extended_attention_mask(
                attention_mask=cond_mask,
                dtype=hidden_state.dtype
            )

        output = self.encoder(
            x=hidden_state,
            attention_mask=attention_mask,
            emb_t=hidden_t,
            cond=cond,
            cond_mask=cond_mask,
        )
        return output


if __name__ == '__main__':
    from diffusion.models.contextual_denoising.planner import bert_config_slava
    se = PlannerEstimator(bert_config_slava)
    se.eval()

    bs = 10
    seq_len_cond = 10
    seq_len_x = 16
    hidden_dim = 768

    cross_encs = torch.randn(bs, seq_len_cond, hidden_dim)
    cross_encs_att_map = torch.ones(bs, seq_len_cond)

    x_t = torch.randn(bs, seq_len_x, hidden_dim)
    att_map = torch.ones(bs, seq_len_x)

    time_t = torch.rand(bs)

    with torch.no_grad():
        outs = se.forward(x_t, time_t, att_map, cross_encs, cross_encs_att_map)
        print(outs.shape)
