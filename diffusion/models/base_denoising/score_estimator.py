import torch
import math
from typing import List, Optional, Tuple, Union

from transformers.models.bert.modeling_bert import BertEncoder, BertConfig


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


class ScoreEstimator(BertEncoder):
    def __init__(self, config: BertConfig):
        super(ScoreEstimator, self).__init__(config)

        input_size = config.hidden_size
        self.input_size = input_size
        hidden_layer_dim = self.config.hidden_size
        self._hidden_layer_dim = hidden_layer_dim
        self.time_emb = torch.nn.Sequential(
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_layer_dim * 2, hidden_layer_dim)
        )

        self._max_position_embeddings = self.config.max_position_embeddings
        self.register_buffer("position_ids", torch.arange(self._max_position_embeddings).expand((1, -1)))
        self.position_embeddings = torch.nn.Embedding(self._max_position_embeddings, self._hidden_layer_dim)

        self.input_up_proj = torch.nn.Sequential(
            torch.nn.Linear(input_size, self._hidden_layer_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self._hidden_layer_dim, self._hidden_layer_dim)
        )
        self.LayerNorm = torch.nn.LayerNorm(self._hidden_layer_dim, eps=self.config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        self.output_down_proj = torch.nn.Sequential(
            torch.nn.Linear(self._hidden_layer_dim, self._hidden_layer_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self._hidden_layer_dim, input_size)
        )

    def get_extended_attention_mask(self, attention_mask, dtype):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(
            self,
            x_t: torch.Tensor,
            time_t: Optional[torch.Tensor] = None,
            *args, **kwargs
    ):
        assert time_t is not None
        input_type = x_t.dtype
        emb_t = timestep_embedding(time_t, self._hidden_layer_dim)
        # emb_t - shape [BATCH_SIZE; EMB_DIM]
        emb_t = self.time_emb(emb_t)
        # emb_t - shape [BATCH_SIZE; EMB_DIM]
        # emb_t[:, None, :] - shape [BATCH_SIZE; 1; EMB_DIM] - for broadcasting only
        # SLAVIK LEGENDA

        emb_x = self.input_up_proj(x_t)

        seq_length = x_t.size(1)
        position_ids = self.position_ids[:, : seq_length]
        emb_pos = self.position_embeddings(position_ids)

        hidden_state = emb_x + emb_t[:, None, :] + emb_pos
        hidden_state = self.dropout(self.LayerNorm(hidden_state))

        attention_mask = kwargs["attention_mask"]
        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(
                attention_mask=attention_mask,
                dtype=hidden_state.dtype
            )

        output = super(ScoreEstimator, self).forward(
            hidden_states=hidden_state,
            attention_mask=attention_mask,
        ).last_hidden_state

        output = self.output_down_proj(output).type(input_type)
        return output
