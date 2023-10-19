import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union

from transformers.models.bert.modeling_bert import (
    BertIntermediate, BertConfig, BertSelfAttention,
    apply_chunking_to_forward, prune_linear_layer,
    find_pruneable_heads_and_indices
)

bert_config_slava = BertConfig(**{
    "hidden_size": 768,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "vocab_size": 30522,
    "hidden_dropout_prob": 0.1,
    "num_attention_heads": 12,
    "type_vocab_size": 2,
    "max_position_embeddings": 512,
    "num_hidden_layers": 12,
    "intermediate_size": 3072,
    "attention_probs_dropout_prob": 0.1,
    "layer_norm_eps": 1e-12,
    "model_type": "bert",
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.6.0.dev0",
    "is_decoder": True,
})



class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps,
            elementwise_affine=False
        )
        self.beta_gamma_alpha_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size * 3)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        input_states = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        beta_mean, gamma_std, alpha_scale = self.beta_gamma_alpha_adaLN(cond_emb).chunk(3, dim=1)
        # [BS; H_DIM]
        hidden_states = hidden_states * (1 + gamma_std[:, None]) * beta_mean[:, None]

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states) * alpha_scale[:, None]

        hidden_states = hidden_states + input_states

        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps,
            elementwise_affine=False
        )
        self.beta_gamma_alpha_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size * 3)
        )

        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        time_emb: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        input_states = hidden_states
        cond_emb = time_emb

        hidden_states = self.LayerNorm(hidden_states)
        beta_mean, gamma_std, alpha_scale = self.beta_gamma_alpha_adaLN(cond_emb).chunk(3, dim=1)
        # [BS; H_DIM]
        hidden_states = hidden_states * (1 + gamma_std[:, None]) * beta_mean[:, None]

        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self_outputs[0]
        result = attention_output * alpha_scale[:, None]
        result = input_states + attention_output

        result = self.output(result, cond_emb)
        return (result,)


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.intermediate = BertIntermediate(config)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps,
            elementwise_affine=False
        )
        self.beta_gamma_alpha_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size * 3)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        input_states = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        beta_mean, gamma_std, alpha_scale = self.beta_gamma_alpha_adaLN(cond_emb).chunk(3, dim=1)
        # [BS; H_DIM]
        hidden_states = hidden_states * (1 + gamma_std[:, None]) * beta_mean[:, None]

        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states) * alpha_scale[:, None]

        hidden_states = hidden_states + input_states

        return hidden_states


class BertBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = bert_config_slava
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        assert self.is_decoder is True, "NE DECODER, BRAT, POCHEMU??"
        if self.is_decoder:
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            time_emb: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            time_emb=time_emb
        )
        attention_output = self_attention_outputs[0]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                hidden_states=attention_output,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                time_emb=time_emb
            )
            attention_output = cross_attention_outputs[0]

        outputs = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim,
            (attention_output, time_emb)
        )

        return outputs

    def feed_forward_chunk(self, inp_for_output_layer):
        hidden_states, time_t = inp_for_output_layer
        layer_output = self.output.forward(hidden_states, time_t)
        return layer_output


TransformerBlock = BertBlock


class TransformerEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_hidden_layers = 12
        self.hidden_size = 768
        self.input_blocks = torch.nn.ModuleList(
            [TransformerBlock(config) for _ in range(0, self.num_hidden_layers // 2)]
        )
        self.output_blocks = torch.nn.ModuleList(
            [TransformerBlock(config) for _ in range(0, self.num_hidden_layers // 2)]
        )
        # self.time_layers = torch.nn.ModuleList(
        #    [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(0, self.num_hidden_layers)]
        # )

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            emb_t=None,
            cond=None,
            cond_mask=None,
    ):

        x_input_list = []

        for i, block in enumerate(self.input_blocks):
            x_input_list.append(x)
            # x = x + self.time_layers[i](emb_t) SLAVYAN
            x = x
            x = block(
                hidden_states=x,
                attention_mask=attention_mask,
                encoder_hidden_states=cond,
                encoder_attention_mask=cond_mask,
                time_emb=emb_t # NOT LIKE U SLAVYANA
            )

        for i, block in enumerate(self.output_blocks):
            # SLAVYAN
            # x = x + x_input_list.pop() + self.time_layers[i + self.num_hidden_layers // 2](emb_t)
            x = x + x_input_list.pop()
            x = block(
                hidden_states=x,
                attention_mask=attention_mask,
                encoder_hidden_states=cond,
                encoder_attention_mask=cond_mask,
                time_emb=emb_t # NOT LIKE U SLAVYANA
            )

        return x