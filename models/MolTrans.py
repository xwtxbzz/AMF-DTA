from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn


class MolTrans(nn.Module):
    """MolTrans model aligned with the official ``BIN_Interaction_Flat`` code.

    Official source used for structure:
    https://github.com/kexinhuang12345/MolTrans/blob/master/models.py

    This wrapper keeps a simple regression interface ``forward(drug, target)``.
    Masks are derived from padding id 0.
    """

    def __init__(
        self,
        drug_vocab_size: int,
        target_vocab_size: int,
        max_drug_seq: int = 120,
        max_protein_seq: int = 1000,
        emb_size: int = 128,
        intermediate_size: int = 512,
        num_attention_heads: int = 8,
        dropout_rate: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_d = max_drug_seq
        self.max_p = max_protein_seq
        self.emb_size = emb_size
        self.dropout_rate = dropout_rate

        self.demb = Embeddings(drug_vocab_size, emb_size, max_drug_seq, dropout_rate)
        self.pemb = Embeddings(target_vocab_size, emb_size, max_protein_seq, dropout_rate)
        self.d_encoder = EncoderMultipleLayers(2, emb_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.p_encoder = EncoderMultipleLayers(2, emb_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.icnn = nn.Conv2d(1, 3, 3, padding=0)
        flat_dim = 3 * max(max_drug_seq - 2, 1) * max(max_protein_seq - 2, 1)
        self.decoder = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
        )

    def forward(self, drug: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        drug = _fit_length(drug.long(), self.max_d)
        target = _fit_length(target.long(), self.max_p)
        d_mask = drug.ne(0).float()
        p_mask = target.ne(0).float()

        ex_d_mask = (1.0 - d_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
        ex_p_mask = (1.0 - p_mask.unsqueeze(1).unsqueeze(2)) * -10000.0

        d_encoded = self.d_encoder(self.demb(drug).float(), ex_d_mask.float())
        p_encoded = self.p_encoder(self.pemb(target).float(), ex_p_mask.float())
        d_aug = d_encoded.unsqueeze(2).repeat(1, 1, self.max_p, 1)
        p_aug = p_encoded.unsqueeze(1).repeat(1, self.max_d, 1, 1)
        interaction = torch.sum(d_aug * p_aug, dim=-1).unsqueeze(1)
        interaction = F.dropout(interaction, p=self.dropout_rate, training=self.training)
        features = self.icnn(interaction).flatten(start_dim=1)
        return self.decoder(features).squeeze(-1)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, variance_epsilon: float = 1e-12) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position_size: int, dropout_rate: float) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        return self.dropout(self.LayerNorm(embeddings))


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_probs_dropout_prob: float) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = self.dropout(nn.Softmax(dim=-1)(attention_scores))
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return context_layer.view(*new_context_shape)


class SelfOutput(nn.Module):
    def __init__(self, hidden_size: int, hidden_dropout_prob: float) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(self.dense(hidden_states))
        return self.LayerNorm(hidden_states + input_tensor)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_probs_dropout_prob: float, hidden_dropout_prob: float) -> None:
        super().__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.output(self.self(input_tensor, attention_mask), input_tensor)


class Encoder(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_attention_heads: int, attention_probs_dropout_prob: float, hidden_dropout_prob: float) -> None:
        super().__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = F.relu(self.intermediate(attention_output))
        layer_output = self.dropout(self.output_dense(intermediate_output))
        return self.output_norm(layer_output + attention_output)


class EncoderMultipleLayers(nn.Module):
    def __init__(self, n_layer: int, hidden_size: int, intermediate_size: int, num_attention_heads: int, attention_probs_dropout_prob: float, hidden_dropout_prob: float) -> None:
        super().__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


def _fit_length(tokens: torch.Tensor, length: int) -> torch.Tensor:
    if tokens.size(1) > length:
        return tokens[:, :length]
    if tokens.size(1) < length:
        pad = tokens.new_zeros(tokens.size(0), length - tokens.size(1))
        return torch.cat([tokens, pad], dim=1)
    return tokens
