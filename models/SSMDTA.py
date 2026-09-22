from __future__ import annotations

import math

import torch
from torch import nn


class SSMDTA(nn.Module):
    """Supervised PyTorch variant of SSM-DTA for local DTA benchmarks.

    The official SSM-DTA implementation is built on fairseq/RoBERTa with
    separate molecule/protein encoders and a lightweight cross-attention module:
    https://github.com/QizhiPei/SSM-DTA

    This module keeps that interaction pattern while using plain PyTorch so it
    can train inside the current SMILES+protein-sequence experiment framework.
    """

    def __init__(
        self,
        drug_vocab_size: int,
        target_vocab_size: int,
        drug_max_len: int,
        target_max_len: int,
        embed_dim: int = 128,
        encoder_layers: int = 2,
        num_attention_heads: int = 4,
        ffn_dim: int = 256,
        cross_attention_heads: int = 4,
        dropout: float = 0.1,
        fc_dims: tuple[int, ...] = (512, 256),
    ) -> None:
        super().__init__()
        self.drug_padding_idx = 0
        self.target_padding_idx = 0

        self.drug_embedding = TokenPositionEmbedding(drug_vocab_size, drug_max_len, embed_dim, dropout)
        self.target_embedding = TokenPositionEmbedding(target_vocab_size, target_max_len, embed_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attention_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.drug_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.target_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        self.drug_to_target = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=cross_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.target_to_drug = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=cross_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drug_pool = AttentionPool(embed_dim, dropout)
        self.target_pool = AttentionPool(embed_dim, dropout)
        self.norm = nn.LayerNorm(embed_dim * 4)
        self.regressor = MLP(embed_dim * 4, fc_dims, dropout)

    def forward(self, drug: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        drug_mask = drug.eq(self.drug_padding_idx)
        target_mask = target.eq(self.target_padding_idx)

        drug_repr = self.drug_embedding(drug)
        target_repr = self.target_embedding(target)
        drug_repr = self.drug_encoder(drug_repr, src_key_padding_mask=drug_mask)
        target_repr = self.target_encoder(target_repr, src_key_padding_mask=target_mask)

        drug_context, _ = self.drug_to_target(
            query=drug_repr,
            key=target_repr,
            value=target_repr,
            key_padding_mask=target_mask,
            need_weights=False,
        )
        target_context, _ = self.target_to_drug(
            query=target_repr,
            key=drug_repr,
            value=drug_repr,
            key_padding_mask=drug_mask,
            need_weights=False,
        )

        drug_vec = self.drug_pool(drug_repr + drug_context, drug_mask)
        target_vec = self.target_pool(target_repr + target_context, target_mask)
        features = torch.cat([drug_vec, target_vec, drug_vec * target_vec, torch.abs(drug_vec - target_vec)], dim=-1)
        return self.regressor(self.norm(features)).squeeze(-1)


class TokenPositionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        return self.dropout(self.token(x) * self.scale + self.position(positions))


class AttentionPool(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(x * weights, dim=1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
