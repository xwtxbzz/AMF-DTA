from __future__ import annotations

import torch
from torch import nn


class AttentionDTA(nn.Module):
    """Sequence-based AttentionDTA implementation for DTA regression.

    Paper: AttentionDTA: Drug-Target Binding Affinity Prediction by
    Sequence-Based Deep Learning With Attention Mechanism.

    The paper uses SMILES strings and amino-acid sequences, 1D-CNN feature
    extractors, and a two-side multi-head attention mechanism. This local module
    keeps that structure while exposing the same ``forward(drug, target)`` API
    used by the unified benchmark framework.
    """

    def __init__(
        self,
        drug_vocab_size: int,
        target_vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        drug_kernel_size: int = 4,
        target_kernel_size: int = 8,
        conv_layers: int = 3,
        attention_heads: int = 4,
        fc_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_encoder = CNNSequenceEncoder(
            drug_vocab_size,
            embedding_dim,
            hidden_dim,
            drug_kernel_size,
            conv_layers,
            dropout,
        )
        self.target_encoder = CNNSequenceEncoder(
            target_vocab_size,
            embedding_dim,
            hidden_dim,
            target_kernel_size,
            conv_layers,
            dropout,
        )
        self.drug_to_target = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.target_to_drug = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.norm_drug = nn.LayerNorm(hidden_dim)
        self.norm_target = nn.LayerNorm(hidden_dim)
        self.regressor = MLP(hidden_dim * 4, fc_dims, dropout)

    def forward(self, drug: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        drug_mask = drug.eq(0)
        target_mask = target.eq(0)
        drug_features = self.drug_encoder(drug)
        target_features = self.target_encoder(target)

        drug_attended, _ = self.drug_to_target(
            drug_features,
            target_features,
            target_features,
            key_padding_mask=target_mask,
            need_weights=False,
        )
        target_attended, _ = self.target_to_drug(
            target_features,
            drug_features,
            drug_features,
            key_padding_mask=drug_mask,
            need_weights=False,
        )
        drug_vec = masked_mean(self.norm_drug(drug_features + drug_attended), drug_mask)
        target_vec = masked_mean(self.norm_target(target_features + target_attended), target_mask)
        features = torch.cat([drug_vec, target_vec, drug_vec * target_vec, torch.abs(drug_vec - target_vec)], dim=-1)
        return self.regressor(features).squeeze(-1)


class CNNSequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, kernel_size: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        blocks: list[nn.Module] = []
        in_channels = embedding_dim
        padding = kernel_size // 2
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=padding),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = hidden_dim
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x).transpose(1, 2)
        z = self.network(z).transpose(1, 2)
        return z[:, : x.size(1)]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (~mask).unsqueeze(-1).to(x.dtype)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (x * valid).sum(dim=1) / denom
