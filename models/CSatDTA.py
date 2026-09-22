from __future__ import annotations

import torch
from torch import nn


class CSatDTA(nn.Module):
    """Convolution plus self-attention DTA model.

    Paper: CSatDTA: Prediction of Drug-Target Binding Affinity Using
    Convolution Model with Self-Attention.

    The paper augments convolutional sequence encoders with self-attention
    feature maps and predicts affinity from SMILES and protein sequences only.
    """

    def __init__(
        self,
        drug_vocab_size: int,
        target_vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        drug_kernel_size: int = 7,
        target_kernel_size: int = 9,
        attention_heads: int = 4,
        blocks: int = 2,
        fc_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_encoder = AttentionAugmentedEncoder(
            drug_vocab_size,
            embedding_dim,
            hidden_dim,
            drug_kernel_size,
            attention_heads,
            blocks,
            dropout,
        )
        self.target_encoder = AttentionAugmentedEncoder(
            target_vocab_size,
            embedding_dim,
            hidden_dim,
            target_kernel_size,
            attention_heads,
            blocks,
            dropout,
        )
        self.regressor = MLP(hidden_dim * 4, fc_dims, dropout)

    def forward(self, drug: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        drug_mask = drug.eq(0)
        target_mask = target.eq(0)
        drug_features = self.drug_encoder(drug, drug_mask)
        target_features = self.target_encoder(target, target_mask)
        drug_vec = masked_max_mean(drug_features, drug_mask)
        target_vec = masked_max_mean(target_features, target_mask)
        pair = torch.cat([drug_vec, target_vec, drug_vec * target_vec, torch.abs(drug_vec - target_vec)], dim=-1)
        return self.regressor(pair).squeeze(-1)


class AttentionAugmentedEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        kernel_size: int,
        attention_heads: int,
        blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            AttentionAugmentedConvBlock(hidden_dim, kernel_size, attention_heads, dropout)
            for _ in range(blocks)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(self.embedding(tokens))
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class AttentionAugmentedConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, attention_heads: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.mix = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        conv_out = conv_out[:, : x.size(1)]
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask, need_weights=False)
        fused = self.mix(torch.cat([conv_out, attn_out], dim=-1))
        return self.norm(x + self.dropout(fused))


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_max_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (~mask).unsqueeze(-1).to(x.dtype)
    mean = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    max_values = x.masked_fill(mask.unsqueeze(-1), torch.finfo(x.dtype).min).max(dim=1).values
    max_values = torch.where(torch.isfinite(max_values), max_values, torch.zeros_like(max_values))
    return 0.5 * (mean + max_values)
