from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class HiFDTA(nn.Module):
    """HiF-DTA-style hierarchical feature model for DTA regression.

    Reference:
    HiF-DTA: Hierarchical Feature Learning Network for Drug-Target Affinity
    Prediction, arXiv:2510.27281.

    The web search on 2026-07-13 did not identify an official GitHub
    implementation. This PyTorch/PyG implementation follows the paper-level
    design: local/topological graph features, global sequence features, and
    multi-scale attention fusion for drug-target affinity regression.
    """

    def __init__(
        self,
        num_node_features: int = 78,
        drug_vocab_size: int = 64,
        target_vocab_size: int = 26,
        hidden_dim: int = 128,
        graph_layers: int = 3,
        sequence_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.2,
        fc_dims: tuple[int, ...] = (512, 256),
    ) -> None:
        super().__init__()
        self.graph_encoder = MultiScaleGraphEncoder(num_node_features, hidden_dim, graph_layers, dropout)
        self.drug_sequence = GlobalLocalSequenceEncoder(drug_vocab_size, hidden_dim, sequence_layers, attention_heads, dropout)
        self.target_sequence = GlobalLocalSequenceEncoder(target_vocab_size, hidden_dim, sequence_layers, attention_heads, dropout)
        self.drug_scale_attention = ScaleAttention(hidden_dim, attention_heads, dropout)
        self.target_scale_attention = ScaleAttention(hidden_dim, attention_heads, dropout)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim * 4)

        dims = [hidden_dim * 4, *fc_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, drug: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        graph_scales = self.graph_encoder(x, edge_index, batch)
        drug_seq_scales = self.drug_sequence(drug.long())
        target_scales = self.target_sequence(target.long())

        drug_scales = torch.stack([*graph_scales, *drug_seq_scales], dim=1)
        target_scales = torch.stack(target_scales, dim=1)
        drug_vec = self.drug_scale_attention(drug_scales)
        target_vec = self.target_scale_attention(target_scales)

        pair_tokens = torch.stack([drug_vec, target_vec], dim=1)
        attended, _ = self.cross_attention(pair_tokens, pair_tokens, pair_tokens, need_weights=False)
        drug_att, target_att = attended[:, 0], attended[:, 1]
        features = torch.cat([drug_att, target_att, drug_att * target_att, torch.abs(drug_att - target_att)], dim=-1)
        return self.regressor(self.norm(features)).squeeze(-1)

    def forward_batch(self, data) -> torch.Tensor:
        return self.forward(data.x, data.edge_index, data.batch, data.drug, data.target)


class MultiScaleGraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(layers))
        self.projections = nn.ModuleList(nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> list[torch.Tensor]:
        h = torch.relu(self.input_proj(x.float()))
        scales = []
        for conv, norm, proj in zip(self.convs, self.norms, self.projections):
            residual = h
            h = torch.relu(norm(conv(h, edge_index)))
            h = self.dropout((h + residual) * 0.5)
            pooled = torch.cat([global_max_pool(h, batch), global_mean_pool(h, batch)], dim=-1)
            scales.append(torch.relu(proj(pooled)))
        return scales


class GlobalLocalSequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.local_convs = nn.ModuleList(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2) for kernel_size in (3, 5, 7))
        self.local_proj = nn.Linear(hidden_dim * len(self.local_convs), hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        mask = tokens.eq(0)
        emb = self.embedding(tokens)
        global_seq = self.global_encoder(emb, src_key_padding_mask=mask)
        global_vec = masked_mean(global_seq, mask)

        local_input = emb.transpose(1, 2)
        local_pooled = []
        for conv in self.local_convs:
            z = torch.relu(conv(local_input)).transpose(1, 2)
            local_pooled.append(masked_max(z, mask))
        local_vec = torch.relu(self.local_proj(torch.cat(local_pooled, dim=-1)))
        return [self.dropout(global_vec), self.dropout(local_vec)]


class ScaleAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, scales: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(scales.size(0), -1, -1)
        pooled, _ = self.attn(query, scales, scales, need_weights=False)
        return self.norm(pooled.squeeze(1))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (~mask).unsqueeze(-1).to(x.dtype)
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = x.masked_fill(mask.unsqueeze(-1), torch.finfo(x.dtype).min).max(dim=1).values
    return torch.where(torch.isfinite(values), values, torch.zeros_like(values))
