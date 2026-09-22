from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch


class MATTDTI(nn.Module):
    """Multiple-attention-block DTA model.

    Paper: Deep drug-target binding affinity prediction with multiple attention
    blocks (MATT-DTI). The local implementation uses a graph atom encoder,
    relation-aware local self-attention over dense atom sequences, and
    multi-head drug-target cross attention.
    """

    def __init__(
        self,
        num_node_features: int = 78,
        target_vocab_size: int = 26,
        hidden_dim: int = 128,
        graph_layers: int = 2,
        protein_embedding_dim: int = 128,
        protein_kernel_size: int = 7,
        attention_heads: int = 4,
        relative_kernel_size: int = 5,
        fc_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(num_node_features, hidden_dim)
        self.graph_layers = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(graph_layers))
        self.graph_norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(graph_layers))
        self.atom_attention = RelationAwareSelfAttention(hidden_dim, attention_heads, relative_kernel_size, dropout)
        self.protein = ProteinEncoder(target_vocab_size, protein_embedding_dim, hidden_dim, protein_kernel_size, dropout)
        self.drug_to_target = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.target_to_drug = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.norm_drug = nn.LayerNorm(hidden_dim)
        self.norm_target = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.regressor = MLP(hidden_dim * 4, fc_dims, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.node_proj(x.float()))
        for conv, norm in zip(self.graph_layers, self.graph_norms):
            residual = h
            h = torch.relu(norm(conv(h, edge_index)))
            h = self.dropout(h + residual)
        drug_tokens, drug_mask = to_dense_batch(h, batch)
        drug_tokens = self.atom_attention(drug_tokens, drug_mask)
        target_tokens = self.protein(target.long())
        target_mask = target.eq(0)

        drug_ctx, _ = self.drug_to_target(
            drug_tokens,
            target_tokens,
            target_tokens,
            key_padding_mask=target_mask,
            need_weights=False,
        )
        target_ctx, _ = self.target_to_drug(
            target_tokens,
            drug_tokens,
            drug_tokens,
            key_padding_mask=~drug_mask,
            need_weights=False,
        )
        drug_vec = masked_mean(self.norm_drug(drug_tokens + drug_ctx), ~drug_mask)
        target_vec = masked_mean(self.norm_target(target_tokens + target_ctx), target_mask)
        features = torch.cat([drug_vec, target_vec, drug_vec * target_vec, torch.abs(drug_vec - target_vec)], dim=-1)
        return self.regressor(features).squeeze(-1)

    def forward_batch(self, data) -> torch.Tensor:
        target = data.target if hasattr(data, "target") else data.protein
        return self.forward(data.x, data.edge_index, data.batch, target)


class RelationAwareSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, relative_kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        padding = relative_kernel_size // 2
        self.local_relation = nn.Conv1d(hidden_dim, hidden_dim, relative_kernel_size, padding=padding, groups=hidden_dim)
        self.mix = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        attn, _ = self.attention(x, x, x, key_padding_mask=~valid_mask, need_weights=False)
        local = self.local_relation(x.transpose(1, 2)).transpose(1, 2)
        local = local[:, : x.size(1)]
        fused = self.mix(torch.cat([attn, local], dim=-1))
        return self.norm(x + self.dropout(fused))


class ProteinEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        x = self.embedding(target).transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        return self.norm(x[:, : target.size(1)])


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (~mask).unsqueeze(-1).to(x.dtype)
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
