from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class ViDTA(nn.Module):
    """ViDTA-style DTA model with virtual graph nodes and attention fusion.

    Reference paper:
    ViDTA: Enhanced Drug-Target Affinity Prediction via Virtual Graph Nodes and
    Attention-based Feature Fusion, arXiv:2412.19589.

    No official GitHub implementation was found during the 2026-07-13 search.
    This class implements the paper-level architecture in PyTorch/PyG for the
    current Davis/KIBA/Metz split: a molecular GNN with a per-graph virtual
    memory node, a CNN protein encoder, and attention-based feature fusion.
    """

    def __init__(
        self,
        num_node_features: int = 78,
        target_vocab_size: int = 26,
        hidden_dim: int = 128,
        graph_layers: int = 4,
        protein_embedding_dim: int = 128,
        protein_filters: int = 128,
        protein_kernel_sizes: tuple[int, ...] = (3, 5, 7),
        attention_heads: int = 4,
        fc_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(graph_layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(graph_layers))
        self.virtual_updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(graph_layers)
        )
        self.virtual_init = nn.Parameter(torch.zeros(1, hidden_dim))
        self.protein = ProteinCNN(target_vocab_size, protein_embedding_dim, protein_filters, protein_kernel_sizes, dropout)
        self.drug_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.protein_proj = nn.Linear(protein_filters * len(protein_kernel_sizes), hidden_dim)
        self.fusion_attention = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_dim * 4)
        dims = [hidden_dim * 4, *fc_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.regressor = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.node_proj(x.float()))
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        virtual = self.virtual_init.expand(batch_size, -1)

        for conv, norm, update in zip(self.convs, self.norms, self.virtual_updates):
            h = h + virtual[batch]
            residual = h
            h = torch.relu(norm(conv(h, edge_index)))
            h = self.dropout((h + residual) * 0.5)
            pooled = global_mean_pool(h, batch)
            virtual = virtual + update(torch.cat([virtual, pooled], dim=-1))

        drug_vec = torch.cat([global_max_pool(h, batch), global_mean_pool(h, batch), virtual], dim=-1)
        drug_vec = self.drug_proj(drug_vec)
        protein_vec = self.protein_proj(self.protein(target.long()))
        tokens = torch.stack([drug_vec, protein_vec], dim=1)
        attended, _ = self.fusion_attention(tokens, tokens, tokens, need_weights=False)
        drug_att, protein_att = attended[:, 0], attended[:, 1]
        features = torch.cat([drug_att, protein_att, drug_att * protein_att, torch.abs(drug_att - protein_att)], dim=-1)
        return self.regressor(self.fusion_norm(features)).squeeze(-1)

    def forward_batch(self, data) -> torch.Tensor:
        target = data.target if hasattr(data, "target") else data.protein
        return self.forward(data.x, data.edge_index, data.batch, target)


class ProteinCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        filters: int,
        kernel_sizes: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(embedding_dim, filters, kernel) for kernel in kernel_sizes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        x = self.embedding(target).transpose(1, 2)
        pooled = [torch.max(torch.relu(conv(x)), dim=-1).values for conv in self.convs]
        return self.dropout(torch.cat(pooled, dim=-1))
