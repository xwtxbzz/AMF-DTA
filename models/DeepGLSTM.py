from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class DeepGLSTM(nn.Module):
    """GCN plus BiLSTM model for DTA regression.

    Paper: DeepGLSTM: Deep Graph Convolutional Network and LSTM based approach
    for predicting drug-target binding affinity.
    """

    def __init__(
        self,
        num_node_features: int = 78,
        target_vocab_size: int = 26,
        hidden_dim: int = 128,
        graph_layers: int = 3,
        protein_embedding_dim: int = 128,
        lstm_layers: int = 1,
        fc_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(num_node_features, hidden_dim)
        self.graph_layers = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(graph_layers))
        self.graph_norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(graph_layers))
        self.protein_embedding = nn.Embedding(target_vocab_size, protein_embedding_dim, padding_idx=0)
        self.protein_lstm = nn.LSTM(
            input_size=protein_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        graph_dim = hidden_dim * 2
        protein_dim = hidden_dim * 4
        self.regressor = MLP(graph_dim + protein_dim, fc_dims, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.node_proj(x.float()))
        for conv, norm in zip(self.graph_layers, self.graph_norms):
            residual = h
            h = torch.relu(norm(conv(h, edge_index)))
            h = self.dropout(h + residual)
        drug_vec = torch.cat([global_max_pool(h, batch), global_mean_pool(h, batch)], dim=-1)

        target_mask = target.eq(0)
        protein_tokens, _ = self.protein_lstm(self.protein_embedding(target.long()))
        protein_mean = masked_mean(protein_tokens, target_mask)
        protein_max = masked_max(protein_tokens, target_mask)
        protein_vec = torch.cat([protein_mean, protein_max], dim=-1)
        return self.regressor(torch.cat([drug_vec, protein_vec], dim=-1)).squeeze(-1)

    def forward_batch(self, data) -> torch.Tensor:
        target = data.target if hasattr(data, "target") else data.protein
        return self.forward(data.x, data.edge_index, data.batch, target)


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


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = x.masked_fill(mask.unsqueeze(-1), torch.finfo(x.dtype).min).max(dim=1).values
    return torch.where(torch.isfinite(values), values, torch.zeros_like(values))
