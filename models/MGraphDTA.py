from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class MGraphDTA(nn.Module):
    """Deep multiscale graph model inspired by MGraphDTA.

    The paper proposes a deep multiscale GNN for molecular graphs and a protein
    sequence branch. This implementation keeps the multiscale graph idea with a
    configurable stack of graph convolution layers and combines early, middle,
    and final graph representations with a CNN protein encoder.
    """

    def __init__(
        self,
        num_node_features: int = 78,
        target_vocab_size: int = 26,
        hidden_dim: int = 128,
        graph_layers: int = 27,
        protein_embedding_dim: int = 128,
        protein_filters: int = 128,
        protein_kernel_size: int = 8,
        fc_dims: tuple[int, ...] = (1024, 512),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(graph_layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(graph_layers))
        self.scale_points = sorted({max(0, graph_layers // 3 - 1), max(0, (2 * graph_layers) // 3 - 1), graph_layers - 1})
        self.protein = ProteinBranch(target_vocab_size, protein_embedding_dim, protein_filters, protein_kernel_size)
        graph_dim = len(self.scale_points) * hidden_dim * 2
        dims = [graph_dim + protein_filters, *fc_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.regressor = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_proj(x.float()))
        multiscale = []
        for idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = h
            h = conv(h, edge_index)
            h = torch.relu(norm(h))
            h = self.dropout((h + residual) * 0.5)
            if idx in self.scale_points:
                multiscale.extend([global_max_pool(h, batch), global_mean_pool(h, batch)])
        graph_vec = torch.cat(multiscale, dim=-1)
        protein_vec = self.protein(target.long())
        return self.regressor(torch.cat([graph_vec, protein_vec], dim=-1)).squeeze(-1)

    def forward_batch(self, data) -> torch.Tensor:
        target = data.target if hasattr(data, "target") else data.protein
        return self.forward(data.x, data.edge_index, data.batch, target)


class ProteinBranch(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, filters: int, kernel_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(embedding_dim, filters, kernel_size)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        x = self.embedding(target).transpose(1, 2)
        x = torch.relu(self.conv(x))
        return torch.max(x, dim=-1).values
