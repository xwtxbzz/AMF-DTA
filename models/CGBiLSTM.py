from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, global_max_pool, global_mean_pool


class CGBiLSTM(nn.Module):
    """Combination graph neural network and BiLSTM DTA model.

    Paper: Drug-Target Binding Affinity Prediction by Combination Graph Neural
    Network and BiLSTM.

    The drug branch combines GCN and GAT graph representations derived from
    SMILES, while the protein branch uses a BiLSTM over amino-acid tokens.
    """

    def __init__(
        self,
        num_node_features: int = 78,
        target_vocab_size: int = 26,
        hidden_dim: int = 128,
        graph_layers: int = 2,
        gat_heads: int = 4,
        protein_embedding_dim: int = 128,
        lstm_layers: int = 1,
        fc_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(num_node_features, hidden_dim)
        self.gcn_layers = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(graph_layers))
        self.gat_layers = nn.ModuleList(
            GATConv(hidden_dim, hidden_dim, heads=gat_heads, concat=False, dropout=dropout)
            for _ in range(graph_layers)
        )
        self.gcn_norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(graph_layers))
        self.gat_norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(graph_layers))
        self.protein_embedding = nn.Embedding(target_vocab_size, protein_embedding_dim, padding_idx=0)
        self.protein_lstm = nn.LSTM(
            protein_embedding_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        graph_dim = hidden_dim * 4
        protein_dim = hidden_dim * 4
        self.regressor = MLP(graph_dim + protein_dim, fc_dims, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        h0 = torch.relu(self.node_proj(x.float()))
        gcn_h = h0
        gat_h = h0
        for gcn, gat, gcn_norm, gat_norm in zip(self.gcn_layers, self.gat_layers, self.gcn_norms, self.gat_norms):
            gcn_h = self.dropout(torch.relu(gcn_norm(gcn(gcn_h, edge_index))) + gcn_h)
            gat_h = self.dropout(torch.relu(gat_norm(gat(gat_h, edge_index))) + gat_h)
        graph_vec = torch.cat(
            [
                global_max_pool(gcn_h, batch),
                global_mean_pool(gcn_h, batch),
                global_max_pool(gat_h, batch),
                global_mean_pool(gat_h, batch),
            ],
            dim=-1,
        )

        mask = target.eq(0)
        protein_tokens, _ = self.protein_lstm(self.protein_embedding(target.long()))
        protein_vec = torch.cat([masked_mean(protein_tokens, mask), masked_max(protein_tokens, mask)], dim=-1)
        return self.regressor(torch.cat([graph_vec, protein_vec], dim=-1)).squeeze(-1)

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
