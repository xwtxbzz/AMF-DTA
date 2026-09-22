from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(torch.tanh(x)).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class CKGraphEncoder(nn.Module):
    """Atom-level graph encoder for CKDTA's molecular modeling branch."""

    def __init__(
        self,
        node_in_feats: int = 78,
        hidden_dim: int = 128,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(node_in_feats, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x.float())
        for conv, norm in zip(self.layers, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = norm(torch.relu(x) + residual)
            x = self.dropout(x)
        return x


class CKProteinEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bigru = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = target.ne(0)
        x = self.embedding(target)
        x = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        x, _ = self.bigru(x)
        x = self.dropout(self.norm(x))
        return x, mask


class CoAttention(nn.Module):
    """Drug graph/protein sequence co-attention inspired by CKDTA."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.seq_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim**-0.5

    def forward(
        self,
        graph_nodes: torch.Tensor,
        graph_mask: torch.Tensor,
        seq_tokens: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.graph_proj(graph_nodes)
        s = self.seq_proj(seq_tokens)
        scores = torch.bmm(g, s.transpose(1, 2)) * self.scale
        scores = scores.masked_fill(~seq_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        graph_to_seq = torch.bmm(torch.softmax(scores, dim=-1), seq_tokens)

        reverse_scores = scores.transpose(1, 2)
        reverse_scores = reverse_scores.masked_fill(~graph_mask.unsqueeze(1), torch.finfo(reverse_scores.dtype).min)
        seq_to_graph = torch.bmm(torch.softmax(reverse_scores, dim=-1), graph_nodes)

        graph_summary = masked_mean(graph_to_seq, graph_mask)
        seq_summary = masked_mean(seq_to_graph, seq_mask)
        return graph_summary, seq_summary


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
    total = (x * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return total / denom


class CKDTA(nn.Module):
    """Runnable PyG reproduction of CKDTA's graph/sequence co-attention core.

    Paper: "CKDTA: A chemical knowledge-enhanced framework for drug-target
    affinity prediction" (Journal of Computational Science, 2025).

    The paper enriches molecules with chemical-element and functional-region
    prior layers. The current Davis/KIBA/Metz CSV pipeline only provides SMILES
    and protein sequences, so this class implements the atom graph, sequence
    encoder, hierarchical attention flavor, and graph-sequence co-attention
    while leaving functional-region priors as a data-pipeline extension.
    """

    def __init__(
        self,
        num_node_features: int = 78,
        target_vocab_size: int = 25,
        hidden_dim: int = 128,
        graph_layers: int = 3,
        graph_heads: int = 4,
        protein_embedding_dim: int = 128,
        protein_kernel_size: int = 5,
        fc_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.graph_encoder = CKGraphEncoder(num_node_features, hidden_dim, graph_layers, graph_heads, dropout)
        self.protein_encoder = CKProteinEncoder(
            target_vocab_size,
            embedding_dim=protein_embedding_dim,
            hidden_dim=hidden_dim,
            kernel_size=protein_kernel_size,
            dropout=dropout,
        )
        self.protein_pool = MaskedAttentionPool(hidden_dim)
        self.co_attention = CoAttention(hidden_dim)

        fusion_dim = hidden_dim * 8
        layers: list[nn.Module] = []
        in_dim = fusion_dim
        for hidden in fc_dims:
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        node_features = self.graph_encoder(x, edge_index)
        graph_dense, graph_mask = to_dense_batch(node_features, batch)
        graph_mean = global_mean_pool(node_features, batch)
        graph_max = global_max_pool(node_features, batch)
        graph_features = 0.5 * (graph_mean + graph_max)

        seq_tokens, seq_mask = self.protein_encoder(target.long())
        seq_features = self.protein_pool(seq_tokens, seq_mask)

        graph_co, seq_co = self.co_attention(graph_dense, graph_mask, seq_tokens, seq_mask)
        interaction = torch.cat(
            [
                graph_features,
                seq_features,
                graph_co,
                seq_co,
                torch.abs(graph_features - seq_features),
                graph_features * seq_features,
                torch.abs(graph_co - seq_co),
                graph_co * seq_co,
            ],
            dim=-1,
        )
        return self.regressor(interaction).squeeze(-1)

    def forward_batch(self, data) -> torch.Tensor:
        target = data.target if hasattr(data, "target") else data.protein
        return self.forward(data.x, data.edge_index, data.batch, target)
