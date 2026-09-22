from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class LaProDTA(nn.Module):
    """LaPro-DTA-style model for current DTA regression benchmarks.

    Reference:
    LaPro-DTA: Latent Dual-View Drug Representations and Salient Protein Feature
    Extraction for Generalizable Drug--Target Affinity Prediction, arXiv:2603.14792.

    No official GitHub implementation was found in the 2026-07-13 web search.
    This PyTorch/PyG reproduction implements the paper-level components that
    fit the local SMILES/protein setup: latent dual-view drug representation,
    salient protein top-k pooling, and cross-view multi-head attention fusion.
    """

    def __init__(
        self,
        num_node_features: int = 78,
        drug_vocab_size: int = 64,
        target_vocab_size: int = 26,
        hidden_dim: int = 128,
        graph_layers: int = 3,
        attention_heads: int = 4,
        topk_ratio: float = 0.1,
        dropout: float = 0.2,
        fc_dims: tuple[int, ...] = (512, 256),
    ) -> None:
        super().__init__()
        self.instance_drug = GraphInstanceEncoder(num_node_features, hidden_dim, graph_layers, dropout)
        self.distribution_drug = SmilesDistributionEncoder(drug_vocab_size, hidden_dim, attention_heads, dropout)
        self.protein = SalientProteinEncoder(target_vocab_size, hidden_dim, attention_heads, topk_ratio, dropout)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim * 4)
        dims = [hidden_dim * 4, *fc_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, drug: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        instance_view = self.instance_drug(x, edge_index, batch)
        distribution_view = self.distribution_drug(drug.long())
        protein_view = self.protein(target.long())
        drug_tokens = torch.stack([instance_view, distribution_view], dim=1)
        drug_view, _ = self.cross_attention(drug_tokens, drug_tokens, drug_tokens, need_weights=False)
        drug_vec = drug_view.mean(dim=1)

        pair_tokens = torch.stack([drug_vec, protein_view], dim=1)
        pair_view, _ = self.cross_attention(pair_tokens, pair_tokens, pair_tokens, need_weights=False)
        drug_att, protein_att = pair_view[:, 0], pair_view[:, 1]
        features = torch.cat([drug_att, protein_att, drug_att * protein_att, torch.abs(drug_att - protein_att)], dim=-1)
        return self.regressor(self.norm(features)).squeeze(-1)

    def forward_batch(self, data) -> torch.Tensor:
        return self.forward(data.x, data.edge_index, data.batch, data.drug, data.target)


class GraphInstanceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(layers))
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_proj(x.float()))
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            h = torch.relu(norm(conv(h, edge_index)))
            h = self.dropout((h + residual) * 0.5)
        pooled = torch.cat([global_max_pool(h, batch), global_mean_pool(h, batch)], dim=-1)
        return torch.relu(self.proj(pooled))


class SmilesDistributionEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.semantic_remap = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.eq(0)
        x = self.semantic_remap(self.embedding(tokens))
        attended, _ = self.attention(x, x, x, key_padding_mask=mask, need_weights=False)
        return self.norm(masked_mean(attended, mask))


class SalientProteinEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, heads: int, topk_ratio: float, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pattern_convs = nn.ModuleList(nn.Conv1d(hidden_dim, hidden_dim, kernel, padding=kernel // 2) for kernel in (3, 5, 9))
        self.pattern_score = nn.Linear(hidden_dim, 1)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.topk_ratio = topk_ratio
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.eq(0)
        emb = self.embedding(tokens)
        conv_input = emb.transpose(1, 2)
        patterns = torch.stack([torch.relu(conv(conv_input)).transpose(1, 2) for conv in self.pattern_convs], dim=0).mean(dim=0)
        scores = self.pattern_score(patterns).squeeze(-1).masked_fill(mask, torch.finfo(patterns.dtype).min)
        seq_len = tokens.size(1)
        k = max(1, min(seq_len, int(seq_len * self.topk_ratio)))
        _, indices = torch.topk(scores, k=k, dim=1)
        gathered = patterns.gather(1, indices.unsqueeze(-1).expand(-1, -1, patterns.size(-1)))
        attended, _ = self.attention(gathered, gathered, gathered, need_weights=False)
        return self.norm(self.dropout(attended.mean(dim=1)))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (~mask).unsqueeze(-1).to(x.dtype)
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
