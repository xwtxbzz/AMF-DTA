from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, global_max_pool


class DrugMGR(nn.Module):
    """Deep multigranular ligand representation model.

    Paper: DrugMGR: a deep bioactive molecule binding method to identify
    compounds targeting proteins.

    The paper combines graph, SMILES-CNN, molecular-transformer ligand
    branches with a gated-CNN VAE-style protein encoder. This supervised local
    reproduction keeps those branches but builds all inputs from the current
    SMILES and amino-acid sequence fields, avoiding external substructure
    corpora or binding-site labels.
    """

    def __init__(
        self,
        num_node_features: int = 78,
        drug_vocab_size: int = 64,
        target_vocab_size: int = 26,
        hidden_dim: int = 128,
        gat_heads: int = 4,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        protein_kernel_size: int = 7,
        fc_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.graph_branch = GraphLigandBranch(num_node_features, hidden_dim, gat_heads, dropout)
        self.smiles_branch = SmilesCNNBranch(drug_vocab_size, hidden_dim, dropout)
        self.substructure_branch = SmilesTransformerBranch(
            drug_vocab_size,
            hidden_dim,
            transformer_layers,
            transformer_heads,
            dropout,
        )
        self.protein_branch = ProteinGatedBranch(target_vocab_size, hidden_dim, protein_kernel_size, dropout)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, transformer_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.regressor = MLP(hidden_dim * 5, fc_dims, dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        drug_tokens: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        graph_vec = self.graph_branch(x, edge_index, batch)
        smiles_vec = self.smiles_branch(drug_tokens)
        sub_vec = self.substructure_branch(drug_tokens)
        protein_tokens, protein_vec = self.protein_branch(target)

        ligand_tokens = torch.stack([graph_vec, smiles_vec, sub_vec], dim=1)
        protein_mask = target.eq(0)
        ligand_ctx, _ = self.cross_attention(
            ligand_tokens,
            protein_tokens,
            protein_tokens,
            key_padding_mask=protein_mask,
            need_weights=False,
        )
        ligand_vec = self.norm(ligand_tokens + self.dropout(ligand_ctx)).mean(dim=1)
        features = torch.cat([graph_vec, smiles_vec, sub_vec, protein_vec, ligand_vec], dim=-1)
        return self.regressor(features).squeeze(-1)

    def forward_batch(self, data) -> torch.Tensor:
        target = data.target if hasattr(data, "target") else data.protein
        drug_tokens = data.drug if hasattr(data, "drug") else torch.zeros_like(target[:, :1])
        return self.forward(data.x, data.edge_index, data.batch, drug_tokens.long(), target.long())


class GraphLigandBranch(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.proj(x.float()))
        h = self.dropout(torch.relu(self.norm1(self.gat1(h, edge_index))) + h)
        h = self.dropout(torch.relu(self.norm2(self.gat2(h, edge_index))) + h)
        return global_max_pool(h, batch)


class SmilesCNNBranch(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens).transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        return masked_mean(x, tokens.eq(0))


class SmilesTransformerBranch(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.eq(0)
        x = self.encoder(self.embedding(tokens), src_key_padding_mask=mask)
        return masked_mean(self.norm(x), mask)


class ProteinGatedBranch(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        padding = kernel_size // 2
        self.feature_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.gate_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.mu = nn.Linear(hidden_dim, hidden_dim)
        self.logvar = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = target.eq(0)
        x = self.embedding(target).transpose(1, 2)
        gated = torch.tanh(self.feature_conv(x)) * torch.sigmoid(self.gate_conv(x))
        tokens = self.norm(self.dropout(gated.transpose(1, 2))[:, : target.size(1)])
        pooled = masked_mean(tokens, mask)
        mu = self.mu(pooled)
        logvar = self.logvar(pooled).clamp(-6.0, 6.0)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return tokens, z


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
