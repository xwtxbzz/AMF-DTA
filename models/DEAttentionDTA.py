from __future__ import annotations

import torch
from torch import nn


class DynamicEmbeddingEncoder(nn.Module):
    """1D-CNN dynamic embedding followed by self-attention pooling."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        kernel_size: int = 5,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dynamic_conv = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.pool_score = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = tokens.ne(0)
        x = self.embedding(tokens).transpose(1, 2)
        x = self.dynamic_conv(x).transpose(1, 2)
        x = self.norm(x)
        key_padding_mask = ~mask
        attended, _ = self.self_attention(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm(x + self.dropout(attended))
        scores = self.pool_score(torch.tanh(x)).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return x, pooled


class DEAttentionDTA(nn.Module):
    """DEAttentionDTA dynamic embedding + self-attention architecture.

    Paper: "DEAttentionDTA: protein-ligand binding affinity prediction based
    on dynamic embedding and self-attention" (Bioinformatics, 2024).

    The original model uses three linear streams: SMILES, global protein
    sequence, and active-pocket local sequence. Davis/KIBA/Metz in this
    workspace do not provide pocket annotations, so the third stream is a
    learned local-protein proxy derived from the target sequence. The class
    keeps the three-stream design so a real pocket tensor can be added later
    without changing the fusion head.
    """

    def __init__(
        self,
        drug_vocab_size: int,
        target_vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        drug_kernel_size: int = 5,
        protein_kernel_size: int = 7,
        pocket_kernel_size: int = 3,
        attention_heads: int = 4,
        fc_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_encoder = DynamicEmbeddingEncoder(
            drug_vocab_size,
            embedding_dim,
            hidden_dim,
            drug_kernel_size,
            attention_heads,
            dropout,
        )
        self.protein_encoder = DynamicEmbeddingEncoder(
            target_vocab_size,
            embedding_dim,
            hidden_dim,
            protein_kernel_size,
            attention_heads,
            dropout,
        )
        self.pocket_proxy_encoder = DynamicEmbeddingEncoder(
            target_vocab_size,
            embedding_dim,
            hidden_dim,
            pocket_kernel_size,
            attention_heads,
            dropout,
        )
        self.cross_attention = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)

        fusion_dim = hidden_dim * 6
        layers: list[nn.Module] = []
        in_dim = fusion_dim
        for hidden in fc_dims:
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, drug: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _drug_tokens, drug_vec = self.drug_encoder(drug)
        protein_tokens, protein_vec = self.protein_encoder(target)
        _pocket_tokens, pocket_vec = self.pocket_proxy_encoder(target)

        query = drug_vec.unsqueeze(1)
        cross, _ = self.cross_attention(query, protein_tokens, protein_tokens, key_padding_mask=~target.ne(0), need_weights=False)
        drug_context = self.cross_norm(query + cross).squeeze(1)

        fusion = torch.cat(
            [
                drug_context,
                protein_vec,
                pocket_vec,
                torch.abs(drug_context - protein_vec),
                drug_context * protein_vec,
                protein_vec * pocket_vec,
            ],
            dim=-1,
        )
        return self.regressor(fusion).squeeze(-1)
