from __future__ import annotations

import torch
from torch import nn


class WideDTA(nn.Module):
    """WideDTA-style word-level CNN model.

    The paper represents SMILES and protein sequences as overlapping words
    rather than single characters. The original model also studies protein
    domains/motifs and ligand MCS words; this implementation keeps the two
    universally available sources, SMILES words and protein sequence words, so
    it can run on Davis, KIBA, and Metz without external annotation files.
    """

    def __init__(
        self,
        drug_vocab_size: int,
        target_vocab_size: int,
        embedding_dim: int = 128,
        filters: int = 32,
        kernel_size: int = 4,
        fc_dims: tuple[int, ...] = (1024, 1024, 512),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_encoder = WordCNN(drug_vocab_size, embedding_dim, filters, kernel_size, dropout)
        self.target_encoder = WordCNN(target_vocab_size, embedding_dim, filters, kernel_size, dropout)
        dims = [filters * 2, *fc_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, drug_words: torch.Tensor, target_words: torch.Tensor) -> torch.Tensor:
        drug_vec = self.drug_encoder(drug_words)
        target_vec = self.target_encoder(target_words)
        return self.regressor(torch.cat([drug_vec, target_vec], dim=-1)).squeeze(-1)


class WordCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, filters: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(embedding_dim, filters, kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x).transpose(1, 2)
        z = torch.relu(self.conv(z))
        z = torch.max(z, dim=-1).values
        return self.dropout(z)
