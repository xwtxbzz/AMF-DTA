from __future__ import annotations

import torch
from torch import nn


class FingerDTA(nn.Module):
    """Fingerprint-embedding DTA model.

    The FingerDTA paper combines global fingerprint information with local CNN
    sequence patterns. This implementation uses Morgan fingerprints for drugs,
    hashed k-mer fingerprints for protein sequences, and a CNN branch over the
    protein sequence, which are all available for Davis, KIBA, and Metz.
    """

    def __init__(
        self,
        fingerprint_dim: int = 2048,
        target_vocab_size: int = 26,
        target_fingerprint_dim: int = 1024,
        hidden_dim: int = 256,
        protein_embedding_dim: int = 128,
        protein_filters: int = 128,
        protein_kernel_sizes: tuple[int, ...] = (3, 5, 7),
        fc_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.drug_fp = FingerprintEncoder(fingerprint_dim, hidden_dim, dropout)
        self.target_fp = FingerprintEncoder(target_fingerprint_dim, hidden_dim, dropout)
        self.protein_cnn = ProteinCNN(target_vocab_size, protein_embedding_dim, protein_filters, protein_kernel_sizes, dropout)
        input_dim = hidden_dim * 2 + protein_filters * len(protein_kernel_sizes)
        dims = [input_dim, *fc_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, drug_fingerprint: torch.Tensor, target: torch.Tensor, target_fingerprint: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [
                self.drug_fp(drug_fingerprint.float()),
                self.target_fp(target_fingerprint.float()),
                self.protein_cnn(target.long()),
            ],
            dim=-1,
        )
        return self.regressor(features).squeeze(-1)


class FingerprintEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ProteinCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, filters: int, kernel_sizes: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(embedding_dim, filters, kernel_size) for kernel_size in kernel_sizes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        x = self.embedding(target).transpose(1, 2)
        features = []
        for conv in self.convs:
            z = torch.relu(conv(x))
            features.append(torch.max(z, dim=-1).values)
        return self.dropout(torch.cat(features, dim=-1))
