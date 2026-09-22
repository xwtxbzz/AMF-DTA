from __future__ import annotations

import torch
from torch import nn


class DeepDTA(nn.Module):
    """PyTorch implementation aligned with the official DeepDTA CNN model.

    Official source used for structure:
    https://github.com/hkmztrk/DeepDTA/blob/master/source/run_experiments.py

    The model expects integer-encoded SMILES and protein sequences:
    - drug input shape: ``[batch_size, max_drug_len]``
    - target input shape: ``[batch_size, max_target_len]``

    Token id 0 is treated as padding. This makes the class directly usable
    with Davis-style SMILES/sequence CSV files after character tokenization.
    """

    def __init__(
        self,
        drug_vocab_size: int,
        target_vocab_size: int,
        drug_max_len: int = 120,
        target_max_len: int = 1000,
        embedding_dim: int = 128,
        drug_kernel_size: int = 4,
        target_kernel_size: int = 8,
        num_filters: tuple[int, int, int] = (32, 64, 96),
        fc_dims: tuple[int, int, int] = (1024, 1024, 512),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_max_len = drug_max_len
        self.target_max_len = target_max_len

        self.drug_embedding = nn.Embedding(drug_vocab_size, embedding_dim, padding_idx=0)
        self.target_embedding = nn.Embedding(target_vocab_size, embedding_dim, padding_idx=0)

        self.drug_conv = self._make_conv_stack(embedding_dim, num_filters, drug_kernel_size)
        self.target_conv = self._make_conv_stack(embedding_dim, num_filters, target_kernel_size)

        self.drug_pool = nn.AdaptiveMaxPool1d(1)
        self.target_pool = nn.AdaptiveMaxPool1d(1)

        fused_dim = num_filters[-1] * 2
        self.fc1 = nn.Linear(fused_dim, fc_dims[0])
        self.fc2 = nn.Linear(fc_dims[0], fc_dims[1])
        self.fc3 = nn.Linear(fc_dims[1], fc_dims[2])
        self.out = nn.Linear(fc_dims[2], 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _make_conv_stack(
        embedding_dim: int,
        num_filters: tuple[int, int, int],
        kernel_size: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = embedding_dim
        for out_channels in num_filters:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size))
            layers.append(nn.ReLU())
            in_channels = out_channels
        return nn.Sequential(*layers)

    def encode_drug(self, drug: torch.Tensor) -> torch.Tensor:
        drug = self.drug_embedding(drug).transpose(1, 2)
        drug = self.drug_conv(drug)
        drug = self.drug_pool(drug).squeeze(-1)
        return drug

    def encode_target(self, target: torch.Tensor) -> torch.Tensor:
        target = self.target_embedding(target).transpose(1, 2)
        target = self.target_conv(target)
        target = self.target_pool(target).squeeze(-1)
        return target

    def forward(self, drug: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        drug_features = self.encode_drug(drug)
        target_features = self.encode_target(target)
        x = torch.cat([drug_features, target_features], dim=1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        return self.out(x).squeeze(-1)
