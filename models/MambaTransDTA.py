from __future__ import annotations

import torch
from torch import nn


class MambaStyleBlock(nn.Module):
    """Lightweight pure-PyTorch selective state-space mixer.

    This follows the official MambaTransDTA design intent while avoiding the
    optional mamba-ssm/CUDA dependency, so it can run in the current setup.
    """

    def __init__(self, d_model: int, d_state: int = 16, kernel_size: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, padding=kernel_size - 1)
        self.delta = nn.Linear(d_model, d_model)
        self.state_in = nn.Linear(d_model, d_state)
        self.state_out = nn.Linear(d_state, d_model)
        self.log_decay = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = torch.nn.functional.silu(self.conv(u.transpose(1, 2)).transpose(1, 2)[:, : x.size(1)])
        step = torch.sigmoid(self.delta(u))
        state = u.new_zeros(u.size(0), self.state_in.out_features)
        outputs = []
        decay = torch.sigmoid(self.log_decay).mean().clamp(1e-4, 0.9999)
        for t in range(u.size(1)):
            candidate = torch.tanh(self.state_in(u[:, t]))
            state = decay * state + (1.0 - decay) * step[:, t].mean(dim=-1, keepdim=True) * candidate
            outputs.append(self.state_out(state))
        y = torch.stack(outputs, dim=1)
        y = y * torch.sigmoid(gate)
        if padding_mask is not None:
            y = y.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return residual + self.dropout(self.out_proj(self.norm(y)))


class HybridTransformerMambaBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mamba = MambaStyleBlock(embed_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attended, _ = self.attention(x, x, x, key_padding_mask=padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attended))
        return self.norm2(self.mamba(x, padding_mask))


class SMILESHybridEncoder(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, embed_dim: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position = nn.Embedding(max_len, embed_dim)
        self.blocks = nn.ModuleList(HybridTransformerMambaBlock(embed_dim, heads, dropout) for _ in range(layers))
        self.output = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens[:, : self.max_len].long()
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        padding_mask = tokens.eq(0)
        x = self.embedding(tokens) + self.position(positions)
        for block in self.blocks:
            x = block(x, padding_mask)
        # The official implementation uses a vocabulary projection followed
        # by max reduction over vocabulary, yielding one value per SMILES site.
        return self.output(x).amax(dim=-1)


class ProteinCNNEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, filters: tuple[int, ...], kernels: tuple[int, ...], dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(embedding_dim, out_channels, kernel) for out_channels, kernel in zip(filters, kernels))
        self.dropout = nn.Dropout(dropout)
        self.output_dim = sum(filters)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens.long()).transpose(1, 2)
        pooled = [torch.relu(conv(x)).amax(dim=-1) for conv in self.convs]
        return self.dropout(torch.cat(pooled, dim=-1))


class MambaTransDTA(nn.Module):
    """MambaTransDTA adapted to the project's sequence batch interface.

    Reference: Lou et al., JCIM 2026, DOI 10.1021/acs.jcim.5c02361;
    official repository: github.com/pdssunny/MambaTransDTA.
    """

    def __init__(
        self,
        drug_vocab_size: int,
        target_vocab_size: int,
        max_drug_len: int = 120,
        embed_dim: int = 64,
        num_layers: int = 1,
        heads: int = 8,
        protein_embedding_dim: int = 128,
        protein_filters: tuple[int, ...] = (32, 64, 96),
        protein_kernels: tuple[int, ...] = (4, 8, 12),
        hidden_dims: tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.drug_encoder = SMILESHybridEncoder(drug_vocab_size, max_drug_len, embed_dim, num_layers, heads, dropout)
        self.protein_encoder = ProteinCNNEncoder(target_vocab_size, protein_embedding_dim, protein_filters, protein_kernels, dropout)
        input_dim = max_drug_len + self.protein_encoder.output_dim
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.LeakyReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, drug: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        drug_features = self.drug_encoder(drug)
        if drug_features.size(1) < self.drug_encoder.max_len:
            drug_features = torch.nn.functional.pad(drug_features, (0, self.drug_encoder.max_len - drug_features.size(1)))
        protein_features = self.protein_encoder(protein)
        features = torch.cat([drug_features, protein_features], dim=-1)
        return self.regressor(features).squeeze(-1)

    def forward_batch(self, data):
        return self.forward(data.drug, data.protein)
