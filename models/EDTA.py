from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SmileAttention(nn.Module):
    """Official EDTA-style lightweight multi-head SMILES attention."""

    def __init__(self, input_dim: int = 256, heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, heads)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.query(x)
        if mask is not None:
            valid = mask.clone()
            empty_rows = ~valid.any(dim=1)
            if empty_rows.any():
                valid[empty_rows, 0] = True
            scores = scores.masked_fill(~valid.unsqueeze(-1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        context = torch.einsum("blh,bld->bhd", weights, x)
        return context.mean(dim=1), weights


def _make_conv_stack(
    kernels: tuple[int, int, int, int],
    dropout: float,
) -> nn.Sequential:
    channels = (128, 32, 64, 128, 256)
    return nn.Sequential(
        nn.Conv1d(channels[0], channels[1], kernels[0]),
        nn.PReLU(),
        nn.Dropout(dropout),
        nn.Conv1d(channels[1], channels[2], kernels[1]),
        nn.PReLU(),
        nn.Dropout(dropout),
        nn.Conv1d(channels[2], channels[3], kernels[2]),
        nn.PReLU(),
        nn.Conv1d(channels[3], channels[4], kernels[3]),
        nn.PReLU(),
    )


class EDTA(nn.Module):
    """EDTA architecture aligned to the released official checkpoints.

    Official repository: https://github.com/macjiffriya/EDTA

    The released repository contains model weights but not the training source.
    This implementation therefore preserves the exact checkpoint-facing module
    names and tensor shapes, and follows the paper's five input streams:

    - SMILES token ids, encoded by a two-layer BiLSTM with attention.
    - Drug atomic descriptors with 24 features per atom.
    - Target residue descriptors with 42 features per residue.
    - Target BPE token ids.
    - Potential residue descriptors with 67 features per residue.

    For the generic CSV loaders in this project, ``forward(drug, target)`` also
    works by using the available SMILES/protein token ids and zero-filled
    descriptor streams. To reproduce the official EDTA inputs, call ``forward``
    with the keyword tensors shown in the method signature.
    """

    def __init__(
        self,
        drug_vocab_size: int | None = None,
        target_vocab_size: int | None = None,
        embedding_dim: int = 128,
        smiles_vocab_size: int = 65,
        bpe_vocab_size: int = 16694,
        atom_feature_dim: int = 24,
        target_feature_dim: int = 42,
        pocket_feature_dim: int = 67,
        smiles_lstm_hidden_dim: int = 128,
        smiles_lstm_layers: int = 2,
        smiles_attention_heads: int = 8,
        max_drug_atoms: int = 100,
        target_feature_len: int = 512,
        pocket_max_len: int = 200,
        conv_ll_kernel_sizes: tuple[int, int, int, int] = (4, 6, 6, 4),
        conv_pp_kernel_sizes: tuple[int, int, int, int] = (4, 8, 12, 8),
        classifier_dims: tuple[int, int, int, int] = (512, 256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if embedding_dim != 128:
            raise ValueError("Official EDTA checkpoints use embedding_dim=128")
        if smiles_lstm_hidden_dim != 128:
            raise ValueError("Official EDTA checkpoints use smiles_lstm_hidden_dim=128")
        if smiles_lstm_layers != 2:
            raise ValueError("Official EDTA checkpoints use smiles_lstm_layers=2")
        if smiles_attention_heads != 8:
            raise ValueError("Official EDTA checkpoints use 8 SMILES attention heads")
        if classifier_dims != (512, 256, 128, 64):
            raise ValueError("Official EDTA checkpoints use classifier dims (512, 256, 128, 64)")

        self.local_drug_vocab_size = drug_vocab_size
        self.local_target_vocab_size = target_vocab_size
        self.embedding_dim = embedding_dim
        self.smiles_vocab_size = smiles_vocab_size
        self.bpe_vocab_size = bpe_vocab_size
        self.atom_feature_dim = atom_feature_dim
        self.target_feature_dim = target_feature_dim
        self.pocket_feature_dim = pocket_feature_dim
        self.max_drug_atoms = max_drug_atoms
        self.target_feature_len = target_feature_len
        self.pocket_max_len = pocket_max_len
        self.conv_ll_min_len = sum(k - 1 for k in conv_ll_kernel_sizes) + 1
        self.conv_pp_min_len = sum(k - 1 for k in conv_pp_kernel_sizes) + 1

        self.w_omega = nn.Parameter(torch.empty(1280, 1280))
        self.ll_embed = nn.Embedding(smiles_vocab_size, embedding_dim, padding_idx=0)
        self.ll_embed_atom = nn.Linear(atom_feature_dim, embedding_dim)
        self.pp_embed = nn.Linear(target_feature_dim, embedding_dim)
        self.pocket_embed = nn.Linear(pocket_feature_dim, embedding_dim)
        self.pp_pbe_embed = nn.Embedding(bpe_vocab_size, embedding_dim, padding_idx=0)

        self.smiles_rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=smiles_lstm_hidden_dim,
            num_layers=smiles_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.smiles_out_fc = nn.Linear(smiles_lstm_hidden_dim * 2, embedding_dim)
        self.smi_attentions = SmileAttention(
            input_dim=smiles_lstm_hidden_dim * 2,
            heads=smiles_attention_heads,
        )

        self.conv_pp = _make_conv_stack(conv_pp_kernel_sizes, dropout)
        self.conv_ll = _make_conv_stack(conv_ll_kernel_sizes, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(1280, classifier_dims[0]),
            nn.PReLU(),
            nn.Linear(classifier_dims[0], classifier_dims[1]),
            nn.PReLU(),
            nn.Linear(classifier_dims[1], classifier_dims[2]),
            nn.PReLU(),
            nn.Linear(classifier_dims[2], classifier_dims[3]),
            nn.PReLU(),
            nn.Linear(classifier_dims[3], 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_omega)

    @classmethod
    def from_official_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: str | torch.device = "cpu",
    ) -> EDTA:
        model = cls()
        model.load_official_checkpoint(checkpoint_path, map_location=map_location)
        return model

    def load_official_checkpoint(
        self,
        checkpoint_path: str,
        map_location: str | torch.device = "cpu",
    ) -> EDTA:
        try:
            state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=map_location)
        self.load_state_dict(state_dict, strict=True)
        return self

    def forward(
        self,
        drug: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        *,
        smiles_tokens: torch.Tensor | None = None,
        drug_atom_features: torch.Tensor | None = None,
        target_features: torch.Tensor | None = None,
        target_bpe_tokens: torch.Tensor | None = None,
        pocket_features: torch.Tensor | None = None,
        smiles_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        smiles_tokens = smiles_tokens if smiles_tokens is not None else drug
        target_bpe_tokens = target_bpe_tokens if target_bpe_tokens is not None else target
        ref = self._reference_tensor(
            smiles_tokens,
            drug_atom_features,
            target_features,
            target_bpe_tokens,
            pocket_features,
        )
        if ref is None:
            raise ValueError("EDTA.forward requires at least SMILES tokens or official feature tensors")

        batch_size = int(ref.shape[0])
        device = ref.device
        feature_dtype = self.ll_embed_atom.weight.dtype

        if smiles_tokens is None:
            smiles_tokens = torch.zeros(batch_size, self.max_drug_atoms, dtype=torch.long, device=device)
        if target_bpe_tokens is None:
            target_bpe_tokens = torch.zeros(batch_size, self.target_feature_len, dtype=torch.long, device=device)
        if drug_atom_features is None:
            drug_atom_features = torch.zeros(
                batch_size,
                self.max_drug_atoms,
                self.atom_feature_dim,
                dtype=feature_dtype,
                device=device,
            )
        if target_features is None:
            target_features = torch.zeros(
                batch_size,
                self.target_feature_len,
                self.target_feature_dim,
                dtype=feature_dtype,
                device=device,
            )
        if pocket_features is None:
            pocket_features = torch.zeros(
                batch_size,
                self.pocket_max_len,
                self.pocket_feature_dim,
                dtype=feature_dtype,
                device=device,
            )

        smiles_tokens = self._tokens(smiles_tokens, self.smiles_vocab_size)
        target_bpe_tokens = self._tokens(target_bpe_tokens, self.bpe_vocab_size)
        if smiles_mask is None:
            smiles_mask = smiles_tokens.ne(0)

        smiles_vec = self._encode_smiles(smiles_tokens, smiles_mask)
        atom_vec = self._conv_pool(
            self.conv_ll,
            self.ll_embed_atom(drug_atom_features.to(device=device, dtype=feature_dtype)),
            self.conv_ll_min_len,
        )
        target_vec = self._conv_pool(
            self.conv_pp,
            self.pp_embed(target_features.to(device=device, dtype=feature_dtype)),
            self.conv_pp_min_len,
        )
        target_bpe_vec = self._conv_pool(
            self.conv_ll,
            self.pp_pbe_embed(target_bpe_tokens),
            self.conv_ll_min_len,
        )
        pocket_vec = self._conv_pool(
            self.conv_pp,
            self.pocket_embed(pocket_features.to(device=device, dtype=feature_dtype)),
            self.conv_pp_min_len,
        )

        features = torch.cat([smiles_vec, atom_vec, target_vec, target_bpe_vec, pocket_vec], dim=-1)
        features = self._self_attention(features)
        return self.classifier(features).squeeze(-1)

    @staticmethod
    def _reference_tensor(*tensors: torch.Tensor | None) -> torch.Tensor | None:
        for tensor in tensors:
            if tensor is not None:
                return tensor
        return None

    @staticmethod
    def _tokens(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
        tokens = tokens.long()
        return tokens.clamp_(min=0, max=vocab_size - 1)

    @staticmethod
    def _pad_sequence_embeddings(x: torch.Tensor, min_len: int) -> torch.Tensor:
        if x.size(1) >= min_len:
            return x
        return F.pad(x, (0, 0, 0, min_len - x.size(1)))

    def _conv_pool(self, conv: nn.Sequential, x: torch.Tensor, min_len: int) -> torch.Tensor:
        x = self._pad_sequence_embeddings(x, min_len)
        x = conv(x.transpose(1, 2))
        return F.adaptive_max_pool1d(x, 1).squeeze(-1)

    def _encode_smiles(self, smiles_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.ll_embed(smiles_tokens)
        rnn_out, _ = self.smiles_rnn(embedded)
        context, _ = self.smi_attentions(rnn_out, mask)
        compact = torch.relu(self.smiles_out_fc(context))
        return torch.cat([context[:, :128], compact], dim=-1)

    def _self_attention(self, features: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(features, self.w_omega) / math.sqrt(features.size(-1))
        weights = torch.softmax(scores, dim=-1)
        return features * weights
