"""AMF-DTA model — paper Sec. II.

Dual-gate multi-view fusion with a Rational-ANOVA interaction layer and a
multi-branch score mixer:

  * three drug fingerprint views (Morgan / feature-Morgan / MACCS) fused by a
    sample-level drug-view reliability gate (Eq. 1-2);
  * two protein contexts (k-mer hash fingerprint, sequence embedding) plus the
    fused drug representation combined by a cross-modal context gate
    (Eq. 3-4);
  * a Rational-ANOVA interaction layer (Eq. 6-7) followed by a multi-branch
    score mixer (Eq. 9):  y = lam1*MLP(f_rail) + lam2*Linear(v_drug)
    + lam3*MLP(f_bias), where the bias branch consumes training-set-only
    statistics (Sec. II-E).

The forward signature matches the batch dict produced by
`framework.data.AMFDTADataset`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .modules import MLPInteractionLayer, PolyInteractionLayer, RailInteractionLayer, RationalUnit1D


class FingerprintEncoder(nn.Module):
    """MLP encoder for a single fingerprint view: Linear-BN-ReLU-Dropout-Linear-ReLU."""

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
        return self.network(x.float())


class ProteinSequenceEncoder(nn.Module):
    """Protein sequence encoder (h_seq in the paper): character embedding +
    masked mean pooling + projection. The paper calls this view the
    "sequence encoder"; every reported result was trained with this module.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        mask = target.ne(0).unsqueeze(-1)
        embedded = self.embedding(target.long())
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.projection(pooled)


def _build_mlp(dims: list[int], dropout: float, last_dropout: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:-1]):
        layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)])
    if len(dims) >= 2:
        layers.extend([nn.Linear(dims[-2], dims[-1])])
    return nn.Sequential(*layers)


class AMFDTA(nn.Module):
    """AMF-DTA: reliability-gated multi-view DTA model with rational-ANOVA
    interaction (paper Sec. II).

    `interaction` selects the interaction block:
      - "rail"               full Rational-ANOVA layer (default; F-A8)
      - "rail_fixed_interact" rational main effects over target/sequence,
                              fixed |diff| and product interactions (F-A7)
      - "fixed"              hand-designed interactions (Eq. 5) + dual gates
                              (A3 baseline)
      - "no_abs" / "no_product"  A3 without the absolute-difference / product
                              terms (A4 / A5)
      - "mlp_interact"       learned per-pair MLP interaction, no rational
                              structure (review F-M1; parameter-matched)
      - "poly_interact"      unconstrained degree-2 polynomial interaction,
                              no denominator (review F-M2; parameter-matched)

    `gate_mode` selects what the two gates are allowed to condition on:
      - "sample"   (default) the gate MLPs read the per-sample concatenated
                   views, so the weights vary from pair to pair (Eq. 1, Eq. 3);
      - "dataset"  each gate MLP is kept but reads a single learned constant
                   instead of the sample, so it emits one weight vector shared
                   by every pair -- the *retrained* counterpart of the post-hoc
                   `both_mean` intervention in `analysis_gate_intervention_fa3.py`,
                   which freezes a trained model's gates to their validation-set
                   mean. Parameter count is unchanged, so the arm isolates what
                   the gate is allowed to condition on (variant F-D1).

    `gate_mode` is consulted only when the corresponding `use_*_gate` is True;
    a disabled gate keeps its constant 1/K behaviour and ignores the mode.
    """

    def __init__(
        self,
        target_vocab_size: int,
        fingerprint_dim: int = 2048,
        target_fingerprint_dim: int = 1024,
        maccs_dim: int = 167,
        bias_feature_dim: int = 6,
        hidden_dim: int = 256,
        fc_dims: tuple[int, ...] = (768, 256),
        dropout: float = 0.2,
        interaction: str = "rail",
        use_drug_gate: bool = True,
        use_context_gate: bool = True,
        use_bias: bool = True,
        positive_denominator: bool = True,
        residual_gate: bool = True,
        gate_mode: str = "sample",
        protein_pretrained_dim: int = 0,
        drug_pretrained_dim: int = 0,
    ) -> None:
        super().__init__()
        if gate_mode not in {"sample", "dataset"}:
            raise ValueError(f"gate_mode must be 'sample' or 'dataset', got {gate_mode!r}")
        self.interaction = interaction
        self.use_drug_gate = use_drug_gate
        self.use_context_gate = use_context_gate
        self.use_bias = use_bias
        self.gate_mode = gate_mode

        # --- drug views -----------------------------------------------------
        self.drug_morgan = FingerprintEncoder(fingerprint_dim, hidden_dim, dropout)
        self.drug_feature_morgan = FingerprintEncoder(fingerprint_dim, hidden_dim, dropout)
        self.drug_maccs = FingerprintEncoder(maccs_dim, hidden_dim, dropout)
        # drug-view reliability gate (Eq. 1): softmax weights over the three
        # fingerprint views. Under gate_mode="dataset" the MLP is kept intact
        # but its input is a single learned constant instead of the sample, so
        # the weights it emits are shared by every pair (F-D1). Keeping the MLP
        # is deliberate: it holds the parameter count fixed and leaves the
        # *only* difference between the two arms the information the gate sees.
        self.drug_view_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.drug_view_gate_const = (
            nn.Parameter(torch.zeros(hidden_dim * 3)) if gate_mode == "dataset" else None
        )

        # --- protein contexts ----------------------------------------------
        self.target_fp = FingerprintEncoder(target_fingerprint_dim, hidden_dim, dropout)
        self.seq_encoder = ProteinSequenceEncoder(
            target_vocab_size, hidden_dim=hidden_dim, dropout=dropout
        )

        # --- optional frozen pretrained views (cold-start study) --------------
        # Opt-in. With both dims 0 nothing below changes shape and the module is
        # parameter-identical to the published model, so every existing number
        # is unaffected. When on, a cached frozen feature is projected and
        # appended to the *population the corresponding gate already weights*,
        # so the gate -- not a separate fusion path -- decides how much the
        # pretrained view contributes, and the existing gate-intervention tools
        # apply to it unchanged. The pretrained and random-init arms build this
        # same graph and differ only in the values of the cached tensor, so any
        # gap between them cannot be attributed to the extra capacity.
        self.n_drug_views = 4 if drug_pretrained_dim > 0 else 3
        self.n_contexts = 4 if protein_pretrained_dim > 0 else 3
        self.protein_pretrained_proj = (
            nn.Sequential(nn.Linear(protein_pretrained_dim, hidden_dim), nn.ReLU())
            if protein_pretrained_dim > 0 else None)
        self.drug_pretrained_proj = (
            nn.Sequential(nn.Linear(drug_pretrained_dim, hidden_dim), nn.ReLU())
            if drug_pretrained_dim > 0 else None)

        # cross-modal context gate (Eq. 3): softmax over drug/target/sequence
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.context_gate_const = (
            nn.Parameter(torch.zeros(hidden_dim * 3)) if gate_mode == "dataset" else None
        )

        # --- interaction block ----------------------------------------------
        self.rail: RailInteractionLayer | None = None
        self.rail_main_t: RationalUnit1D | None = None
        self.rail_main_s: RationalUnit1D | None = None
        self.mlp_interaction: MLPInteractionLayer | None = None
        self.poly_interaction: PolyInteractionLayer | None = None
        if interaction == "rail":
            self.rail = RailInteractionLayer(
                hidden_dim,
                positive_denominator=positive_denominator,
                residual_gate=residual_gate,
            )
        elif interaction == "rail_fixed_interact":
            self.rail_main_t = RationalUnit1D(hidden_dim)
            self.rail_main_s = RationalUnit1D(hidden_dim)
        elif interaction == "mlp_interact":
            self.mlp_interaction = MLPInteractionLayer(hidden_dim, dropout=dropout)
        elif interaction == "poly_interact":
            self.poly_interaction = PolyInteractionLayer(hidden_dim)

        # --- scoring ---------------------------------------------------------
        if interaction in {"no_abs", "no_product"}:
            interaction_terms = 5
        else:
            interaction_terms = 7  # (6 interaction features + context) x H
        input_dim = hidden_dim * interaction_terms
        dims = [input_dim, *fc_dims, 1]
        self.regressor = _build_mlp(dims, dropout)
        self.drug_score = nn.Linear(hidden_dim, 1)
        self.bias_score = nn.Sequential(
            nn.Linear(bias_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.score_mixer = nn.Linear(3 if use_bias else 2, 1)

    def forward(
        self,
        drug_fingerprint: torch.Tensor,
        feature_fingerprint: torch.Tensor,
        maccs_fingerprint: torch.Tensor,
        bias_features: torch.Tensor,
        target: torch.Tensor,
        target_fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        # --- 1) per-view encodings -> [B, 3, H] -----------------------------
        drug_views = torch.stack(
            [
                self.drug_morgan(drug_fingerprint),
                self.drug_feature_morgan(feature_fingerprint),
                self.drug_maccs(maccs_fingerprint),
            ],
            dim=1,
        )

        # --- 2) drug-view reliability gate (Eq. 1-2) -------------------------
        if self.drug_view_gate_const is not None:
            drug_gate = torch.softmax(
                self.drug_view_gate(
                    self.drug_view_gate_const.expand(drug_views.size(0), -1)
                ),
                dim=-1,
            )
        elif self.use_drug_gate:
            drug_gate = torch.softmax(self.drug_view_gate(drug_views.flatten(1)), dim=-1)
        else:
            drug_gate = torch.full(
                (drug_views.size(0), 3), 1.0 / 3.0, device=drug_views.device
            )
        v_drug = (drug_views * drug_gate.unsqueeze(-1)).sum(dim=1)

        # --- 3) protein contexts ---------------------------------------------
        h_target = self.target_fp(target_fingerprint)
        h_seq = self.seq_encoder(target)

        # --- 4) cross-modal context gate (Eq. 3-4) ----------------------------
        contexts = torch.stack([v_drug, h_target, h_seq], dim=1)
        if self.context_gate_const is not None:
            context_gate = torch.softmax(
                self.context_gate(
                    self.context_gate_const.expand(contexts.size(0), -1)
                ),
                dim=-1,
            )
        elif self.use_context_gate:
            context_gate = torch.softmax(self.context_gate(contexts.flatten(1)), dim=-1)
        else:
            context_gate = torch.full(
                (contexts.size(0), 3), 1.0 / 3.0, device=contexts.device
            )
        h_context = (contexts * context_gate.unsqueeze(-1)).sum(dim=1)

        # --- 5) interaction features (Eq. 5-7) --------------------------------
        if self.rail is not None:
            rail_features = self.rail(v_drug, h_target, h_seq)  # [B, 6H]
            features = torch.cat([rail_features, h_context], dim=-1)
        elif self.mlp_interaction is not None:  # learned MLP interaction (F-M1)
            features = torch.cat(
                [self.mlp_interaction(v_drug, h_target, h_seq), h_context], dim=-1)
        elif self.poly_interaction is not None:  # polynomial interaction (F-M2)
            features = torch.cat(
                [self.poly_interaction(v_drug, h_target, h_seq), h_context], dim=-1)
        elif self.rail_main_t is not None:  # rail_fixed_interact (F-A7)
            feature_parts = [
                self.rail_main_t(h_target),
                self.rail_main_s(h_seq),
                torch.abs(v_drug - h_target),
                torch.abs(v_drug - h_seq),
                v_drug * h_target,
                v_drug * h_seq,
            ]
            feature_parts.append(h_context)
            features = torch.cat(feature_parts, dim=-1)
        else:  # hand-designed interactions (Eq. 5)
            feature_parts = [h_target, h_seq]
            if self.interaction != "no_abs":
                feature_parts.extend(
                    [torch.abs(v_drug - h_target), torch.abs(v_drug - h_seq)]
                )
            if self.interaction != "no_product":
                feature_parts.extend([v_drug * h_target, v_drug * h_seq])
            feature_parts.append(h_context)
            features = torch.cat(feature_parts, dim=-1)

        # --- 6) multi-branch score mixer (Eq. 9) ------------------------------
        score_parts = [self.regressor(features), self.drug_score(v_drug)]
        if self.use_bias:
            score_parts.append(self.bias_score(bias_features.float()))
        scores = torch.cat(score_parts, dim=-1)
        return self.score_mixer(scores).squeeze(-1)


class FingerDTA(nn.Module):
    """Plain FingerDTA baseline (no gates, fixed interactions) re-implemented
    in this framework for matched-protocol comparison (Table I, dagger row).
    """

    def __init__(
        self,
        target_vocab_size: int,
        fingerprint_dim: int = 2048,
        target_fingerprint_dim: int = 1024,
        maccs_dim: int = 167,
        bias_feature_dim: int = 6,
        hidden_dim: int = 256,
        fc_dims: tuple[int, ...] = (768, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.drug_fp = FingerprintEncoder(fingerprint_dim * 2 + maccs_dim, hidden_dim, dropout)
        self.target_fp = FingerprintEncoder(target_fingerprint_dim, hidden_dim, dropout)
        self.seq_encoder = ProteinSequenceEncoder(
            target_vocab_size, hidden_dim=hidden_dim, dropout=dropout
        )
        self.bias_score = nn.Sequential(
            nn.Linear(bias_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        dims = [hidden_dim * 6, *fc_dims, 1]
        self.regressor = _build_mlp(dims, dropout)
        self.drug_score = nn.Linear(hidden_dim, 1)
        self.score_mixer = nn.Linear(3, 1)

    def forward(
        self,
        drug_fingerprint: torch.Tensor,
        feature_fingerprint: torch.Tensor,
        maccs_fingerprint: torch.Tensor,
        bias_features: torch.Tensor,
        target: torch.Tensor,
        target_fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([drug_fingerprint, feature_fingerprint, maccs_fingerprint], dim=-1)
        drug_features = self.drug_fp(combined)
        target_features = self.target_fp(target_fingerprint)
        sequence_features = self.seq_encoder(target)
        features = torch.cat(
            [
                target_features,
                sequence_features,
                torch.abs(drug_features - target_features),
                drug_features * target_features,
                torch.abs(drug_features - sequence_features),
                drug_features * sequence_features,
            ],
            dim=-1,
        )
        scores = torch.cat(
            [
                self.regressor(features),
                self.drug_score(drug_features),
                self.bias_score(bias_features.float()),
            ],
            dim=-1,
        )
        return self.score_mixer(scores).squeeze(-1)
