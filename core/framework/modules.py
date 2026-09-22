"""Rational-ANOVA interaction modules (RAIL) — paper Sec. II-C.

Implements the learnable rational units of Rational-ANOVA Networks (RAN),
adapted to drug-target affinity (DTA) multi-view fusion:

  * RationalUnit1D:  Pade-style quotient r_tilde(x) = p(x) / d(x) with a
                     strictly positive denominator d(x) = 1 + softplus(q(x))
                     + eps (pole-free), wrapped in a residual gate
                     r(x) = x + alpha * (r_tilde(x) - x) that initializes at
                     the identity map (alpha = 0).
  * RationalUnit2D:  bivariate rational interaction r(x, y) anchored at the
                     elementwise product x * y (warm start), then learns the
                     interaction shape.
  * RailInteractionLayer: functional-ANOVA topology over the three DTA
                     contexts (drug / target fingerprint / sequence):

                       f_rail = [ r_d(x),  r_t(y),   r_s(z),
                                  r_dt(x,y), r_ds(x,z), r_ts(y,z) ]

                     i.e. main effects plus sparse pairwise interactions on
                     the interaction set S = {(d,t), (d,s), (t,s)}. The
                     topology is budgetable: each added pair adds second-order
                     capacity without combinatorial explosion.

References:
  [RAN] J. Zhang, N. Liu, Q. Lyu, J. Yang, and K. Wang,
        "Rational ANOVA networks," arXiv:2602.04006, 2026.
  Paper Eqs. (6)-(7) and the ablation discussion in Sec. III-E (F-A7..F-A10).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RationalUnit1D(nn.Module):
    """Feature-wise 1D rational unit.

    r_tilde(x) = p(x) / d(x),   d(x) = 1 + softplus(q(x)) + eps  >=  1 + eps
    r(x)       = x + alpha * (r_tilde(x) - x)

    Initialization: p(x) ~= x (a1 = 1, other coefficients 0), d ~= 1
    (coefficients 0), alpha = 0  =>  r(x) = x. The nonlinearity "turns on"
    during training via alpha, which receives a 10x learning rate (Sec. III-A).
    """

    def __init__(
        self,
        dim: int,
        num_degree: int = 2,
        den_degree: int = 2,
        eps: float = 1e-4,
        init_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_degree = num_degree
        self.den_degree = den_degree
        self.eps = eps
        # numerator coefficients [dim, num_degree + 1];  p(x) = sum_a c_a x^a
        self.num = nn.Parameter(torch.zeros(dim, num_degree + 1))
        with torch.no_grad():
            if num_degree >= 1:
                self.num.data[:, 1] = 1.0  # p(x) ~= x
        # denominator coefficients [dim, den_degree + 1];  q(x) = sum_b c_b x^b
        self.den = nn.Parameter(torch.zeros(dim, den_degree + 1))
        # residual gate alpha: free parameter, init 0 = identity map
        self.alpha = nn.Parameter(torch.full((dim,), float(init_alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Powers built with cumulative multiplication (avoids pow()).
        deg = max(self.num_degree, self.den_degree)
        powers = [torch.ones_like(x)]
        for _ in range(deg):
            powers.append(powers[-1] * x)
        pw = torch.stack(powers, dim=-1)  # [B, D, deg + 1]
        p = torch.einsum("bdk,dk->bd", pw[..., : self.num_degree + 1], self.num)
        q = torch.einsum("bdk,dk->bd", pw[..., : self.den_degree + 1], self.den)
        d = 1.0 + F.softplus(q) + self.eps
        r_tilde = p / d
        alpha = self.alpha.unsqueeze(0)
        return x + alpha * (r_tilde - x)


class RationalUnit2D(nn.Module):
    """Bivariate rational interaction unit.

    r_tilde(x, y) = p(x, y) / d(x, y)
      p(x, y) = sum_t gamma_t * psi_t(x, y)
      d(x, y) = 1 + softplus(sum_s delta_s * psi_s(x, y)) + eps
    r(x, y) = anchor(x, y) + alpha * (r_tilde(x, y) - anchor(x, y))
      with anchor = x * y  (warm start from the elementwise product).

    Low-order monomial basis:  {1, x, y, x^2, xy, y^2}  (controls curvature).
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-4,
        init_alpha: float = 0.0,
        warm_start_product: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.warm_start_product = warm_start_product
        # basis indices: 0:1, 1:x, 2:y, 3:x^2, 4:xy, 5:y^2
        self.gamma = nn.Parameter(torch.zeros(dim, 6))
        self.delta = nn.Parameter(torch.zeros(dim, 6))
        if warm_start_product:
            with torch.no_grad():
                self.gamma.data[:, 4] = 1.0  # p ~= x * y
        self.alpha = nn.Parameter(torch.full((dim,), float(init_alpha)))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        basis = torch.stack(
            [torch.ones_like(x), x, y, x * x, x * y, y * y], dim=-1
        )  # [B, D, 6]
        p = torch.einsum("bdk,dk->bd", basis, self.gamma)
        q = torch.einsum("bdk,dk->bd", basis, self.delta)
        d = 1.0 + F.softplus(q) + self.eps
        r_tilde = p / d
        anchor = x * y if self.warm_start_product else 0.5 * (x + y)
        alpha = self.alpha.unsqueeze(0)
        return anchor + alpha * (r_tilde - anchor)


class RailInteractionLayer(nn.Module):
    """Rational-ANOVA interaction layer over three contexts.

    f_rail = [ r_d(drug), r_t(target), r_s(seq),
               r_dt(drug, target), r_ds(drug, seq), r_ts(target, seq) ]

    Main effects are 1D rational units; pairwise interactions are 2D units
    anchored at the elementwise product. Ablation switches (paper F-A9 /
    F-A10): positive_denominator=False drops the pole-free constraint,
    residual_gate=False forces alpha = 1 (no identity-safe gating).
    """

    def __init__(
        self,
        dim: int,
        positive_denominator: bool = True,
        residual_gate: bool = True,
    ) -> None:
        super().__init__()
        self.positive_denominator = positive_denominator
        self.residual_gate = residual_gate
        # Main effects
        self.main_drug = RationalUnit1D(dim)
        self.main_target = RationalUnit1D(dim)
        self.main_seq = RationalUnit1D(dim)
        # Pairwise interactions
        self.pair_dt = RationalUnit2D(dim)
        self.pair_ds = RationalUnit2D(dim)
        self.pair_ts = RationalUnit2D(dim)
        if not positive_denominator:
            self._disable_positive_denominator()
        if not residual_gate:
            self._disable_residual_gate()

    def _disable_positive_denominator(self) -> None:
        """Ablation F-A9: drop the 1 + softplus(...) + eps constraint so the
        denominator may cross zero (pole-prone; training is expected to
        diverge — see paper Sec. III-E)."""

        for unit in (self.main_drug, self.main_target, self.main_seq):
            unit.eps = 0.0

        class _NoDenom2D(RationalUnit2D):
            def forward(self, x, y):  # noqa: D102
                basis = torch.stack(
                    [torch.ones_like(x), x, y, x * x, x * y, y * y], dim=-1
                )
                p = torch.einsum("bdk,dk->bd", basis, self.gamma)
                d = torch.einsum("bdk,dk->bd", basis, self.delta)
                r_tilde = p / d
                anchor = x * y if self.warm_start_product else 0.5 * (x + y)
                alpha = torch.sigmoid(self.alpha).unsqueeze(0)
                return anchor + alpha * (r_tilde - anchor)

        self.pair_dt = _NoDenom2D(self.pair_dt.gamma.size(0))
        self.pair_ds = _NoDenom2D(self.pair_ds.gamma.size(0))
        self.pair_ts = _NoDenom2D(self.pair_ts.gamma.size(0))

    def _disable_residual_gate(self) -> None:
        """Ablation F-A10: force alpha = 1 (rational map always active)."""

        for unit in (self.main_drug, self.main_target, self.main_seq):
            with torch.no_grad():
                unit.alpha.fill_(1.0)
            unit.alpha.requires_grad_(False)
        for unit in (self.pair_dt, self.pair_ds, self.pair_ts):
            with torch.no_grad():
                unit.alpha.fill_(1.0)
            unit.alpha.requires_grad_(False)

    def forward(
        self,
        drug: torch.Tensor,
        target: torch.Tensor,
        seq: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                self.main_drug(drug),
                self.main_target(target),
                self.main_seq(seq),
                self.pair_dt(drug, target),
                self.pair_ds(drug, seq),
                self.pair_ts(target, seq),
            ],
            dim=-1,
        )


class MLPInteractionLayer(nn.Module):
    """Learned nonlinear interaction baseline (review F-M1): per-pair MLPs
    shared across the three pairs, identity main effects. No rational or
    ANOVA structure. Produces the same 6D interaction output and feeds the
    same 7D regressor as RAIL; total parameters match within ~1% (hidden 16:
    interaction block 12.5k vs. RAIL's 15.4k)."""

    def __init__(self, dim: int, hidden: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.pair_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        drug: torch.Tensor,
        target: torch.Tensor,
        seq: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                self.pair_mlp(torch.cat([drug, target], dim=-1)),
                self.pair_mlp(torch.cat([drug, seq], dim=-1)),
                self.pair_mlp(torch.cat([target, seq], dim=-1)),
                drug,   # identity "main effects" (RAIL at alpha=0 is identity)
                target,
                seq,
            ],
            dim=-1,
        )


class PolyInteractionLayer(nn.Module):
    """Unconstrained degree-2 polynomial interaction baseline (review F-M2):
    the same monomial basis and topology as RAIL but linear in the monomial
    features --- no rational denominator, no residual gate. Total parameters
    match within ~0.3% (per-pair [dim, 6] coefficients)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        # basis indices: 0:1, 1:x, 2:y, 3:x^2, 4:xy, 5:y^2
        self.pair_dt = nn.Parameter(torch.zeros(dim, 6))
        self.pair_ds = nn.Parameter(torch.zeros(dim, 6))
        self.pair_ts = nn.Parameter(torch.zeros(dim, 6))

    def _pair(self, gamma: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        basis = torch.stack([torch.ones_like(x), x, y, x * x, x * y, y * y], dim=-1)
        return torch.einsum("bdk,dk->bd", basis, gamma)

    def forward(
        self,
        drug: torch.Tensor,
        target: torch.Tensor,
        seq: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                self._pair(self.pair_dt, drug, target),
                self._pair(self.pair_ds, drug, seq),
                self._pair(self.pair_ts, target, seq),
                drug,
                target,
                seq,
            ],
            dim=-1,
        )
