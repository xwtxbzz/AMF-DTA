"""Retrained no-gate vs post-hoc gating removal, compared on the same 15 pairs.

The paper's methodological claim compares two ways of asking "what does
gating contribute?":

  post-hoc  -- take a model trained *with* gates and replace the gates by
               constants at test time (`analysis_gate_intervention.py`);
  retrained -- train the architecture from the start under those same
               constants (`run_no_gate_5seeds.py`, variant F-A11).

Only the `uniform` post-hoc condition (constant 1/K) is the *same
condition* as the no-gate architecture: `model.py`'s no-gate branch emits
exactly 1/K. That is the pair the claim rests on. `both_mean` (freeze to
the validation-set mean) is a different, milder intervention -- it removes
sample-level adaptivity but keeps the learned dataset-level preference --
and has no retrained counterpart in the ablation table, so it is reported
separately rather than mixed into the comparison.

Both deltas are computed per (seed, dataset) pair as
    100 * (rmse_ablated - rmse_full) / rmse_full
against the *same* seed's full model, so the two 15-value sets are directly
comparable, and both are paired within seed.

Full model = F-A3 (fixed interactions + both gates); it is the configuration
whose seed-42 average reproduces Table II's A3 (0.4237).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from scipy.stats import mannwhitneyu, wilcoxon

OUT = Path(__file__).resolve().parent / "outputs"
DATASETS = ("davis", "kiba", "metz")
SEEDS = (42, 52, 62, 72, 82)
NO_GATE = "F-A11"
FULL = "F-A3"

# Table II (upper group) A1: the seed-42-only no-gate run this sweep replaces.
TABLE_A1_AVG = 0.4311


def variant_rmse(variant: str, seed: int, ds: str) -> float | None:
    tag = variant.replace("-", "")
    p = OUT / f"rail_ab_{tag}_seed{seed}_200" / ds / "test_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["rmse"]


def retrained_deltas() -> list[float]:
    """15 paired deltas: no-gate trained from scratch vs the full model."""
    out = []
    for seed in SEEDS:
        for ds in DATASETS:
            full = variant_rmse(FULL, seed, ds)
            nogate = variant_rmse(NO_GATE, seed, ds)
            if full is None or nogate is None:
                continue
            out.append(100.0 * (nogate - full) / full)
    return out


def posthoc_deltas(condition: str) -> list[float]:
    """15 paired deltas from the frozen-gate intervention, same convention."""
    d = json.loads((OUT / "gate_intervention.json").read_text(encoding="utf-8"))
    out = []
    for seed in SEEDS:
        for ds in DATASETS:
            leaf = d["per_seed"].get(str(seed), {}).get(ds)
            if leaf is None:
                continue
            out.append(leaf["conditions"][condition]["delta_pct"])
    return out


def describe(name: str, vals: list[float]) -> str:
    if not vals:
        return f"{name:26s} n= 0  (not run yet)"
    return (f"{name:26s} n={len(vals):2d}  median={statistics.median(vals):+6.1f}%  "
            f"mean={statistics.mean(vals):+6.1f}%  "
            f"min={min(vals):+6.1f}%  max={max(vals):+6.1f}%")


def main() -> None:
    post = posthoc_deltas("uniform")
    retr = retrained_deltas()

    print("=== RMSE cost of removing gating, per (seed, dataset) pair ===")
    print(describe("post-hoc uniform 1/K", post))
    print(describe(f"retrained no gate ({NO_GATE})", retr))

    if len(retr) < 15:
        print(f"\nNOTE: only {len(retr)}/15 retrained pairs on disk -- the "
              f"no-gate sweep is still running; numbers below are partial.")

    if not retr:
        return

    ratio_med = statistics.median(post) / statistics.median(retr)
    ratio_mean = statistics.mean(post) / statistics.mean(retr)
    print(f"\noverstatement: median {statistics.median(post):.1f}% vs "
          f"{statistics.median(retr):.1f}%  ->  {ratio_med:.1f}x")
    print(f"               mean   {statistics.mean(post):.1f}% vs "
          f"{statistics.mean(retr):.1f}%  ->  {ratio_mean:.1f}x")

    if len(retr) == len(post) == 15:
        u, p = mannwhitneyu(post, retr, alternative="greater")
        print(f"               Mann-Whitney (post-hoc > retrained): p = {p:.2e}")
        # Even the mildest post-hoc pair vs the retrained cost.
        print(f"mildest post-hoc pair {min(post):.1f}% is "
              f"{min(post) / statistics.median(retr):.1f}x the retrained median")

    # Representative case, in the units of the ablation table rather than %.
    seed, ds = 42, "davis"
    full, nogate = variant_rmse(FULL, seed, ds), variant_rmse(NO_GATE, seed, ds)
    if full and nogate:
        print(f"\n{ds} seed {seed}: full {full:.4f} -> no-gate {nogate:.4f}")

    # Built-in correctness check: seed 42 must reproduce Table II's A1.
    fa3_42 = [variant_rmse(FULL, 42, ds) for ds in DATASETS]
    ng_42 = [variant_rmse(NO_GATE, 42, ds) for ds in DATASETS]
    if all(fa3_42) and all(ng_42):
        avg42 = statistics.mean(ng_42)
        print(f"\nseed-42 no-gate avg RMSE = {avg42:.4f}  "
              f"(Table II A1 = {TABLE_A1_AVG:.4f}, "
              f"diff {100 * (avg42 - TABLE_A1_AVG) / TABLE_A1_AVG:+.1f}%)")
        print("  ^ if this does not match A1, the config in "
              "run_no_gate_5seeds.py is not the one that produced the table")


if __name__ == "__main__":
    main()
