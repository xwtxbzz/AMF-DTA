"""The paper's §3.6 numbers, from the architecture-matched intervention.

Three sets of deltas, all as 100 * (rmse_ablated - rmse_reference) / rmse_reference:

  A. FA3 post-hoc  -- gates frozen on `rail_ab_FA3_seed*_200` checkpoints
                      (`gate_intervention_fa3.json`). This is the set that
                      pairs with the retrained arm, because F-A11 is built on
                      the same `interaction="fixed"` config.
  B. RAIL post-hoc -- the same intervention on `rail_single_seed*_200`
                      (`gate_intervention.json`), whose config has
                      `interaction="rail"`. Kept as a robustness check: the
                      two architectures differ only in the interaction slot,
                      which Sec. III-D shows is neutral.
  C. Retrained     -- F-A11 (no gates, trained from scratch) vs F-A3, the
                      same-seed full model (`run_no_gate_5seeds.py`).

The placebo is reported next to `uniform` because it decides how the pair
should be read: it keeps the multiset of gate weights each batch uses and
changes only which sample receives which, so a placebo that reproduces most
of the uniform cost means the post-hoc number measures sensitivity to the
sample-gate pairing rather than to the removal of the gate.

Deltas in A and B are paired within (seed, dataset); C is paired within seed
and dataset too, since F-A11 and F-A3 share a seed's partition and init.

Usage: python analysis_posthoc_compare.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from scipy.stats import wilcoxon

OUT = Path(__file__).resolve().parent / "outputs"
DATASETS = ("davis", "kiba", "metz")
SEEDS = (42, 52, 62, 72, 82)
FULL = "F-A3"
NO_GATE = "F-A11"
TABLE_A1_AVG = 0.4311


def intervention_set(path: Path, condition: str) -> dict[tuple[int, str], float]:
    """{(seed, dataset): delta_pct} for one condition."""
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for seed in SEEDS:
        for ds in DATASETS:
            leaf = d.get("per_seed", {}).get(str(seed), {}).get(ds)
            if leaf and condition in leaf["conditions"]:
                out[(seed, ds)] = leaf["conditions"][condition]["delta_pct"]
    return out


def retrained_set() -> dict[tuple[int, str], float]:
    out = {}
    for seed in SEEDS:
        for ds in DATASETS:
            a = OUT / f"rail_ab_{FULL.replace('-', '')}_seed{seed}_200" / ds / "test_metrics.json"
            b = OUT / f"rail_ab_{NO_GATE.replace('-', '')}_seed{seed}_200" / ds / "test_metrics.json"
            if a.exists() and b.exists():
                full = json.loads(a.read_text(encoding="utf-8"))["rmse"]
                ng = json.loads(b.read_text(encoding="utf-8"))["rmse"]
                out[(seed, ds)] = 100.0 * (ng - full) / full
    return out


def describe(label: str, vals: list[float]) -> None:
    if not vals:
        print(f"{label:34s} n= 0  (not available)")
        return
    print(f"{label:34s} n={len(vals):2d}  median={statistics.median(vals):+7.2f}%  "
          f"mean={statistics.mean(vals):+7.2f}%  "
          f"min={min(vals):+7.2f}%  max={max(vals):+7.2f}%")


def by_dataset(label: str, d: dict[tuple[int, str], float]) -> None:
    if not d:
        return
    parts = []
    for ds in DATASETS:
        v = [x for (s, x_ds), x in d.items() if x_ds == ds]
        if v:
            parts.append(f"{ds}={statistics.median(v):+.1f}/{statistics.mean(v):+.1f}")
    print(f"  {label:32s} median/mean per dataset: " + "  ".join(parts))


def main() -> None:
    fa3_uniform = intervention_set(OUT / "gate_intervention_fa3.json", "uniform")
    fa3_placebo = intervention_set(OUT / "gate_intervention_fa3.json", "placebo")
    fa3_both = intervention_set(OUT / "gate_intervention_fa3.json", "both_mean")
    rail_uniform = intervention_set(OUT / "gate_intervention.json", "uniform")
    retr = retrained_set()

    print("=== RMSE cost of removing or perturbing the gates ===")
    describe("FA3 post-hoc uniform 1/K", list(fa3_uniform.values()))
    describe("FA3 post-hoc permutation placebo", list(fa3_placebo.values()))
    describe("FA3 post-hoc freeze to val mean", list(fa3_both.values()))
    describe("RAIL post-hoc uniform 1/K (robustness)", list(rail_uniform.values()))
    describe(f"retrained no gate ({NO_GATE})", list(retr.values()))

    if retr and len(retr) < 15:
        print(f"\nNOTE: {len(retr)}/15 retrained pairs on disk; sweep still running.")

    print("\n=== per dataset (median/mean %) ===")
    by_dataset("FA3 uniform", fa3_uniform)
    by_dataset("FA3 placebo", fa3_placebo)
    by_dataset("FA3 freeze-to-mean", fa3_both)
    by_dataset("retrained no gate", retr)

    # Placebo as a share of the uniform cost, pair by pair -- this is the
    # number that decides how the post-hoc result is read.
    shared = sorted(set(fa3_uniform) & set(fa3_placebo))
    if shared:
        shares = [100.0 * fa3_placebo[k] / fa3_uniform[k] for k in shared
                  if fa3_uniform[k] != 0]
        print(f"\nplacebo as a share of the uniform cost: median "
              f"{statistics.median(shares):.0f}%  mean {statistics.mean(shares):.0f}%  "
              f"(n={len(shares)})")

    # Sign tests: how many pairs get worse.
    for label, d in (("FA3 uniform", fa3_uniform), ("FA3 placebo", fa3_placebo),
                     ("retrained", retr)):
        if d:
            worse = sum(1 for v in d.values() if v > 0)
            print(f"{label:14s} worse in {worse}/{len(d)} pairs")

    # Paired tests on the same 15 pairs.
    print("\n=== paired Wilcoxon vs the un-modified model (two-sided) ===")
    for label, d in (("FA3 uniform", fa3_uniform), ("FA3 placebo", fa3_placebo),
                     ("FA3 both_mean", fa3_both)):
        if d:
            a = [v for _, v in sorted(d.items())]
            stat, p = wilcoxon(a)
            print(f"{label:14s} p={p:.3e} (n={len(a)})")

    if fa3_uniform and retr:
        common = sorted(set(fa3_uniform) & set(retr))
        if common:
            u = [fa3_uniform[k] for k in common]
            r = [retr[k] for k in common]
            print(f"\noverstatement on the {len(common)} shared pairs: "
                  f"median {statistics.median(u):.1f}% vs {statistics.median(r):.1f}%"
                  f"  ->  {statistics.median(u) / statistics.median(r):.1f}x")

    # Built-in correctness check: seed 42 must reproduce Table II's A1.
    ng_42 = [OUT / f"rail_ab_{NO_GATE.replace('-', '')}_seed42_200" / ds / "test_metrics.json"
             for ds in DATASETS]
    if all(p.exists() for p in ng_42):
        avg42 = statistics.mean(json.loads(p.read_text(encoding="utf-8"))["rmse"]
                                for p in ng_42)
        print(f"\nseed-42 no-gate avg RMSE = {avg42:.4f}  "
              f"(Table II A1 = {TABLE_A1_AVG:.4f}, "
              f"{100 * (avg42 - TABLE_A1_AVG) / TABLE_A1_AVG:+.1f}%)")


if __name__ == "__main__":
    main()
