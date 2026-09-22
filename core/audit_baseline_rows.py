"""Verify the reproduced-baseline rows of Table II against the run artefacts.

The lower group of Table II (FingerDTA, WideDTA, DeepDTA, DrugMGR,
MambaTransDTA) comes from the *root* `outputs/` tree, not from
`rail_dta/outputs/` -- these are baselines reimplemented in the main
framework, run by the root-level multiseed scripts. An earlier audit looked
only under `rail_dta/outputs/` and wrongly recorded them as unverifiable.

Two naming generations coexist and use different seed-42 conventions:

  * `{ds}_{model}_all200`   is the seed-42 run of the `run_all_experiments`
                            generation, and `compile_all_results.py` pairs it
                            with `_seed{52,62,72,82}_200`;
  * `{ds}_{model}_seed42_200` is the run_2024/2025 generation, which names
                            all five seeds explicitly (DrugMGR,
                            MambaTransDTA use this one).

This script resolves each model by trying the explicit-`seed42` form first
and falling back to `_all200`, prints every seed used, and compares the
resulting per-dataset means and seed-wise average against the paper.

Usage: python audit_baseline_rows.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs"
DATASETS = ("davis", "kiba", "metz")
SEEDS = (42, 52, 62, 72, 82)

# Per-dataset mean RMSE and the three-dataset average (paper Table II).
# The stds must be *sample* std (n-1), matching the caption and the upper
# group. The original lower-group values were population std (n), a factor
# sqrt(5/4) = 1.118 too small; this script is what caught it.
CLAIMED = {
    "fingerdta":      ([.4658, .4143, .5380], .4727, .0106, .8574),
    "widedta":        ([.4728, .4131, .5520], .4793, .0053, .8576),
    "deepdta":        ([.4756, .4202, .5538], .4832, .0091, .8543),
    "drugmgr":        ([.5178, .4274, .5505], .4986, .0099, .8481),
    "mambatransdta":  ([.4800, .4669, .5772], .5080, .0054, .8399),
}

OK, BAD = [], []
PINNED: list[float] = []      # see audit_coverage.py


def check(label: str, claimed: float, actual: float | None,
          tol: float = 5e-4) -> None:
    if isinstance(claimed, (int, float)) and not isinstance(claimed, bool):
        PINNED.append(round(float(claimed), 6))
    if actual is None:
        good, shown = False, f"paper={claimed:.4f}  derived=missing"
    else:
        good = abs(claimed - actual) <= tol
        shown = f"paper={claimed:.4f}  derived={actual:.4f}"
    (OK if good else BAD).append(label)
    print(f"  [{'ok ' if good else 'MISMATCH'}] {label:34s} {shown}")


def seed_dir(ds: str, model: str, seed: int) -> Path:
    """Resolve the directory for one (dataset, model, seed)."""
    explicit = OUT / f"{ds}_{model}_seed{seed}_200"
    if explicit.exists():
        return explicit
    if seed == 42:                       # older generation's name for seed 42
        return OUT / f"{ds}_{model}_all200"
    return explicit


def metrics(ds: str, model: str, seed: int) -> dict | None:
    p = seed_dir(ds, model, seed) / "test_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    for model, (per_ds, avg, std, ci) in CLAIMED.items():
        print(f"=== {model} ===")
        for i, ds in enumerate(DATASETS):
            vals = []
            for s in SEEDS:
                m = metrics(ds, model, s)
                if m is None:
                    print(f"    missing: {ds} seed {s} "
                          f"({seed_dir(ds, model, s).name})")
                else:
                    vals.append(m["rmse"])
            check(f"{model} {ds} mean (n={len(vals)})", per_ds[i],
                  statistics.mean(vals) if vals else None)

        # Three-dataset average, taken per seed then averaged -- the same
        # convention as the upper group and Table I, so the two groups are
        # built identically.
        per_seed, cis = [], []
        for s in SEEDS:
            row = [metrics(ds, model, s) for ds in DATASETS]
            if all(row):
                per_seed.append(statistics.mean(r["rmse"] for r in row))
                cis.append(statistics.mean(r["ci"] for r in row))
        check(f"{model} avg RMSE (n={len(per_seed)})", avg,
              statistics.mean(per_seed) if per_seed else None)
        check(f"{model} avg std", std,
              statistics.stdev(per_seed) if len(per_seed) > 1 else None)
        check(f"{model} mean CI", ci,
              statistics.mean(cis) if cis else None,
              tol=5e-4 if cis else 1)
        print()

    # The four baselines the paper *omits* from Table I are quoted only as a
    # range ("trail at 0.4970--0.6218"). Two of them follow the same naming
    # split as the checked rows -- seed 42 lives under `*_all200` -- so a
    # reader (or an audit) that globs only `*_seed{42}_200` silently drops
    # seed 42 and gets 0.4997/0.5318 instead of 0.4970/0.5291.
    print("=== omitted baselines: quoted range ===")
    omitted = {}
    for model in ("attentiondta", "graphdta", "ckdta", "edta"):
        per_seed = []
        for s in SEEDS:
            row = [metrics(ds, model, s) for ds in DATASETS]
            if all(row):
                per_seed.append(statistics.mean(r["rmse"] for r in row))
        omitted[model] = statistics.mean(per_seed)
        print(f"  {model:14s} n_seeds={len(per_seed)} avg={omitted[model]:.4f}")
    check("omitted range low", 0.4970, round(min(omitted.values()), 4))
    check("omitted range high", 0.6218, round(max(omitted.values()), 4))
    print()

    print(f"=== {len(OK)} checks ok, {len(BAD)} mismatched ===")
    (Path(__file__).resolve().parent / "outputs"
     / "audit_pinned_baseline_rows.json").write_text(
        json.dumps(sorted(set(PINNED))), encoding="utf-8")
    for label in BAD:
        print(f"  MISMATCH {label}")


if __name__ == "__main__":
    main()
