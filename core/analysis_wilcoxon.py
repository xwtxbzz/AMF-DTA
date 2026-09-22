"""Paired Wilcoxon signed-rank tests over per-(seed, dataset) RMSE pairs.

Pairs share the identical split at each seed, so differences are paired.
Alternative 'greater': diff = b - a > 0 means a (first arg) is better.
"""
import json
import statistics
from pathlib import Path

from scipy.stats import wilcoxon

OUT = Path(__file__).resolve().parent / "outputs"
DATASETS = ("davis", "kiba", "metz")
SEEDS = (42, 52, 62, 72, 82)

summary = json.loads((OUT / "summary_raildta_5seeds_200.json").read_text(encoding="utf-8"))


def ens(seed, ds):
    return summary["ensemble_per_seed"][str(seed)][ds]["rmse"]


def swa(seed, ds):
    return summary["single_swa_per_seed"][str(seed)][ds]["rmse"]


def fdt(seed, ds):
    return summary["fingerdta_dagger_per_seed"][str(seed)][ds]["rmse"]


def ab(variant, seed, ds):
    tag = variant.replace("-", "")
    p = OUT / f"rail_ab_{tag}_seed{seed}_200" / ds / "test_metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))["rmse"]


def paired(a_fn, b_fn, label):
    diffs = [b_fn(seed, ds) - a_fn(seed, ds) for seed in SEEDS for ds in DATASETS]
    _, p = wilcoxon(diffs, alternative="greater")
    wins = sum(1 for d in diffs if d > 0)
    print(f"{label}: mean b-a = {statistics.mean(diffs):+.4f} "
          f"(a better in {15 - wins}/15 pairs), Wilcoxon p = {p:.4f}")


if __name__ == "__main__":
    paired(ens, fdt, "Ens vs FingerDTA-dagger")
    paired(swa, fdt, "SWA  vs FingerDTA-dagger")
    paired(lambda s, d: ab("F-A8", s, d), lambda s, d: ab("F-A3", s, d),
           "F-A8 (RAIL) vs F-A3 (fixed)")
    paired(ens, lambda s, d: ab("F-A3", s, d), "Ens vs F-A3 (fixed)")
    paired(ens, lambda s, d: ab("F-A8", s, d), "Ens vs F-A8 (RAIL)")
