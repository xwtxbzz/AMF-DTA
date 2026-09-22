"""Paired inference for the paper's parity and gate claims (review round 1).

Why this exists
---------------
Reviewers of both rounds converged on the same objection to the *inference*,
not the experiments: the paper reads a non-significant paired Wilcoxon
(p=0.15, five seeds) as "parity", and it pools 15 (seed, dataset) pairs whose
three datasets share one initialization and one training run, so the pairs are
not independent and the p-values are optimistic. Three concrete asks followed:

  * an equivalence test (TOST) with a pre-specified margin, instead of
    "we failed to reject the null";
  * a hierarchical / seed-level treatment instead of the pooled one;
  * per-dataset breakdowns, effect sizes and confidence intervals.

All three are answerable from artifacts already on disk -- `test_metrics.json`
per (family, seed, dataset) and the intervention JSON -- so this script adds no
training. It reports, for each contrast, the paired difference in native RMSE
units, its 95% CI under two error models (i.i.d. over the 15 pairs, and a
seed-level cluster bootstrap that treats each seed as one independent draw),
Wilcoxon, Cohen's d_z, and TOST at several margins.

What "margin" means here
------------------------
Equivalence is only meaningful against a margin fixed in advance. The paper's
own seed-noise floor is 1.7-2.1% of RMSE (Sec. III-D), so a difference below
that is not resolvable at five seeds; 1% is roughly the resolution a reviewer
would demand and is *not* attainable at this sample size for the pooled
contrast. The script therefore reports the full margin curve rather than
picking the favourable one, and states the seed count each margin would need.

Sign convention: d = RMSE(treatment) - RMSE(reference). Negative means the
treatment is better. `delta_pct` is 100*d/mean(RMSE reference per pair).

Usage:  python analysis_equivalence.py [--out outputs/equivalence.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
SEEDS = (42, 52, 62, 72, 82)
DATASETS = ("davis", "kiba", "metz")

# contrast name -> (treatment family, reference family, run_tag pattern)
# Families are the ablation arms shipped in outputs/; the tag pattern is
# "{family}_seed{seed}_200" except for the two intervention-derived arms below.
CONTRASTS = {
    "M1_vs_fingerdta_style": ("rail_ab_FM1", "fingerdta"),
    "M2_vs_fingerdta_style": ("rail_ab_FM2", "fingerdta"),
    "RAIL_vs_fixed": ("rail_ab_FA8", "rail_ab_FA3"),
    "retrained_nogate_vs_gated": ("rail_ab_FA11", "rail_ab_FA3"),
}


def load_rmse(family: str) -> dict[tuple[int, str], float]:
    """RMSE per (seed, dataset) for one run family; missing pairs are skipped."""
    out: dict[tuple[int, str], float] = {}
    for seed in SEEDS:
        for ds in DATASETS:
            p = OUTPUT_ROOT / f"{family}_seed{seed}_200" / ds / "test_metrics.json"
            if p.exists():
                out[(seed, ds)] = json.loads(p.read_text(encoding="utf-8"))["rmse"]
    return out


def paired_arrays(treatment: str, reference: str):
    t, r = load_rmse(treatment), load_rmse(reference)
    keys = sorted(set(t) & set(r))
    d = np.array([t[k] - r[k] for k in keys], dtype=float)
    ref = np.array([r[k] for k in keys], dtype=float)
    return keys, d, ref


def cluster_bootstrap_ci(d: np.ndarray, keys, n_boot: int = 20000,
                         alpha: float = 0.05, rng_seed: int = 0):
    """CI for mean(d) resampling *seeds*, not pairs.

    The 15 pairs are three datasets observed under five seeds; conditioning on
    a seed fixes the partition, the initialization and the training run, so the
    three datasets within a seed move together. Treating the seed as the
    sampling unit is the conservative reading a reviewer asked for.
    """
    rng = np.random.default_rng(rng_seed)
    by_seed: dict[int, list[float]] = {}
    for k, v in zip(keys, d):
        by_seed.setdefault(k[0], []).append(float(v))
    seed_ids = sorted(by_seed)
    # a seed's contribution is the mean over its datasets, so every resample
    # keeps the 3-datasets-per-seed structure intact
    per_seed = np.array([statistics.mean(by_seed[s]) for s in seed_ids])
    n = len(per_seed)
    draws = np.array([
        rng.choice(per_seed, size=n, replace=True).mean() for _ in range(n_boot)
    ])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), n


def tost(d: np.ndarray, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided t-tests for |mean(d)| < margin.

    Equivalence is declared only when *both* one-sided tests reject, which is
    the conservative reading: it requires the CI to lie inside (-margin, margin).
    """
    n = len(d)
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / math.sqrt(n))
    if se == 0:
        return {"margin": margin, "equivalent": abs(mean) < margin,
                "p_lower": 0.0, "p_upper": 0.0, "df": n - 1}
    t_crit = stats.t.ppf(1 - alpha, n - 1)
    # H0_lower: mean <= -margin  -> reject when t_lo > t_crit
    t_lo = (mean + margin) / se
    # H0_upper: mean >= +margin  -> reject when t_hi < -t_crit
    t_hi = (mean - margin) / se
    p_lo = float(1 - stats.t.cdf(t_lo, n - 1))
    p_hi = float(stats.t.cdf(t_hi, n - 1))
    return {
        "margin": margin,
        "equivalent": bool(t_lo > t_crit and t_hi < -t_crit),
        "p_lower": p_lo,
        "p_upper": p_hi,
        "df": n - 1,
    }


def required_seeds(d: np.ndarray, margin: float, power: float = 0.8,
                   alpha: float = 0.05):
    """Seeds needed to *establish equivalence* at `margin` with `power`.

    Returns None when the observed effect already exceeds the margin: the
    equivalence null is then not merely under-powered but false, so no sample
    size converges -- reporting a number there would be meaningless.
    """
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return 0.0
    effect = abs(float(np.mean(d)))
    if effect >= margin:
        return None
    z_a = stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    gap = margin - effect
    return float(((z_a + z_b) * sd / gap) ** 2)


def contrast_report(treatment: str, reference: str) -> dict:
    keys, d, ref = paired_arrays(treatment, reference)
    n = len(d)
    if n == 0:
        return {"error": "no overlapping runs"}
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    ci = (mean - stats.t.ppf(0.975, n - 1) * se,
          mean + stats.t.ppf(0.975, n - 1) * se) if n > 1 else (mean, mean)
    wil = stats.wilcoxon(d) if n > 1 and np.any(d != 0) else None
    base = float(np.mean(ref))
    d_z = mean / sd if sd > 0 else 0.0
    lo_s, hi_s, n_seeds = cluster_bootstrap_ci(d, keys)

    per_ds = {}
    for ds in DATASETS:
        sub = np.array([v for k, v in zip(keys, d) if k[1] == ds], dtype=float)
        sub_ref = np.array([v for k, v in zip(keys, ref) if k[1] == ds], dtype=float)
        if len(sub) == 0:
            continue
        per_ds[ds] = {
            "n": len(sub),
            "mean_diff": float(np.mean(sub)),
            "mean_diff_pct": float(100 * np.mean(sub) / np.mean(sub_ref)),
            "std_diff_pct": float(100 * np.std(sub, ddof=1) / np.mean(sub_ref))
            if len(sub) > 1 else 0.0,
            "wins": int(np.sum(sub < 0)),
        }

    return {
        "treatment": treatment,
        "reference": reference,
        "n_pairs": n,
        "n_seeds": n_seeds,
        "mean_diff_rmse": mean,
        "mean_diff_pct": float(100 * mean / base),
        "median_diff_pct": float(100 * np.median(d) / base),
        "ci95_iid": [float(ci[0]), float(ci[1])],
        "ci95_cluster_by_seed": [lo_s, hi_s],
        "wilcoxon_p": float(wil.pvalue) if wil is not None else None,
        "cohens_dz": float(d_z),
        "tost": {str(m): tost(d, m) for m in (0.005, 0.01, 0.02, 0.03)},
        "required_seeds_at_1pct_margin": required_seeds(d, 0.01 * base),
        "per_dataset": per_ds,
    }


def print_contrast(name: str, rep: dict) -> None:
    if "error" in rep:
        print(f"{name}: {rep['error']}")
        return
    print(f"\n=== {name}  (n={rep['n_pairs']} pairs / {rep['n_seeds']} seeds)")
    print(f"  mean diff {rep['mean_diff_rmse']:+.4f} RMSE "
          f"({rep['mean_diff_pct']:+.2f}%, median {rep['median_diff_pct']:+.2f}%)")
    print(f"  95% CI iid      [{rep['ci95_iid'][0]:+.4f}, {rep['ci95_iid'][1]:+.4f}]")
    print(f"  95% CI by seed  [{rep['ci95_cluster_by_seed'][0]:+.4f}, "
          f"{rep['ci95_cluster_by_seed'][1]:+.4f}]")
    need = rep["required_seeds_at_1pct_margin"]
    need_txt = "n/a (effect exceeds margin)" if need is None else f"{need:.0f}"
    print(f"  Wilcoxon p={rep['wilcoxon_p']:.4f}  d_z={rep['cohens_dz']:+.2f}  "
          f"seeds for 1% margin={need_txt}")
    for m, res in rep["tost"].items():
        print(f"  TOST margin {float(m) * 100:>4.1f}%: "
              f"{'EQUIVALENT' if res['equivalent'] else 'not established'}"
              f"  (p_lower={res['p_lower']:.4f}, p_upper={res['p_upper']:.4f})")
    for ds, v in rep["per_dataset"].items():
        print(f"    {ds:6s} n={v['n']} diff {v['mean_diff_pct']:+.2f}% "
              f"± {v['std_diff_pct']:.2f}  wins {v['wins']}/{v['n']}")


def gate_panel() -> dict:
    """Gate contrasts at the state the paper actually quotes.

    Sec. III-F measures every gate delta from each run's *best-validation*
    checkpoint, not the final-fit SWA state that `test_metrics.json` holds, and
    says so. The two states do not give the same retrained penalty
    (0.87% vs 1.18% median), so mixing them would misstate the paper. This
    panel therefore reads the matched-state artifact and the intervention
    artifact -- both already on disk -- and gives them the same CI treatment as
    the model contrasts, which is what review round 1 asked for.
    """
    out: dict = {"matched_state": None, "posthoc": {}}

    ms_path = OUTPUT_ROOT / "retrained_matched_state.json"
    if ms_path.exists():
        ms = json.loads(ms_path.read_text(encoding="utf-8"))
        rows = ms["rows"]
        keys = [(r["seed"], r["dataset"]) for r in rows]
        d = np.array([r["gated"] - r["nogate"] for r in rows], dtype=float)
        ref = np.array([r["gated"] for r in rows], dtype=float)
        lo_s, hi_s, n_seeds = cluster_bootstrap_ci(d, keys)
        wil = stats.wilcoxon(d)
        per_ds = {}
        for ds in DATASETS:
            sub = np.array([v for k, v in zip(keys, d) if k[1] == ds])
            sub_ref = np.array([v for k, v in zip(keys, ref) if k[1] == ds])
            if len(sub):
                per_ds[ds] = {
                    "n": len(sub),
                    "mean_diff_pct": float(100 * np.mean(sub) / np.mean(sub_ref)),
                    "std_diff_pct": float(100 * np.std(sub, ddof=1) / np.mean(sub_ref)),
                    "wins": int(np.sum(sub < 0)),
                }
        sd = float(np.std(d, ddof=1))
        out["matched_state"] = {
            "description": "retrained no-gate (F-A11) minus gated (F-A3), best-validation state",
            "n_pairs": len(d),
            "n_seeds": n_seeds,
            "mean_diff_pct": float(100 * np.mean(d) / np.mean(ref)),
            "median_diff_pct": float(ms["median_delta_pct"]),
            "ci95_cluster_by_seed_pct": [float(100 * lo_s / np.mean(ref)),
                                         float(100 * hi_s / np.mean(ref))],
            "wilcoxon_p": float(wil.pvalue),
            "cohens_dz": float(np.mean(d) / sd) if sd else 0.0,
            "per_dataset": per_ds,
        }

    iv_path = OUTPUT_ROOT / "gate_intervention_fa3.json"
    if iv_path.exists():
        iv = json.loads(iv_path.read_text(encoding="utf-8"))
        conditions = ["drug_mean", "context_mean", "both_mean", "uniform", "placebo"]
        for cond in conditions:
            keys, d, ref = [], [], []
            for seed, per_ds in iv["per_seed"].items():
                for ds, node in per_ds.items():
                    c = node["conditions"]
                    if cond not in c:
                        continue
                    keys.append((int(seed), ds))
                    d.append(c[cond]["rmse"] - c["sample"]["rmse"])
                    ref.append(c["sample"]["rmse"])
            d = np.array(d, dtype=float)
            ref = np.array(ref, dtype=float)
            if len(d) == 0:
                continue
            lo_s, hi_s, n_seeds = cluster_bootstrap_ci(d, keys)
            wil = stats.wilcoxon(d) if np.any(d != 0) else None
            per_ds = {}
            for ds in DATASETS:
                sub = np.array([v for k, v in zip(keys, d) if k[1] == ds])
                sub_ref = np.array([v for k, v in zip(keys, ref) if k[1] == ds])
                if len(sub):
                    per_ds[ds] = {
                        "n": len(sub),
                        "mean_diff_pct": float(100 * np.mean(sub) / np.mean(sub_ref)),
                        "median_diff_pct": float(100 * np.median(sub) / np.mean(sub_ref)),
                    }
            sd = float(np.std(d, ddof=1))
            out["posthoc"][cond] = {
                "description": f"post-hoc `{cond}` minus sample gates, best-validation state",
                "n_pairs": len(d),
                "n_seeds": n_seeds,
                "mean_diff_pct": float(100 * np.mean(d) / np.mean(ref)),
                "median_diff_pct": float(100 * np.median(d) / np.mean(ref)),
                "ci95_cluster_by_seed_pct": [float(100 * lo_s / np.mean(ref)),
                                             float(100 * hi_s / np.mean(ref))],
                "wilcoxon_p": float(wil.pvalue) if wil is not None else None,
                "cohens_dz": float(np.mean(d) / sd) if sd else 0.0,
                "per_dataset": per_ds,
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/equivalence.json")
    args = ap.parse_args()

    report = {"note": "d = RMSE(treatment) - RMSE(reference); negative = treatment better",
              "contrasts": {}, "gate_panel": {}}
    print("### model contrasts (final-fit state, Table I/II families)")
    for name, (t, r) in CONTRASTS.items():
        rep = contrast_report(t, r)
        report["contrasts"][name] = rep
        print_contrast(name, rep)

    print("\n\n### gate panel (best-validation matched state, as the paper quotes)")
    gp = gate_panel()
    report["gate_panel"] = gp
    ms = gp.get("matched_state")
    if ms:
        print(f"\n=== retrained_no_gate_vs_gated [MATCHED STATE] "
              f"(n={ms['n_pairs']} / {ms['n_seeds']} seeds)")
        print(f"  mean {ms['mean_diff_pct']:+.2f}%  median {ms['median_diff_pct']:+.2f}%  "
              f"95% CI by seed [{ms['ci95_cluster_by_seed_pct'][0]:+.2f}%, "
              f"{ms['ci95_cluster_by_seed_pct'][1]:+.2f}%]")
        print(f"  Wilcoxon p={ms['wilcoxon_p']:.5f}  d_z={ms['cohens_dz']:+.2f}")
        for ds, v in ms["per_dataset"].items():
            print(f"    {ds:6s} {v['mean_diff_pct']:+.2f}% ± {v['std_diff_pct']:.2f}  "
                  f"wins {v['wins']}/{v['n']}")
    for cond, v in gp.get("posthoc", {}).items():
        print(f"\n=== posthoc_{cond} (n={v['n_pairs']} / {v['n_seeds']} seeds)")
        print(f"  mean {v['mean_diff_pct']:+.2f}%  median {v['median_diff_pct']:+.2f}%  "
              f"95% CI by seed [{v['ci95_cluster_by_seed_pct'][0]:+.2f}%, "
              f"{v['ci95_cluster_by_seed_pct'][1]:+.2f}%]")
        print(f"  Wilcoxon p={v['wilcoxon_p']:.3g}  d_z={v['cohens_dz']:+.2f}")
        for ds, dv in v["per_dataset"].items():
            print(f"    {ds:6s} mean {dv['mean_diff_pct']:+.2f}%  "
                  f"median {dv['median_diff_pct']:+.2f}%")

    out = OUTPUT_ROOT / "equivalence.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
