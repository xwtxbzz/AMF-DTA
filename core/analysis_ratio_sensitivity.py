"""Sensitivity of the headline "~24x overstatement" ratio itself.

The paper's headline is a ratio:

    overstatement = (what the post-hoc probe says the gates are worth)
                    / (what the gates are actually worth)

with the numerator read off `gate_intervention_fa3.json` (freeze a trained
model's gates to uniform 1/K) and the denominator off
`retrained_matched_state.json` (retrain the identical architecture with no
gates). Both arms share the same baseline state -- the `gated` column of the
retrained file and the `sample` condition of the intervention file are the
same RMSE to float precision -- so the ratio is same-state and paired by
construction.

Sec. III-F already trims the *numerator* by dropping diverging pairs. That is
only half the job: the headline is a ratio of two medians, and a trimmed
numerator over an untrimmed denominator is not the ratio a reader would
compute after applying the paper's own exclusion rule. This script applies
each trim to BOTH arms and reports the ratio under each.

Estimators reported, because a ratio of medians is not the only defensible
one and a reviewer should be able to see whether the conclusion is
estimator-dependent:

    ratio_of_medians  -- median(post-hoc) / median(no-gate)   [the paper's]
    ratio_of_means    -- mean(post-hoc)   / mean(no-gate)
    median_of_ratios  -- median over pairs of per-pair ratios

All are in percent-delta units, so the baseline cancels and the ratio is
dimensionless.

Trims mirror Sec. III-F exactly:
    none        all 15 (seed, dataset) pairs
    gt50        drop pairs whose post-hoc delta exceeds +50%   (n=10)
    gt100       drop pairs whose post-hoc delta exceeds +100%  (n=13)

Provenance: read-only, over completed artefacts -- nothing is re-derived here.
The best_state.pt checkpoints these numbers came from are NOT in this bundle,
because `package_release.py` copies `outputs/` with `KEEP_SUFFIX =
{".json", ".csv"}`; they are intact in the full working tree
(`rail_dta/outputs/`), so both arms are recomputable from there.

Usage: python analysis_ratio_sensitivity.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
POST_HOC = OUT / "gate_intervention_fa3.json"
RETAINED = OUT / "retrained_matched_state.json"
DEST = OUT / "ratio_sensitivity.json"

DATASETS = ("davis", "kiba", "metz")
TRIMS = (("none", None), ("gt50", 50.0), ("gt100", 100.0))


def load_pairs() -> dict[tuple[int, str], dict[str, float]]:
    """Per-(seed, dataset) deltas for both arms, keyed identically."""
    post = json.loads(POST_HOC.read_text(encoding="utf-8"))["per_seed"]
    retr = json.loads(RETAINED.read_text(encoding="utf-8"))["rows"]

    nogate = {(int(r["seed"]), r["dataset"]): r["delta_pct"] for r in retr}
    pairs: dict[tuple[int, str], dict[str, float]] = {}
    for seed, per_ds in post.items():
        for ds, block in per_ds.items():
            key = (int(seed), ds)
            if key not in nogate:
                continue
            pairs[key] = {
                "post_hoc": block["conditions"]["uniform"]["delta_pct"],
                "no_gate": nogate[key],
                # Carried so the baseline identity check below is explicit
                # rather than asserted in a comment.
                "gated_rmse": block["conditions"]["sample"]["rmse"],
                "nogate_gated_rmse": next(
                    r["gated"] for r in retr
                    if int(r["seed"]) == int(seed) and r["dataset"] == ds),
            }
    return pairs


def estimators(rows: list[dict[str, float]]) -> dict[str, float]:
    num = [r["post_hoc"] for r in rows]
    den = [r["no_gate"] for r in rows]
    ratios = [r["post_hoc"] / r["no_gate"] for r in rows
              if r["no_gate"] != 0]
    return {
        "n": len(rows),
        "median_post_hoc_pct": statistics.median(num),
        "median_no_gate_pct": statistics.median(den),
        "ratio_of_medians": statistics.median(num) / statistics.median(den),
        "ratio_of_means": statistics.mean(num) / statistics.mean(den),
        "median_of_ratios": statistics.median(ratios),
    }


def main() -> None:
    pairs = load_pairs()

    # The two arms must be measured on the same baseline state, or the ratio
    # is not same-state and the whole exercise is void.
    worst = max(abs(v["gated_rmse"] - v["nogate_gated_rmse"]) for v in pairs.values())
    print(f"baseline identity check: max |gated_rmse difference| = {worst:.3e}")
    if worst > 1e-5:
        raise SystemExit("arms are not on the same baseline state; aborting")

    out: dict = {"n_pairs_total": len(pairs), "baseline_max_abs_diff": worst,
                 "trims": {}, "per_dataset": {}}

    print("\n=== trims (applied to BOTH arms) ===")
    for name, cutoff in TRIMS:
        keep = [k for k, v in pairs.items()
                if cutoff is None or v["post_hoc"] <= cutoff]
        est = estimators([pairs[k] for k in keep])
        out["trims"][name] = est
        print(f"{name:6s} n={est['n']:2d}  "
              f"post-hoc med={est['median_post_hoc_pct']:7.2f}%  "
              f"no-gate med={est['median_no_gate_pct']:5.2f}%  "
              f"-> ratio {est['ratio_of_medians']:5.1f}x "
              f"(means {est['ratio_of_means']:5.1f}x, "
              f"median-of-ratios {est['median_of_ratios']:5.1f}x)")

    print("\n=== per dataset (no trim) ===")
    for ds in DATASETS:
        rows = [v for k, v in pairs.items() if k[1] == ds]
        est = estimators(rows)
        out["per_dataset"][ds] = est
        print(f"{ds:6s} n={est['n']}  post-hoc med={est['median_post_hoc_pct']:7.2f}%  "
              f"no-gate med={est['median_no_gate_pct']:5.2f}%  "
              f"-> ratio {est['ratio_of_medians']:5.1f}x")

    full = out["trims"]["none"]["ratio_of_medians"]
    trim = out["trims"]["gt50"]["ratio_of_medians"]
    lo = min(v["ratio_of_medians"] for v in out["per_dataset"].values())
    hi = max(v["ratio_of_medians"] for v in out["per_dataset"].values())
    print(f"\nheadline {full:.0f}x; dropping diverging pairs {trim:.0f}x; "
          f"per-dataset range {lo:.0f}x-{hi:.0f}x")

    # The floor over EVERY variant, not just the estimator the paper leads
    # with. The manuscript claims the overstatement is "an order of magnitude
    # whichever way it is formed", which is a claim about this minimum -- so
    # it has to be computed over all three estimators x all three trims x
    # all three datasets, and the weakest cell named. Reporting only the
    # pooled estimator's floor would let the sentence overstate: the weakest
    # cell here is a per-dataset median-of-ratios, well below the pooled one.
    ESTIMATORS = ("ratio_of_medians", "ratio_of_means", "median_of_ratios")
    cells = [(est, grp, key, block[est])
             for grp in ("trims", "per_dataset")
             for key, block in out[grp].items()
             for est in ESTIMATORS]
    est, grp, key, val = min(cells, key=lambda c: c[3])
    out["global_min"] = {"value": val, "estimator": est,
                         "group": grp, "key": key, "n_variants": len(cells)}
    print(f"floor over all {len(cells)} variants: {val:.2f}x "
          f"({key}, {est}, {grp})")

    DEST.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {DEST}")


if __name__ == "__main__":
    main()
