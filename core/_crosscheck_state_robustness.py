r"""Final-fit cross-check of the matched-state retrained finding.

`analysis_retrained_matched_state.py` measures the retrained arm at the
best-validation state, because that is the state the post-hoc arm lives on.
But every run also wrote a `test_metrics.json` at the *final-fit* state, and
nothing stops us from comparing the two arms there for free.

This is worth doing because review W1's objection was that the published
retrained delta came from a different state than the post-hoc delta. If the
retrained effect is ~1.6% at the matched state and also ~1.5% at final-fit,
then the *retrained* side of the ratio is not a state artifact -- which is a
stronger statement than either state alone, and it is checkable from data
already on disk.

What this cannot do is produce a final-fit *ratio*: the post-hoc arm's 21.0%
exists only on `best_state.pt` (that is the point of the intervention), so
there is no final-fit post-hoc number to divide by. So this is a robustness
check on the denominator, not a second headline.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scipy.stats import wilcoxon  # noqa: E402
from analysis_gate_intervention import DATASETS, SEEDS  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"


def rmse(run: str, seed: int, ds: str) -> float | None:
    p = OUT / f"{run}_seed{seed}_200" / ds / "test_metrics.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text(encoding="utf-8"))["rmse"])


rows = []
for seed in SEEDS:
    for ds in DATASETS:
        g, n = rmse("rail_ab_FA3", seed, ds), rmse("rail_ab_FA11", seed, ds)
        if g is None or n is None:
            print(f"  [skip] seed {seed} {ds}")
            continue
        rows.append({"seed": seed, "dataset": ds, "gated": g, "nogate": n,
                     "delta_pct": 100 * (n - g) / g})

deltas = [r["delta_pct"] for r in rows]
print(f"\n=== retrained no-gate, FINAL-FIT state, n={len(rows)} ===")
print(f"  gated (F-A3)  mean = {statistics.mean([r['gated'] for r in rows]):.4f}"
      f"   (Table II lower group quotes .4363)")
print(f"  nogate (F-A11) mean = {statistics.mean([r['nogate'] for r in rows]):.4f}")
print(f"  delta median={statistics.median(deltas):+.2f}%  "
      f"mean={statistics.mean(deltas):+.2f}%  "
      f"worse in {sum(1 for d in deltas if d > 0)}/{len(deltas)}")
print(f"  two-sided paired Wilcoxon p="
      f"{wilcoxon([r['gated'] for r in rows], [r['nogate'] for r in rows]).pvalue:.4g}")
for ds in DATASETS:
    v = [r["delta_pct"] for r in rows if r["dataset"] == ds]
    print(f"  {ds:6s} median={statistics.median(v):+6.2f}% "
          f"mean={statistics.mean(v):+.2f}%")

print("\n  per-seed:")
for seed in SEEDS:
    v = [r["delta_pct"] for r in rows if r["seed"] == seed]
    print(f"    seed {seed}: " + "  ".join(f"{d:+.2f}%" for d in v))
