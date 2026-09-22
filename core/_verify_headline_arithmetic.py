r"""Recompute the headline statistics from the raw per-pair values.

`analysis_retrained_matched_state.py` writes `delta_pct` per row and three
aggregates on top of them, and the abstract quotes those aggregates. Nothing
else in the tree recomputes them from the rows -- the audit compares the
manuscript to the artefact, and the chain test compares the artefact to the
manuscript, so a bug in the aggregation itself would agree with itself all
the way to the camera-ready.

The specific thing worth pinning is the p-value's *sidedness*. The manuscript
says "two-sided paired", which is `scipy.stats.wilcoxon`'s default only by
convention -- if that default ever differed, the quoted p would be wrong by a
factor of two while every downstream check still passed, because they all
compare against the same stored number. So this computes both and reports
which one the artefact holds.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from scipy.stats import wilcoxon

sys.path.insert(0, ".")
import analysis_retrained_matched_state as M  # noqa: E402

def artefact_path() -> Path:
    """The canonical artefact if it exists, else the partial.

    Preferred order matters: once the full sweep lands, the partial is a
    stale 8-pair measurement, and verifying *it* would report a green result
    for numbers the paper no longer quotes.
    """
    canonical = M.OUT / "retrained_matched_state.json"
    if canonical.exists():
        return canonical
    return M.OUT / "retrained_matched_state.partial.json"


path = artefact_path()
art = json.loads(path.read_text(encoding="utf-8"))
print(f"verifying {path.name}  (canonical)"
      if path.name == "retrained_matched_state.json"
      else f"verifying {path.name}  (PARTIAL -- canonical not yet written)")
rows = art["rows"]
deltas = [r["delta_pct"] for r in rows]
gated = [r["gated"] for r in rows]
nogate = [r["nogate"] for r in rows]

fails: list[str] = []


def cmp(name: str, stored: float, recomputed: float, tol: float) -> None:
    ok = abs(stored - recomputed) <= tol
    if not ok:
        fails.append(name)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:34s} stored={stored:<12.6g}"
          f" recomputed={recomputed:<12.6g} tol={tol:g}")


print(f"artefact: {len(rows)} pairs\n")

print("per-row delta formula  delta_pct == 100*(nogate-gated)/gated")
worst_row = max(abs(r["delta_pct"] - 100 * (r["nogate"] - r["gated"])
                    / r["gated"]) for r in rows)
cmp("max per-row formula error", 0.0, worst_row, 1e-9)

print("\naggregates")
cmp("median_delta_pct", art["median_delta_pct"], statistics.median(deltas), 1e-9)
cmp("mean_delta_pct", art["mean_delta_pct"], statistics.mean(deltas), 1e-9)
if "n_pairs" in art:
    cmp("n_pairs", art["n_pairs"], len(rows), 0)
else:
    # The partial artefact predates the n_pairs/complete keys: it was written
    # by a process that had already loaded the module before that edit landed.
    # The full run writes the current schema, so this branch should not be
    # taken once the sweep completes -- if it is, the artefact is stale.
    print("  [note] n_pairs absent (pre-schema artefact); "
          f"len(rows) = {len(rows)}")

print("\np-value sidedness")
two = float(wilcoxon(gated, nogate).pvalue)
# The claim-aligned direction is "less": the paper says the model *loses*
# accuracy without gates, so nogate RMSE is the larger one. `greater` here
# would test the opposite hypothesis and print ~1.0 -- a number a reader
# would reasonably take as "no effect", which is the reverse of the finding.
one = float(wilcoxon(gated, nogate, alternative="less").pvalue)
stored = float(art["p_value"])
print(f"  two-sided                p = {two:.6g}")
print(f"  one-sided (gated<nogate) p = {one:.6g}  <- claim-aligned direction")
print(f"  one-sided (gated>nogate) p = "
      f"{float(wilcoxon(gated, nogate, alternative='greater').pvalue):.6g}  "
      f"(opposite; ~1 means the effect is in the claimed direction)")
cmp("p_value is the two-sided one", stored, two, 1e-12)
if abs(stored - one) <= 1e-12 and abs(two - one) > 1e-12:
    fails.append("sidedness")
    print("  !! artefact holds the ONE-SIDED p; the manuscript says two-sided")

print("\nmatched-state identity (the paper's methodological core)")
# The artefact's `gated` column must equal the post-hoc run's own recorded
# unmodified-baseline RMSE for the same (seed, dataset), because that is what
# makes the two arms comparable: same checkpoint, same loader, same
# denormalisation, differing only in whether gates existed during training.
# Recomputed here from the recorded JSON rather than trusted from the
# analysis script's own printed verdict, so the check does not share a code
# path with the thing it checks.
ref_path = M.OUT / "gate_intervention_fa3.json"
if not ref_path.exists():
    fails.append("matched-state reference missing")
    print(f"  [FAIL] {ref_path.name} not found -- cannot verify state matching")
else:
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    worst, n_cmp, missing = 0.0, 0, []
    for r in rows:
        try:
            want = ref["per_seed"][str(r["seed"])][r["dataset"]]["conditions"]["sample"]["rmse"]
        except KeyError:
            missing.append(f"{r['seed']}/{r['dataset']}")
            continue
        worst = max(worst, abs(r["gated"] - float(want)))
        n_cmp += 1
    ok = worst < 1e-4 and n_cmp == len(rows)
    if not ok:
        fails.append("matched-state identity")
    print(f"  [{'ok ' if ok else 'FAIL'}] gated arm == post-hoc baseline on "
          f"{n_cmp}/{len(rows)} pairs, max |diff| = {worst:.2e}")
    if missing:
        print(f"         no reference for: {', '.join(missing)}")

print("\nper-dataset medians (recomputed from rows)")
ds_med = {ds: statistics.median([r["delta_pct"] for r in rows
                                 if r["dataset"] == ds]) for ds in M.DATASETS}
for ds, v in ds_med.items():
    print(f"  {ds:6s} {v:+.4f}%")
ratio = 21.0 / statistics.median(deltas)
print(f"  ratio  21.0 / {statistics.median(deltas):.4f} = {ratio:.4f}"
      f"  -> rounds to {round(ratio)}")

print()
print("HEADLINE ARITHMETIC VERIFIED" if not fails
      else f"FAILURES: {fails}")
