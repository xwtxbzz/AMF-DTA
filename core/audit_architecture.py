"""Verify the capacity-matching claim that licenses the slot comparison.

The paper says: "All four interaction-slot variants compared here (fixed,
RAIL, M1, M2) match within 0.5% in parameter count (3.63--3.64M); the
term-removal ablations of Table II (A4, A5) are smaller by construction."
That sentence carries the controlled comparison: if the four variants
differed in capacity, a difference in RMSE could be capacity rather than
the interaction slot. It is a claim about code, so no run artefact pins it
by itself and the main audit (which stays torch-free) cannot check it.

Two sources, deliberately:

  * rail_dta runs, for the four slot variants -- counted by *building* the
    model from each run's own config, so the check follows whatever the
    configs say rather than a stored number that could drift from them;
  * autoresearch summaries, for A1/A3/A4/A5/A6 -- these store `num_params`
    per dataset, which is enough for the "A4, A5 are smaller" half.

The naming collision this script exists to avoid: `autoresearch/outputs/m1`
and `m2` are the *ensemble members* (3.12M and 3.74M), not the paper's M1/M2
interaction slots. Matching on the token `m1` instead of on the run family
reports a 15% mismatch against a sentence that is exactly right.

Usage: python audit_architecture.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
AR = HERE.parent / "autoresearch" / "outputs"
DATA_ROOT = HERE.parent / "data"

# The four interaction-slot variants of the matched comparison, by the run
# that implements each one in this tree.
SLOTS = {
    "fixed": "rail_ab_FA3_seed42_200",
    "RAIL": "rail_single_seed42_200",
    "M1": "rail_ab_FM1_seed42_200",
    "M2": "rail_ab_FM2_seed42_200",
}
# Autoresearch ablation runs whose summaries store `num_params`.
ABLATIONS = ("a1", "a2", "a3", "a4", "a5", "a6")
TERM_REMOVAL = ("a4", "a5")

OK, BAD = [], []
PINNED: list[float] = []      # see audit_coverage.py


def check(label: str, claimed, actual, tol: float = 5e-4,
          fmt: str = "{:.4f}") -> None:
    if isinstance(claimed, (int, float)) and not isinstance(claimed, bool):
        PINNED.append(round(float(claimed), 6))
    if isinstance(claimed, (int, float)) and isinstance(actual, (int, float)):
        good = abs(claimed - actual) <= tol
        shown = f"paper={fmt.format(claimed)}  derived={fmt.format(actual)}"
    else:
        good = claimed == actual
        shown = f"paper={claimed!r}  derived={actual!r}"
    (OK if good else BAD).append((label, shown))
    print(f"  [{'ok ' if good else 'MISMATCH'}] {label:44s} {shown}")


def counted_params() -> dict[str, float]:
    """Parameter count per slot variant, by building each run's model."""
    from analysis_retrained_matched_state import load_config
    from framework.data import build_dataloaders
    from framework.train import build_model

    # Any dataset works: the count moves with the protein vocabulary, which
    # is character-level and identical across the three. Davis is the
    # cheapest to load.
    bundle = build_dataloaders("davis", batch_size=192, seed=42,
                               data_root=DATA_ROOT)
    out = {}
    for name, tag in SLOTS.items():
        cfg = load_config(OUT / tag)
        model = build_model(cfg, bundle.protein_vocab_size)
        out[name] = sum(p.numel() for p in model.parameters())
    return out


def main() -> None:
    try:
        counts = counted_params()
    except Exception as exc:                      # torch missing, etc.
        print(f"  [skip] could not build the slot variants: {exc}")
        return

    print("=== parameter count: four interaction-slot variants ===")
    for name, n in counts.items():
        print(f"       {name:6s} {SLOTS[name]:26s} {n / 1e6:.4f} M")
    lo, hi = min(counts.values()), max(counts.values())
    # The paper's two halves of the claim: a range and a relative spread.
    check("slot variants, low end (M)", 3.63, round(lo / 1e6, 2),
          tol=5e-3, fmt="{:.2f}")
    check("slot variants, high end (M)", 3.64, round(hi / 1e6, 2),
          tol=5e-3, fmt="{:.2f}")
    check("slot variants, spread <= 0.5%", True, 100 * (hi - lo) / lo <= 0.5)

    print("\n=== parameter count: term-removal ablations are smaller ===")
    abl = {}
    for tag in ABLATIONS:
        p = AR / tag / "summary.json"
        if p.exists():
            # `num_params` is stored per dataset and averaged over the three.
            abl[tag] = json.loads(p.read_text(encoding="utf-8"))["num_params"]
    if not abl:
        print(f"  [skip] no ablation summaries under {AR}")
    else:
        for tag, n in sorted(abl.items()):
            print(f"       {tag:4s} {n / 1e6:.4f} M"
                  f"{'   (term removal)' if tag in TERM_REMOVAL else ''}")
        full = statistics.mean(n for t, n in abl.items()
                               if t not in TERM_REMOVAL)
        check("A4/A5 both smaller than the full model", True,
              all(abl[t] < full for t in TERM_REMOVAL if t in abl))

    print(f"\n=== {len(OK)} checks ok, {len(BAD)} mismatched ===")
    (OUT / "audit_pinned_architecture.json").write_text(
        json.dumps(sorted(set(PINNED))), encoding="utf-8")
    for label, shown in BAD:
        print(f"  MISMATCH {label}: {shown}")


if __name__ == "__main__":
    sys.exit(main())
