"""Retrained no-gate delta measured at the *same* checkpoint state as the
post-hoc arm.

Review W1: the headline "post-hoc overstates gating by ~12x" compares a
post-hoc delta measured on `best_state.pt` (best-validation) with a
retrained delta (A1 vs A3) taken from the final-fit SWA state. Those are
different states, and the gap between them (F-A3 0.4469 at best-val vs
0.4363 at final-fit, 2.4%) is larger than the retrained effect itself, so
the ratio mixes protocols.

The final-fit state cannot be re-derived: `train.py` builds its shuffle
with the *global* RNG (no generator), after 200 epochs of training, and
keeps the optimizer/scaler state in memory only. Replaying it would mean
re-running the whole training.

But it does not need to be: `best_state.pt` is saved for every variant, so
both arms can be evaluated at the best-validation state instead. That is
exactly the state the post-hoc intervention uses
(`analysis_gate_intervention_fa3.py`), so this script makes the two arms
state-matched by moving the *retrained* arm, not the post-hoc one.

  post-hoc arm  : F-A3 best_state, gates zeroed to 1/K, vs the same
                  checkpoint unmodified            (already measured)
  retrained arm : F-A11 best_state vs F-A3 best_state, both unmodified,
                  differing only in whether gates existed during training

Both are paired over the same 15 (seed, dataset) pairs, so the ratio
between them is now a like-for-like comparison.

Note the two numbers answer a slightly different question from Table II's
A1/A3 (final-fit); this script prints the state it used so the paper can
say which is which.

Cross-family calibration (worth knowing before reading the output as
"the sweep failed"): Table II's upper group lives in the *autoresearch*
tree and this script's runs in *rail_dta*, two implementations of the same
protocol. For the **gated** pair the two agree closely -- `a3` vs F-A3 at
seed 42 is 0.4287/0.3438/0.4987 against 0.4280/0.3430/0.4995 per dataset,
a three-dataset average of 0.42371 against 0.42350 (0.05%). For the
**no-gate** pair they do not: `a1` davis is 0.4492 against F-A11's 0.4569
(+1.72%), about ten times the cross-implementation gap seen with gates on.
So `run_no_gate_5seeds.py`'s stated self-check -- that F-A11 seed 42 must
land on A1's 0.4311 average -- is too strict to hold and should not be read
as a failure signal. The comparison this script reports is unaffected: both
of *its* arms are rail_dta runs on the same 15 pairs, so the ratio it
computes is within one implementation.

Requires the F-A11 sweep to have finished. Usage:
  python analysis_retrained_matched_state.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

from analysis_gate_intervention import DATASETS, SEEDS, run_loader
from analysis_gate_intervention_fa3 import load_config

OUT = Path(__file__).resolve().parent / "outputs"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
GATED = "rail_ab_FA3_seed{seed}_200"      # gates present
NOGATE = "rail_ab_FA11_seed{seed}_200"    # gates absent (uniform 1/K)
OUT_JSON = OUT / "retrained_matched_state.json"


def evaluate(run_dir: Path, ds: str, seed: int, device, bundle=None,
             cfg=None) -> float | None:
    """Test RMSE of one run's best-validation checkpoint.

    `bundle`/`cfg` may be passed in: the gated and no-gate arms at a given
    (seed, dataset) share a split and a batch size, so building the pipeline
    once and reusing it halves the work. That matters -- `build_dataloaders`
    rebuilds the pair-level bias cache from the raw frame on every call, and
    on KIBA (118k pairs) that dominates the runtime of the whole sweep. The
    batch size is the only field of `cfg` the bundle depends on, and
    `audit_paper_numbers.py` asserts the two configs differ *only* in the two
    gate flags, so sharing is safe.
    """
    from framework.data import build_dataloaders
    from framework.train import build_model

    state_path = run_dir / ds / "best_state.pt"
    if not state_path.exists():
        return None
    if cfg is None:
        cfg = load_config(run_dir)
    if bundle is None:
        bundle = build_dataloaders(ds, batch_size=cfg.batch_size, seed=seed,
                                   data_root=DATA_ROOT)
    model = build_model(cfg, bundle.protein_vocab_size).to(device)
    model.load_state_dict(torch.load(state_path, map_location="cpu",
                                     weights_only=False))
    model.eval()
    if ds in {"kiba", "metz"}:
        labels = bundle.train_loader.dataset.labels
        label_mean, label_std = float(labels.mean()), float(labels.std())
    else:
        label_mean, label_std = 0.0, 1.0
    y, pred = run_loader(model, bundle.test_loader, device,
                         label_mean, label_std)
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def build_bundle(ds: str, seed: int, cfg):
    """The shared test pipeline for one (seed, dataset)."""
    from framework.data import build_dataloaders

    return build_dataloaders(ds, batch_size=cfg.batch_size, seed=seed,
                             data_root=DATA_ROOT)


def check_matched_state(gated: dict[tuple[int, str], float]) -> bool:
    """Verify the gated arm here *is* the post-hoc arm's baseline.

    State-matching is the whole point of this script, and a reader cannot
    check it from the paper. It can be checked from the artefacts: the
    post-hoc run recorded its own unmodified-baseline RMSE per (seed,
    dataset) under `conditions.sample`, evaluated on the same
    `best_state.pt`. If `evaluate()` here agrees with those numbers to
    <1e-4, then the retrained arm's gated side sits on exactly the
    checkpoint, loader and denormalisation the post-hoc arm used, so the
    only difference between the two arms is the training-time presence of
    the gates.
    """
    record = json.loads((OUT / "gate_intervention_fa3.json").read_text(
        encoding="utf-8"))
    worst, n = 0.0, 0
    for (seed, ds), value in sorted(gated.items()):
        ref = record["per_seed"][str(seed)][ds]["conditions"]["sample"]["rmse"]
        worst = max(worst, abs(value - ref))
        n += 1
    ok = worst < 1e-4 and n > 0
    print(f"  [{'ok ' if ok else 'FAIL'}] gated arm reproduces the post-hoc "
          f"baseline on {n} pairs, max |diff| = {worst:.2e}")
    return ok


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    gated_values: dict[tuple[int, str], float] = {}
    for seed in SEEDS:
        gated_dir = OUT / GATED.format(seed=seed)
        nogate_dir = OUT / NOGATE.format(seed=seed)
        for ds in DATASETS:
            gated_ck = (gated_dir / ds / "best_state.pt").exists()
            nogate_ck = (nogate_dir / ds / "best_state.pt").exists()
            if not (gated_ck and nogate_ck):
                print(f"  [skip] seed {seed} {ds}: "
                      f"gated={'ok' if gated_ck else 'missing'} "
                      f"nogate={'ok' if nogate_ck else 'missing'}", flush=True)
                continue
            # One pipeline for both arms: same (seed, dataset) => same split,
            # and the audit pins the two configs to differ only in the gates.
            cfg = load_config(gated_dir)
            bundle = build_bundle(ds, seed, cfg)
            print(f"  [build] seed {seed} {ds} ready; evaluating arms",
                  flush=True)
            g = evaluate(gated_dir, ds, seed, device, bundle=bundle, cfg=cfg)
            n = evaluate(nogate_dir, ds, seed, device, bundle=bundle)
            if g is None or n is None:
                print(f"  [skip] seed {seed} {ds}: state load failed",
                      flush=True)
                continue
            rows.append({"seed": seed, "dataset": ds, "gated": g,
                         "nogate": n, "delta_pct": 100 * (n - g) / g})
            gated_values[(seed, ds)] = g
            print(f"  seed {seed} {ds:6s} gated={g:.4f} nogate={n:.4f} "
                  f"delta={100 * (n - g) / g:+6.2f}%", flush=True)

    if not rows:
        print("No paired runs available yet -- has the F-A11 sweep finished?")
        return

    check_matched_state(gated_values)

    deltas = [r["delta_pct"] for r in rows]
    gated_rmse = [r["gated"] for r in rows]
    paired = wilcoxon([r["gated"] for r in rows], [r["nogate"] for r in rows])

    print(f"\n=== retrained no-gate, best-validation state, n={len(rows)} ===")
    # The mean over the *per-dataset* RMSEs -- the same statistic the
    # post-hoc arm reports as `overall.sample.rmse_mean` (the paper's
    # 0.4469), not a mean of three-dataset averages. The count is printed
    # from the data rather than hard-coded: a stale "15" on a partial sweep
    # reads an 8-pair mean as the full-sweep value.
    print(f"  gated baseline (mean of {len(gated_rmse)} per-dataset RMSEs): "
          f"{statistics.mean(gated_rmse):.4f}")
    print(f"  delta median={statistics.median(deltas):+.2f}%  "
          f"mean={statistics.mean(deltas):+.2f}%  "
          f"worse in {sum(1 for d in deltas if d > 0)}/{len(deltas)}")
    print(f"  paired Wilcoxon p={paired.pvalue:.4g}")
    for ds in DATASETS:
        v = [r["delta_pct"] for r in rows if r["dataset"] == ds]
        if v:
            print(f"  {ds:6s} median={statistics.median(v):+6.2f}% "
                  f"mean={statistics.mean(v):+.2f}%")

    # Write the canonical path only for a complete sweep. The audit treats
    # `retrained_matched_state.json` as the artefact backing the paper's
    # claim, and a partial file there would be *nearly* right -- correct
    # enough to be believed, wrong in every aggregate. A partial run gets a
    # differently named file so it can never be mistaken for the real one.
    n_expected = len(SEEDS) * len(DATASETS)
    target = OUT_JSON if len(rows) == n_expected else OUT_JSON.with_name(
        OUT_JSON.stem + ".partial.json")
    target.write_text(json.dumps({"rows": rows,
                                  "n_pairs": len(rows),
                                  "complete": len(rows) == n_expected,
                                  "median_delta_pct": statistics.median(deltas),
                                  "mean_delta_pct": statistics.mean(deltas),
                                  "p_value": float(paired.pvalue)},
                                 indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"\nSaved {target}")
    if len(rows) != n_expected:
        print(f"  PARTIAL ({len(rows)}/{n_expected} pairs) -- not the "
              f"artefact of record; do not quote these numbers.")

    if len(rows) == n_expected:
        emit_snippet(rows, float(paired.pvalue))
    else:
        # Suppressed rather than printed-with-a-warning: the fragments are
        # designed to be pasted straight into the manuscript, and a partial
        # median is close enough to the final one to survive a careless read.
        print(f"\n(partial run: {len(rows)}/{n_expected} pairs -- paste "
              f"fragments suppressed, run again when the sweep finishes)")


def emit_snippet(rows: list[dict], p_value: float) -> None:
    """Print the quoted values in the paper's own notation.

    Printed rather than transcribed by hand because four separate places in
    the manuscript repeat them (abstract, contribution list, Sec. III-D,
    conclusion) and the ratio has to stay consistent with the post-hoc median
    the same section reports (the 0.4469-baseline run, 21.0%).
    """
    post_hoc_median = 21.0
    deltas = [r["delta_pct"] for r in rows]

    def pct(x: float) -> str:
        """Unsigned: the paper always writes the direction in words ("loses
        only 1.7%", "costs a median 21.0%"), never as a leading sign."""
        return f"{-x:.1f}\\%" if x < 0 else f"{x:.1f}\\%"

    print("\n--- paper snippet (copy into Sec. III-D) ---")
    # The three ways the sweep can fail to support the sentences as written.
    # Each prints a distinct warning, because the fix differs: a sign flip
    # needs rewriting, a split direction needs the disagreement disclosed, and
    # non-significance needs the claim downgraded from "costs X%" to a
    # description of the point estimate.
    n_worse = sum(1 for d in deltas if d > 0)
    frac = n_worse / len(deltas)
    if statistics.median(deltas) < 0:
        print("  !! median delta is NEGATIVE: gating *hurts* at matched state.")
        print("     The fragments below assume the opposite sign -- rewrite,"
              " do not paste.")
    if 0.2 < frac < 0.8:
        print(f"  !! direction is split: only {n_worse}/{len(deltas)} pairs "
              f"are worse without gates.")
        print("     A median would hide this; the paper must state the split"
              " or drop the median claim.")
    if p_value > 0.05:
        print(f"  !! p={p_value:.3g} is not significant at 0.05: report the"
              " point estimate, do not")
        print("     imply the retrained cost is established. (This weakens"
              " the overstatement claim,")
        print("     it does not reverse it.)")
    print(f"  median      {pct(statistics.median(deltas))}")
    print(f"  mean        {pct(statistics.mean(deltas))}")
    # Two significant figures here too, matching the literal emitted below.
    # These are the same number, and a one-digit render above the two-digit
    # literal is exactly the kind of divergence a reader copies by hand.
    mantissa, exponent = f"{p_value:.1e}".split("e")
    print(f"  p           {mantissa}\\times10^{{{int(exponent)}}}")
    for ds in DATASETS:
        v = [r["delta_pct"] for r in rows if r["dataset"] == ds]
        if v:
            print(f"  {ds:11s} {pct(statistics.median(v))}")
    print(f"  ratio       {post_hoc_median / statistics.median(deltas):.0f}x "
          f"({post_hoc_median:.1f} / {statistics.median(deltas):.1f})")

    # The same two values appear in four places, phrased differently. Printed
    # as whole fragments so the substitution is a paste rather than four
    # hand-computed roundings that can disagree.
    med = statistics.median(deltas)
    ratio = round(post_hoc_median / med)
    # `\sim` on all four, not just the abstract. The printed operands are
    # one-decimal (21.0 / 0.9) and their quotient is 23.3 -> 23, while the
    # quotient of the underlying measurements is 24.08 -> 24. A bare "24x" is
    # therefore off by one against the paper's own numbers -- the first thing
    # a reviewer recomputes -- whereas "~24x" is right under both readings.
    print("\n--- paste targets (all four must round the same way) ---")
    print(f"  abstract   : loses only {pct(med)}: post-hoc interventions "
          f"overstate gating's\n               architectural value by "
          f"$\\sim{ratio}\\times$")
    print(f"  contribution: loses only {pct(med)}, so a post-hoc intervention "
          f"of this kind\n               overstates gating's architectural "
          f"value by $\\sim{ratio}\\times$")
    print(f"  III-D       : the $\\sim{ratio}\\times$ gap to the post-hoc median")
    print(f"  conclusion  : costs a median {post_hoc_median:.1f}\\% while "
          f"retraining under those\n               weights costs {pct(med)}"
          f"---a $\\sim{ratio}\\times$ overstatement")

    # Ready-to-paste RETRAINED_CLAIM for audit_paper_numbers.py.
    #
    # The audit compares the claim to the artefact at tol 5e-2 and to the
    # manuscript at tol 1e-9. The second tolerance is the binding one: the claim
    # has to be the number the manuscript *prints*, i.e. `pct()`'s one-decimal
    # rounding, not the raw statistic. Recording the raw statistic instead
    # fails by up to 0.05 on every site -- which is precisely the transcription
    # error this print exists to remove, so it is emitted as a literal rather
    # than left to be retyped.
    def r1(x: float) -> float:
        """The numeric value `pct()` renders, via the same formatting path."""
        return float(f"{x:.1f}")

    # Two significant figures, not one. The audit compares the claim to the
    # measured p at 5% relative tolerance, and a one-digit mantissa can round
    # by up to a third (mantissa 1.49 -> 1). At p=0.003357 that is the
    # difference between passing and failing, which is how this was caught.
    mantissa, exponent = f"{p_value:.1e}".split("e")
    p_rendered = float(mantissa) * 10 ** int(exponent)
    ds_site = {ds: r1(statistics.median([r["delta_pct"] for r in rows
                                         if r["dataset"] == ds]))
               for ds in DATASETS}
    claim = {"median": r1(med), "mean": r1(statistics.mean(deltas)),
             "p": p_rendered, **ds_site, "ratio": float(ratio)}
    raw = {"median": med, "mean": statistics.mean(deltas), "p": p_value,
           **{ds: statistics.median([r["delta_pct"] for r in rows
                                     if r["dataset"] == ds])
              for ds in DATASETS},
           "ratio": post_hoc_median / med}

    print("\n--- paste into audit_paper_numbers.py ---")
    print("RETRAINED_CLAIM: dict[str, float | None] = {")
    # `%g`, not `:.0e`: the exponent format collapses the two-significant-
    # figure value back to one digit (0.0034 -> 3e-03), silently discarding
    # the precision the line above went to the trouble of preserving.
    print(f'    "median": {claim["median"]}, "mean": {claim["mean"]}, '
          f'"p": {claim["p"]:g},')
    print(f'    "davis": {claim["davis"]}, "kiba": {claim["kiba"]}, '
          f'"metz": {claim["metz"]}, "ratio": {claim["ratio"]:.0f},')
    print("}")
    for name, value in claim.items():
        tol = (5e-2 if name not in ("p", "ratio") else abs(value) * 0.05)
        if abs(value - raw[name]) > tol:
            print(f"  !! {name}: claim {value:g} is {abs(value-raw[name]):.4g} "
                  f"from the measured {raw[name]:.4g}, outside the audit's "
                  f"tol {tol:.4g}")
            print("     the audit will fail; do not paste this line")


if __name__ == "__main__":
    main()
