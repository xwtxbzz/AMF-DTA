r"""F-D1 (retrained dataset-level gate) measured at the post-hoc arm's state.

Why this script exists
----------------------
The paper's interpretation of its own gates -- "most of the penalty is the
lost dataset-level weighting rather than the broken pairing", "genuine
per-sample assignment only on KIBA" -- rests on *post-hoc* probes
(`drug_mean`, `context_mean`, `both_mean`, `placebo`). That is the same
instrument the paper's headline says is biased by ~24x. Nothing in the
manuscript answers why those probes should be trusted, and a reviewer who
notices has a clean objection: the paper cannot exempt its own interventions
from its own critique.

F-D1 is the answer for the one condition that can also be formed by
retraining. `run_dataset_gate_5seeds.py` keeps both gate MLPs and their
parameter count exactly as in F-A3, and replaces the only thing that made
the weights sample-dependent -- their input -- with one learned constant.
So F-D1 is `both_mean`'s retrained counterpart: same intervention class
(remove sample-level adaptivity, keep a dataset-level preference), reached
by training rather than by freezing.

**State matching is the whole point, and it is not automatic.** The post-hoc
numbers come from `best_state.pt`; the numbers in `test_metrics.json` come
from the final-fit state. `analysis_retrained_matched_state.py`'s docstring
records the review comment (W1) that forced this distinction for the
no-gate arm: the best-val/final-fit gap (F-A3 0.4469 vs 0.4363, 2.4%) is
*larger* than the effect being measured here (~1%), so quoting a
`test_metrics.json` delta against a post-hoc delta would compare protocols,
not interventions. Both arms are therefore evaluated at the best-validation
checkpoint, on the same split and the same loader.

The F-A3 side is taken from `retrained_matched_state.json`'s `gated` column,
which is F-A3 re-evaluated through this same `evaluate()` path, rather than
from the post-hoc record's own `sample` condition. Pairing against a
re-evaluated baseline is what makes `check_matched_state` (imported from the
no-gate script) a real cross-artefact check instead of a file compared with
itself; it currently holds to 1.4e-6 across all 15 pairs.

Mode correctness needs no separate test, because loading a dataset-mode
checkpoint into a sample-mode model cannot succeed: `drug_view_gate_const`
and `context_gate_const` are `nn.Parameter`s created only under
`gate_mode="dataset"` (`framework/model.py:158,193`), so they would arrive
as unexpected keys and `evaluate()`'s strict `load_state_dict` would raise.
The numbers below are therefore computed with the dataset-mode forward --
gates reading the learned constant, not the sample -- or not at all.

**The arm covers 13 of the 15 (seed, dataset) pairs, and the gap is recorded
rather than papered over.** Seed 72's run was stopped after Davis and KIBA
and before Metz: `(72, kiba)` has a checkpoint on disk that was never
evaluated, and `(72, metz)` was never trained. A fixed `SEEDS x DATASETS`
expectation would therefore leave the arm permanently `partial`, and the
paper would have to quote a `.partial.json` -- the failure mode the rule
exists to prevent.

The resolution is to let the file state its own gap. `--finalize` writes the
canonical name and is refused unless every missing pair is written into the
record alongside it (`unevaluated_checkpoints`, `untrained_pairs`), so
"canonical" means *closed and fully disclosed*, not *nothing missing*. Two
alternatives were rejected as worse: dropping seed 72 to reach a balanced
$4\times3=12$ discards a real measurement (`(72, davis)` is evaluated) to fix
a presentational problem, and dropping the pair count entirely would let a
reader assume the neighbouring sentence's $15$. `(72, kiba)`'s RMSE is
recoverable in about a minute of forward passes on the existing checkpoint if
anyone wants the 14th pair.

The paper's sentence says "$13$ pairs" for this reason. Headroom if it ever
changes: over the four complete seeds the median is $+0.90\%$ against the
probe's $+1.46\%$ ($p=0.15$) -- same conclusion, so the choice of arm is not
load-bearing.

Usage (needs the F-D1 sweep finished and the `*.pt` checkpoints present,
so this cannot run from the released bundle):
  python analysis_dataset_gate_matched_state.py
"""
from __future__ import annotations

import argparse
import inspect
import json
import statistics
from pathlib import Path

import torch
from scipy.stats import wilcoxon

from analysis_gate_intervention import DATASETS, SEEDS
from analysis_gate_intervention_fa3 import load_config
from analysis_retrained_matched_state import OUT, build_bundle, evaluate

DATASET_GATE = "rail_ab_FD1_seed{seed}_200"
POST_HOC_JSON = OUT / "gate_intervention_fa3.json"
OUT_JSON = OUT / "dataset_gate_matched_state.json"


def load_post_hoc() -> dict:
    return json.loads(POST_HOC_JSON.read_text(encoding="utf-8"))


def load_baselines() -> tuple[dict[tuple[int, str], float], dict[tuple[int, str], float]]:
    """(F-A3 at best_state, post-hoc both_mean), both keyed by (seed, dataset).

    The F-A3 side is deliberately *not* read from the post-hoc record's own
    `sample` condition. It comes from `retrained_matched_state.json`, which
    holds F-A3 re-evaluated through the same `evaluate()` path F-D1 uses --
    so pairing F-D1 against it validates the evaluator by construction, and
    `check_matched_state` below then becomes a real cross-artefact check
    rather than comparing a file with itself. (Verified: that column matches
    the post-hoc baseline to 1.4e-6 across all 15 pairs.)
    """
    matched = json.loads(
        (OUT / "retrained_matched_state.json").read_text(encoding="utf-8"))
    post = load_post_hoc()
    base = {(r["seed"], r["dataset"]): r["gated"] for r in matched["rows"]}
    both = {(s, ds): post["per_seed"][str(s)][ds]["conditions"]["both_mean"]["rmse"]
            for (s, ds) in base}
    return base, both


def _framework_defaults() -> dict:
    from framework.data import build_dataloaders
    return {n: p.default for n, p in
            inspect.signature(build_dataloaders).parameters.items()
            if p.default is not inspect.Parameter.empty}


def effective(cfg: dict, defaults: dict) -> dict:
    """Fill framework defaults in, so configs written under different schema
    revisions compare equal when they describe the same experiment.

    This is not hypothetical: F-D1 seeds 42/52/62 were launched before
    `run_dataset_gate_5seeds.py` started recording `split_method`/`cold_key`,
    and 72/82 after. Both forms are the random split -- the values 72/82
    record are exactly `build_dataloaders`'s defaults -- so a raw key-set
    comparison reports a difference that does not exist. Comparing effective
    configs keeps the check honest about the experiment while staying blind
    to serialization drift.
    """
    out = dict(cfg)
    for key, value in defaults.items():
        out.setdefault(key, value)
    return out


def verify_isolation() -> bool:
    """Abort-worthy check that F-D1 differs from F-A3 only in `gate_mode`.

    The whole interpretation rests on the two arms being the same
    architecture at the same parameter count on the same split, so this reads
    both runs' recorded configs rather than trusting the launcher's
    OVERRIDES. A stray difference here would make F-D1 measure something else
    entirely and would invalidate the number in a way no delta would reveal.
    """
    defaults = _framework_defaults()
    drift: list[str] = []
    n = 0
    for seed in SEEDS:
        a3 = OUT / f"rail_ab_FA3_seed{seed}_200" / "config.json"
        d1 = OUT / f"rail_ab_FD1_seed{seed}_200" / "config.json"
        if not (a3.exists() and d1.exists()):
            continue
        ca, cd = (effective(json.loads(p.read_text(encoding="utf-8")), defaults)
                  for p in (a3, d1))
        n += 1
        for key in sorted(set(ca) | set(cd)):
            if key == "gate_mode":
                continue
            if ca.get(key) != cd.get(key):
                drift.append(f"seed {seed}: {key} F-A3={ca.get(key)!r} "
                             f"F-D1={cd.get(key)!r}")
    ok = not drift and n > 0
    print(f"  [{'ok ' if ok else 'FAIL'}] F-D1 differs from F-A3 only in "
          f"gate_mode ({n} seed config pairs)")
    for line in drift:
        print(f"       {line}")
    return ok


def prior_rows() -> dict[tuple[int, str], float]:
    """Reuse already-evaluated pairs so the final run is incremental.

    A full sweep re-forwards 15 test sets through freshly built dataloaders,
    which is slow enough that redoing it to pick up one late seed would cost
    more than the seed did. Keyed on (seed, dataset) only: the value is a
    deterministic function of the checkpoint and the split, so a hit is the
    same number a re-run would produce, not an approximation of it.
    """
    found: dict[tuple[int, str], float] = {}
    for path in (OUT_JSON, OUT_JSON.with_name(OUT_JSON.stem + ".partial.json")):
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            found[(row["seed"], row["dataset"])] = row["fd1"]
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="restrict to these seeds (default: all five). Use "
                         "after a late seed lands to evaluate only its pairs.")
    ap.add_argument("--no-reuse", action="store_true",
                    help="re-evaluate pairs already present in the output "
                         "JSON instead of reusing them")
    ap.add_argument("--finalize", action="store_true",
                    help="close the arm and write the canonical filename "
                         "even though pairs are missing. Refused silently if "
                         "it would hide the gap: every missing pair is "
                         "written into the record next to the numbers, so "
                         "'canonical' reads as 'closed and disclosed'.")
    ap.add_argument("--reuse-only", action="store_true",
                    help="write the record from reused pairs alone and "
                         "evaluate nothing. Use this to finalise the arm "
                         "after the sweep has been stopped: the pairs whose "
                         "checkpoints exist but were never evaluated are "
                         "left out rather than re-forwarded.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_map, post_map = load_baselines()
    verify_isolation()

    reuse = {} if args.no_reuse else prior_rows()
    seeds = args.seeds if args.seeds else list(SEEDS)
    if args.seeds:
        unknown = set(seeds) - set(SEEDS)
        if unknown:
            raise SystemExit(f"unknown seeds {sorted(unknown)}; "
                             f"the sweep defines {list(SEEDS)}")
    if reuse:
        print(f"  [reuse] {len(reuse)} pair(s) already evaluated")

    rows: list[dict] = []
    base_values: dict[tuple[int, str], float] = {}
    # Iterate every (seed, dataset) rather than only the requested seeds, so a
    # `--seeds 72 --reuse` top-up writes a complete 15-row file instead of a
    # three-row one that would look like a regressed sweep.
    for seed in SEEDS:
        run_dir = OUT / DATASET_GATE.format(seed=seed)
        for ds in DATASETS:
            if (seed, ds) not in base_map:
                print(f"  [skip] seed {seed} {ds}: no matched-state baseline",
                      flush=True)
                continue
            base, post = base_map[(seed, ds)], post_map[(seed, ds)]
            base_values[(seed, ds)] = base
            if (seed, ds) in reuse:
                v = reuse[(seed, ds)]
            elif seed not in seeds or args.reuse_only:
                continue
            else:
                if not (run_dir / ds / "best_state.pt").exists():
                    print(f"  [skip] seed {seed} {ds}: no best_state.pt",
                          flush=True)
                    continue
                cfg = load_config(run_dir)
                bundle = build_bundle(ds, seed, cfg)
                print(f"  [build] seed {seed} {ds} ready; evaluating F-D1",
                      flush=True)
                v = evaluate(run_dir, ds, seed, device, bundle=bundle, cfg=cfg)
                if v is None:
                    print(f"  [skip] seed {seed} {ds}: state load failed",
                          flush=True)
                    continue
            rows.append({"seed": seed, "dataset": ds, "fd1": v, "base": base,
                         "post_hoc": post,
                         "delta_pct": 100 * (v - base) / base,
                         "post_hoc_delta_pct": 100 * (post - base) / base})
            print(f"  seed {seed} {ds:6s} base={base:.4f} F-D1={v:.4f} "
                  f"delta={100 * (v - base) / base:+6.2f}%   "
                  f"(post-hoc both_mean {100 * (post - base) / base:+6.2f}%)",
                  flush=True)

    if not rows:
        print("No paired runs available yet -- has the F-D1 sweep finished?")
        return

    # Re-deriving the post-hoc baselines from the same loader is the check
    # that the two arms share a state; reusing the no-gate script's helper
    # keeps one definition of "matched" in the tree.
    from analysis_retrained_matched_state import check_matched_state
    check_matched_state(base_values)

    deltas = [r["delta_pct"] for r in rows]
    post_deltas = [r["post_hoc_delta_pct"] for r in rows]
    # Does the retrained arm differ from the post-hoc arm? This is the test
    # the paper's sentence needs: a non-significant result means the probe
    # and the retraining agree, which is what licenses quoting the post-hoc
    # interpretation for the dataset-level condition.
    agree = wilcoxon([r["fd1"] for r in rows], [r["post_hoc"] for r in rows])

    print(f"\n=== F-D1 retrained dataset-level gate, best-validation state, "
          f"n={len(rows)} ===")
    print(f"  F-D1    median={statistics.median(deltas):+.2f}%  "
          f"mean={statistics.mean(deltas):+.2f}%  "
          f"worse in {sum(1 for d in deltas if d > 0)}/{len(deltas)}")
    print(f"  post-hoc both_mean  median={statistics.median(post_deltas):+.2f}%  "
          f"mean={statistics.mean(post_deltas):+.2f}%  "
          f"worse in {sum(1 for d in post_deltas if d > 0)}/{len(post_deltas)}")
    print(f"  F-D1 vs post-hoc, paired Wilcoxon p={agree.pvalue:.4g}")
    for ds in DATASETS:
        v = [r["delta_pct"] for r in rows if r["dataset"] == ds]
        p = [r["post_hoc_delta_pct"] for r in rows if r["dataset"] == ds]
        if v:
            print(f"  {ds:6s} F-D1 median={statistics.median(v):+6.2f}%   "
                  f"post-hoc median={statistics.median(p):+6.2f}%")

    # Same discipline as the no-gate script: only a complete sweep may claim
    # the canonical filename, so a partial file can never be mistaken for the
    # artefact the paper quotes. "Complete" is measured against the runs that
    # exist on disk, not against SEEDS x DATASETS -- see the docstring on the
    # 13-pair arm. A pair still counts as expected only once its checkpoint
    # exists *and* its matched-state baseline is available, so absent
    # training cannot masquerade as a finished sweep; what this rule drops is
    # only the pair whose checkpoint exists but was never evaluated.
    # A checkpoint that exists but was never evaluated and a pair that was
    # never trained are different facts, and a reader checking this file needs
    # to see which is which -- so they are recorded separately.
    evaluated = {(r["seed"], r["dataset"]) for r in rows}
    on_disk = {(seed, ds) for seed in SEEDS for ds in DATASETS
               if (OUT / DATASET_GATE.format(seed=seed) / ds
                   / "best_state.pt").exists()}
    unevaluated = sorted(on_disk - evaluated)
    untrained = sorted({(seed, ds) for seed in SEEDS for ds in DATASETS
                        if (seed, ds) in base_map} - on_disk - evaluated)
    complete = not unevaluated and not untrained
    if not complete:
        print(f"  [warn] {len(unevaluated)} checkpoint(s) never evaluated: "
              f"{unevaluated}")
        print(f"  [warn] {len(untrained)} pair(s) never trained: {untrained}")
    # `--finalize` closes the arm with its gap disclosed *inside* the record,
    # rather than leaving the canonical name reserved for a sweep that can no
    # longer be finished. The distinction the filename carries is "is this the
    # artefact of record", and a file that names its own missing pairs answers
    # that honestly.
    finalized = complete or args.finalize
    target = OUT_JSON if finalized else OUT_JSON.with_name(
        OUT_JSON.stem + ".partial.json")
    target.write_text(json.dumps(
        {"rows": rows, "n_pairs": len(rows),
         "finalized": finalized, "complete": complete,
         "unevaluated_checkpoints": [[s, d] for s, d in unevaluated],
         "untrained_pairs": [[s, d] for s, d in untrained],
         "gap_note": ("seed 72's F-D1 run was stopped after Davis and KIBA, "
                      "so (72, kiba) has an unevaluated checkpoint and "
                      "(72, metz) has none"
                      if unevaluated or untrained else ""),
         "seeds": sorted({r["seed"] for r in rows}),
         "pairs_per_seed": {str(s): sorted(r["dataset"] for r in rows
                                           if r["seed"] == s)
                            for s in sorted({r["seed"] for r in rows})},
         "median_delta_pct": statistics.median(deltas),
         "mean_delta_pct": statistics.mean(deltas),
         "post_hoc_median_delta_pct": statistics.median(post_deltas),
         "agree_p_value": float(agree.pvalue)},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {target}")
    if not finalized:
        print(f"  PARTIAL ({len(rows)}/{len(on_disk) + len(untrained)} pairs) "
              f"-- not the artefact of record; do not quote these numbers.")
    else:
        # One-decimal, matching the paper's own rounding, so the audit's
        # claim literal and the printed sentence cannot diverge.
        print(f"\n--- paper fragment ---")
        print(f"  F-D1 median delta {statistics.median(deltas):+.1f}\\%  "
              f"(post-hoc both_mean {statistics.median(post_deltas):+.1f}\\%)  "
              f"p={agree.pvalue:.1e}")


if __name__ == "__main__":
    main()
