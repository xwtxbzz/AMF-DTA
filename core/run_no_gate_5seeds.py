"""Retrained no-gate control (5 seeds) for the post-hoc vs retrained comparison.

The paper reports two numbers for "how much do the gates matter":

  * post-hoc: replacing the gates of a *trained* model by constant 1/K costs
    a median +21.0% RMSE over 15 (seed, dataset) pairs
    (`analysis_gate_intervention_fa3.py`, which reads the FA3 checkpoints so
    that the intervention shares this script's `interaction="fixed"` config;
    the earlier RAIL-config intervention answered a different question);
  * retrained: a model *trained* with the gates removed costs a median
    +0.9% over five seeds at matched state
    (`analysis_retrained_matched_state.py`, which is what makes the two
    figures comparable); Table II's F-A11 row shows +1.4% at the final-fit
    state, and the seed-42 screen A1 vs A3 is +1.7%. A1 is seed-42 only, so
    this script's seed-42 run is also a built-in correctness check: it must
    land on A1's 0.4311 avg. RMSE. A mismatch means the config below does
    not reproduce the table's A1, and the 5-seed numbers would be answering
    a different question -- check seed 42 before trusting the rest.

The gap between them (~24x, never below 9.7x across the estimator/trim
variants) is the paper's methodological point: post-hoc interventions
overstate the architectural value of gating, because the downstream MLP
partially re-learns the weighting when the gate is absent from the start.

This script trains the no-gate configuration under the *same* protocol as
Table I and the F-A* ablation -- 200 epochs + final-fit SWA, seeds
42/52/62/72/82, three datasets -- which is what lets the retrained delta be
stated over five seeds and paired like-for-like with the +21.0% post-hoc
figure on this same grid.

"No gate" is exactly uniform weighting: `model.py` substitutes a constant
1/K weight vector when `use_drug_gate` / `use_context_gate` is False, so
this variant matches the intervention's `uniform` condition as an
architecture rather than as a test-time override.

Output tag: `rail_ab_FA11_seed{seed}_200` (same layout as the other F-A*
variants). 5 seeds x 3 datasets = 15 runs.

Resumable at two levels: a seed whose three datasets are all present is
skipped, and `train_all_datasets(skip_existing=True)` skips individual
datasets within a partially finished run.

Usage:
  python run_no_gate_5seeds.py            # all 5 seeds
  python run_no_gate_5seeds.py 42 52      # selected seeds
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

DATASETS = ("davis", "kiba", "metz")
SEEDS = (42, 52, 62, 72, 82)
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

VARIANT = "F-A11"          # A1 under the 5-seed protocol: no gates, fixed interactions
MAX_PARALLEL = 3
# `interaction="fixed"` is essential, not incidental: A1 lives in the
# fixed-interaction family (A3 = interaction="fixed" + both gates; A2 = A3
# minus the drug gate; A1 = A2 minus the context gate), and F-A3 reproduces A3
# (seed-42 avg RMSE 0.4235 vs the table's 0.4237). Leaving the default
# interaction="rail" would instead build an F-A8-without-gates variant and
# answer a different question.
OVERRIDES = {"interaction": "fixed", "use_drug_gate": False,
             "use_context_gate": False}


def variant_dir(seed: int) -> Path:
    return OUTPUT_ROOT / f"rail_ab_{VARIANT.replace('-', '')}_seed{seed}_200"


def is_complete(seed: int) -> bool:
    base = variant_dir(seed)
    return all((base / ds / "test_metrics.json").exists() for ds in DATASETS)


def run_one(seed: int) -> None:
    from framework.train import TrainConfig, train_all_datasets

    if is_complete(seed):
        print(f"===== Skipping completed {VARIANT}/seed{seed} =====", flush=True)
        return
    cfg = TrainConfig()
    for key, value in OVERRIDES.items():
        setattr(cfg, key, value)
    cfg.seed = seed
    run_tag = f"rail_ab_{VARIANT.replace('-', '')}_seed{seed}_200"
    print(f"===== Running {VARIANT} (no gates, uniform) seed {seed} =====",
          flush=True)
    train_all_datasets(cfg, datasets=DATASETS, output_root=OUTPUT_ROOT,
                       data_root=DATA_ROOT, run_tag=run_tag)
    print(f"===== Completed {VARIANT}/seed{seed} =====", flush=True)


def write_summary() -> None:
    out: dict = {"variant": VARIANT, "overrides": OVERRIDES, "per_dataset": {}}
    all_rmse: list[float] = []
    for ds in DATASETS:
        vals = []
        for seed in SEEDS:
            p = variant_dir(seed) / ds / "test_metrics.json"
            if p.exists():
                vals.append(json.loads(p.read_text(encoding="utf-8")))
        if not vals:
            continue
        rmses = [v["rmse"] for v in vals]
        all_rmse.extend(rmses)
        out["per_dataset"][ds] = {
            "rmse_mean": statistics.mean(rmses),
            "rmse_std": statistics.stdev(rmses) if len(rmses) > 1 else 0.0,
            "ci_mean": statistics.mean(v["ci"] for v in vals),
            "n": len(vals),
        }
    if all_rmse:
        out["avg_rmse_mean"] = statistics.mean(all_rmse)
        out["avg_rmse_std"] = (statistics.stdev(all_rmse)
                               if len(all_rmse) > 1 else 0.0)
    path = OUTPUT_ROOT / "summary_no_gate_5seeds_200.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"Saved summary to {path}", flush=True)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("avg_rmse_mean", "avg_rmse_std")}, indent=2),
          flush=True)


def main() -> None:
    seeds = tuple(int(x) for x in sys.argv[1:]) or SEEDS
    pending = [s for s in seeds if not is_complete(s)]
    print(f"No-gate control ({VARIANT}) seeds: {seeds}; "
          f"{len(pending)} pending", flush=True)
    if not pending:
        print("All requested no-gate runs already complete.", flush=True)
        write_summary()
        return

    running: list[tuple[subprocess.Popen, int]] = []
    start = time.time()
    while pending or running:
        while pending and len(running) < MAX_PARALLEL:
            seed = pending.pop(0)
            code = ("from run_no_gate_5seeds import run_one; "
                    f"run_one({seed})")
            process = subprocess.Popen([sys.executable, "-c", code],
                                       cwd=str(Path(__file__).resolve().parent))
            running.append((process, seed))
            print(f"[no-gate] Started seed {seed}, pid={process.pid}",
                  flush=True)
        time.sleep(20)
        for process, seed in list(running):
            if process.poll() is not None:
                running.remove((process, seed))
                status = ("ok" if process.returncode == 0
                          else f"FAILED rc={process.returncode}")
                print(f"[no-gate] Finished seed {seed} ({status}), "
                      f"elapsed={time.time() - start:.0f}s", flush=True)

    missing = [s for s in seeds if not is_complete(s)]
    if missing:
        print(f"WARNING: incomplete seeds: {missing}", flush=True)
    else:
        print("All no-gate runs complete.", flush=True)
    write_summary()


if __name__ == "__main__":
    main()
