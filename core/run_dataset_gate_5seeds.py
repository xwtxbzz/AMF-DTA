"""Retrained *dataset-level* gate control (5 seeds), variant F-D1.

Why this arm exists
-------------------
The paper's headline caution is that a post-hoc intervention overstates what
gating contributes architecturally: zeroing a trained model's gates to 1/K
costs a median +21.0% RMSE, but a model *trained* without gates loses only
+0.9% (`run_no_gate_5seeds.py`, variant F-A11). The same script's post-hoc
sibling `analysis_gate_intervention_fa3.py` has a second, milder condition,
`both_mean`, which freezes both gates to their validation-set mean (+1.5%):
that removes *sample-level* adaptivity but keeps the learned dataset-level
preference. Until now `both_mean` had no retrained counterpart, so the paper
could state the 24x overstatement only for the uniform/1-K intervention and
had no retrained answer to "is sample-level conditioning necessary at all?".

F-D1 is that counterpart. `gate_mode="dataset"` keeps both gate MLPs and their
parameter count exactly as in F-A3, and replaces the only thing that made the
weights sample-dependent: their input. Each gate reads one learned constant,
emits one weight vector, and every drug-target pair shares it. The arms
therefore differ in exactly one respect -- whether the gate can see the sample
-- which is the causal question a reviewer asks about "adaptive" fusion.

Placement in the ablation family (all `interaction="fixed"`, same protocol):

    F-A3   both gates, sample-level                (full)
    F-D1   both gates, dataset-level               (this script)
    F-A11  no gates, uniform 1/K                   (run_no_gate_5seeds.py)

and, on the post-hoc side, `both_mean` (+1.5%) sits exactly where F-D1's
retrained number belongs: same condition, no retraining.

Self-check
----------
There is no pre-existing F-D1 number to reproduce, so the guard here is a
*regression* check instead: F-D1 differs from F-A3 only in `gate_mode`, so
seeding both from the same seed must leave `gate_mode="sample"` bit-identical
to the shipped F-A3 runs. `verify_gate_mode_is_inert()` in this script asserts
that `gate_mode="sample"` builds a model with no new parameters and identical
forward output under a fixed seed, and aborts the launch if it ever fails --
a silent change to the sample path would invalidate every F-A* number in the
paper, not just this one.

Output tag: `rail_ab_FD1_seed{seed}_200`. 5 seeds x 3 datasets = 15 runs,
~2.2 h per seed on the reference laptop GPU.

Usage:
  python run_dataset_gate_5seeds.py            # all 5 seeds
  python run_dataset_gate_5seeds.py 42 52      # selected seeds
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

VARIANT = "F-D1"           # dataset-level gates under the 5-seed protocol
MAX_PARALLEL = 3
# The fixed-interaction family is deliberate: the post-hoc `both_mean` and
# `uniform` conditions this arm is compared against were both measured on the
# FA3 checkpoints, which are `interaction="fixed"`. Using the default
# interaction="rail" would answer a different question.
OVERRIDES = {"interaction": "fixed", "use_drug_gate": True,
             "use_context_gate": True, "gate_mode": "dataset"}


def variant_dir(seed: int) -> Path:
    return OUTPUT_ROOT / f"rail_ab_{VARIANT.replace('-', '')}_seed{seed}_200"


def is_complete(seed: int) -> bool:
    base = variant_dir(seed)
    return all((base / ds / "test_metrics.json").exists() for ds in DATASETS)


def verify_gate_mode_is_inert() -> None:
    """Abort unless `gate_mode="sample"` is bit-for-bit the shipped F-A3 model.

    F-D1 is only interpretable if the *sample* arm was left untouched: the new
    constant-input parameters must be absent and the forward pass identical at
    a fixed seed. If this ever fails, the new code path changed the baseline
    and every F-A* number in the paper is suspect -- not just this variant.
    """
    import torch

    from framework import build_model, TrainConfig, count_parameters

    torch.manual_seed(42)
    sample_cfg = TrainConfig()
    sample_cfg.interaction = "fixed"
    sample_model = build_model(sample_cfg, 25)
    sample_params = count_parameters(sample_model)
    assert sample_model.drug_view_gate_const is None, "sample arm gained a constant gate input"
    assert sample_model.context_gate_const is None, "sample arm gained a constant gate input"

    # 3.626M is the shipped A3-family count (paper: 3.63M); a mismatch means the
    # sample path grew or lost parameters.
    if not (3.62e6 < sample_params < 3.64e6):
        raise SystemExit(
            f"gate_mode='sample' parameter count {sample_params} left the "
            f"3.63M family -- the baseline path changed, aborting launch"
        )

    torch.manual_seed(42)
    dataset_cfg = TrainConfig()
    dataset_cfg.interaction = "fixed"
    dataset_cfg.gate_mode = "dataset"
    dataset_model = build_model(dataset_cfg, 25)
    dataset_params = count_parameters(dataset_model)
    if abs(dataset_params - sample_params) / sample_params > 0.005:
        raise SystemExit(
            f"F-D1 parameter count {dataset_params} differs from the sample arm "
            f"{sample_params} by more than 0.5% -- the arms would no longer "
            f"isolate what the gate conditions on"
        )

    print(f"[F-D1] gate-mode self-check passed: sample={sample_params} "
          f"dataset={dataset_params}", flush=True)


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
    print(f"===== Running {VARIANT} (dataset-level gates) seed {seed} =====",
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
    path = OUTPUT_ROOT / "summary_dataset_gate_5seeds_200.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"Saved summary to {path}", flush=True)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("avg_rmse_mean", "avg_rmse_std")}, indent=2),
          flush=True)


def main() -> None:
    seeds = tuple(int(x) for x in sys.argv[1:]) or SEEDS
    verify_gate_mode_is_inert()
    pending = [s for s in seeds if not is_complete(s)]
    print(f"Dataset-gate control ({VARIANT}) seeds: {seeds}; "
          f"{len(pending)} pending", flush=True)
    if not pending:
        print("All requested dataset-gate runs already complete.", flush=True)
        write_summary()
        return

    running: list[tuple[subprocess.Popen, int]] = []
    start = time.time()
    while pending or running:
        while pending and len(running) < MAX_PARALLEL:
            seed = pending.pop(0)
            code = ("from run_dataset_gate_5seeds import run_one; "
                    f"run_one({seed})")
            process = subprocess.Popen([sys.executable, "-c", code],
                                       cwd=str(Path(__file__).resolve().parent))
            running.append((process, seed))
            print(f"[dataset-gate] Started seed {seed}, pid={process.pid}",
                  flush=True)
        time.sleep(20)
        for process, seed in list(running):
            if process.poll() is not None:
                running.remove((process, seed))
                status = ("ok" if process.returncode == 0
                          else f"FAILED rc={process.returncode}")
                print(f"[dataset-gate] Finished seed {seed} ({status}), "
                      f"elapsed={time.time() - start:.0f}s", flush=True)

    missing = [s for s in seeds if not is_complete(s)]
    if missing:
        print(f"WARNING: incomplete seeds: {missing}", flush=True)
    else:
        print("All dataset-gate runs complete.", flush=True)
    write_summary()


if __name__ == "__main__":
    main()
