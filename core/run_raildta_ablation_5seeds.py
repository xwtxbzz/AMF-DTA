"""
RAIL-DTA ablation (Table II lower group) 5-seed 200-epoch training script.

Variants (paper Table II, lower group; reviewer request: F-A3/F-A7/F-A8/F-A9/
F-A10 under the same protocol as Table I):
  F-A3  fixed hand-designed interactions + dual gates (interaction="fixed")
  F-A7  rational main effects, fixed |diff| and product (rail_fixed_interact)
  F-A8  full RAIL (default; interaction="rail")
  F-A9  RAIL without the positive denominator (diverges -> NaN)
  F-A10 RAIL without the residual gate (residual_gate=False)

Protocol: 200 epochs + final-fit SWA (12-13 epochs), seeds 42/52/62/72/82,
same as the main five-seed runs (Table I). F-A9 is known to diverge at 40
epochs, so we run only davis/seed42 to confirm and mark it "diverges (NaN)".

Tasks: 4 variants x 5 seeds x 3 datasets = 60 + 1 (F-A9 probe) = 61 runs.
Output: rail_dta/outputs/rail_ab_{variant}_seed{seed}_200/<dataset>/

Run AFTER the main 5-seed queue (by4m3cd0w) finishes: both schedulers would
otherwise oversubscribe the 8 GB GPU.
"""

from __future__ import annotations

import dataclasses
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

EPOCHS = 200
MAX_PARALLEL = 3

# variant name -> TrainConfig overrides
VARIANTS: dict[str, dict] = {
    "F-A3": {"interaction": "fixed"},
    "F-A7": {"interaction": "rail_fixed_interact"},
    "F-A8": {"interaction": "rail"},
    "F-A10": {"interaction": "rail", "residual_gate": False},
}
# F-A9 probe: davis only, seed 42, no positive denominator
F9_PROBE = ("davis", 42)


def variant_dir(variant: str, seed: int) -> Path:
    tag = variant.replace("-", "")
    return OUTPUT_ROOT / f"rail_ab_{tag}_seed{seed}_200"


def is_complete(variant: str, seed: int) -> bool:
    base = variant_dir(variant, seed)
    return all((base / ds / "test_metrics.json").exists() for ds in DATASETS)


def run_one(variant: str, seed: int) -> None:
    from framework.train import train_all_datasets, TrainConfig

    cfg = TrainConfig()
    for key, value in VARIANTS[variant].items():
        setattr(cfg, key, value)
    cfg.seed = seed
    tag = variant.replace("-", "")
    run_tag = f"rail_ab_{tag}_seed{seed}_200"

    if is_complete(variant, seed):
        print(f"===== Skipping completed {variant}/seed{seed} =====", flush=True)
        return

    print(f"===== Running {variant}/seed{seed} for {EPOCHS} epochs =====", flush=True)
    train_all_datasets(cfg, datasets=DATASETS, output_root=OUTPUT_ROOT,
                       data_root=DATA_ROOT, run_tag=run_tag)
    print(f"===== Completed {variant}/seed{seed} =====", flush=True)


def run_f9_probe() -> None:
    """F-A9 (no positive denominator) is known to diverge; confirm on davis/42."""
    from framework.train import train_all_datasets, TrainConfig

    ds, seed = F9_PROBE
    cfg = TrainConfig(positive_denominator=False, seed=seed)
    probe_dir = OUTPUT_ROOT / "rail_ab_fa9_seed42_200"
    if (probe_dir / "diverged.json").exists() or (probe_dir / ds / "test_metrics.json").exists():
        print("F-A9 probe already done, skipping", flush=True)
        return
    print("===== Running F-A9 probe (davis/seed42, expect divergence) =====", flush=True)
    try:
        train_all_datasets(cfg, datasets=(ds,), output_root=OUTPUT_ROOT,
                           data_root=DATA_ROOT, run_tag="rail_ab_fa9_seed42_200")
        (probe_dir / "diverged.json").write_text(
            json.dumps({"diverged": False}), encoding="utf-8")
        print("F-A9 probe: did NOT diverge (unexpected)", flush=True)
    except RuntimeError as exc:
        (probe_dir / "diverged.json").write_text(
            json.dumps({"diverged": True, "error": str(exc)}), encoding="utf-8")
        print(f"F-A9 probe diverged as expected: {exc}", flush=True)


def write_summary() -> None:
    def load(ds: str, variant: str, seed: int) -> dict | None:
        p = variant_dir(variant, seed) / ds / "test_metrics.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    summary: dict = {}
    for variant in VARIANTS:
        per_ds: dict[str, dict] = {}
        for ds in DATASETS:
            vals = [load(ds, variant, s) for s in SEEDS]
            vals = [x for x in vals if x]
            if vals:
                rmses = [x["rmse"] for x in vals]
                per_ds[ds] = {
                    "rmse_mean": statistics.mean(rmses),
                    "rmse_std": statistics.stdev(rmses) if len(rmses) > 1 else 0.0,
                    "ci_mean": statistics.mean(x["ci"] for x in vals),
                    "n": len(vals),
                }
        summary[variant] = per_ds
    probe = OUTPUT_ROOT / "rail_ab_fa9_seed42_200" / "diverged.json"
    summary["F-A9"] = {"probe": json.loads(probe.read_text(encoding="utf-8"))} if probe.exists() else None
    path = OUTPUT_ROOT / "summary_raildta_ablation_5seeds_200.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {path}", flush=True)


def main() -> None:
    jobs = [(v, s) for v in VARIANTS for s in SEEDS]
    pending = [j for j in jobs if not is_complete(*j)]
    if not pending and (OUTPUT_ROOT / "rail_ab_fa9_seed42_200" / "diverged.json").exists():
        print("All RAIL-DTA ablation runs are complete!", flush=True)
        write_summary()
        return

    print(f"RAIL-DTA ablation plan: {len(pending)} variant runs + F-A9 probe", flush=True)

    log_dir = OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    remaining = list(pending)
    running: dict[tuple[str, int], tuple[subprocess.Popen, object]] = {}

    try:
        while remaining or running:
            while remaining and len(running) < MAX_PARALLEL:
                variant, seed = remaining.pop(0)
                log_path = log_dir / f"raildta_ab_{variant.replace('-','')}_seed{seed}_200.log"
                log_file = log_path.open("a", encoding="utf-8")
                code = (f"from run_raildta_ablation_5seeds import run_one; "
                        f"run_one({variant!r}, {seed})")
                process = subprocess.Popen(
                    [sys.executable, "-u", "-c", code],
                    cwd=Path.cwd(),
                    stdout=log_file, stderr=subprocess.STDOUT, text=True,
                )
                running[(variant, seed)] = (process, log_file)
                print(f"[{len(pending) - len(remaining)}/{len(pending)}] "
                      f"Started {variant}/seed{seed}, pid={process.pid}", flush=True)

            for job, (process, log_file) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log_file.close()
                print(f"[{len(pending) - len(remaining)}/{len(pending)}] "
                      f"Finished {job[0]}/seed{job[1]}, return_code={code}", flush=True)
                del running[job]

            time.sleep(10)

        # F-A9 divergence probe (single davis/seed42 run)
        run_f9_probe()

    except KeyboardInterrupt:
        print("\nInterrupted. Killing running processes...", flush=True)
        for job, (process, log_file) in running.items():
            process.kill(); log_file.close()
        sys.exit(1)

    print(f"\nAll {len(pending)} ablation jobs + F-A9 probe completed!", flush=True)
    write_summary()


if __name__ == "__main__":
    main()
