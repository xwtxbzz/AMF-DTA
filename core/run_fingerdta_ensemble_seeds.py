"""Train the remaining FingerDTA ensemble members at seeds 52-82.

Reviewer R2/R3 (and the meta-review) flag the ensemble-vs-single confound in
Table I: AMF-DTA's reported row is a five-member ensemble, while the
FingerDTA row is a single model. Making that comparison matched requires a
FingerDTA ensemble averaged over the *same* five seeds.

Member "single" already exists for every seed as `fingerdta_seed{s}_200`
(the `FingerDTA_dagger` runs of run_raildta_5seeds.py). This script trains
the four remaining members m1-m4 (member configs identical to the AMF-DTA
ones in rail_dta.ensemble.MEMBERS, only `model_type="fingerdta"` and the
seed changed): 4 members x 4 seeds = 16 runs, each 3 datasets, 200 epochs.

Output tag: `fingerdta_{tag}_seed{seed}_200`, i.e. the same naming the
existing seed-42 members use, so an ensemble evaluator can glob all five
members across all five seeds.

Resumable: a (member, seed) pair is skipped when all three datasets already
have test_metrics.json.

Usage:
  python run_fingerdta_ensemble_seeds.py               # all 16 runs
  python run_fingerdta_ensemble_seeds.py 52            # pilot: seed 52 only
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DATASETS = ("davis", "kiba", "metz")
SEEDS = (52, 62, 72, 82)  # 42 already trained
MEMBER_TAGS = ("m1", "m2", "m3", "m4")
MAX_PARALLEL = 3


def member_dir(tag: str, seed: int) -> Path:
    return OUTPUT_ROOT / f"fingerdta_{tag}_seed{seed}_200"


def is_complete(tag: str, seed: int) -> bool:
    base = member_dir(tag, seed)
    return all((base / ds / "test_metrics.json").exists() for ds in DATASETS)


def run_one(tag: str, seed: int) -> None:
    from framework.ensemble import MEMBERS
    from framework.train import train_all_datasets

    if is_complete(tag, seed):
        print(f"===== Skipping completed FingerDTA {tag}/seed{seed} =====", flush=True)
        return
    cfg = MEMBERS[tag]
    cfg.model_type = "fingerdta"
    cfg.seed = seed
    print(f"===== Running FingerDTA member {tag} (seed {seed}) =====", flush=True)
    train_all_datasets(cfg, datasets=DATASETS, output_root=OUTPUT_ROOT,
                       data_root=DATA_ROOT,
                       run_tag=f"fingerdta_{tag}_seed{seed}_200")
    print(f"===== Completed FingerDTA member {tag}/seed{seed} =====", flush=True)


def main() -> None:
    seeds = tuple(int(x) for x in sys.argv[1:]) or SEEDS
    print(f"Seeds: {seeds}", flush=True)
    pending = [(t, s) for s in seeds for t in MEMBER_TAGS if not is_complete(t, s)]
    if not pending:
        print("All FingerDTA ensemble members already trained.")
        return
    print(f"{len(pending)} FingerDTA member runs to train "
          f"({len(MEMBER_TAGS)} members x {len(SEEDS)} seeds)", flush=True)

    running: list[tuple[subprocess.Popen, str]] = []
    start = time.time()
    while pending or running:
        while pending and len(running) < MAX_PARALLEL:
            tag, seed = pending.pop(0)
            code = (f"from run_fingerdta_ensemble_seeds import run_one; "
                    f"run_one({tag!r}, {seed})")
            process = subprocess.Popen([sys.executable, "-c", code],
                                       cwd=str(Path(__file__).resolve().parent))
            running.append((process, f"{tag}/seed{seed}"))
            print(f"[F-Ens] Started member {tag} seed {seed}, pid={process.pid}",
                  flush=True)
        time.sleep(20)
        for process, label in list(running):
            if process.poll() is not None:
                running.remove((process, label))
                status = "ok" if process.returncode == 0 else f"FAILED rc={process.returncode}"
                print(f"[F-Ens] Finished {label} ({status}), "
                      f"elapsed={time.time() - start:.0f}s", flush=True)

    missing = [(t, s) for t in MEMBER_TAGS for s in seeds
               if not is_complete(t, s)]
    if missing:
        print(f"WARNING: incomplete runs: {missing}", flush=True)
    else:
        print("All FingerDTA ensemble members (seeds 42-82) complete.", flush=True)


if __name__ == "__main__":
    main()
