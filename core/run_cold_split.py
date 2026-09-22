"""Cold-split evaluation for the paper's Sec. III cold-start paragraph.

The paper's main protocol is a *random* split, so nearly every test drug and
target is also seen in training. The existing `icasp_*cold*` artefacts under
`autoresearch/outputs/` are not usable for a cold-start claim: the scripts
that produced them are absent from the working tree and from git history,
they carry no `config.json`, they are seed-42 only, and `framework.data`
records that they were built with `cold_key="id"`, which is not a cold split
-- the frame is not injective, so 8.9% of Davis cold-target test pairs and
2.5% of KIBA cold-drug pairs keep a training partner with the identical
sequence/SMILES.

This script produces the replacement: the same arms as the random-split
tables, under `cold_key="entity"`, which partitions on the sequence/SMILES
string itself and therefore leaks nothing by construction.

Arms (`run_amfdta_ablation_5seeds.py`'s F-A* naming, so the cold rows line
up with Table II row-for-row):

  F-A3   fixed hand-designed interactions + dual gates  (the reference)
  F-A8   full RAIL                                      (interaction)
  F-A11  no gates, fixed interactions                   (the gate arm)

Seed 42 only, 200 epochs, the same protocol as the rest of the paper. This
is a pilot: one seed cannot bound anything, and the script's summary records
`n_seeds` so the paper cannot quote it as if it were a five-seed result.

Usage:
  python run_cold_split.py                # all arms x splits
  python run_cold_split.py F-A8           # selected arms
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "outputs"
DATA_ROOT = HERE.parent / "data"

DATASETS = ("davis", "kiba", "metz")
SPLITS = ("cold_drug", "cold_target")
SEED = 42
EPOCHS = 200
MAX_PARALLEL = 2

# variant name -> TrainConfig overrides. F-A11 is the same configuration as
# `run_no_gate_5seeds.py`, which is what makes the pair interpretable.
ARMS: dict[str, dict] = {
    "F-A3": {"interaction": "fixed"},
    "F-A8": {"interaction": "rail"},
    "F-A11": {"interaction": "fixed", "use_drug_gate": False, "use_context_gate": False},
}


def run_tag(arm: str, split: str) -> str:
    return f"rail_cold_{arm.replace('-', '')}_{split}_seed{SEED}_{EPOCHS}"


def run_dir(arm: str, split: str) -> Path:
    return OUTPUT_ROOT / run_tag(arm, split)


def is_complete(arm: str, split: str) -> bool:
    base = run_dir(arm, split)
    return all((base / ds / "test_metrics.json").exists() for ds in DATASETS)


def run_one(arm: str, split: str) -> None:
    from framework.train import TrainConfig, train_all_datasets

    if is_complete(arm, split):
        print(f"===== Skipping completed {arm}/{split} =====", flush=True)
        return

    cfg = TrainConfig()
    for key, value in ARMS[arm].items():
        setattr(cfg, key, value)
    cfg.seed = SEED
    cfg.max_epochs = EPOCHS
    cfg.split_method = split
    cfg.cold_key = "entity"

    print(f"===== Running {arm}/{split}, seed {SEED}, {EPOCHS} epochs =====", flush=True)
    train_all_datasets(cfg, datasets=DATASETS, output_root=OUTPUT_ROOT,
                       data_root=DATA_ROOT, run_tag=run_tag(arm, split))
    print(f"===== Completed {arm}/{split} =====", flush=True)


def write_summary() -> None:
    summary: dict = {"seed": SEED, "epochs": EPOCHS, "cold_key": "entity",
                     "n_seeds": 1, "arms": {}}
    for arm in ARMS:
        for split in SPLITS:
            row: dict = {}
            rmses, cis = [], []
            for ds in DATASETS:
                p = run_dir(arm, split) / ds / "test_metrics.json"
                if not p.exists():
                    continue
                m = json.loads(p.read_text(encoding="utf-8"))
                row[ds] = {"rmse": m["rmse"], "ci": m["ci"]}
                rmses.append(m["rmse"])
                cis.append(m["ci"])
            if rmses:
                row["avg_rmse"] = statistics.mean(rmses)
                row["avg_ci"] = statistics.mean(cis)
                row["n_datasets"] = len(rmses)
            summary["arms"][f"{arm}_{split}"] = row

    # Paired deltas against the F-A3 reference, per split, so the two
    # comparisons (interaction, gating) are read off the same baseline.
    for split in SPLITS:
        ref = summary["arms"].get(f"F-A3_{split}", {})
        for arm in ("F-A8", "F-A11"):
            cur = summary["arms"].get(f"{arm}_{split}", {})
            if "avg_rmse" in ref and "avg_rmse" in cur:
                cur["delta_vs_FA3_pct"] = (
                    100.0 * (cur["avg_rmse"] - ref["avg_rmse"]) / ref["avg_rmse"])

    path = OUTPUT_ROOT / "summary_cold_split.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {path}", flush=True)


def main() -> None:
    selected = [a for a in sys.argv[1:] if a in ARMS] or list(ARMS)
    jobs = [(a, s) for a in selected for s in SPLITS]
    pending = [j for j in jobs if not is_complete(*j)]
    if not pending:
        print("All cold-split runs are complete!", flush=True)
        write_summary()
        return

    print(f"Cold-split plan: {len(pending)} runs "
          f"(arms {selected}, splits {list(SPLITS)}, seed {SEED})", flush=True)

    log_dir = OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    remaining = list(pending)
    running: dict[tuple[str, str], tuple[subprocess.Popen, object]] = {}
    started = time.time()

    try:
        while remaining or running:
            while remaining and len(running) < MAX_PARALLEL:
                arm, split = remaining.pop(0)
                log_path = log_dir / f"{run_tag(arm, split)}.log"
                log_file = log_path.open("a", encoding="utf-8")
                code = (f"from run_cold_split import run_one; "
                        f"run_one({arm!r}, {split!r})")
                process = subprocess.Popen(
                    [sys.executable, "-u", "-c", code],
                    cwd=HERE, stdout=log_file, stderr=subprocess.STDOUT, text=True,
                )
                running[(arm, split)] = (process, log_file)
                print(f"Started {arm}/{split}, pid={process.pid}", flush=True)

            for job, (process, log_file) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log_file.close()
                del running[job]
                print(f"Finished {job[0]}/{job[1]}, return_code={code} "
                      f"({(time.time() - started) / 60:.0f} min elapsed)", flush=True)

            if running:
                time.sleep(15)
    except KeyboardInterrupt:
        for process, log_file in running.values():
            process.terminate()
            log_file.close()
        raise

    write_summary()


if __name__ == "__main__":
    main()
