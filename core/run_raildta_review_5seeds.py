"""
Review-response training queue (ICASSP 2027 revision, addresses reviewer asks):

  F-M1: learned per-pair MLP interaction (interaction="mlp_interact")
  F-M2: unconstrained degree-2 polynomial interaction ("poly_interact")
        Both are parameter-matched to RAIL (<0.4% total params, same 7D
        regressor input) and trained 5 seeds x 3 datasets, 200 epochs +
        final-fit SWA -- the Table II lower-group protocol.

  F-Ens: FingerDTA-dagger 5-member ensemble at seed 42. Member "single" is
        the existing fingerdta_seed42_200 run; members m1-m4 are trained
        here with model_type="fingerdta" and the RAIL member configs.
        Answers "evaluate against a five-member ensemble of the strongest
        baseline" (reviewer Exp. point 2).

Protocol matches run_raildta_ablation_5seeds.py: each (variant, seed) job
trains all three datasets sequentially; MAX_PARALLEL=3 on the 8 GB GPU.

Outputs:
  outputs/rail_ab_fm1_seed{seed}_200/<dataset>/
  outputs/rail_ab_fm2_seed{seed}_200/<dataset>/
  outputs/fingerdta_{m1,m2,m3,m4}_seed42_200/<dataset>/
"""
from __future__ import annotations

import csv
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

VARIANTS: dict[str, dict] = {
    "F-M1": {"interaction": "mlp_interact"},
    "F-M2": {"interaction": "poly_interact"},
}
# FingerDTA ensemble members to train (member "single" already exists as
# fingerdta_seed42_200)
FENS_MEMBERS = ("m1", "m2", "m3", "m4")


def variant_dir(variant: str, seed: int) -> Path:
    return OUTPUT_ROOT / f"rail_ab_{variant.replace('-', '')}_seed{seed}_200"


def is_complete(variant: str, seed: int) -> bool:
    base = variant_dir(variant, seed)
    return all((base / ds / "test_metrics.json").exists() for ds in DATASETS)


def fens_dir(tag: str) -> Path:
    return OUTPUT_ROOT / f"fingerdta_{tag}_seed42_200"


def fens_complete(tag: str) -> bool:
    return all((fens_dir(tag) / ds / "test_metrics.json").exists() for ds in DATASETS)


def run_one(variant: str, seed: int) -> None:
    from framework.train import train_all_datasets, TrainConfig

    cfg = TrainConfig()
    for key, value in VARIANTS[variant].items():
        setattr(cfg, key, value)
    cfg.seed = seed
    run_tag = f"rail_ab_{variant.replace('-', '')}_seed{seed}_200"

    if is_complete(variant, seed):
        print(f"===== Skipping completed {variant}/seed{seed} =====", flush=True)
        return

    print(f"===== Running {variant}/seed{seed} for {EPOCHS} epochs =====", flush=True)
    train_all_datasets(cfg, datasets=DATASETS, output_root=OUTPUT_ROOT,
                       data_root=DATA_ROOT, run_tag=run_tag)
    print(f"===== Completed {variant}/seed{seed} =====", flush=True)


def run_one_fens(tag: str) -> None:
    from framework.ensemble import MEMBERS
    from framework.train import train_all_datasets

    if fens_complete(tag):
        print(f"===== Skipping completed FingerDTA member {tag} =====", flush=True)
        return
    cfg = MEMBERS[tag]
    cfg.model_type = "fingerdta"
    cfg.seed = 42
    print(f"===== Running FingerDTA member {tag} (seed 42) =====", flush=True)
    train_all_datasets(cfg, datasets=DATASETS, output_root=OUTPUT_ROOT,
                       data_root=DATA_ROOT, run_tag=f"fingerdta_{tag}_seed42_200")
    print(f"===== Completed FingerDTA member {tag} =====", flush=True)


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

    # --- F-Ens: average the 5 FingerDTA members' seed-42 predictions --------
    def read_preds(root: Path) -> tuple[list[float], list[float]]:
        yt, yp = [], []
        with (root / "test_predictions.csv").open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                yt.append(float(row["y_true"]))
                yp.append(float(row["y_pred"]))
        return yt, yp

    from lifelines.utils import concordance_index
    fens = {}
    for ds in DATASETS:
        y, _ = read_preds(fens_dir("m1") / ds)
        preds = []
        for tag in ("single", "m1", "m2", "m3", "m4"):
            root = OUTPUT_ROOT / f"fingerdta_seed42_200" if tag == "single" else fens_dir(tag)
            yt, yp = read_preds(root / ds)
            assert yt == y, f"y_true mismatch {tag} {ds}"
            preds.append(yp)
        mu = [sum(p[i] for p in preds) / len(preds) for i in range(len(y))]
        fens[ds] = {
            "rmse": float((sum((y[i] - mu[i]) ** 2 for i in range(len(y))) / len(y)) ** 0.5),
            "ci": float(concordance_index(y, mu)),
        }
    summary["F-Ens"] = fens

    path = OUTPUT_ROOT / "summary_raildta_review_5seeds_200.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {path}", flush=True)


def main() -> None:
    jobs = [(v, s) for v in VARIANTS for s in SEEDS]
    pending = [j for j in jobs if not is_complete(*j)]
    fens_pending = [t for t in FENS_MEMBERS if not fens_complete(t)]

    if not pending and not fens_pending:
        print("All review-response runs are complete!", flush=True)
        write_summary()
        return

    print(f"Review queue: {len(pending)} variant runs + {len(fens_pending)} FingerDTA members", flush=True)

    log_dir = OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    remaining = list(pending)
    running: dict[tuple[str, int], tuple[subprocess.Popen, object]] = {}

    def spawn_variant(variant: str, seed: int) -> None:
        log_path = log_dir / f"raildta_{variant.replace('-', '')}_seed{seed}_200.log"
        log_file = log_path.open("a", encoding="utf-8")
        code = (f"from run_raildta_review_5seeds import run_one; "
                f"run_one({variant!r}, {seed})")
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", code],
            cwd=Path.cwd(),
            stdout=log_file, stderr=subprocess.STDOUT, text=True,
        )
        running[(variant, seed)] = (process, log_file)
        print(f"[{len(pending) - len(remaining)}/{len(pending)}] "
              f"Started {variant}/seed{seed}, pid={process.pid}", flush=True)

    try:
        while remaining or running or fens_pending:
            while remaining and len(running) < MAX_PARALLEL:
                variant, seed = remaining.pop(0)
                spawn_variant(variant, seed)
            while fens_pending and len(running) < MAX_PARALLEL:
                tag = fens_pending.pop(0)
                log_path = log_dir / f"fingerdta_{tag}_seed42_200.log"
                log_file = log_path.open("a", encoding="utf-8")
                code = (f"from run_raildta_review_5seeds import run_one_fens; "
                        f"run_one_fens({tag!r})")
                process = subprocess.Popen(
                    [sys.executable, "-u", "-c", code],
                    cwd=Path.cwd(),
                    stdout=log_file, stderr=subprocess.STDOUT, text=True,
                )
                running[(tag, 42)] = (process, log_file)
                print(f"[F-Ens] Started FingerDTA member {tag}, pid={process.pid}", flush=True)

            for job, (process, log_file) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log_file.close()
                print(f"Finished {job[0]}/seed{job[1]}, return_code={code}", flush=True)
                del running[job]

            time.sleep(10)

    except KeyboardInterrupt:
        print("\nInterrupted. Killing running processes...", flush=True)
        for job, (process, log_file) in running.items():
            process.kill()
            log_file.close()
        sys.exit(1)

    print("\nAll review-response jobs completed!", flush=True)
    write_summary()


if __name__ == "__main__":
    main()
