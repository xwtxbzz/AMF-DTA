"""
AMF-DTA 5-seed (42,52,62,72,82) 200-epoch training script.

Tasks:
  - 5 ensemble members (single, m1, m2, m3, m4) x 5 seeds = 25 runs

The FingerDTA$^\\dagger$ arm that this script used to train alongside the
members is gone, along with its artefacts. The paper's FingerDTA row is the
*cited baseline* re-implemented in the root tree, at
`outputs/{ds}_fingerdta_seed{s}_200/` (audited by `audit_baseline_rows.py`),
not a run of this driver; the ensemble it fed (`F-Ens`) has no claim in the
manuscript either. Re-adding the tag would regenerate the deleted
directories, so the tag is gone from `main()` as well.

Protocol (matches run.py / train.py paper protocol):
  - 200 epochs + final-fit SWA (12-13 epochs); reported metrics are the SWA
    state test metrics (test_metrics.json)
  - output: rail_dta/outputs/rail_{tag}_seed{seed}_200/<dataset>/

After training, summarize (`outputs/summary_raildta_5seeds_200.json`):
  - ensemble_per_seed / ensemble_avg = equal-weight ensemble of the 5 members
    per seed, then averaged over seeds
  - single_swa_per_seed = single-member SWA test_metrics, per seed

Parallelism: MAX_PARALLEL=3 (RTX 4060 8GB).
"""

from __future__ import annotations

import json
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


def member_dir(tag: str, seed: int) -> Path:
    return OUTPUT_ROOT / f"rail_{tag}_seed{seed}_200"


def is_complete(tag: str, seed: int) -> bool:
    """A run is complete when all three datasets have test_metrics.json."""
    base = member_dir(tag, seed)
    return all((base / ds / "test_metrics.json").exists() for ds in DATASETS)


def run_one(tag: str, seed: int) -> None:
    from framework.train import train_all_datasets
    from framework.ensemble import MEMBERS

    cfg = MEMBERS[tag]
    run_tag = f"rail_{tag}_seed{seed}_200"

    if is_complete(tag, seed):
        print(f"===== Skipping completed {tag}/seed{seed} =====", flush=True)
        return

    cfg.seed = seed
    print(f"===== Running {tag}/seed{seed} for {EPOCHS} epochs =====", flush=True)
    train_all_datasets(cfg, datasets=DATASETS, output_root=OUTPUT_ROOT,
                       data_root=DATA_ROOT, run_tag=run_tag)
    print(f"===== Completed {tag}/seed{seed} =====", flush=True)


def evaluate_best_state(tag: str, seed: int, dataset: str) -> dict:
    """Evaluate best_state.pt (best-validation checkpoint) on the test split.

    Mirrors train.py's test protocol exactly: dict batches, predict_batch(),
    and kiba/metz label z-scoring de-normalized by train-set mean/std
    (Davis labels are already normalized).
    """
    import torch
    from framework.data import build_dataloaders, regression_metrics
    from framework.train import build_model, predict_batch
    from framework.ensemble import MEMBERS

    cfg = MEMBERS[tag]
    cfg.seed = seed
    bundle = build_dataloaders(dataset, batch_size=cfg.batch_size, seed=cfg.seed,
                               data_root=DATA_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, bundle.protein_vocab_size).to(device)
    state = torch.load(member_dir(tag, seed) / dataset / "best_state.pt",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    if dataset in {"kiba", "metz"}:
        labels = bundle.train_loader.dataset.labels
        label_mean = labels.mean().to(device)
        label_std = labels.std(unbiased=False).clamp_min(1.0e-6).to(device)
    else:  # Davis labels are already normalized
        label_mean = torch.tensor(0.0, device=device)
        label_std = torch.tensor(1.0, device=device)
    preds, y_true = [], []
    with torch.no_grad():
        for batch in bundle.test_loader:
            batch = {k: v.to(device, non_blocking=device.type == "cuda")
                     for k, v in batch.items()}
            y_pred = predict_batch(model, batch)
            preds.append((y_pred.float() * label_std + label_mean).cpu())
            y_true.append(batch["label"].cpu())
    y_pred = torch.cat(preds)
    y = torch.cat(y_true)
    return regression_metrics(y.numpy(), y_pred.numpy())


def write_summary() -> None:
    from framework.ensemble import evaluate_ensemble, MEMBERS

    summary: dict = {"protocol": "5 seeds x 200 epochs, SWA test state",
                     "seeds": list(SEEDS), "datasets": list(DATASETS)}
    # --- per-seed ensemble (Ens row) ---
    ens_per_seed = {}
    for seed in SEEDS:
        res = evaluate_ensemble(OUTPUT_ROOT, datasets=DATASETS,
                                member_tags=list(MEMBERS), seed=seed)
        ens_per_seed[str(seed)] = {ds: {k: res["per_dataset"][ds][k]
                                        for k in ("rmse", "ci")} for ds in DATASETS}
    summary["ensemble_per_seed"] = ens_per_seed
    summary["ensemble_avg"] = {
        "rmse_mean": float(sum(e["davis"]["rmse"] for e in ens_per_seed.values()) / 5),
        "ci_mean": float(sum(e["davis"]["ci"] for e in ens_per_seed.values()) / 5),
    }

    # --- single member SWA (SWA row) ---
    swa = {}
    for seed in SEEDS:
        swa[str(seed)] = {ds: {k: json.loads(
            (member_dir("single", seed) / ds / "test_metrics.json").read_text(encoding="utf-8"))[k]
            for k in ("rmse", "ci")} for ds in DATASETS}
    summary["single_swa_per_seed"] = swa

    path = OUTPUT_ROOT / "summary_raildta_5seeds_200.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {path}", flush=True)


def main() -> None:
    jobs = [(tag, seed)
            for tag in ("single", "m1", "m2", "m3", "m4")
            for seed in SEEDS]
    pending = [j for j in jobs if not is_complete(*j)]

    if not pending:
        print("All AMF-DTA 5-seed runs are complete!", flush=True)
        write_summary()
        return

    print(f"AMF-DTA 5-seed training plan: {len(pending)} remaining", flush=True)
    print(f"  Max parallel: {MAX_PARALLEL}", flush=True)

    log_dir = OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    remaining = list(pending)
    running: dict[tuple[str, int], tuple[subprocess.Popen, object]] = {}

    try:
        while remaining or running:
            while remaining and len(running) < MAX_PARALLEL:
                tag, seed = remaining.pop(0)
                log_path = log_dir / f"raildta_{tag}_seed{seed}_200.log"
                log_file = log_path.open("a", encoding="utf-8")
                code = (f"from run_amfdta_5seeds import run_one; "
                        f"run_one({tag!r}, {seed})")
                process = subprocess.Popen(
                    [sys.executable, "-u", "-c", code],
                    cwd=Path.cwd(),
                    stdout=log_file, stderr=subprocess.STDOUT, text=True,
                )
                running[(tag, seed)] = (process, log_file)
                print(f"[{len(pending) - len(remaining)}/{len(pending)}] "
                      f"Started {tag}/seed{seed}, pid={process.pid}", flush=True)

            for job, (process, log_file) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log_file.close()
                print(f"[{len(pending) - len(remaining)}/{len(pending)}] "
                      f"Finished {job[0]}/seed{job[1]}, return_code={code}", flush=True)
                del running[job]

            time.sleep(10)

    except KeyboardInterrupt:
        print("\nInterrupted. Killing running processes...", flush=True)
        for job, (process, log_file) in running.items():
            process.kill(); log_file.close()
        sys.exit(1)

    print(f"\nAll {len(pending)} jobs completed!", flush=True)
    write_summary()


if __name__ == "__main__":
    main()
