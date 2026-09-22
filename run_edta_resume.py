"""重新启动 EDTA 剩余训练 + seed62 梯度裁剪"""
from __future__ import annotations

import json, shutil, subprocess, sys, time
from pathlib import Path

import pandas as pd

HAS_NAN_CACHE: dict[str, bool] = {}  # cache NaN checks across function calls

DATASETS = ("davis", "kiba", "metz")
SEEDS = (42, 52, 62, 72, 82)
MODELS = ("ckdta", "edta")
EPOCHS = 200
MAX_PARALLEL = 3
BATCH_SIZE = {"ckdta": 32, "edta": 32}


def is_complete(dataset: str, model: str, seed: int) -> bool:
    output_dir = Path("outputs") / f"{dataset}_{model}_seed{seed}_200"
    history = output_dir / "history.csv"
    test = output_dir / "test_metrics.json"
    if not history.exists() or not test.exists():
        return False
    return len(pd.read_csv(history)) >= EPOCHS


def run_one(dataset: str, model_name: str, seed: int) -> None:
    config_path = Path("configs/generated") / f"{dataset}_{model_name}.yaml"
    if not config_path.exists():
        print(f"SKIP: config not found {config_path}", flush=True)
        return

    from dta_experiment.config import load_config
    from dta_experiment.train import train

    config = load_config(str(config_path))
    config["seed"] = seed
    output_dir = Path(f"{config['train']['output_dir']}_seed{seed}_200")
    config["train"]["output_dir"] = str(output_dir)
    config["train"]["epochs"] = EPOCHS
    config["train"]["early_stopping_patience"] = EPOCHS
    config["train"]["resume"] = True
    config["train"]["batch_size"] = BATCH_SIZE.get(model_name, 32)
    config["train"]["checkpoint_interval"] = 5
    config["data"]["pin_memory"] = True

    # 对 kiba/edta/seed62 加数值稳定措施（NaN crash 修复）
    # 根因: float16 AMP + seed62 初始权重导致梯度爆炸 → NaN
    if dataset == "kiba" and model_name == "edta" and seed == 62:
        print("===== Applying NaN fix for kiba/edta/seed62: "
              "disable AMP, grad_clip=1.0, lr=5e-4 =====", flush=True)
        config["train"]["amp"] = False
        config["train"]["grad_clip_norm"] = 1.0
        config["train"]["learning_rate"] = 5e-4

    test_path = output_dir / "test_metrics.json"
    if test_path.exists():
        print(f"===== Skipping completed {dataset}/{model_name}/seed{seed} =====", flush=True)
        return

    # 检测 history.csv 中是否有 NaN train_loss → 删除目录从头训练
    history_path = output_dir / "history.csv"
    if history_path.exists():
        try:
            h = pd.read_csv(history_path)
            if h["train_loss"].isna().any():
                cache_key = f"{dataset}/{model_name}/seed{seed}"
                if not HAS_NAN_CACHE.get(cache_key):
                    HAS_NAN_CACHE[cache_key] = True
                    print(f"===== Detected NaN in history.csv for {dataset}/{model_name}/seed{seed}, "
                          f"deleting {output_dir} and restarting fresh =====", flush=True)
                    shutil.rmtree(output_dir, ignore_errors=True)
                    # After deletion, don't resume
                    config["train"]["resume"] = False
                else:
                    print(f"===== Already cleaned {dataset}/{model_name}/seed{seed}, restarting fresh =====", flush=True)
        except Exception as e:
            print(f"===== Error reading history.csv: {e}, continuing =====", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"===== Running {dataset}/{model_name}/seed{seed} for {EPOCHS} epochs =====", flush=True)
    train(config)
    print(f"===== Completed {dataset}/{model_name}/seed{seed} =====", flush=True)


def main() -> None:
    pending = [
        (dataset, model, seed)
        for model in MODELS
        for dataset in DATASETS
        for seed in SEEDS
        if not is_complete(dataset, model, seed)
    ]

    if not pending:
        print("All jobs are complete!", flush=True)
        return

    total = len(pending)
    completed_total = (len(MODELS) * len(DATASETS) * len(SEEDS)) - total

    print(f"Remaining: {total} jobs", flush=True)
    for ds, mn, sd in pending:
        print(f"  {ds}/{mn}/seed{sd}", flush=True)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    remaining = list(pending)
    running: dict[tuple[str, str, int], tuple[subprocess.Popen, object]] = {}

    try:
        while remaining or running:
            while remaining and len(running) < MAX_PARALLEL:
                dataset, model_name, seed = remaining.pop(0)
                log_path = log_dir / f"2025_{dataset}_{model_name}_seed{seed}_200.log"
                log_file = log_path.open("a", encoding="utf-8")
                code = (
                    f"from run_edta_resume import run_one; "
                    f"run_one({dataset!r}, {model_name!r}, {seed})"
                )
                process = subprocess.Popen(
                    [sys.executable, "-u", "-c", code],
                    cwd=Path.cwd(),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running[(dataset, model_name, seed)] = (process, log_file)
                print(f"[{completed_total + len(remaining) + len(running)}/{total}] "
                      f"Started {dataset}/{model_name}/seed{seed}, pid={process.pid}", flush=True)

            for job, (process, log_file) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log_file.close()
                ds, mn, sd = job
                completed_total += 1
                print(f"[{completed_total}/{total}] Finished {ds}/{mn}/seed{sd}, "
                      f"return_code={code}", flush=True)
                del running[job]

            time.sleep(10)

    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        for job, (process, log_file) in running.items():
            process.kill()
            log_file.close()
        sys.exit(1)

    print(f"\nAll {total} jobs completed!", flush=True)


if __name__ == "__main__":
    main()
