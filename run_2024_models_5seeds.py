"""
2024-2026 模型 5种子 200轮 训练启动脚本

需要训练的模型（按年份从近到远）:
  2026: (无)
  2025: (无)
  2024: DrugMGR

  说明: 本脚本原先还驱动另外两个 2024 模型，二者已从仓库移除（代码与结果均已删除），
        因此模型列表已裁剪到只剩 DrugMGR；本脚本仍作为 DrugMGR 的多种子驱动保留。

每模型: 3 datasets × 5 seeds (42,52,62,72,82) = 15 个训练任务
总计: 1 × 15 = 15 个任务

已有 seed42 结果的模型: 无（MambaTransDTA 已全部完成）

并行度: MAX_PARALLEL=3
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


DATASETS = ("davis", "kiba", "metz")
SEEDS = (42, 52, 62, 72, 82)
MODELS = (
    "drugmgr",    # 2024, Bioinformatics
)

EPOCHS = 200
MAX_PARALLEL = 3
BATCH_SIZE = {"drugmgr": 32}


def is_complete(dataset: str, model: str, seed: int) -> bool:
    output_dir = Path("outputs") / f"{dataset}_{model}_seed{seed}_200"
    history = output_dir / "history.csv"
    test = output_dir / "test_metrics.json"
    if not history.exists() or not test.exists():
        return False
    try:
        return len(pd.read_csv(history)) >= EPOCHS
    except Exception:
        return False


def history_count(output_dir: Path) -> int:
    history_path = output_dir / "history.csv"
    if not history_path.exists():
        return 0
    return int(len(pd.read_csv(history_path)))


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

    test_path = output_dir / "test_metrics.json"
    if test_path.exists() and history_count(output_dir) >= EPOCHS:
        print(f"===== Skipping completed {dataset}/{model_name}/seed{seed} =====", flush=True)
        return

    print(f"===== Running {dataset}/{model_name}/seed{seed} for {EPOCHS} epochs =====", flush=True)
    train(config)
    print(f"===== Completed {dataset}/{model_name}/seed{seed} =====", flush=True)


def write_summary() -> None:
    summary: dict[str, dict[str, dict[str, dict]]] = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for model_name in MODELS:
            summary[dataset][model_name] = {}
            for seed in SEEDS:
                path = Path("outputs") / f"{dataset}_{model_name}_seed{seed}_200" / "test_metrics.json"
                if path.exists():
                    summary[dataset][model_name][str(seed)] = json.loads(path.read_text(encoding="utf-8"))
    path = Path("outputs") / "summary_2024_models_5seeds_200.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {path}", flush=True)


def main() -> None:
    # Build pending list: newest model first
    pending = [
        (dataset, model, seed)
        for model in MODELS
        for dataset in DATASETS
        for seed in SEEDS
        if not is_complete(dataset, model, seed)
    ]

    if not pending:
        print("All 2024-2026 model seed runs are complete!", flush=True)
        write_summary()
        return

    total = len(pending)
    completed_total = (len(MODELS) * len(DATASETS) * len(SEEDS)) - total

    print(f"2024-2026 models training plan:", flush=True)
    print(f"  Models: {', '.join(MODELS)}", flush=True)
    print(f"  Datasets: {', '.join(DATASETS)}", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)
    print(f"  Total jobs: {total} remaining + {completed_total} completed", flush=True)
    print(f"  Max parallel: {MAX_PARALLEL}", flush=True)
    print(flush=True)

    # Use subprocess parallel execution (similar to run_comparison_models_parallel.py)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    remaining = list(pending)
    running: dict[tuple[str, str, int], tuple[subprocess.Popen, object]] = {}

    try:
        while remaining or running:
            while remaining and len(running) < MAX_PARALLEL:
                dataset, model_name, seed = remaining.pop(0)
                log_path = log_dir / f"2024_{dataset}_{model_name}_seed{seed}_200.log"
                log_file = log_path.open("a", encoding="utf-8")
                code = (
                    f"from run_2024_models_5seeds import run_one; "
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
        print("\nInterrupted. Killing running processes...", flush=True)
        for job, (process, log_file) in running.items():
            process.kill()
            log_file.close()
            print(f"Killed {job[0]}/{job[1]}/seed{job[2]}", flush=True)
        sys.exit(1)

    print(f"\nAll {total} jobs completed!", flush=True)
    write_summary()

    # Print simple ranking after completion
    print("\n=== Quick ranking (avg RMSE) ===", flush=True)
    for model_name in MODELS:
        rmses = []
        for dataset in DATASETS:
            for seed in SEEDS:
                path = Path("outputs") / f"{dataset}_{model_name}_seed{seed}_200" / "test_metrics.json"
                if path.exists():
                    data = json.loads(path.read_text(encoding="utf-8"))
                    rmses.append(data["rmse"])
        if rmses:
            avg = sum(rmses) / len(rmses)
            print(f"  {model_name}: avg_rmse={avg:.4f} (across {len(rmses)} seeds)", flush=True)


if __name__ == "__main__":
    main()
