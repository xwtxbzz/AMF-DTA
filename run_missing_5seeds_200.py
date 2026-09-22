"""
缺失 5 种子的模型补跑 200 轮 5 种子训练启动脚本。

需要训练的模型（仅有 seed42 单次 200 轮结果，无 seed52/62/72/82）:
  attentiondta, graphdta, drugban, mgraphdta
  csatdta 仅 davis（kiba/metz 无 seed42 结果，不补）

协议: seed42 沿用已有 200 轮结果（all200 / all200_stable）;
      种子 52/62/72/82 各训练 200 轮, 取验证集最佳 checkpoint (best.pt) 测试。
不包含 MolTrans、ssmdta、deattentiondta。

每模型: 3 datasets x 4 seeds = 12 个训练任务 (csatdta: davis x 4 = 4)
总计: 4 x 12 + 4 = 52 个任务
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
SEEDS = (52, 62, 72, 82)   # seed 42 已有 200 轮结果

# model -> 需要补跑的数据集 (csatdta 仅 davis 有 seed42)
MODEL_DATASETS = {
    "attentiondta": DATASETS,
    "graphdta": DATASETS,
    "drugban": DATASETS,
    "mgraphdta": DATASETS,
    "csatdta": ("davis",),
}

# 各模型 seed42 (已有 200 轮结果) 所在目录, 用于汇总
SEED42_SRC = {
    "attentiondta": {ds: f"{ds}_attentiondta_all200" for ds in DATASETS},
    "graphdta": {ds: f"{ds}_graphdta_all200" for ds in DATASETS},
    "drugban": {"davis": "davis_drugban_all200", "kiba": "kiba_drugban_all200",
                "metz": "metz_drugban_all200_stable"},
    "mgraphdta": {ds: f"{ds}_mgraphdta_all200" for ds in DATASETS},
    "csatdta": {"davis": "davis_csatdta_all200"},
}

EPOCHS = 200
MAX_PARALLEL = 1


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
    try:
        return int(len(pd.read_csv(history_path)))
    except Exception:
        return 0


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
    config["train"]["early_stopping_patience"] = EPOCHS   # 跑满 200 轮, 取验证最佳
    config["train"]["resume"] = True
    config["train"]["checkpoint_interval"] = 5
    config["data"]["pin_memory"] = True

    if test_path_exists(output_dir) and history_count(output_dir) >= EPOCHS:
        print(f"===== Skipping completed {dataset}/{model_name}/seed{seed} =====", flush=True)
        return

    print(f"===== Running {dataset}/{model_name}/seed{seed} for {EPOCHS} epochs =====", flush=True)
    train(config)
    print(f"===== Completed {dataset}/{model_name}/seed{seed} =====", flush=True)


def test_path_exists(output_dir: Path) -> bool:
    return (output_dir / "test_metrics.json").exists()


def write_summary() -> None:
    summary: dict[str, dict[str, dict[str, dict]]] = {}
    for model_name, datasets in MODEL_DATASETS.items():
        for dataset in datasets:
            summary.setdefault(dataset, {})[model_name] = {}
            # seed42 = 已有 200 轮结果
            src = SEED42_SRC.get(model_name, {}).get(dataset)
            if src:
                p = Path("outputs") / src / "test_metrics.json"
                if p.exists():
                    summary[dataset][model_name]["42"] = json.loads(p.read_text(encoding="utf-8"))
            for seed in SEEDS:
                path = Path("outputs") / f"{dataset}_{model_name}_seed{seed}_200" / "test_metrics.json"
                if path.exists():
                    summary[dataset][model_name][str(seed)] = json.loads(path.read_text(encoding="utf-8"))
    path = Path("outputs") / "summary_missing_5seeds_200.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to {path}", flush=True)


def main() -> None:
    jobs = [
        (dataset, model, seed)
        for model, datasets in MODEL_DATASETS.items()
        for dataset in datasets
        for seed in SEEDS
    ]
    pending = [j for j in jobs if not is_complete(*j)]

    if not pending:
        print("All missing-5seed runs are complete!", flush=True)
        write_summary()
        return

    total = len(pending)
    completed_total = len(jobs) - total

    print(f"Missing-5seed training plan:", flush=True)
    for model, datasets in MODEL_DATASETS.items():
        print(f"  {model}: {', '.join(datasets)}", flush=True)
    print(f"  Seeds: {SEEDS} (200 epochs, best-validation checkpoint)", flush=True)
    print(f"  Total jobs: {total} remaining + {completed_total} completed", flush=True)
    print(f"  Max parallel: {MAX_PARALLEL}", flush=True)
    print(flush=True)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    remaining = list(pending)
    running: dict[tuple[str, str, int], tuple[subprocess.Popen, object]] = {}

    try:
        while remaining or running:
            while remaining and len(running) < MAX_PARALLEL:
                dataset, model_name, seed = remaining.pop(0)
                log_path = log_dir / f"missing_{dataset}_{model_name}_seed{seed}_200.log"
                log_file = log_path.open("a", encoding="utf-8")
                code = (
                    f"from run_missing_5seeds_200 import run_one; "
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

    print("\n=== Quick ranking (avg RMSE) ===", flush=True)
    for model_name in MODEL_DATASETS:
        rmses = []
        for dataset in MODEL_DATASETS[model_name]:
            for seed in (*SEEDS,):
                path = Path("outputs") / f"{dataset}_{model_name}_seed{seed}_200" / "test_metrics.json"
                if path.exists():
                    data = json.loads(path.read_text(encoding="utf-8"))
                    rmses.append(data["rmse"])
        if rmses:
            avg = sum(rmses) / len(rmses)
            print(f"  {model_name}: avg_rmse={avg:.4f} (across {len(rmses)} seeds)", flush=True)


if __name__ == "__main__":
    main()
