"""AMF-DTA command-line entry points.

Usage:
    python run.py train --tag single [--datasets davis,kiba,metz] [--device auto]
    python run.py train --tag single --smoke            # 2 epochs, davis only
    python run.py train-all                             # 5 ensemble members
    python run.py ensemble [--tags single,m1,m2,m3,m4]
    python run.py params [--tag single]                 # parameter count
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from framework import build_model, count_parameters, train_all_datasets, TrainConfig
from framework.ensemble import MEMBERS, evaluate_ensemble
from framework.data import build_dataloaders

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def _config_from_tag(tag: str, interaction: str = "rail") -> TrainConfig:
    if tag == "fingerdta":
        return TrainConfig(model_type="fingerdta")
    if tag not in MEMBERS:
        raise SystemExit(f"Unknown tag {tag!r}; choose from {list(MEMBERS)} or 'fingerdta'")
    cfg = MEMBERS[tag]
    cfg.interaction = interaction
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="AMF-DTA")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="train one model on all datasets")
    train_p.add_argument("--tag", default="single")
    train_p.add_argument("--datasets", default="davis,kiba,metz")
    train_p.add_argument("--interaction", default="rail")
    train_p.add_argument("--epochs", type=int, default=200)
    train_p.add_argument("--device", default="auto")
    train_p.add_argument("--smoke", action="store_true", help="2 epochs, davis only")

    sub.add_parser("train-all", help="train the five ensemble members")
    ens_p = sub.add_parser("ensemble", help="equal-weight ensemble from saved predictions")
    ens_p.add_argument("--tags", default="single,m1,m2,m3,m4")
    ens_p.add_argument("--datasets", default="davis,kiba,metz")

    params_p = sub.add_parser("params", help="print parameter count of a config")
    params_p.add_argument("--tag", default="single")

    for p in (train_p, ens_p, params_p):
        p.add_argument("--data-root", default=None, help="overrides DTA_DATA_ROOT")

    args = parser.parse_args()

    if args.command == "train":
        cfg = _config_from_tag(args.tag, args.interaction)
        cfg.max_epochs = 2 if args.smoke else args.epochs
        datasets = ("davis",) if args.smoke else tuple(args.datasets.split(","))
        run_tag = "fingerdta" if args.tag == "fingerdta" else f"rail_{args.tag}"
        summary = train_all_datasets(cfg, datasets=datasets, output_root=OUTPUT_ROOT,
                                     data_root=args.data_root, run_tag=run_tag)
        print("---")
        for key in ("avg_mse", "avg_rmse", "avg_mae", "avg_r2", "avg_rm2",
                    "avg_ci", "avg_pearson", "avg_spearman"):
            print(f"{key}: {summary['avg'][key]:.6f}")
        print(f"num_params_M: {summary['avg_num_params'] / 1e6:.3f}")

    elif args.command == "train-all":
        for tag, cfg in MEMBERS.items():
            print(f"\n===== training member {tag} =====", flush=True)
            train_all_datasets(cfg, datasets=("davis", "kiba", "metz"), output_root=OUTPUT_ROOT,
                               data_root=args.data_root, run_tag=f"rail_{tag}")
        print("\n===== all members done; run `python run.py ensemble` =====")

    elif args.command == "ensemble":
        result = evaluate_ensemble(OUTPUT_ROOT, datasets=tuple(args.datasets.split(",")),
                                   member_tags=args.tags.split(","))
        print("--- equal-weight ensemble ---")
        for key, value in result["avg"].items():
            print(f"{key}: {value:.6f}")
        for dataset, metrics in result["per_dataset"].items():
            print(f"{dataset}: rmse={metrics['rmse']:.4f} ci={metrics['ci']:.4f}")

    elif args.command == "params":
        cfg = _config_from_tag(args.tag)
        bundle = build_dataloaders("davis", batch_size=cfg.batch_size, seed=cfg.seed)
        model = build_model(cfg, bundle.protein_vocab_size)
        print(f"{args.tag}: {count_parameters(model) / 1e6:.3f}M parameters")


if __name__ == "__main__":
    main()
