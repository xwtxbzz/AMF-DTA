from __future__ import annotations

import argparse
import json
from pathlib import Path

from dta_experiment.config import load_config
from dta_experiment.train import train


DEFAULT_CONFIGS = [
    "configs/davis_deepdta.yaml",
    "configs/kiba_deepdta.yaml",
    "configs/metz_deepdta.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all configs.")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience for all configs.")
    parser.add_argument("--output-suffix", default="", help="Append a suffix to each configured output_dir.")
    args = parser.parse_args()

    summary = {}
    for config_path in args.configs:
        config = load_config(config_path)
        if args.epochs is not None:
            config["train"]["epochs"] = args.epochs
        if args.patience is not None:
            config["train"]["early_stopping_patience"] = args.patience
        if args.output_suffix:
            config["train"]["output_dir"] = f"{config['train']['output_dir']}{args.output_suffix}"
        name = config["data"]["name"]
        print(f"===== Running {name}: {config_path} =====")
        summary[name] = train(config)

    summary_name = f"summary_metrics{args.output_suffix}.json" if args.output_suffix else "summary_metrics.json"
    output_path = Path("outputs") / summary_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
