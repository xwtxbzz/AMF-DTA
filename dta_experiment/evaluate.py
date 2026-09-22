from __future__ import annotations

import argparse

from .config import load_config
from .train import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate_checkpoint(config, args.checkpoint, split=args.split)


if __name__ == "__main__":
    main()
