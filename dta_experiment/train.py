from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from .config import ensure_dir, load_config
from .data import build_dataloaders
from .metrics import regression_metrics
from .models import build_model
from .utils import resolve_device, save_json, set_seed


def save_history(history: list[dict], output_dir: Path) -> None:
    history_path = output_dir / "history.csv"
    try:
        pd.DataFrame(history).to_csv(history_path, index=False)
    except PermissionError:
        fallback = output_dir / "history_latest.csv"
        pd.DataFrame(history).to_csv(fallback, index=False)


def _move_batch(batch, device: torch.device, non_blocking: bool):
    if isinstance(batch, dict):
        return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}
    try:
        return batch.to(device, non_blocking=non_blocking)
    except TypeError:
        return batch.to(device)


def _predict_batch(model: nn.Module, batch):
    if isinstance(batch, dict):
        protein = batch["protein"]
        if "fingerprint" in batch:
            if "target_fingerprint" in batch:
                return model(batch["fingerprint"], protein, batch["target_fingerprint"])
            return model(batch["fingerprint"], protein)
        return model(batch["drug"], protein)
    if hasattr(model, "forward_batch"):
        return model.forward_batch(batch)
    return model(batch.x, batch.edge_index, batch.batch, batch.target)


def _batch_label(batch) -> torch.Tensor:
    if isinstance(batch, dict):
        return batch["label"]
    return batch.label.view(-1)


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    optimizer=None,
    grad_clip_norm: float | None = None,
    scaler: GradScaler | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    non_blocking: bool = False,
):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    labels = []
    preds = []

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, leave=False, disable=True):
            batch = _move_batch(batch, device, non_blocking)
            label = _batch_label(batch)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred = _predict_batch(model, batch)
                loss = criterion(pred.float(), label.float())

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    if grad_clip_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

            total_loss += loss.item() * label.size(0)
            labels.append(label.detach().cpu())
            preds.append(pred.detach().cpu())

    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(preds).numpy()
    return total_loss / len(loader.dataset), y_true, y_pred


def train(config: dict) -> dict:
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config["train"].get("cudnn_benchmark", True))
        try:
            torch.set_float32_matmul_precision(config["train"].get("matmul_precision", "high"))
        except Exception:
            pass
    output_dir = ensure_dir(config["train"]["output_dir"])

    bundle = build_dataloaders(config)
    model = build_model(config, bundle.drug_tokenizer.vocab_size, bundle.protein_tokenizer.vocab_size).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    criterion = nn.MSELoss()
    amp_enabled = bool(config["train"].get("amp", device.type == "cuda")) and device.type == "cuda"
    amp_dtype_name = str(config["train"].get("amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    scaler = GradScaler(device.type, enabled=amp_enabled and amp_dtype == torch.float16)
    non_blocking = device.type == "cuda"
    checkpoint_interval = max(1, int(config["train"].get("checkpoint_interval", 1)))
    ci_higher_is_better = bool(config.get("evaluation", {}).get("ci_higher_is_better", True))
    metric_for_best = config["train"].get("metric_for_best", "rmse")
    lower_is_better = metric_for_best in {"loss", "mse", "rmse", "mae"}
    best_score = float("inf") if lower_is_better else -float("inf")
    patience = int(config["train"].get("early_stopping_patience", 10))
    bad_epochs = 0
    history = []
    start_epoch = 1

    latest_path = output_dir / "latest.pt"
    if config["train"].get("resume", False) and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = float(config["train"]["learning_rate"])
            group["weight_decay"] = float(config["train"].get("weight_decay", 0.0))
        history = checkpoint.get("history", [])
        if checkpoint.get("scaler_state_dict") and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_score = float(checkpoint.get("best_score", best_score))
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, int(config["train"]["epochs"]) + 1):
        train_loss, _, _ = run_epoch(
            model,
            bundle.train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip_norm=float(config["train"].get("grad_clip_norm", 0.0)) or None,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            non_blocking=non_blocking,
        )
        val_loss, val_y, val_pred = run_epoch(
            model,
            bundle.val_loader,
            criterion,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            non_blocking=non_blocking,
        )
        val_metrics = regression_metrics(val_y, val_pred, ci_higher_is_better)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        save_history(history, output_dir)
        print(row)

        current = val_loss if metric_for_best == "loss" else val_metrics[metric_for_best]
        improved = current < best_score if lower_is_better else current > best_score
        if improved:
            best_score = current
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "drug_vocab": bundle.drug_tokenizer.stoi,
                    "protein_vocab": bundle.protein_tokenizer.stoi,
                    "best_epoch": epoch,
                    "best_score": best_score,
                },
                output_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        if epoch % checkpoint_interval == 0 or epoch == int(config["train"]["epochs"]):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "config": config,
                    "drug_vocab": bundle.drug_tokenizer.stoi,
                    "protein_vocab": bundle.protein_tokenizer.stoi,
                    "epoch": epoch,
                    "history": history,
                    "best_score": best_score,
                    "bad_epochs": bad_epochs,
                },
                latest_path,
            )

    save_history(history, output_dir)
    test_metrics = evaluate_checkpoint(config, output_dir / "best.pt", split="test")
    save_json(test_metrics, output_dir / "test_metrics.json")
    return test_metrics


def evaluate_checkpoint(config: dict, checkpoint_path: str | Path, split: str = "test") -> dict:
    device = resolve_device(config.get("device", "auto"))
    bundle = build_dataloaders(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(config, bundle.drug_tokenizer.vocab_size, bundle.protein_tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = {"train": bundle.train_loader, "val": bundle.val_loader, "test": bundle.test_loader}[split]
    amp_enabled = bool(config["train"].get("amp", device.type == "cuda")) and device.type == "cuda"
    amp_dtype_name = str(config["train"].get("amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    _, y_true, y_pred = run_epoch(
        model,
        loader,
        nn.MSELoss(),
        device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        non_blocking=device.type == "cuda",
    )
    metrics = regression_metrics(y_true, y_pred, bool(config.get("evaluation", {}).get("ci_higher_is_better", True)))

    if config.get("evaluation", {}).get("save_predictions", True):
        output_dir = ensure_dir(config["train"]["output_dir"])
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(output_dir / f"{split}_predictions.csv", index=False)
    print({f"{split}_{k}": v for k, v in metrics.items()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
