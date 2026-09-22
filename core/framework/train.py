"""Training protocol — paper Sec. III-A.

Fixed 200-epoch protocol (identical for AMF-DTA and every re-implemented
baseline):
  * Adam, lr 2e-4 (RAIL residual-gate alphas 10x = 2e-3, no weight decay),
    weight decay 1e-5, gradient clipping at 5.0;
  * batch size 192, FP16 mixed precision on CUDA;
  * per-dataset label z-normalization (KIBA/Metz from train statistics;
    Davis labels are already normalized);
  * best-validation-RMSE checkpoint (`best_state.pt`);
  * final fit: 12 epochs on train+val, stochastic weight averaging from
    epoch 10 (SWA state used for the reported test numbers);
  * seed 42.

The five-member equal-weight ensemble (paper Sec. III-G) is computed from the
members' test predictions by `framework.ensemble`.
"""
from __future__ import annotations

import csv
import gc
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader

from .data import (
    DataBundle,
    average_metrics,
    build_dataloaders,
    regression_metrics,
    resolve_device,
    seed_everything,
)
from .model import FingerDTA, AMFDTA


@dataclass
class TrainConfig:
    """One model configuration. `model_type` is "rail" (AMF-DTA) or
    "fingerdta" (plain baseline)."""

    model_type: str = "rail"                     # "rail" | "fingerdta"
    interaction: str = "rail"                    # paper Sec. II-C / III-E
    use_drug_gate: bool = True
    use_context_gate: bool = True
    use_bias: bool = True
    positive_denominator: bool = True            # False = F-A9 (diverges)
    residual_gate: bool = True                   # False = F-A10

    hidden_dim: int = 256
    fc_dims: tuple[int, int] = (768, 256)
    dropout: float = 0.2

    batch_size: int = 192
    learning_rate: float = 2.0e-4
    alpha_lr_multiplier: float = 10.0
    weight_decay: float = 1.0e-5
    grad_clip_norm: float = 5.0
    max_epochs: int = 200
    final_fit_epochs: int = 12
    final_fit_swa_start_epoch: int = 10
    amp: str = "fp16"                            # "fp16" | "bf16" | "off"
    seed: int = 42

    # extra hidden dimensions passed through to the model constructor
    model_kwargs: dict = field(default_factory=dict)

    @property
    def tag(self) -> str:
        """Short config fingerprint for output directory naming."""
        return f"h{self.hidden_dim}_f{'-'.join(map(str, self.fc_dims))}_d{int(self.dropout * 100)}"


def build_model(cfg: TrainConfig, protein_vocab_size: int) -> nn.Module:
    if cfg.model_type == "fingerdta":
        return FingerDTA(
            protein_vocab_size,
            hidden_dim=cfg.hidden_dim,
            fc_dims=cfg.fc_dims,
            dropout=cfg.dropout,
        )
    return AMFDTA(
        protein_vocab_size,
        hidden_dim=cfg.hidden_dim,
        fc_dims=cfg.fc_dims,
        dropout=cfg.dropout,
        interaction=cfg.interaction,
        use_drug_gate=cfg.use_drug_gate,
        use_context_gate=cfg.use_context_gate,
        use_bias=cfg.use_bias,
        positive_denominator=cfg.positive_denominator,
        residual_gate=cfg.residual_gate,
        **cfg.model_kwargs,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def predict_batch(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch["drug_fingerprint"],
        batch["feature_fingerprint"],
        batch["maccs_fingerprint"],
        batch["bias_features"],
        batch["target"],
        batch["target_fingerprint"],
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_mean: torch.Tensor,
    label_std: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    labels, preds = [], []

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = move_batch(batch, device)
            label = batch["label"]
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=scaler is not None):
                pred = predict_batch(model, batch)
                loss = criterion(pred.float(), (label.float() - label_mean) / label_std)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

            total_loss += loss.item() * label.size(0)
            labels.append(label.detach().cpu())
            preds.append((pred.float() * label_std + label_mean).detach().cpu())

    return total_loss / len(loader.dataset), torch.cat(labels), torch.cat(preds)


def _clone_state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _restore_state_dict(model: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({key: value.to(device) for key, value in state.items()})


def _write_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions(path: Path, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred"])
        for true_value, pred_value in zip(y_true.tolist(), y_pred.tolist()):
            writer.writerow([true_value, pred_value])


def train_dataset(
    dataset_name: str,
    cfg: TrainConfig,
    device: torch.device,
    output_dir: Path,
    data_root: Path | None = None,
) -> dict:
    """Train one model on one dataset with the paper protocol; returns test
    metrics and saves history.csv / best_state.pt / test_predictions.csv /
    test_metrics.json / config.json."""
    from .data import DEFAULT_DATA_ROOT
    bundle = build_dataloaders(dataset_name, batch_size=cfg.batch_size, seed=cfg.seed,
                               data_root=data_root or DEFAULT_DATA_ROOT)
    if dataset_name in {"kiba", "metz"}:
        train_labels = bundle.train_loader.dataset.labels
        label_mean = train_labels.mean().to(device)
        label_std = train_labels.std(unbiased=False).clamp_min(1.0e-6).to(device)
    else:  # Davis labels are already normalized
        label_mean = torch.tensor(0.0, device=device)
        label_std = torch.tensor(1.0, device=device)

    model = build_model(cfg, bundle.protein_vocab_size).to(device)
    # RAIL residual-gate alphas get a 10x learning rate so the rational
    # nonlinearity can "turn on" from its identity initialization; the
    # numerator/denominator coefficients keep the base rate for stability.
    alpha_params = [
        p for name, p in model.named_parameters()
        if "rail" in name and name.endswith(".alpha")
    ]
    other_params = [
        p for name, p in model.named_parameters()
        if not ("rail" in name and name.endswith(".alpha"))
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
            {"params": alpha_params, "lr": cfg.learning_rate * cfg.alpha_lr_multiplier,
             "weight_decay": 0.0},
        ],
    )
    criterion = nn.MSELoss()

    amp_dtype = torch.bfloat16 if cfg.amp == "bf16" else torch.float16
    scaler = GradScaler(device.type, enabled=device.type == "cuda" and cfg.amp == "fp16")

    history = []
    best_val_rmse = float("inf")
    best_state = None
    epoch = 0
    train_start = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        epoch_start = time.time()
        train_loss, _, _ = run_epoch(model, bundle.train_loader, criterion, device,
                                     label_mean, label_std, optimizer=optimizer,
                                     scaler=scaler, amp_dtype=amp_dtype)
        val_loss, val_y, val_pred = run_epoch(model, bundle.val_loader, criterion, device,
                                              label_mean, label_std, amp_dtype=amp_dtype)
        val_metrics = regression_metrics(val_y.numpy(), val_pred.numpy())
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "epoch_seconds": time.time() - epoch_start,
        })
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = _clone_state_dict_cpu(model)
        print(f"{dataset_name} epoch {epoch:03d} | val_rmse={val_metrics['rmse']:.6f} "
              f"val_ci={val_metrics['ci']:.6f}", flush=True)

        if math.isnan(train_loss) or train_loss > 1e6:
            raise RuntimeError(f"{dataset_name} training diverged")

    if best_state is not None:
        _restore_state_dict(model, best_state, device)

    # Final fit (train + val) with SWA; the SWA state is what gets evaluated.
    if cfg.final_fit_epochs > 0:
        final_fit_loader = DataLoader(
            ConcatDataset([bundle.train_loader.dataset, bundle.val_loader.dataset]),
            batch_size=cfg.batch_size, shuffle=True, num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        final_swa_state = None
        final_swa_count = 0
        for final_epoch in range(1, cfg.final_fit_epochs + 1):
            run_epoch(model, final_fit_loader, criterion, device, label_mean, label_std,
                      optimizer=optimizer, scaler=scaler, amp_dtype=amp_dtype)
            if final_epoch >= cfg.final_fit_swa_start_epoch:
                current_state = _clone_state_dict_cpu(model)
                if final_swa_state is None:
                    final_swa_state = current_state
                else:
                    for key, value in current_state.items():
                        if torch.is_floating_point(value):
                            final_swa_state[key].mul_(final_swa_count / (final_swa_count + 1))
                            final_swa_state[key].add_(value / (final_swa_count + 1))
                        else:
                            final_swa_state[key] = value
                final_swa_count += 1
        if final_swa_state is not None:
            _restore_state_dict(model, final_swa_state, device)

    test_loss, test_y, test_pred = run_epoch(model, bundle.test_loader, criterion, device,
                                             label_mean, label_std, amp_dtype=amp_dtype)
    test_metrics = regression_metrics(test_y.numpy(), test_pred.numpy())

    dataset_dir = output_dir / dataset_name
    elapsed = time.time() - train_start
    num_params = count_parameters(model)
    sizes = {"train": bundle.train_size, "val": bundle.val_size, "test": bundle.test_size}
    _write_history(dataset_dir / "history.csv", history)
    _write_predictions(dataset_dir / "test_predictions.csv", test_y, test_pred)
    (dataset_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    # Sidecar for train_all_datasets(skip_existing=True): everything needed to
    # rebuild a result dict for a dataset that is not retrained. `test_metrics
    # .json` alone is not enough -- it carries no cost/parameter accounting.
    (dataset_dir / "run_meta.json").write_text(
        json.dumps({"test_loss": test_loss, "epochs": epoch,
                    "training_seconds": elapsed, "num_params": num_params,
                    "sizes": sizes}, indent=2, default=str),
        encoding="utf-8",
    )
    if best_state is not None:
        torch.save(best_state, dataset_dir / "best_state.pt")
    (output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )

    return {
        "dataset": dataset_name,
        "metrics": test_metrics,
        "test_loss": test_loss,
        "epochs": epoch,
        "training_seconds": elapsed,
        "num_params": num_params,
        "sizes": sizes,
    }


def _load_existing_result(dataset_dir: Path, dataset_name: str) -> dict | None:
    """Rebuild a `train_dataset` result from artifacts already on disk, or
    None when that dataset has not been trained yet. `run_meta.json` is only
    present for runs from 2026-09-21 onward; older directories still skip on
    `test_metrics.json` but report no parameter count."""
    metrics_path = dataset_dir / "test_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(metrics, dict) or "rmse" not in metrics:
        return None
    meta: dict = {}
    meta_path = dataset_dir / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    num_params = meta.get("num_params")
    return {
        "dataset": dataset_name,
        "metrics": metrics,
        "test_loss": meta.get("test_loss"),
        "epochs": meta.get("epochs"),
        "training_seconds": float(meta.get("training_seconds") or 0.0),
        "num_params": int(num_params) if isinstance(num_params, (int, float)) else None,
        "sizes": meta.get("sizes"),
        "reused": True,
    }


def train_all_datasets(
    cfg: TrainConfig,
    datasets: tuple[str, ...] = ("davis", "kiba", "metz"),
    output_root: Path = Path("outputs"),
    data_root: Path | None = None,
    run_tag: str | None = None,
    skip_existing: bool = True,
) -> dict:
    """Train one config on all datasets (paper protocol). Returns averaged
    metrics; everything is saved under output_root/<run_tag>/.

    `skip_existing` (default) makes the sweep resumable per dataset: a dataset
    whose test_metrics.json is already present is read back from disk instead
    of retrained, so an interrupted multi-run job only redoes what it must.
    Pass False to force a full retrain (e.g. after changing the model).
    NOTE: with skip_existing, `total_seconds` and `avg_num_params` cover only
    the datasets actually trained in this call, and avg_num_params is 0 when
    every dataset was reused from a pre-run_meta.json directory."""
    seed_everything(cfg.seed)
    device = resolve_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if run_tag is None:
        run_tag = f"rail_{cfg.tag}" if cfg.model_type == "rail" else f"fingerdta_{cfg.tag}"
    output_dir = output_root / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    total_seconds = 0.0
    total_params = 0
    params_counted = 0
    for dataset_name in datasets:
        existing = (_load_existing_result(output_dir / dataset_name, dataset_name)
                    if skip_existing else None)
        if existing is not None:
            rmse = existing["metrics"].get("rmse")
            detail = f", RMSE={rmse:.4f}" if isinstance(rmse, (int, float)) else ""
            print(f"[{run_tag}] Skipping {dataset_name}: test_metrics.json "
                  f"already present{detail}", flush=True)
            results[dataset_name] = existing
            continue
        result = train_dataset(dataset_name, cfg, device, output_dir, data_root)
        results[dataset_name] = result
        total_seconds += result["training_seconds"]
        if result["num_params"] is not None:
            total_params += result["num_params"]
            params_counted += 1
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_dataset = {name: r["metrics"] for name, r in results.items()}
    summary = {"run_tag": run_tag, "avg": average_metrics(per_dataset),
               "datasets": results, "total_seconds": total_seconds,
               "avg_num_params": total_params / params_counted if params_counted else 0}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
