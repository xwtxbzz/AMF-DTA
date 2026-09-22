"""Five-member uncertainty-aware ensemble — paper Sec. II-F / III-G.

Members are structurally complementary (fusion-MLP width, final-fit schedule,
dropout, hidden width) and are aggregated by equal-weight averaging of test
predictions, which the paper shows is already optimal on the validation set
(inverse-RMSE and sample-disagreement weighting match it within 0.1-0.2%).
The per-member standard deviation serves as the ensemble's uncertainty
estimate (Fig. 3 calibration analysis).
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch

from .data import regression_metrics
from .train import TrainConfig

# The five members of the reported ensemble (paper Sec. III-G, Table I).
MEMBERS: dict[str, TrainConfig] = {
    "single": TrainConfig(hidden_dim=256, fc_dims=(768, 256), dropout=0.2, final_fit_epochs=12),
    "m1": TrainConfig(hidden_dim=256, fc_dims=(512, 256), dropout=0.2, final_fit_epochs=13),
    "m2": TrainConfig(hidden_dim=384, fc_dims=(768, 384), dropout=0.2, final_fit_epochs=13),
    "m3": TrainConfig(hidden_dim=256, fc_dims=(512, 256), dropout=0.25, final_fit_epochs=13),
    "m4": TrainConfig(hidden_dim=320, fc_dims=(512, 256), dropout=0.2, final_fit_epochs=13),
}


def read_predictions(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    y_true, y_pred = [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            y_true.append(float(row["y_true"]))
            y_pred.append(float(row["y_pred"]))
    return (torch.tensor(y_true, dtype=torch.float32),
            torch.tensor(y_pred, dtype=torch.float32))


def ensemble_predictions(
    output_root: Path,
    member_tags: list[str],
    dataset: str,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equal-weight average of member test predictions. All members must share
    the identical y_true ordering (fixed split); std across members is
    returned as the uncertainty estimate."""
    predictions = []
    y_true = None
    for tag in member_tags:
        member_dir = f"rail_{tag}" if seed is None else f"rail_{tag}_seed{seed}_200"
        path = output_root / member_dir / dataset / "test_predictions.csv"
        current_y, current_pred = read_predictions(path)
        if y_true is None:
            y_true = current_y
        elif not torch.equal(y_true, current_y):
            raise ValueError(f"y_true mismatch between members for {dataset}: {path}")
        predictions.append(current_pred)
    assert y_true is not None
    stacked = torch.stack(predictions, dim=0)  # [members, N]
    std = stacked.std(dim=0) if stacked.size(0) > 1 else torch.zeros_like(stacked[0])
    return y_true, stacked.mean(dim=0), std


def evaluate_ensemble(
    output_root: Path,
    datasets: tuple[str, ...] = ("davis", "kiba", "metz"),
    member_tags: list[str] | None = None,
    seed: int | None = None,
) -> dict:
    member_tags = member_tags or list(MEMBERS)
    per_dataset = {}
    for dataset in datasets:
        y_true, y_pred, _ = ensemble_predictions(output_root, member_tags, dataset, seed=seed)
        per_dataset[dataset] = regression_metrics(y_true.numpy(), y_pred.numpy())
    from .data import average_metrics
    return {"avg": average_metrics(per_dataset), "per_dataset": per_dataset,
            "members": member_tags, "seed": seed}
