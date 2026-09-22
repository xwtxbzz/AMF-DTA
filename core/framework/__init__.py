"""AMF-DTA: Adaptive Multi-view Fusion Network for Drug-Target Affinity
Prediction — official PyTorch implementation (paper code release).

Quick start:
    from framework import AMFDTA, build_dataloaders, TrainConfig
    cfg = TrainConfig()
    bundle = build_dataloaders("davis")
    model = AMFDTA(bundle.protein_vocab_size)
"""
from .data import (
    DEFAULT_DATA_ROOT,
    DataBundle,
    average_metrics,
    build_dataloaders,
    regression_metrics,
    seed_everything,
)
from .model import FingerDTA, AMFDTA
from .modules import RailInteractionLayer, RationalUnit1D, RationalUnit2D
from .train import TrainConfig, build_model, count_parameters, train_all_datasets, train_dataset

__version__ = "1.0.0"
__all__ = [
    "AMFDTA",
    "FingerDTA",
    "RailInteractionLayer",
    "RationalUnit1D",
    "RationalUnit2D",
    "TrainConfig",
    "build_model",
    "build_dataloaders",
    "train_dataset",
    "train_all_datasets",
    "count_parameters",
    "regression_metrics",
    "average_metrics",
    "DataBundle",
    "DEFAULT_DATA_ROOT",
    "seed_everything",
]
