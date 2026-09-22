"""Data pipeline: fingerprints, tokenization, fixed 8:1:1 split, and the
training-set-only bias statistics (paper Sec. III-A / II-E).

Dataset layout (same as the dta_experiment framework):
    <data_root>/<dataset>/{smiles,sequence,affinity}.csv
with columns DRUGID, SMILES / TARGETID, SEQUENCE / DRUGID, TARGETID, VALUE
(column names configurable via DATASET_FILES below).

Split: 8:1:1, two-stage so val:test are disjoint 10% each. `split_frame`
offers three methods. "random" (default, and the protocol behind every number
in the paper) splits pairs, so test entities are usually seen in training.
"cold_drug" / "cold_target" split the unique entity ids instead, so a test
drug (resp. target) never appears in training. Tokenizers and all statistics
are fit on the training split only.
"""
from __future__ import annotations

import math
import os
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

try:
    from rdkit import Chem, DataStructs
    from rdkit import RDLogger
    from rdkit.Chem import AllChem, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    FEATURE_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    )
    MACCSkeys = None
    try:
        from rdkit.Chem import MACCSkeys  # noqa: F811
    except Exception:  # pragma: no cover
        pass
except Exception:  # pragma: no cover - RDKit optional; hashed fallback below
    Chem = None
    DataStructs = None
    FEATURE_MORGAN_GENERATOR = None
    MACCSkeys = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_ROOT = Path(
    os.environ.get("DTA_DATA_ROOT", Path(__file__).resolve().parent.parent / "data")
)

FINGERPRINT_DIM = 2048
FINGERPRINT_RADIUS = 2
MACCS_DIM = 167
TARGET_FINGERPRINT_DIM = 1024
TARGET_FINGERPRINT_K = 3
MAX_PROTEIN_LEN = 1000
BIAS_FEATURE_DIM = 6

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

DATASET_FILES = {
    "davis": {"smiles_file": "smiles.csv", "sequence_file": "sequence.csv",
              "affinity_file": "affinity.csv"},
    "kiba": {"smiles_file": "smiles.csv", "sequence_file": "sequence.csv",
             "affinity_file": "affinity.csv"},
    "metz": {"smiles_file": "smiles.csv", "sequence_file": "sequence.csv",
             "affinity_file": "affinity.csv"},
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Tokenizer and fingerprints
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Character-level tokenizer; 0 = padding, 1 = unknown (fit on train only)."""

    def __init__(self) -> None:
        self.stoi = {"<pad>": 0, "<unk>": 1}

    def fit(self, values: list[str]) -> None:
        chars = sorted({ch for value in values for ch in str(value)})
        self.stoi.update({ch: i + 2 for i, ch in enumerate(chars)})

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, value: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(ch, 1) for ch in str(value)[:max_len]]
        if len(ids) < max_len:
            ids.extend([0] * (max_len - len(ids)))
        return ids


def hashed_string_fingerprint(value: str, n_bits: int, k: int) -> torch.Tensor:
    """k-mer hashing into a fixed-length bit vector (zlib crc32). Used for the
    protein fingerprint view and as a no-RDKit fallback for drugs."""
    arr = np.zeros((n_bits,), dtype=np.float32)
    value = str(value)
    if not value:
        return torch.from_numpy(arr)
    if len(value) < k:
        arr[zlib.crc32(value.encode("utf-8")) % n_bits] = 1.0
        return torch.from_numpy(arr)
    for i in range(len(value) - k + 1):
        arr[zlib.crc32(value[i : i + k].encode("utf-8")) % n_bits] = 1.0
    return torch.from_numpy(arr)


def smiles_to_morgan(smiles: str) -> torch.Tensor:
    arr = np.zeros((FINGERPRINT_DIM,), dtype=np.float32)
    if Chem is None or DataStructs is None:
        return hashed_string_fingerprint(smiles, FINGERPRINT_DIM, k=3)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return torch.from_numpy(arr)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_DIM)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.from_numpy(arr)


def smiles_to_feature_morgan(smiles: str) -> torch.Tensor:
    arr = np.zeros((FINGERPRINT_DIM,), dtype=np.float32)
    if FEATURE_MORGAN_GENERATOR is None:
        return hashed_string_fingerprint(smiles, FINGERPRINT_DIM, k=3)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return torch.from_numpy(arr)
    fp = FEATURE_MORGAN_GENERATOR.GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.from_numpy(arr)


def smiles_to_maccs(smiles: str) -> torch.Tensor:
    arr = np.zeros((MACCS_DIM,), dtype=np.float32)
    if MACCSkeys is None or Chem is None:
        return hashed_string_fingerprint(smiles, MACCS_DIM, k=3)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return torch.from_numpy(arr)
    fp = MACCSkeys.GenMACCSKeys(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.from_numpy(arr)


def sequence_to_hashed_fingerprint(sequence: str) -> torch.Tensor:
    return hashed_string_fingerprint(sequence, TARGET_FINGERPRINT_DIM, TARGET_FINGERPRINT_K)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AMFDTADataset(Dataset):
    """Returns one dict per (drug, target) pair with all five views plus the
    training-set-only bias statistics. Fingerprints are cached per unique
    molecule / sequence.

    When `protein_embedding_cache` / `drug_embedding_cache` are given (frozen
    pretrained view features from `extract_pretrained_views.py`) the
    corresponding keys are added to the returned dict; otherwise they are
    absent and every existing caller is unaffected.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        protein_tokenizer: CharTokenizer,
        bias_feature_cache: dict[tuple[str, str], torch.Tensor],
        protein_embedding_cache: dict[str, torch.Tensor] | None = None,
        drug_embedding_cache: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.protein_tokenizer = protein_tokenizer
        self.smiles = self.frame["smiles"].astype(str).tolist()
        self.sequences = self.frame["sequence"].astype(str).tolist()
        self.labels = torch.tensor(
            self.frame["label"].astype(float).to_numpy(), dtype=torch.float32
        )
        self.protein_tokens = {
            value: torch.tensor(
                self.protein_tokenizer.encode(value, MAX_PROTEIN_LEN), dtype=torch.long
            )
            for value in pd.unique(pd.Series(self.sequences))
        }
        self.bias_feature_cache = bias_feature_cache
        self.protein_embedding_cache = protein_embedding_cache
        self.drug_embedding_cache = drug_embedding_cache
        if protein_embedding_cache is not None:
            missing = set(self.sequences) - set(protein_embedding_cache)
            if missing:
                raise KeyError(
                    f"{len(missing)} protein sequences have no cached embedding "
                    f"(e.g. {next(iter(missing))[:40]!r}); re-run "
                    f"extract_pretrained_views.py for this dataset"
                )
        if drug_embedding_cache is not None:
            missing = set(self.smiles) - set(drug_embedding_cache)
            if missing:
                raise KeyError(
                    f"{len(missing)} SMILES have no cached embedding "
                    f"(e.g. {next(iter(missing))[:40]!r}); re-run "
                    f"extract_pretrained_views.py for this dataset"
                )

        self._morgan_cache: dict[str, torch.Tensor] = {}
        self._feature_cache: dict[str, torch.Tensor] = {}
        self._maccs_cache: dict[str, torch.Tensor] = {}
        self._target_fp_cache: dict[str, torch.Tensor] = {}
        for smiles in set(self.smiles):
            self._morgan_cache[smiles] = smiles_to_morgan(smiles)
            self._feature_cache[smiles] = smiles_to_feature_morgan(smiles)
            self._maccs_cache[smiles] = smiles_to_maccs(smiles)
        for sequence in set(self.sequences):
            self._target_fp_cache[sequence] = sequence_to_hashed_fingerprint(sequence)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        smiles = self.smiles[idx]
        sequence = self.sequences[idx]
        item = {
            "drug_fingerprint": self._morgan_cache[smiles],
            "feature_fingerprint": self._feature_cache[smiles],
            "maccs_fingerprint": self._maccs_cache[smiles],
            "target_fingerprint": self._target_fp_cache[sequence],
            "target": self.protein_tokens[sequence],
            "bias_features": self.bias_feature_cache[(smiles, sequence)],
            "label": self.labels[idx],
        }
        if self.protein_embedding_cache is not None:
            item["protein_embedding"] = self.protein_embedding_cache[sequence]
        if self.drug_embedding_cache is not None:
            item["drug_embedding"] = self.drug_embedding_cache[smiles]
        return item


def build_bias_feature_cache(
    train_frame: pd.DataFrame, frames: list[pd.DataFrame]
) -> dict[tuple[str, str], torch.Tensor]:
    """6-dim training-set statistics (paper Sec. II-E), computed once from the
    training labels and reused for val/test:
      [drug-z, target-z, 0.5*(drug-z+target-z), drug-count-norm,
       target-count-norm, has-both]
    Unseen drugs/targets fall back to the global mean (cold-start safety)."""
    train_smiles = train_frame["smiles"].astype(str).tolist()
    train_sequences = train_frame["sequence"].astype(str).tolist()
    train_labels = train_frame["label"].astype(float).to_numpy()
    global_mean = float(np.mean(train_labels))
    global_std = float(np.std(train_labels)) if len(train_labels) > 1 else 1.0
    if global_std < 1e-6:
        global_std = 1.0

    drug_values: dict[str, list[float]] = {}
    target_values: dict[str, list[float]] = {}
    for smiles, sequence, label in zip(train_smiles, train_sequences, train_labels):
        drug_values.setdefault(smiles, []).append(float(label))
        target_values.setdefault(sequence, []).append(float(label))
    drug_mean = {k: float(np.mean(v)) for k, v in drug_values.items()}
    target_mean = {k: float(np.mean(v)) for k, v in target_values.items()}
    max_drug_count = max((len(v) for v in drug_values.values()), default=1)
    max_target_count = max((len(v) for v in target_values.values()), default=1)

    cache: dict[tuple[str, str], torch.Tensor] = {}
    for frame in frames:
        for smiles, sequence in zip(frame["smiles"].astype(str), frame["sequence"].astype(str)):
            d_count = len(drug_values.get(smiles, []))
            t_count = len(target_values.get(sequence, []))
            d_z = (drug_mean.get(smiles, global_mean) - global_mean) / global_std
            t_z = (target_mean.get(sequence, global_mean) - global_mean) / global_std
            features = np.array(
                [
                    d_z,
                    t_z,
                    0.5 * (d_z + t_z),
                    math.log1p(d_count) / math.log1p(max_drug_count),
                    math.log1p(t_count) / math.log1p(max_target_count),
                    float(d_count > 0 and t_count > 0),
                ],
                dtype=np.float32,
            )
            cache[(smiles, sequence)] = torch.from_numpy(features)
    return cache


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------


def load_frame(dataset_name: str, data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    if dataset_name not in DATASET_FILES:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    root = data_root / dataset_name
    smiles = pd.read_csv(root / "smiles.csv")
    sequences = pd.read_csv(root / "sequence.csv")
    affinity = pd.read_csv(root / "affinity.csv")
    smiles = smiles.rename(columns={"DRUGID": "drug_id", "SMILES": "smiles"})
    sequences = sequences.rename(columns={"TARGETID": "target_id", "SEQUENCE": "sequence"})
    affinity = affinity.rename(
        columns={"DRUGID": "drug_id", "TARGETID": "target_id", "VALUE": "label"}
    )
    frame = affinity.merge(smiles[["drug_id", "smiles"]], on="drug_id", how="inner")
    frame = frame.merge(sequences[["target_id", "sequence"]], on="target_id", how="inner")
    frame = frame.dropna(subset=["smiles", "sequence", "label"])
    frame["label"] = frame["label"].astype(float)
    return frame.reset_index(drop=True)


def split_frame(
    frame: pd.DataFrame, seed: int = SEED, method: str = "random",
    cold_key: str = "entity",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test.

    `method="random"` (the paper's protocol) splits *pairs* 8:1:1, so nearly
    every test drug and target is also seen in training. The two cold methods
    partition the held-out *side* 8:1:1 first and then filter the frame by
    membership, so no test drug (resp. target) group occurs in training.

    `cold_key` chooses what that partition is keyed on, and it is not a
    detail -- keying on `"id"` does **not** produce a cold split:

      * `"id"`     partition unique `drug_id` / `target_id` (this is what
                   `dta_experiment/data.py` does, and hence what the existing
                   `icasp_*cold*` baselines were measured under). The tables
                   are not injective -- Davis has 442 targets but 379 unique
                   sequences, KIBA 2111 drugs but ~2077 unique SMILES -- so a
                   held-out id can share its exact sequence/SMILES with a
                   training id. Measured on the shipped baselines this leaks
                   8.9% of Davis cold-target test pairs and 2.5% of KIBA
                   cold-drug pairs, and the leaked pairs are exactly the
                   memorisable ones. Do not read that as a bound in either
                   direction: keying on the id also selects a *different*
                   partition, so the id-keyed and entity-keyed means differ
                   by more than the leak (measured on F-A3 at seed 42,
                   cold_target 0.6678 id-keyed against 0.6223 entity-keyed --
                   the leaky run is the *worse* one). The two numbers measure
                   different test sets, not the leak's size, and no cold-start
                   figure should be quoted from the id-keyed run.
      * `"entity"` (default) partition by the `smiles` / `sequence` string
                   itself, so every row carrying a given entity lands in one
                   split. Leakage is zero by construction.

    Use `cold_key="id"` only to reproduce the older numbers.

    The random branch is byte-for-byte the previous implementation and remains
    the default: every number already reported was produced through it, and
    the `method` parameter is opt-in so those runs cannot shift.
    """
    if method == "random":
        train, temp = train_test_split(frame, train_size=TRAIN_RATIO, random_state=seed, shuffle=True)
        relative_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val, test = train_test_split(temp, train_size=relative_val, random_state=seed, shuffle=True)
        return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))

    if method not in {"cold_drug", "cold_target"}:
        raise ValueError(f"Unknown split method: {method!r}")
    if cold_key not in {"id", "entity"}:
        raise ValueError(f"cold_key must be 'id' or 'entity', got {cold_key!r}")

    is_drug = method == "cold_drug"
    id_col = "drug_id" if is_drug else "target_id"
    entity_col = "smiles" if is_drug else "sequence"
    key = id_col if cold_key == "id" else entity_col
    if key not in frame.columns:
        raise ValueError(f"Split {method!r} needs column {key!r}; frame has {list(frame.columns)}")

    unique_values = frame[key].drop_duplicates()
    train_keys, temp_keys = train_test_split(
        unique_values, train_size=TRAIN_RATIO, random_state=seed, shuffle=True
    )
    relative_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_keys, test_keys = train_test_split(
        temp_keys, train_size=relative_val, random_state=seed, shuffle=True
    )
    return (
        frame[frame[key].isin(train_keys)].reset_index(drop=True),
        frame[frame[key].isin(val_keys)].reset_index(drop=True),
        frame[frame[key].isin(test_keys)].reset_index(drop=True),
    )


@dataclass
class DataBundle:
    dataset_name: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    protein_vocab_size: int
    train_size: int
    val_size: int
    test_size: int
    # 0 when the corresponding frozen pretrained view is not in use
    protein_pretrained_dim: int = 0
    drug_pretrained_dim: int = 0


def build_dataloaders(
    dataset_name: str,
    batch_size: int = 192,
    seed: int = SEED,
    data_root: Path = DEFAULT_DATA_ROOT,
    num_workers: int = 0,
    split_method: str = "random",
    cold_key: str = "entity",
    pretrained_protein: str = "",
    pretrained_drug: str = "",
) -> DataBundle:
    frame = load_frame(dataset_name, data_root)
    train, val, test = split_frame(frame, seed, method=split_method, cold_key=cold_key)

    protein_tokenizer = CharTokenizer()
    protein_tokenizer.fit(train["sequence"].astype(str).tolist())
    bias_cache = build_bias_feature_cache(train, [train, val, test])

    # Frozen pretrained view features, cached per entity string by
    # extract_pretrained_views.py. Loaded once and shared across the three
    # loaders; the *same* file is used for every split of a dataset, so the
    # features themselves cannot leak split information.
    def load_embeddings(path: str) -> dict[str, torch.Tensor] | None:
        if not path:
            return None
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        return {k: v.float() for k, v in loaded.items()}

    protein_embeddings = load_embeddings(pretrained_protein)
    drug_embeddings = load_embeddings(pretrained_drug)

    def make_loader(df: pd.DataFrame, shuffle: bool) -> DataLoader:
        dataset = AMFDTADataset(df, protein_tokenizer, bias_cache,
                                protein_embedding_cache=protein_embeddings,
                                drug_embedding_cache=drug_embeddings)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    return DataBundle(
        dataset_name=dataset_name,
        train_loader=make_loader(train, True),
        val_loader=make_loader(val, False),
        test_loader=make_loader(test, False),
        protein_vocab_size=protein_tokenizer.vocab_size,
        train_size=len(train),
        val_size=len(val),
        test_size=len(test),
        protein_pretrained_dim=(next(iter(protein_embeddings.values())).numel()
                                if protein_embeddings else 0),
        drug_pretrained_dim=(next(iter(drug_embeddings.values())).numel()
                             if drug_embeddings else 0),
    )


# ---------------------------------------------------------------------------
# Regression metrics (identical to the paper's evaluation)
# ---------------------------------------------------------------------------


def _pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    sorter = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[sorter]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[sorter[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _pearson_corr(_rankdata(y_true), _rankdata(y_pred))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if total == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / total)


def _pearson_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = _pearson_corr(y_true, y_pred)
    return float(r * r) if not np.isnan(r) else float("nan")


def _zero_intercept_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.sum(y_pred**2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0 or total == 0:
        return float("nan")
    k = np.sum(y_true * y_pred) / denominator
    return float(1.0 - np.sum((y_true - k * y_pred) ** 2) / total)


def rm2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r2 = _pearson_r2(y_true, y_pred)
    r02 = _zero_intercept_r2(y_true, y_pred)
    if np.isnan(r2) or np.isnan(r02):
        return float("nan")
    return float(r2 * (1.0 - np.sqrt(abs(r2 * r2 - r02 * r02))))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    try:
        ci = float(concordance_index(y_true, y_pred))
    except ZeroDivisionError:
        ci = float("nan")
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2": _r2_score(y_true, y_pred),
        "rm2": rm2_score(y_true, y_pred),
        "ci": ci,
        "pearson": _pearson_corr(y_true, y_pred),
        "spearman": _spearman_corr(y_true, y_pred),
    }


def average_metrics(per_dataset: dict[str, dict[str, float]]) -> dict[str, float]:
    metric_names = ("mse", "rmse", "mae", "r2", "rm2", "ci", "pearson", "spearman")
    return {
        f"avg_{metric}": float(np.nanmean([m[metric] for m in per_dataset.values()]))
        for metric in metric_names
    }
