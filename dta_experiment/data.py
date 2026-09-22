from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import warnings
import zlib
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover - optional dependency for graph/fingerprint models.
    Chem = None
    AllChem = None
    DataStructs = None
try:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.loader import DataLoader as PyGDataLoader
except Exception:  # pragma: no cover - optional dependency for graph models.
    PyGData = None
    PyGDataLoader = None

from .tokenizers import CharTokenizer, KmerTokenizer

warnings.filterwarnings("ignore", message="The usage of `scatter\\(reduce='max'\\).*")


@dataclass(frozen=True)
class DTADataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    drug_tokenizer: CharTokenizer
    protein_tokenizer: CharTokenizer


class DTADataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        drug_tokenizer: CharTokenizer,
        protein_tokenizer: CharTokenizer,
        max_smiles_len: int,
        max_protein_len: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.drug_tokenizer = drug_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.max_smiles_len = max_smiles_len
        self.max_protein_len = max_protein_len
        self.smiles = self.frame["smiles"].astype(str).tolist()
        self.sequences = self.frame["sequence"].astype(str).tolist()
        self.labels = torch.tensor(self.frame["label"].astype(float).to_numpy(), dtype=torch.float32)
        self._drug_token_cache = {
            value: torch.tensor(self.drug_tokenizer.encode(value, self.max_smiles_len), dtype=torch.long)
            for value in pd.unique(pd.Series(self.smiles))
        }
        self._protein_token_cache = {
            value: torch.tensor(self.protein_tokenizer.encode(value, self.max_protein_len), dtype=torch.long)
            for value in pd.unique(pd.Series(self.sequences))
        }

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        smiles = self.smiles[idx]
        sequence = self.sequences[idx]
        return {
            "drug": self._drug_token_cache[smiles],
            "protein": self._protein_token_cache[sequence],
            "label": self.labels[idx],
        }


class FingerprintDTADataset(DTADataset):
    def __init__(
        self,
        *args,
        fingerprint_dim: int = 2048,
        fingerprint_radius: int = 2,
        target_fingerprint_dim: int = 0,
        target_fingerprint_k: int = 3,
        fingerprint_cache: dict[str, torch.Tensor] | None = None,
        target_fingerprint_cache: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fingerprint_dim = fingerprint_dim
        self.fingerprint_radius = fingerprint_radius
        self.target_fingerprint_dim = target_fingerprint_dim
        self.target_fingerprint_k = target_fingerprint_k
        self._fingerprint_cache = fingerprint_cache if fingerprint_cache is not None else {}
        for smiles in set(self.smiles):
            if smiles not in self._fingerprint_cache:
                self._fingerprint_cache[smiles] = smiles_to_fingerprint(smiles, n_bits=self.fingerprint_dim, radius=self.fingerprint_radius)
        self._target_fingerprint_cache = target_fingerprint_cache if target_fingerprint_cache is not None else {}
        if self.target_fingerprint_dim > 0:
            for sequence in set(self.sequences):
                if sequence not in self._target_fingerprint_cache:
                    self._target_fingerprint_cache[sequence] = sequence_to_hashed_fingerprint(
                        sequence,
                        n_bits=self.target_fingerprint_dim,
                        k=self.target_fingerprint_k,
                    )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        smiles = self.smiles[idx]
        item["fingerprint"] = self._fingerprint_cache[smiles]
        if self.target_fingerprint_dim > 0:
            sequence = self.sequences[idx]
            item["target_fingerprint"] = self._target_fingerprint_cache[sequence]
        return item


class GraphDTADataset(DTADataset):
    def __init__(self, *args, graph_cache: dict[str, object] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._graph_cache = graph_cache if graph_cache is not None else {}
        for smiles in set(self.smiles):
            if smiles not in self._graph_cache:
                self._graph_cache[smiles] = smiles_to_graph(smiles)

    def __getitem__(self, idx: int):
        if PyGData is None:
            raise ImportError("Graph models require torch-geometric.")
        smiles = self.smiles[idx]
        sequence = self.sequences[idx]
        graph = self._graph_cache[smiles].clone()
        graph.drug = self._drug_token_cache[smiles].unsqueeze(0)
        graph.target = self._protein_token_cache[sequence].unsqueeze(0)
        graph.protein = graph.target
        graph.label = self.labels[idx]
        graph.y = graph.label
        return graph


def one_of_k_encoding(value, choices: list) -> list[int]:
    return [int(value == choice) for choice in choices]


def one_of_k_encoding_unk(value, choices: list) -> list[int]:
    if value not in choices:
        value = choices[-1]
    return one_of_k_encoding(value, choices)


def atom_features(atom) -> list[float]:
    features = (
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C",
                "N",
                "O",
                "S",
                "F",
                "Si",
                "P",
                "Cl",
                "Br",
                "Mg",
                "Na",
                "Ca",
                "Fe",
                "As",
                "Al",
                "I",
                "B",
                "V",
                "K",
                "Tl",
                "Yb",
                "Sb",
                "Sn",
                "Ag",
                "Pd",
                "Co",
                "Se",
                "Ti",
                "Zn",
                "H",
                "Li",
                "Ge",
                "Cu",
                "Au",
                "Ni",
                "Cd",
                "In",
                "Mn",
                "Zr",
                "Cr",
                "Pt",
                "Hg",
                "Pb",
                "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
        + [int(atom.GetIsAromatic())]
    )
    features = [float(v) for v in features]
    if len(features) < 78:
        features.extend([0.0] * (78 - len(features)))
    return features[:78]


def smiles_to_graph(smiles: str):
    if Chem is None or PyGData is None:
        raise ImportError("Graph models require rdkit and torch-geometric.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        x = torch.zeros((1, 78), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return PyGData(x=x, edge_index=edge_index)
    features = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(features, dtype=torch.float32)
    edges: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edges.append((start, end))
        edges.append((end, start))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    return PyGData(x=x, edge_index=edge_index)


def smiles_to_fingerprint(smiles: str, n_bits: int = 2048, radius: int = 2) -> torch.Tensor:
    if Chem is None or AllChem is None or DataStructs is None:
        raise ImportError("Fingerprint models require rdkit.")
    arr = np.zeros((n_bits,), dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.from_numpy(arr)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.from_numpy(arr)


def sequence_to_hashed_fingerprint(sequence: str, n_bits: int = 1024, k: int = 3) -> torch.Tensor:
    arr = np.zeros((n_bits,), dtype=np.float32)
    sequence = str(sequence)
    if not sequence:
        return torch.from_numpy(arr)
    if len(sequence) < k:
        arr[zlib.crc32(sequence.encode("utf-8")) % n_bits] = 1.0
        return torch.from_numpy(arr)
    for i in range(len(sequence) - k + 1):
        arr[zlib.crc32(sequence[i : i + k].encode("utf-8")) % n_bits] = 1.0
    return torch.from_numpy(arr)


def load_dta_frame(data_cfg: dict) -> pd.DataFrame:
    root = Path(data_cfg["root"])
    smiles = pd.read_csv(root / data_cfg["smiles_file"])
    sequences = pd.read_csv(root / data_cfg["sequence_file"])
    affinity = pd.read_csv(root / data_cfg["affinity_file"])

    smiles = smiles.rename(
        columns={
            data_cfg["drug_id_col"]: "drug_id",
            data_cfg["smiles_col"]: "smiles",
        }
    )
    sequences = sequences.rename(
        columns={
            data_cfg["target_id_col"]: "target_id",
            data_cfg["sequence_col"]: "sequence",
        }
    )
    affinity = affinity.rename(
        columns={
            data_cfg["drug_id_col"]: "drug_id",
            data_cfg["target_id_col"]: "target_id",
            data_cfg["label_col"]: "label",
        }
    )

    frame = affinity.merge(smiles[["drug_id", "smiles"]], on="drug_id", how="inner")
    frame = frame.merge(sequences[["target_id", "sequence"]], on="target_id", how="inner")
    frame = frame.dropna(subset=["smiles", "sequence", "label"])
    frame["label"] = frame["label"].astype(float)
    return frame.reset_index(drop=True)


def split_frame(frame: pd.DataFrame, split_cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method = split_cfg.get("method", "random")
    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if method == "random":
        train, temp = train_test_split(frame, train_size=train_ratio, random_state=seed, shuffle=True)
        relative_val = val_ratio / (val_ratio + test_ratio)
        val, test = train_test_split(temp, train_size=relative_val, random_state=seed, shuffle=True)
        return train, val, test

    if method not in {"cold_drug", "cold_target"}:
        raise ValueError(f"Unsupported split method: {method}")

    key = "drug_id" if method == "cold_drug" else "target_id"
    unique_ids = frame[key].drop_duplicates()
    train_ids, temp_ids = train_test_split(unique_ids, train_size=train_ratio, random_state=seed, shuffle=True)
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(temp_ids, train_size=relative_val, random_state=seed, shuffle=True)
    return (
        frame[frame[key].isin(train_ids)],
        frame[frame[key].isin(val_ids)],
        frame[frame[key].isin(test_ids)],
    )


def build_dataloaders(config: dict) -> DTADataBundle:
    data_cfg = config["data"]
    frame = load_dta_frame(data_cfg)
    train, val, test = split_frame(frame, data_cfg["split"], int(config.get("seed", 42)))

    input_mode = config.get("model", {}).get("input_mode", "sequence")
    if input_mode == "kmer":
        drug_tokenizer = KmerTokenizer(int(config["model"].get("drug_kmer", 8)))
        protein_tokenizer = KmerTokenizer(int(config["model"].get("protein_kmer", 3)))
    else:
        drug_tokenizer = CharTokenizer()
        protein_tokenizer = CharTokenizer()
    drug_tokenizer.fit(train["smiles"].astype(str).tolist())
    protein_tokenizer.fit(train["sequence"].astype(str).tolist())

    max_smiles_len = int(data_cfg["max_smiles_len"])
    max_protein_len = int(data_cfg["max_protein_len"])
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", True)) and torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": bool(data_cfg.get("persistent_workers", num_workers > 0)) if num_workers > 0 else False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    graph_cache: dict[str, object] = {}
    fingerprint_cache: dict[str, torch.Tensor] = {}
    target_fingerprint_cache: dict[str, torch.Tensor] = {}

    def make_loader(df: pd.DataFrame, shuffle: bool) -> DataLoader:
        if input_mode == "graph":
            dataset = GraphDTADataset(
                df,
                drug_tokenizer,
                protein_tokenizer,
                max_smiles_len,
                max_protein_len,
                graph_cache=graph_cache,
            )
            if PyGDataLoader is None:
                raise ImportError("Graph models require torch-geometric.")
            return PyGDataLoader(dataset, shuffle=shuffle, **loader_kwargs)
        if input_mode == "fingerprint":
            dataset = FingerprintDTADataset(
                df,
                drug_tokenizer,
                protein_tokenizer,
                max_smiles_len,
                max_protein_len,
                fingerprint_dim=int(config["model"].get("fingerprint_dim", 2048)),
                fingerprint_radius=int(config["model"].get("fingerprint_radius", 2)),
                target_fingerprint_dim=int(config["model"].get("target_fingerprint_dim", 0)),
                target_fingerprint_k=int(config["model"].get("target_fingerprint_k", 3)),
                fingerprint_cache=fingerprint_cache,
                target_fingerprint_cache=target_fingerprint_cache,
            )
            return DataLoader(dataset, shuffle=shuffle, **loader_kwargs)
        dataset = DTADataset(df, drug_tokenizer, protein_tokenizer, max_smiles_len, max_protein_len)
        return DataLoader(dataset, shuffle=shuffle, **loader_kwargs)

    return DTADataBundle(
        train_loader=make_loader(train, True),
        val_loader=make_loader(val, False),
        test_loader=make_loader(test, False),
        drug_tokenizer=drug_tokenizer,
        protein_tokenizer=protein_tokenizer,
    )
