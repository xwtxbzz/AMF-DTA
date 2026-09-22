from __future__ import annotations

import torch
from torch import nn

from models.AttentionDTA import AttentionDTA
from models.CKDTA import CKDTA
from models.DeepDTA import DeepDTA
from models.DrugMGR import DrugMGR
from models.EDTA import EDTA
from models.FingerDTA import FingerDTA
from models.GraphDTA import GraphDTA
from models.MambaTransDTA import MambaTransDTA
from models.WideDTA import WideDTA


class ConvEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        filters: list[int],
        kernel_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if len(filters) != len(kernel_sizes):
            raise ValueError("filters and kernel_sizes must have the same length")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, out_channels, kernel_size)
            for out_channels, kernel_size in zip(filters, kernel_sizes)
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = sum(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).transpose(1, 2)
        features = []
        for conv in self.convs:
            z = torch.relu(conv(x))
            z = torch.max(z, dim=-1).values
            features.append(z)
        return self.dropout(torch.cat(features, dim=-1))


class DeepDTACNN(nn.Module):
    def __init__(self, model_cfg: dict, drug_vocab_size: int, protein_vocab_size: int) -> None:
        super().__init__()
        dropout = float(model_cfg.get("dropout", 0.2))
        self.drug_encoder = ConvEncoder(
            vocab_size=drug_vocab_size,
            embedding_dim=int(model_cfg["drug_embedding_dim"]),
            filters=list(model_cfg["drug_filters"]),
            kernel_sizes=list(model_cfg["drug_kernel_sizes"]),
            dropout=dropout,
        )
        self.protein_encoder = ConvEncoder(
            vocab_size=protein_vocab_size,
            embedding_dim=int(model_cfg["protein_embedding_dim"]),
            filters=list(model_cfg["protein_filters"]),
            kernel_sizes=list(model_cfg["protein_kernel_sizes"]),
            dropout=dropout,
        )
        dims = [self.drug_encoder.output_dim + self.protein_encoder.output_dim]
        dims.extend(int(d) for d in model_cfg["hidden_dims"])
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, drug: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        drug_features = self.drug_encoder(drug)
        protein_features = self.protein_encoder(protein)
        features = torch.cat([drug_features, protein_features], dim=-1)
        return self.regressor(features).squeeze(-1)


def build_model(config: dict, drug_vocab_size: int, protein_vocab_size: int) -> nn.Module:
    model_cfg = config["model"]
    name = model_cfg.get("name", "deepdta_cnn").lower()
    if name == "deepdta_cnn":
        return DeepDTACNN(model_cfg, drug_vocab_size, protein_vocab_size)
    if name == "deepdta":
        return DeepDTA(
            drug_vocab_size=drug_vocab_size,
            target_vocab_size=protein_vocab_size,
            drug_max_len=int(config["data"]["max_smiles_len"]),
            target_max_len=int(config["data"]["max_protein_len"]),
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            drug_kernel_size=int(model_cfg.get("drug_kernel_size", 4)),
            target_kernel_size=int(model_cfg.get("target_kernel_size", 8)),
            num_filters=tuple(model_cfg.get("num_filters", (32, 64, 96))),
            fc_dims=tuple(model_cfg.get("fc_dims", (1024, 1024, 512))),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if name == "attentiondta":
        return AttentionDTA(
            drug_vocab_size=drug_vocab_size,
            target_vocab_size=protein_vocab_size,
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            drug_kernel_size=int(model_cfg.get("drug_kernel_size", 4)),
            target_kernel_size=int(model_cfg.get("target_kernel_size", 8)),
            conv_layers=int(model_cfg.get("conv_layers", 3)),
            attention_heads=int(model_cfg.get("attention_heads", 4)),
            fc_dims=tuple(model_cfg.get("fc_dims", (512, 256))),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if name == "edta":
        return EDTA(
            drug_vocab_size=drug_vocab_size,
            target_vocab_size=protein_vocab_size,
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            smiles_vocab_size=int(model_cfg.get("smiles_vocab_size", 65)),
            bpe_vocab_size=int(model_cfg.get("bpe_vocab_size", 16694)),
            atom_feature_dim=int(model_cfg.get("atom_feature_dim", 24)),
            target_feature_dim=int(model_cfg.get("target_feature_dim", 42)),
            pocket_feature_dim=int(model_cfg.get("pocket_feature_dim", 67)),
            smiles_lstm_hidden_dim=int(model_cfg.get("smiles_lstm_hidden_dim", 128)),
            smiles_lstm_layers=int(model_cfg.get("smiles_lstm_layers", 2)),
            smiles_attention_heads=int(model_cfg.get("smiles_attention_heads", 8)),
            max_drug_atoms=int(model_cfg.get("max_drug_atoms", config["data"]["max_smiles_len"])),
            target_feature_len=int(model_cfg.get("target_feature_len", config["data"]["max_protein_len"])),
            pocket_max_len=int(model_cfg.get("pocket_max_len", 200)),
            conv_ll_kernel_sizes=tuple(model_cfg.get("conv_ll_kernel_sizes", (4, 6, 6, 4))),
            conv_pp_kernel_sizes=tuple(model_cfg.get("conv_pp_kernel_sizes", (4, 8, 12, 8))),
            classifier_dims=tuple(model_cfg.get("classifier_dims", (512, 256, 128, 64))),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    if name == "widedta":
        return WideDTA(
            drug_vocab_size=drug_vocab_size,
            target_vocab_size=protein_vocab_size,
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            filters=int(model_cfg.get("filters", 32)),
            kernel_size=int(model_cfg.get("kernel_size", 4)),
            fc_dims=tuple(model_cfg.get("fc_dims", (1024, 1024, 512))),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if name == "graphdta":
        return GraphDTA(
            num_node_features=int(model_cfg.get("num_node_features", 78)),
            target_vocab_size=protein_vocab_size,
            variant=model_cfg.get("variant", "gcn"),
            n_filters=int(model_cfg.get("n_filters", 32)),
            embed_dim=int(model_cfg.get("embed_dim", 128)),
            output_dim=int(model_cfg.get("output_dim", 128)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            target_max_len=int(config["data"]["max_protein_len"]),
        )
    if name == "ckdta":
        return CKDTA(
            num_node_features=int(model_cfg.get("num_node_features", 78)),
            target_vocab_size=protein_vocab_size,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            graph_layers=int(model_cfg.get("graph_layers", 3)),
            graph_heads=int(model_cfg.get("graph_heads", 4)),
            protein_embedding_dim=int(model_cfg.get("protein_embedding_dim", 128)),
            protein_kernel_size=int(model_cfg.get("protein_kernel_size", 5)),
            fc_dims=tuple(model_cfg.get("fc_dims", (512, 256))),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    if name == "drugmgr":
        return DrugMGR(
            num_node_features=int(model_cfg.get("num_node_features", 78)),
            drug_vocab_size=drug_vocab_size,
            target_vocab_size=protein_vocab_size,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            gat_heads=int(model_cfg.get("gat_heads", 4)),
            transformer_layers=int(model_cfg.get("transformer_layers", 2)),
            transformer_heads=int(model_cfg.get("transformer_heads", 4)),
            protein_kernel_size=int(model_cfg.get("protein_kernel_size", 7)),
            fc_dims=tuple(model_cfg.get("fc_dims", (512, 256))),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    if name == "fingerdta":
        return FingerDTA(
            fingerprint_dim=int(model_cfg.get("fingerprint_dim", 2048)),
            target_vocab_size=protein_vocab_size,
            target_fingerprint_dim=int(model_cfg.get("target_fingerprint_dim", 1024)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            protein_embedding_dim=int(model_cfg.get("protein_embedding_dim", 128)),
            protein_filters=int(model_cfg.get("protein_filters", 128)),
            protein_kernel_sizes=tuple(model_cfg.get("protein_kernel_sizes", (3, 5, 7))),
            fc_dims=tuple(model_cfg.get("fc_dims", (512, 256))),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    if name in {"mambatransdta", "mamba_trans_dta"}:
        return MambaTransDTA(
            drug_vocab_size=drug_vocab_size,
            target_vocab_size=protein_vocab_size,
            max_drug_len=int(config["data"]["max_smiles_len"]),
            embed_dim=int(model_cfg.get("embed_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 1)),
            heads=int(model_cfg.get("heads", 8)),
            protein_embedding_dim=int(model_cfg.get("protein_embedding_dim", 128)),
            protein_filters=tuple(model_cfg.get("protein_filters", (32, 64, 96))),
            protein_kernels=tuple(model_cfg.get("protein_kernels", (4, 8, 12))),
            hidden_dims=tuple(model_cfg.get("hidden_dims", (1024, 512))),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"Unsupported model: {model_cfg.get('name')}")
