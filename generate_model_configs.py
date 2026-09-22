from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from dta_experiment.config import load_config


DATASET_CONFIGS = {
    "davis": "configs/davis_deepdta.yaml",
    "kiba": "configs/kiba_deepdta.yaml",
    "metz": "configs/metz_deepdta.yaml",
}

MODEL_CONFIGS = {
    "deepdta": {
        "name": "deepdta",
        "input_mode": "sequence",
        "embedding_dim": 128,
        "drug_kernel_size": 4,
        "target_kernel_size": 8,
        "num_filters": [32, 64, 96],
        "fc_dims": [1024, 1024, 512],
        "dropout": 0.1,
    },
    "graphdta": {
        "name": "graphdta",
        "input_mode": "graph",
        "variant": "gcn",
        "num_node_features": 78,
        "n_filters": 32,
        "embed_dim": 128,
        "output_dim": 128,
        "dropout": 0.2,
    },
    "ckdta": {
        "name": "ckdta",
        "input_mode": "graph",
        "num_node_features": 78,
        "hidden_dim": 128,
        "graph_layers": 3,
        "graph_heads": 4,
        "protein_embedding_dim": 128,
        "protein_kernel_size": 5,
        "fc_dims": [512, 256],
        "dropout": 0.2,
    },
    "drugmgr": {
        "name": "drugmgr",
        "input_mode": "graph",
        "num_node_features": 78,
        "hidden_dim": 128,
        "gat_heads": 4,
        "transformer_layers": 2,
        "transformer_heads": 4,
        "protein_kernel_size": 7,
        "fc_dims": [512, 256],
        "dropout": 0.2,
    },
    "attentiondta": {
        "name": "attentiondta",
        "input_mode": "sequence",
        "embedding_dim": 128,
        "hidden_dim": 128,
        "drug_kernel_size": 4,
        "target_kernel_size": 8,
        "conv_layers": 3,
        "attention_heads": 4,
        "fc_dims": [512, 256],
        "dropout": 0.1,
    },
    "edta": {
        "name": "edta",
        "input_mode": "sequence",
        "embedding_dim": 128,
        "smiles_vocab_size": 65,
        "bpe_vocab_size": 16694,
        "atom_feature_dim": 24,
        "target_feature_dim": 42,
        "pocket_feature_dim": 67,
        "smiles_lstm_hidden_dim": 128,
        "smiles_lstm_layers": 2,
        "smiles_attention_heads": 8,
        "max_drug_atoms": 100,
        "target_feature_len": 512,
        "pocket_max_len": 200,
        "conv_ll_kernel_sizes": [4, 6, 6, 4],
        "conv_pp_kernel_sizes": [4, 8, 12, 8],
        "classifier_dims": [512, 256, 128, 64],
        "dropout": 0.2,
    },
    "widedta": {
        "name": "widedta",
        "input_mode": "kmer",
        "drug_kmer": 8,
        "protein_kmer": 3,
        "embedding_dim": 128,
        "filters": 32,
        "kernel_size": 4,
        "fc_dims": [1024, 1024, 512],
        "dropout": 0.1,
    },
    "fingerdta": {
        "name": "fingerdta",
        "input_mode": "fingerprint",
        "fingerprint_dim": 2048,
        "fingerprint_radius": 2,
        "target_fingerprint_dim": 1024,
        "target_fingerprint_k": 3,
        "hidden_dim": 256,
        "protein_embedding_dim": 128,
        "protein_filters": 128,
        "protein_kernel_sizes": [3, 5, 7],
        "fc_dims": [512, 256],
        "dropout": 0.2,
    },
}

# 只保留论文报告的模型；已从仓库移除的模型条目已一并删除。
# 注意: main() 直接用 MODEL_BATCH_SIZE[model_name] 取值，其键集必须与 MODEL_CONFIGS 一致。
MODEL_BATCH_SIZE = {
    "deepdta": 128,
    "graphdta": 64,
    "ckdta": 32,
    "drugmgr": 32,
    "attentiondta": 64,
    "edta": 32,
    "widedta": 128,
    "fingerdta": 128,
}

# 未列出的模型沿用数据集基础配置里的序列长度
MODEL_LENGTHS = {
    "ckdta": {"max_smiles_len": 120, "max_protein_len": 512},
    "drugmgr": {"max_smiles_len": 120, "max_protein_len": 512},
    "widedta": {"max_smiles_len": 100, "max_protein_len": 512},
    "fingerdta": {"max_smiles_len": 120, "max_protein_len": 1000},
    "edta": {"max_smiles_len": 100, "max_protein_len": 512},
}


def main() -> None:
    output_dir = Path("configs/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, config_path in DATASET_CONFIGS.items():
        base = load_config(config_path)
        for model_name, model_cfg in MODEL_CONFIGS.items():
            config = deepcopy(base)
            config["model"] = deepcopy(model_cfg)
            config["train"]["batch_size"] = MODEL_BATCH_SIZE[model_name]
            config["train"]["output_dir"] = f"outputs/{dataset}_{model_name}"
            config["train"]["epochs"] = int(config["train"].get("epochs", 50))
            config["train"]["early_stopping_patience"] = int(config["train"].get("early_stopping_patience", 10))
            config["train"]["amp"] = True
            config["train"]["amp_dtype"] = "float16"
            config["train"]["cudnn_benchmark"] = True
            config["train"]["checkpoint_interval"] = 5
            config["data"]["pin_memory"] = True
            config["data"].update(MODEL_LENGTHS.get(model_name, {}))
            path = output_dir / f"{dataset}_{model_name}.yaml"
            with path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()
