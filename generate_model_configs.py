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
    "moltrans": {
        "name": "moltrans",
        "input_mode": "sequence",
        "emb_size": 64,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "hidden_dropout": 0.1,
    },
    "drugban": {
        "name": "drugban",
        "input_mode": "graph",
        "node_in_feats": 78,
        "node_embedding": 128,
        "drug_hidden_feats": [128, 128, 128],
        "protein_embedding_dim": 128,
        "protein_num_filters": [128, 128, 128],
        "protein_kernel_size": [3, 6, 9],
        "mlp_in_dim": 256,
        "mlp_hidden_dim": 512,
        "mlp_out_dim": 128,
        "ban_heads": 2,
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
    "csatdta": {
        "name": "csatdta",
        "input_mode": "sequence",
        "embedding_dim": 128,
        "hidden_dim": 128,
        "drug_kernel_size": 7,
        "target_kernel_size": 9,
        "attention_heads": 4,
        "blocks": 2,
        "fc_dims": [512, 256],
        "dropout": 0.1,
    },
    "cgbilstm": {
        "name": "cgbilstm",
        "input_mode": "graph",
        "num_node_features": 78,
        "hidden_dim": 128,
        "graph_layers": 2,
        "gat_heads": 4,
        "protein_embedding_dim": 128,
        "lstm_layers": 1,
        "fc_dims": [512, 256],
        "dropout": 0.2,
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
    "deattentiondta": {
        "name": "deattentiondta",
        "input_mode": "sequence",
        "embedding_dim": 128,
        "hidden_dim": 128,
        "drug_kernel_size": 5,
        "protein_kernel_size": 7,
        "pocket_kernel_size": 3,
        "attention_heads": 4,
        "fc_dims": [512, 256],
        "dropout": 0.1,
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
    "mgraphdta": {
        "name": "mgraphdta",
        "input_mode": "graph",
        "num_node_features": 78,
        "hidden_dim": 128,
        "graph_layers": 27,
        "protein_embedding_dim": 128,
        "protein_filters": 128,
        "protein_kernel_size": 8,
        "fc_dims": [1024, 512],
        "dropout": 0.2,
    },
    "mattdti": {
        "name": "mattdti",
        "input_mode": "graph",
        "num_node_features": 78,
        "hidden_dim": 128,
        "graph_layers": 2,
        "protein_embedding_dim": 128,
        "protein_kernel_size": 7,
        "attention_heads": 4,
        "relative_kernel_size": 5,
        "fc_dims": [512, 256],
        "dropout": 0.2,
    },
    "deepglstm": {
        "name": "deepglstm",
        "input_mode": "graph",
        "num_node_features": 78,
        "hidden_dim": 128,
        "graph_layers": 3,
        "protein_embedding_dim": 128,
        "lstm_layers": 1,
        "fc_dims": [512, 256],
        "dropout": 0.2,
    },
    "ssmdta": {
        "name": "ssmdta",
        "input_mode": "sequence",
        "embed_dim": 128,
        "encoder_layers": 2,
        "num_attention_heads": 4,
        "ffn_dim": 256,
        "cross_attention_heads": 4,
        "dropout": 0.1,
        "fc_dims": [512, 256],
    },
}

MODEL_BATCH_SIZE = {
    "deepdta": 128,
    "graphdta": 64,
    "ckdta": 32,
    "moltrans": 16,
    "drugban": 32,
    "drugmgr": 32,
    "attentiondta": 64,
    "csatdta": 64,
    "cgbilstm": 32,
    "edta": 32,
    "deattentiondta": 32,
    "widedta": 128,
    "fingerdta": 128,
    "mgraphdta": 32,
    "mattdti": 32,
    "deepglstm": 32,
    "ssmdta": 32,
}

MODEL_LENGTHS = {
    "moltrans": {"max_smiles_len": 80, "max_protein_len": 512},
    "ckdta": {"max_smiles_len": 120, "max_protein_len": 512},
    "drugmgr": {"max_smiles_len": 120, "max_protein_len": 512},
    "csatdta": {"max_smiles_len": 120, "max_protein_len": 512},
    "cgbilstm": {"max_smiles_len": 120, "max_protein_len": 512},
    "widedta": {"max_smiles_len": 100, "max_protein_len": 512},
    "fingerdta": {"max_smiles_len": 120, "max_protein_len": 1000},
    "mgraphdta": {"max_smiles_len": 120, "max_protein_len": 1000},
    "mattdti": {"max_smiles_len": 120, "max_protein_len": 512},
    "deepglstm": {"max_smiles_len": 120, "max_protein_len": 512},
    "ssmdta": {"max_smiles_len": 100, "max_protein_len": 512},
    "edta": {"max_smiles_len": 100, "max_protein_len": 512},
    "deattentiondta": {"max_smiles_len": 100, "max_protein_len": 512},
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
