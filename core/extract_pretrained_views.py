"""Frozen pretrained view features + their random-init twins (cold-start study).

Why this exists
---------------
The paper's cold-start gap is large -- the three-dataset mean RMSE moves from
~0.44 (random split) to 0.65-0.68 when drugs or targets are held out entirely
-- and the paper currently declares it out of scope. The hypothesis under test
is mechanistic, not "a bigger encoder wins":

    under a cold split the burden of generalisation falls on the view on the
    *held-out* side. The handcrafted views are composition-level and carry no
    transferable semantics (Morgan/MACCS bits; a 3-mer hash and a mean-pooled
    character embedding). A pretrained encoder does.

That prediction is per-dataset and falsifiable -- Davis has only 68 drugs, so
its binding constraint is the held-out drug; KIBA has only 229 targets, so
its binding constraint is the held-out target.

The causal control
------------------
Showing "pretrained features help" is not enough: a wider feature vector and
more parameters also help, and a reviewer will say so. So every pretrained
feature matrix ships with a **random-init twin**: identical architecture,
identical tokenizer, identical pooling, identical dimension and dtype --
only the weights differ (and they are never trained; see below). Any gain
that survives that contrast is attributable to *pretraining*, not capacity.

Frozen, not fine-tuned
----------------------
Both variants are frozen and the features are cached to disk. This is what
makes the contrast exact: the two arms differ in one tensor's *values* and in
nothing else -- same graph, same parameter count, same training schedule. It
also makes the arms cheap, which matters at one 8 GB laptop GPU shared with a
running sweep.

Pooling
-------
Masked mean over residues, excluding BOS/EOS. Sequences longer than the
model's positional limit are split into non-overlapping windows, each pooled
independently, and the window vectors are averaged -- a plain truncation would
silently discard the C-terminal half of 181 sequences across the three
datasets (109 Davis, 42 KIBA, 30 Metz) and is a known reviewer target.

Output
------
    pretrained/<tag>/<dataset>_<side>.pt
a torch.save'd dict {entity_string: float32 tensor[dim]}. `<tag>` is either
the HF model id or `<model-id>__randinit<seed>`.

Usage:
    python extract_pretrained_views.py                       # both sides, all sets
    python extract_pretrained_views.py --sides protein
    python extract_pretrained_views.py --models esm2_t12_35M_UR50D
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Offline by default: every model this script needs is already in the local
# HF cache, and a silent download would make the run unreproducible on a
# machine without network. Override with --allow-download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"
OUT_ROOT = ROOT / "pretrained"

DATASETS = ("davis", "kiba", "metz")

# side -> (default HF model id, entity column, entity column values)
SIDES = {
    "protein": {
        "model": "facebook/esm2_t12_35M_UR50D",
        "file": "sequence.csv",
        "id_col": "TARGETID",
        "val_col": "SEQUENCE",
        # residue window; the model's positional limit minus BOS/EOS
        "window": 1022,
    },
    "drug": {
        "model": "DeepChem/ChemBERTa-77M-MTR",
        "file": "smiles.csv",
        "id_col": "DRUGID",
        "val_col": "SMILES",
        "window": 510,
    },
}

RANDINIT_SEED = 42


def load_entities(side: str, dataset: str) -> list[str]:
    """Unique entity strings for one (side, dataset), in a stable order."""
    spec = SIDES[side]
    frame = pd.read_csv(DATA_ROOT / dataset / spec["file"])
    values = frame[spec["val_col"]].astype(str)
    return list(pd.unique(values))


def build_model(model_id: str, random_init: bool):
    """Return (model, tokenizer) in eval mode.

    `random_init=True` builds the *same config* with fresh weights: same
    depth, width, tokenizer and pooling, no pretrained knowledge. That is the
    control arm, and it is why the two arms can be compared at all.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if random_init:
        # Seed before construction so the twin is reproducible across runs.
        torch.manual_seed(RANDINIT_SEED)
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModel.from_config(config)
        # from_config leaves the HF default init, which is what we want: a
        # parameter-matched encoder with no learned structure. The LM head is
        # absent in AutoModel, so nothing here has seen a corpus.
    else:
        model = AutoModel.from_pretrained(model_id)
    model.eval()
    return model, tokenizer


def pooled_embedding(model, tokenizer, values: list[str], window: int,
                     device: torch.device, batch_size: int = 8) -> torch.Tensor:
    """Masked-mean embeddings for `values`, windowing over-long inputs.

    Returns float32 [len(values), dim] on CPU. Each value contributes the mean
    of its window vectors, so length is normalised away without truncation.
    """
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            chunk = values[start:start + batch_size]
            # window each string, remembering how many windows it contributes
            flat: list[str] = []
            counts: list[int] = []
            for value in chunk:
                pieces = [value[i:i + window] for i in range(0, max(len(value), 1), window)]
                flat.extend(pieces)
                counts.append(len(pieces))
            enc = tokenizer(flat, return_tensors="pt", padding=True,
                            truncation=True, max_length=window + 2)
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state.float()           # [N, L, D]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            # drop BOS/EOS from the mean: they carry no residue identity
            mask[:, 0, :] = 0.0
            lengths = enc["attention_mask"].sum(dim=1).clamp_min(1)
            mask[torch.arange(mask.size(0)), lengths - 1, :] = 0.0
            per_window = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            per_window = per_window.cpu()
            # average the windows belonging to each original value
            cursor = 0
            for count in counts:
                out.append(per_window[cursor:cursor + count].mean(dim=0))
                cursor += count
            done = min(start + batch_size, len(values))
            print(f"      {done}/{len(values)}", flush=True)
    return torch.stack(out, dim=0)


def tag_for(model_id: str, random_init: bool) -> str:
    short = model_id.split("/")[-1]
    return f"{short}__randinit{RANDINIT_SEED}" if random_init else short


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sides", default="protein,drug")
    ap.add_argument("--models", default=None,
                    help="comma-separated HF ids; overrides the per-side default")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--allow-download", action="store_true")
    args = ap.parse_args()

    if args.allow_download:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else args.device if args.device != "auto" else "cpu")
    print(f"device={device}", flush=True)

    override = args.models.split(",") if args.models else None
    manifest: dict = {"device": str(device), "randinit_seed": RANDINIT_SEED, "runs": []}

    for side in args.sides.split(","):
        if side not in SIDES:
            raise SystemExit(f"unknown side {side!r}; choose from {list(SIDES)}")
        spec = SIDES[side]
        model_id = override[0] if override else spec["model"]

        # Both variants are built and run in the same process so tokenizer,
        # pooling and dtype cannot drift between the arm and its control.
        for random_init in (False, True):
            tag = tag_for(model_id, random_init)
            print(f"\n=== {side}: {tag} ===", flush=True)
            t0 = time.time()
            model, tokenizer = build_model(model_id, random_init)
            model.to(device)
            dim = int(model.config.hidden_size)
            print(f"    dim={dim}  loaded in {time.time() - t0:.1f}s", flush=True)

            for dataset in args.datasets.split(","):
                entities = load_entities(side, dataset)
                print(f"    {dataset}: {len(entities)} unique {side} entities", flush=True)
                feats = pooled_embedding(model, tokenizer, entities, spec["window"], device)

                # z-score per dimension. Both variants get the *same*
                # treatment, so feature scale cannot explain a difference --
                # otherwise "pretrained wins" could just mean "pretrained has
                # a friendlier scale for the downstream Linear layer".
                mean = feats.mean(dim=0, keepdim=True)
                std = feats.std(dim=0, keepdim=True).clamp_min(1e-6)
                feats = (feats - mean) / std

                out_dir = OUT_ROOT / tag
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{dataset}_{side}.pt"
                torch.save({k: v.clone() for k, v in zip(entities, feats)}, out_path)
                manifest["runs"].append({
                    "side": side, "tag": tag, "dataset": dataset,
                    "model_id": model_id, "random_init": random_init,
                    "dim": dim, "n_entities": len(entities),
                    "mean_norm": float(feats.norm(dim=1).mean()),
                    "path": str(out_path.relative_to(ROOT)),
                })
                print(f"      -> {out_path}  (mean |z| norm {feats.norm(dim=1).mean():.3f})",
                      flush=True)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT_ROOT / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
