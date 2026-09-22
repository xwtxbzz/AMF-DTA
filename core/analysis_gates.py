"""Per-sample gate weight analysis (reviewer Exp. point 5).

Extracts the drug-view and cross-modal context gate weights for every test
sample of the seed-42 single member, and reports:
  - mean weight per view (dataset-level preference, cf. Fig. 2)
  - within-dataset variation: std and entropy across samples
  - how often the per-sample argmax disagrees with the dataset argmax
    (evidence of sample-level adaptation vs. a fixed global weight vector)
  - correlation of per-view weights and gate entropy with |error|
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent / "outputs"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DATASETS = ("davis", "kiba", "metz")
DEFAULT_RUN = "rail_single_seed42_200"


def run_config(run_dir: Path):
    """The config that produced `run_dir`.

    Reading it from the run's own config.json matters for the F-A* family,
    whose `interaction` differs from the default: building the model with
    the wrong interaction would give differently shaped interaction modules
    and the checkpoint load would fail rather than silently measure another
    graph. Falls back to the ensemble's `single` member for older runs that
    predate config.json.
    """
    from framework.train import TrainConfig
    path = run_dir / "config.json"
    if not path.exists():
        from framework.ensemble import MEMBERS
        return MEMBERS["single"]
    cfg = TrainConfig()
    for key, value in json.loads(path.read_text(encoding="utf-8")).items():
        if hasattr(cfg, key):
            setattr(cfg, key, tuple(value) if key == "fc_dims" else value)
    return cfg


def main() -> None:
    from framework.data import build_dataloaders
    from framework.train import build_model

    run = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN
    run_dir = OUT / run
    cfg = run_config(run_dir)
    results = {}
    for ds in DATASETS:
        bundle = build_dataloaders(ds, batch_size=cfg.batch_size, seed=cfg.seed,
                                   data_root=DATA_ROOT)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg, bundle.protein_vocab_size).to(device)
        state = torch.load(run_dir / ds / "best_state.pt",
                           map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.eval()

        captured: dict[str, list] = {"drug": [], "context": []}

        def make_hook(key):
            def hook(module, inp, out):
                captured[key].append(torch.softmax(out, dim=-1).cpu().numpy())
            return hook

        h_drug = model.drug_view_gate.register_forward_hook(make_hook("drug"))
        h_ctx = model.context_gate.register_forward_hook(make_hook("context"))

        y_all, pred_all = [], []
        with torch.no_grad():
            for batch in bundle.test_loader:
                batch = {k: v.to(device, non_blocking=device.type == "cuda")
                         for k, v in batch.items()}
                pred = model(batch["drug_fingerprint"], batch["feature_fingerprint"],
                             batch["maccs_fingerprint"], batch["bias_features"],
                             batch["target"], batch["target_fingerprint"])
                pred_all.append(pred.float().cpu().numpy())
                y_all.append(batch["label"].cpu().numpy())
        h_drug.remove()
        h_ctx.remove()
        y = np.concatenate(y_all)
        pred = np.concatenate(pred_all)
        err = np.abs(y - pred)
        drug_w = np.concatenate(captured["drug"])      # [N, 3]
        ctx_w = np.concatenate(captured["context"])    # [N, 3]

        def entropy(w):
            return -np.sum(w * np.log(np.clip(w, 1e-12, 1.0)), axis=-1)

        out = {}
        for name, w in (("drug", drug_w), ("context", ctx_w)):
            mean_w = w.mean(axis=0)
            switching = np.mean(np.argmax(w, axis=-1) != np.argmax(mean_w))
            ent = entropy(w)
            rho_w = [float(np.corrcoef(w[:, i], err)[0, 1]) for i in range(w.shape[1])]
            rho_ent = float(np.corrcoef(ent, err)[0, 1])
            out[name] = {
                "mean_weights": [float(x) for x in mean_w],
                "std_across_samples": [float(x) for x in w.std(axis=0)],
                "mean_entropy": float(ent.mean()),
                "argmax_switching_vs_global": float(switching),
                "rho_weight_error": rho_w,
                "rho_entropy_error": rho_ent,
            }
            print(f"{ds} {name}: mean={np.round(mean_w, 3).tolist()} "
                  f"std={np.round(w.std(0), 3).tolist()} "
                  f"entropy={ent.mean():.3f} switching={switching:.3f} "
                  f"rho(w,err)={np.round(rho_w, 3).tolist()} rho(ent,err)={rho_ent:.3f}")
        results[ds] = out

    # The default run keeps its original filename -- the paper's Fig. 2
    # paragraph is written against it; other runs get their own file so
    # comparing variants cannot silently overwrite the reported numbers.
    name = ("gate_analysis.json" if run == DEFAULT_RUN
            else f"gate_analysis_{run}.json")
    (OUT / name).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved outputs/{name}")


if __name__ == "__main__":
    main()
