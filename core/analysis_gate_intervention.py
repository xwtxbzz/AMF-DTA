"""Gate intervention analysis (reviewer Exp. point 5 / meta-review).

Turns the correlational gate evidence of `analysis_gates.py` into a causal
test. Each trained single-member checkpoint is re-evaluated on the test
split with each gate overwritten by

  (a) the dataset-level mean weight vector, estimated on the *validation*
      split only (no test information), and
  (b) uniform weights 1/K, i.e. gating removed entirely.

Decomposition of the two effects:
  - uniform  -> frozen-mean : value of a learned *dataset-level* preference
  - frozen-mean -> sample   : value of *sample-level* adaptation

If sample-level adaptation carries information, replacing the per-sample
weight vector by its dataset mean should degrade test RMSE.

Protocol notes:
  - Checkpoints are `best_state.pt` (best-validation epoch). The paper's
    Table rows use the final-fit SWA state, which train.py does not save,
    so absolute RMSE here is anchored to its own `sample` baseline and
    only the *deltas* are comparable across conditions.
  - kiba/metz labels are z-scored for the loss, so predictions are
    de-normalized by the train-set mean/std exactly as
    `train.py::run_epoch` does (Davis labels are already normalized).
  - Inference only, no randomness: deltas are deterministic for a given
    checkpoint, so variation across seeds is the only sampling noise.

Usage: python analysis_gate_intervention.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dta_experiment.metrics import regression_metrics

OUT = Path(__file__).resolve().parent / "outputs"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DATASETS = ("davis", "kiba", "metz")
SEEDS = (42, 52, 62, 72, 82)
CONDITIONS = ("sample", "drug_mean", "context_mean", "both_mean", "uniform")
# drug gate: 0 = Morgan, 1 = feature-Morgan, 2 = MACCS
DRUG_VIEWS = ("morgan", "feature_morgan", "maccs")
# context gate: 0 = fused drug view, 1 = target k-mer fingerprint, 2 = sequence
CONTEXT_VIEWS = ("drug", "target_fp", "seq_emb")


class FrozenGate(nn.Module):
    """Emits a constant pre-softmax logit vector, so the downstream
    softmax returns exactly the requested weight vector."""

    def __init__(self, weights: np.ndarray) -> None:
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float32).clamp_min(1e-12)
        self.register_buffer("logw", torch.log(w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logw.expand(x.shape[0], -1)


def run_loader(model, loader, device, label_mean: float, label_std: float):
    """Mirror `train.py::run_epoch`: kiba/metz labels are z-scored for the
    loss, so model outputs are de-normalized by the train-set mean/std
    before any metric is computed (Davis labels are already normalized,
    where label_mean=0, label_std=1)."""
    y_all, pred_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=device.type == "cuda")
                     for k, v in batch.items()}
            pred = model(batch["drug_fingerprint"], batch["feature_fingerprint"],
                         batch["maccs_fingerprint"], batch["bias_features"],
                         batch["target"], batch["target_fingerprint"])
            pred_all.append((pred.float() * label_std + label_mean).cpu().numpy())
            y_all.append(batch["label"].cpu().numpy())
    return np.concatenate(y_all), np.concatenate(pred_all)


def main() -> None:
    from framework.data import build_dataloaders
    from framework.ensemble import MEMBERS
    from framework.train import build_model

    cfg = MEMBERS["single"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict = {}

    for seed in SEEDS:
        results[seed] = {}
        for ds in DATASETS:
            bundle = build_dataloaders(ds, batch_size=cfg.batch_size, seed=seed,
                                       data_root=DATA_ROOT)
            model = build_model(cfg, bundle.protein_vocab_size).to(device)
            state = torch.load(OUT / f"rail_single_seed{seed}_200/{ds}/best_state.pt",
                               map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            model.eval()

            if ds in {"kiba", "metz"}:
                labels = bundle.train_loader.dataset.labels
                label_mean = float(labels.mean())
                label_std = float(labels.std(unbiased=False).clamp_min(1.0e-6))
            else:  # Davis labels are already normalized
                label_mean, label_std = 0.0, 1.0

            # --- pass 1: dataset-level mean gate weights on validation ------
            captured: dict[str, list] = {"drug": [], "context": []}

            def make_hook(key):
                def hook(module, inp, out):
                    captured[key].append(
                        torch.softmax(out, dim=-1).float().cpu().numpy())
                return hook

            h_drug = model.drug_view_gate.register_forward_hook(make_hook("drug"))
            h_ctx = model.context_gate.register_forward_hook(make_hook("context"))
            run_loader(model, bundle.val_loader, device, label_mean, label_std)
            h_drug.remove()
            h_ctx.remove()

            drug_mean = np.concatenate(captured["drug"]).mean(axis=0)
            ctx_mean = np.concatenate(captured["context"]).mean(axis=0)
            k_drug, k_ctx = len(drug_mean), len(ctx_mean)

            orig_drug, orig_ctx = model.drug_view_gate, model.context_gate
            conditions = {
                "sample": (None, None),
                "drug_mean": (drug_mean, None),
                "context_mean": (None, ctx_mean),
                "both_mean": (drug_mean, ctx_mean),
                "uniform": (np.full(k_drug, 1.0 / k_drug),
                            np.full(k_ctx, 1.0 / k_ctx)),
            }

            base = None
            per_ds: dict = {
                "val_mean_drug_weights": [float(x) for x in drug_mean],
                "val_mean_context_weights": [float(x) for x in ctx_mean],
                "conditions": {},
            }
            for name, (w_drug, w_ctx) in conditions.items():
                model.drug_view_gate = (orig_drug if w_drug is None
                                        else FrozenGate(w_drug).to(device))
                model.context_gate = (orig_ctx if w_ctx is None
                                      else FrozenGate(w_ctx).to(device))
                y, pred = run_loader(model, bundle.test_loader, device,
                                     label_mean, label_std)
                m = regression_metrics(y, pred)
                if base is None:
                    base = m["rmse"]
                per_ds["conditions"][name] = {
                    "rmse": float(m["rmse"]),
                    "ci": float(m["ci"]),
                    "delta_rmse_vs_sample": float(m["rmse"] - base),
                    "delta_pct": float(100.0 * (m["rmse"] - base) / base),
                }
                print(f"  s{seed} {ds:6s} {name:13s} RMSE={m['rmse']:.4f} "
                      f"CI={m['ci']:.4f} d={100.0 * (m['rmse'] - base) / base:+.2f}%",
                      flush=True)

            model.drug_view_gate, model.context_gate = orig_drug, orig_ctx
            results[seed][ds] = per_ds
            print(f"s{seed} {ds}: val-mean drug="
                  f"{[round(float(x), 3) for x in drug_mean]} context="
                  f"{[round(float(x), 3) for x in ctx_mean]}\n", flush=True)

    # --- aggregation: per-dataset over seeds, and overall ------------------
    def agg_over(pairs):
        out = {}
        for cond in CONDITIONS:
            r = np.array([results[s][d]["conditions"][cond]["rmse"] for s, d in pairs])
            c = np.array([results[s][d]["conditions"][cond]["ci"] for s, d in pairs])
            base_r = np.array([results[s][d]["conditions"]["sample"]["rmse"]
                               for s, d in pairs])
            out[cond] = {
                "rmse_mean": float(r.mean()),
                "rmse_std": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
                "ci_mean": float(c.mean()),
                "ci_std": float(c.std(ddof=1)) if len(c) > 1 else 0.0,
                "mean_pct_vs_sample": float(100.0 * np.mean((r - base_r) / base_r)),
                "pct_std": float(100.0 * (r / base_r - 1).std(ddof=1)) if len(r) > 1 else 0.0,
            }
        return out

    summary = {"per_dataset": {}, "overall": {}}
    for ds in DATASETS:
        summary["per_dataset"][ds] = agg_over([(s, ds) for s in SEEDS])
    summary["overall"] = agg_over([(s, d) for s in SEEDS for d in DATASETS])

    # --- paired Wilcoxon over the 15 (seed, dataset) pairs ------------------
    from scipy.stats import wilcoxon

    tests = {}
    for ref in ("both_mean", "drug_mean", "context_mean", "uniform"):
        a = np.array([results[s][d]["conditions"][ref]["rmse"]
                      for s in SEEDS for d in DATASETS])
        b = np.array([results[s][d]["conditions"]["sample"]["rmse"]
                      for s in SEEDS for d in DATASETS])
        stat, p = wilcoxon(a, b)
        tests[f"{ref}_vs_sample"] = {"statistic": float(stat), "p_value": float(p),
                                     "n_pairs": int(len(a))}
    summary["wilcoxon"] = tests

    print("\n=== overall (15 pairs: 5 seeds x 3 datasets) ===")
    for cond in CONDITIONS:
        v = summary["overall"][cond]
        print(f"{cond:13s} RMSE={v['rmse_mean']:.4f}+-{v['rmse_std']:.4f} "
              f"CI={v['ci_mean']:.4f} ({v['mean_pct_vs_sample']:+.2f}% "
              f"+-{v['pct_std']:.2f})")
    print("\n=== per dataset (5 seeds) ===")
    for ds in DATASETS:
        row = " ".join(
            f"{c}={summary['per_dataset'][ds][c]['mean_pct_vs_sample']:+.2f}%"
            for c in CONDITIONS[1:])
        print(f"{ds:6s} {row}")
    print("\n=== paired Wilcoxon vs sample ===")
    for k, v in tests.items():
        print(f"{k:22s} p={v['p_value']:.5f} (n={v['n_pairs']})")

    out = {
        "per_seed": results,
        "summary": summary,
        "views": {"drug_gate": DRUG_VIEWS, "context_gate": CONTEXT_VIEWS},
        "protocol": "single member best_state.pt, seeds 42/52/62/72/82; gate "
                    "weights frozen to validation-split mean; test-set "
                    "inference only; predictions de-normalized by train-set "
                    "label mean/std (kiba/metz)",
    }
    path = OUT / "gate_intervention.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
