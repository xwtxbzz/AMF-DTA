"""Gate intervention on the *fixed-interaction* (A3/F-A3) checkpoints.

Why this exists, and why the original script is not enough
---------------------------------------------------------
`analysis_gate_intervention.py` runs on `rail_single_seed*_200`, whose
`config.json` has `interaction="rail"`. The retrained arm the paper compares
against (F-A11) is `interaction="fixed"` with the gates removed. So the two
arms of the headline contrast differ in the interaction slot as well as in
the gating condition, and the paper's sentence "the post-hoc and retrained
conditions are identical" is not true of the artifacts as built.

This script re-runs the same test-time intervention on
`rail_ab_FA3_seed*_200` (`interaction="fixed"`, both gates, bias) -- the run
whose seed-42 average reproduces Table II's A3. Post-hoc and retrained then
differ in exactly one thing: whether the gates are present during training.

It also adds the **placebo** condition the adversarial review asked for. The
reviewer's objection is that freezing gates to 1/K changes the *input
distribution* of everything downstream, so a large RMSE hit may measure
"sensitivity of a trained network to a perturbed gate" rather than "loss of
learned per-sample adaptivity". The placebo separates these: each sample is
given **another sample's gate logits** (a permutation within the batch), which
preserves the batch's empirical distribution of gate outputs and their scale
exactly, and destroys only the sample-to-gate correspondence.

Interpretation:
  uniform   (+huge)  and placebo (+small)  -> the 1/K result is dominated by
                                             losing the learned weight vector
  uniform   (+huge)  and placebo (+huge)   -> the post-hoc family measures
                                             perturbation sensitivity in general

Either way the paper's *retrained* number is unaffected; what changes is how
much of the post-hoc effect can be attributed to removing adaptivity.

Permutation noise: a single draw is a noisy estimate, so the placebo is
averaged over R independent permutations and the individual RMSEs are stored
alongside the mean.

Protocol: identical to `analysis_gate_intervention.py` -- `best_state.pt`,
seeds 42/52/62/72/82, test-set inference only, predictions de-normalized by
the train-set label mean/std exactly as `train.py::run_epoch` does.

Usage: python analysis_gate_intervention_fa3.py [--permutations 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis_gate_intervention import (
    CONTEXT_VIEWS,
    DATASETS,
    DRUG_VIEWS,
    SEEDS,
    FrozenGate,
    run_loader,
)

OUT = Path(__file__).resolve().parent / "outputs"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RUN_PREFIX = "rail_ab_FA3_seed{seed}_200"
OUT_JSON = OUT / "gate_intervention_fa3.json"

# The two gates are frozen/shuffled together, matching `both_mean` and
# `uniform`, so the three "both gates" conditions are directly comparable.
CONDITIONS = ("sample", "drug_mean", "context_mean", "both_mean", "uniform",
              "placebo")


class PlaceboGate(torch.nn.Module):
    """Permutation placebo: each sample receives another sample's gate logits.

    Implemented inside the batch, so the multiset of gate outputs actually
    used is unchanged -- only which sample gets which output is destroyed.
    The wrapped `source` is the real trained gate, left untouched.
    """

    def __init__(self, source: torch.nn.Module) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.source(x)
        if out.shape[0] < 2:
            return out
        perm = torch.randperm(out.shape[0], device=out.device)
        return out[perm]


def load_config(run_dir: Path):
    """Rebuild the TrainConfig the checkpoints were trained under, so the
    architecture matches even if the defaults drift."""
    from framework.train import TrainConfig

    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = TrainConfig()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, tuple(value) if key == "fc_dims" else value)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=5,
                        help="independent placebo permutations to average")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = parser.parse_args()

    from framework.data import build_dataloaders
    from framework.train import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict = {}

    for seed in args.seeds:
        run_dir = OUT / RUN_PREFIX.format(seed=seed)
        cfg = load_config(run_dir)
        print(f"=== seed {seed}: {run_dir.name} "
              f"interaction={cfg.interaction} "
              f"gates={cfg.use_drug_gate}/{cfg.use_context_gate} ===",
              flush=True)
        results[seed] = {}
        for ds in DATASETS:
            bundle = build_dataloaders(ds, batch_size=cfg.batch_size, seed=seed,
                                       data_root=DATA_ROOT)
            model = build_model(cfg, bundle.protein_vocab_size).to(device)
            state = torch.load(run_dir / ds / "best_state.pt",
                               map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            model.eval()

            if ds in {"kiba", "metz"}:
                labels = bundle.train_loader.dataset.labels
                label_mean = float(labels.mean())
                label_std = float(labels.std(unbiased=False).clamp_min(1.0e-6))
            else:  # Davis labels are already normalized
                label_mean, label_std = 0.0, 1.0

            # --- validation-set mean gate weights --------------------------
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
            fixed = {
                "sample": (None, None),
                "drug_mean": (drug_mean, None),
                "context_mean": (None, ctx_mean),
                "both_mean": (drug_mean, ctx_mean),
                "uniform": (np.full(k_drug, 1.0 / k_drug),
                            np.full(k_ctx, 1.0 / k_ctx)),
            }
            per_ds: dict = {
                "val_mean_drug_weights": [float(x) for x in drug_mean],
                "val_mean_context_weights": [float(x) for x in ctx_mean],
                "conditions": {},
            }
            base = None
            for name, (w_drug, w_ctx) in fixed.items():
                model.drug_view_gate = (orig_drug if w_drug is None
                                        else FrozenGate(w_drug).to(device))
                model.context_gate = (orig_ctx if w_ctx is None
                                      else FrozenGate(w_ctx).to(device))
                y, pred = run_loader(model, bundle.test_loader, device,
                                     label_mean, label_std)
                from dta_experiment.metrics import regression_metrics
                m = regression_metrics(y, pred)
                if base is None:
                    base = m["rmse"]
                per_ds["conditions"][name] = {
                    "rmse": float(m["rmse"]),
                    "ci": float(m["ci"]),
                    "delta_pct": float(100.0 * (m["rmse"] - base) / base),
                }
                print(f"  s{seed} {ds:6s} {name:11s} RMSE={m['rmse']:.4f} "
                      f"d={100.0 * (m['rmse'] - base) / base:+7.2f}%", flush=True)

            # --- placebo: R independent within-batch permutations ----------
            from dta_experiment.metrics import regression_metrics
            placebo_rmses = []
            for r in range(args.permutations):
                torch.manual_seed(10_000 * seed + 100 * r + len(ds))
                model.drug_view_gate = PlaceboGate(orig_drug).to(device)
                model.context_gate = PlaceboGate(orig_ctx).to(device)
                y, pred = run_loader(model, bundle.test_loader, device,
                                     label_mean, label_std)
                placebo_rmses.append(float(regression_metrics(y, pred)["rmse"]))
            placebo_mean = float(np.mean(placebo_rmses))
            per_ds["conditions"]["placebo"] = {
                "rmse": placebo_mean,
                "rmse_per_permutation": placebo_rmses,
                "rmse_std_over_permutations": float(np.std(placebo_rmses, ddof=1)),
                "delta_pct": float(100.0 * (placebo_mean - base) / base),
                "delta_pct_per_permutation": [
                    float(100.0 * (r - base) / base) for r in placebo_rmses],
            }
            model.drug_view_gate, model.context_gate = orig_drug, orig_ctx
            print(f"  s{seed} {ds:6s} {'placebo':11s} RMSE={placebo_mean:.4f} "
                  f"d={100.0 * (placebo_mean - base) / base:+7.2f}% "
                  f"(over {args.permutations} perms, "
                  f"sd={np.std(placebo_rmses, ddof=1):.4f})", flush=True)

            results[seed][ds] = per_ds

    # --- aggregation --------------------------------------------------------
    def agg(pairs):
        out = {}
        for cond in CONDITIONS:
            r = np.array([results[s][d]["conditions"][cond]["rmse"]
                          for s, d in pairs])
            base_r = np.array([results[s][d]["conditions"]["sample"]["rmse"]
                               for s, d in pairs])
            pct = 100.0 * (r - base_r) / base_r
            out[cond] = {
                "rmse_mean": float(r.mean()),
                "mean_pct": float(pct.mean()),
                "median_pct": float(np.median(pct)),
                "pct_std": float(pct.std(ddof=1)) if len(pct) > 1 else 0.0,
                "min_pct": float(pct.min()),
                "max_pct": float(pct.max()),
                "n_worse": int((pct > 0).sum()),
            }
        return out

    from scipy.stats import wilcoxon

    summary = {"per_dataset": {ds: agg([(s, ds) for s in args.seeds])
                               for ds in DATASETS},
               "overall": agg([(s, d) for s in args.seeds for d in DATASETS]),
               "wilcoxon": {}}
    for ref in ("uniform", "both_mean", "drug_mean", "context_mean", "placebo"):
        a = np.array([results[s][d]["conditions"][ref]["rmse"]
                      for s in args.seeds for d in DATASETS])
        b = np.array([results[s][d]["conditions"]["sample"]["rmse"]
                      for s in args.seeds for d in DATASETS])
        stat, p = wilcoxon(a, b)
        summary["wilcoxon"][ref] = {"statistic": float(stat),
                                    "p_value": float(p),
                                    "n_pairs": int(len(a))}

    print("\n=== overall ===")
    for cond in CONDITIONS:
        v = summary["overall"][cond]
        print(f"{cond:11s} mean={v['mean_pct']:+7.2f}% "
              f"median={v['median_pct']:+7.2f}% "
              f"[{v['min_pct']:+.1f},{v['max_pct']:+.1f}] "
              f"worse in {v['n_worse']}/{len(args.seeds) * 3}")
    print("\n=== per dataset (mean %) ===")
    for ds in DATASETS:
        row = " ".join(f"{c}={summary['per_dataset'][ds][c]['mean_pct']:+.2f}%"
                       for c in CONDITIONS[1:])
        print(f"{ds:6s} {row}")
    print("\n=== paired Wilcoxon vs sample (two-sided) ===")
    for k, v in summary["wilcoxon"].items():
        print(f"{k:11s} p={v['p_value']:.3e} (n={v['n_pairs']})")

    out = {
        "per_seed": results,
        "summary": summary,
        "views": {"drug_gate": DRUG_VIEWS, "context_gate": CONTEXT_VIEWS},
        "protocol": f"{RUN_PREFIX} (interaction=fixed, both gates, bias); "
                    "best_state.pt, seeds 42/52/62/72/82; test-set inference "
                    "only; placebo = within-batch permutation of gate logits, "
                    f"averaged over {args.permutations} draws",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
