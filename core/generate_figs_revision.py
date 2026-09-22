"""Regenerate Fig. 2 (gate weights) and Fig. 3 (uncertainty calibration) from
the CURRENT rail_dta outputs, and print the exact numbers for the paper text.

Fig. 2: per-sample drug-view and context gate weights (seed-42 single member),
        mean +- std per view per dataset.
Fig. 3: ensemble std deciles vs mean |error| (5-member ensemble, seed 42);
        prints rho(std, |err|), decile-1/decile-10 RMSE ratios, Gaussian NLL,
        ENCE, 90% PI coverage, sharpness.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "outputs"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "autoresearch" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATASETS = ("davis", "kiba", "metz")
LABELS = {"davis": "Davis", "kiba": "KIBA", "metz": "Metz"}
VIEWS = ("Morgan", "feature-Morgan", "MACCS")
CTX = ("drug", "target FP", "sequence")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "axes.linewidth": 0.7,
})
C = ["#4C72B0", "#DD8452", "#55A868"]


# ─────────────────────────── data loading ────────────────────────────────
def read_preds(root: Path, ds: str):
    data = np.loadtxt(root / ds / "test_predictions.csv", delimiter=",", skiprows=1)
    if data.ndim == 1:
        return np.array([data[0]]), np.array([data[1]])
    return data[:, 0], data[:, 1]


def gate_samples(ds: str):
    """Per-sample gate weights from the seed-42 single member (best_state)."""
    from framework.data import build_dataloaders
    from framework.ensemble import MEMBERS
    from framework.train import build_model

    cfg = MEMBERS["single"]
    bundle = build_dataloaders(ds, batch_size=cfg.batch_size, seed=cfg.seed,
                               data_root=DATA_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, bundle.protein_vocab_size).to(device)
    state = torch.load(OUT / f"rail_single_seed42_200/{ds}/best_state.pt",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    drug_w, ctx_w, err = [], [], []
    with torch.no_grad():
        for batch in bundle.test_loader:
            batch = {k: v.to(device, non_blocking=device.type == "cuda")
                     for k, v in batch.items()}
            views = torch.stack([model.drug_morgan(batch["drug_fingerprint"]),
                                 model.drug_feature_morgan(batch["feature_fingerprint"]),
                                 model.drug_maccs(batch["maccs_fingerprint"])], dim=1)
            w1 = torch.softmax(model.drug_view_gate(views.flatten(1)), dim=-1)
            drug = (views * w1.unsqueeze(-1)).sum(dim=1)
            tgt = model.target_fp(batch["target_fingerprint"])
            seq = model.seq_encoder(batch["target"])
            contexts = torch.stack([drug, tgt, seq], dim=1)
            w2 = torch.softmax(model.context_gate(contexts.flatten(1)), dim=-1)
            pred = model(batch["drug_fingerprint"], batch["feature_fingerprint"],
                         batch["maccs_fingerprint"], batch["bias_features"],
                         batch["target"], batch["target_fingerprint"])
            drug_w.append(w1.cpu().numpy())
            ctx_w.append(w2.cpu().numpy())
            err.append((pred.float().cpu().numpy() - batch["label"].cpu().numpy()))
    return (np.concatenate(drug_w), np.concatenate(ctx_w),
            np.abs(np.concatenate(err)))


# ─────────────────────────── Fig. 2 ──────────────────────────────────────
def fig2() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.6))
    for ax, (name, ws, labels) in zip(
            axes, (("drug-view gate", None, VIEWS), ("context gate", None, CTX))):
        means, stds = [], []
        for ds in DATASETS:
            dw, cw, _ = gate_samples(ds)
            w = dw if "drug" in name else cw
            means.append(w.mean(0))
            stds.append(w.std(0))
        means, stds = np.array(means), np.array(stds)
        x = np.arange(len(DATASETS))
        for i in range(3):
            ax.bar(x + (i - 1) * 0.27, means[:, i], width=0.25, yerr=stds[:, i],
                   color=C[i], label=labels[i], capsize=1.5, error_kw={"lw": 0.6})
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[d] for d in DATASETS])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("mean gate weight")
        ax.set_title(name, fontsize=9)
    axes[0].legend(ncol=3, fontsize=6.5, loc="upper center", frameon=False)
    axes[1].legend(ncol=3, fontsize=6.5, loc="upper center", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_gate_weights.pdf")
    plt.close(fig)


# ─────────────────────────── Fig. 3 ──────────────────────────────────────
def gaussian_nll(y, mu, var):
    return float((0.5 * (math.log(2 * math.pi) + np.log(var) + (y - mu) ** 2 / var)).mean())


def ence(y, mu, sigma, nbins=10):
    order = np.argsort(sigma)
    per = len(y) // nbins
    diffs = []
    for i in range(nbins):
        idx = order[i * per:(i + 1) * per]
        diffs.append(abs(np.sqrt(np.mean((y[idx] - mu[idx]) ** 2)) - sigma[idx].mean()))
    return float(np.mean(diffs) / sigma.mean())


def fig3() -> dict:
    stats = {}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    members = ("single", "m1", "m2", "m3", "m4")
    for ax, ds in zip(axes, DATASETS):
        y, _ = read_preds(OUT / f"rail_single_seed42_200", ds)
        preds = np.stack([read_preds(OUT / f"rail_{m}_seed42_200", ds)[1] for m in members])
        mu, sigma = preds.mean(0), preds.std(0)
        err = np.abs(y - mu)
        order = np.argsort(sigma)
        nb = len(y) // 10
        bin_err, bin_sig, bin_rmse = [], [], []
        for i in range(10):
            idx = order[i * nb:(i + 1) * nb]
            bin_sig.append(sigma[idx].mean())
            bin_err.append(err[idx].mean())
            bin_rmse.append(np.sqrt(np.mean((y[idx] - mu[idx]) ** 2)))
        ax.bar(np.arange(10), bin_err, color=C[0], alpha=0.85)
        ax.plot(np.arange(10), bin_sig, color=C[2], marker="o", ms=2.5, lw=0.9)
        idx1 = order[:nb]
        idx10 = order[-nb:]
        r1 = np.sqrt(np.mean((y[idx1] - mu[idx1]) ** 2))
        r10 = np.sqrt(np.mean((y[idx10] - mu[idx10]) ** 2))
        rho = float(np.corrcoef(sigma, err)[0, 1])
        var = sigma ** 2 + 1e-8
        st = {
            "rho": rho,
            "q1_rmse": float(r1), "q10_rmse": float(r10), "ratio": float(r10 / r1),
            "nll": gaussian_nll(y, mu, var),
            "coverage90": float(np.mean(np.abs(y - mu) <= 1.645 * sigma)),
            "sharpness": float(sigma.mean()),
            "ence": ence(y, mu, sigma),
        }
        stats[ds] = st
        ax.set_title(f"{LABELS[ds]}  ($\\rho$={rho:.2f})", fontsize=8)
        ax.set_xticks(range(0, 10, 2))
        ax.set_xlabel("std decile")
    axes[0].set_ylabel("mean abs. error / std")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_uncertainty_calibration.pdf")
    plt.close(fig)
    return stats


if __name__ == "__main__":
    print("generating fig2 (gate weights)...")
    fig2()
    stats = fig3()
    print("\nFig3 stats (seed 42, deciles):")
    for ds, s in stats.items():
        print(f"  {ds}: rho={s['rho']:.3f} q1={s['q1_rmse']:.3f} q10={s['q10_rmse']:.3f} "
              f"ratio={s['ratio']:.1f}x nll={s['nll']:.2f} cov90={s['coverage90']:.2f} "
              f"sharp={s['sharpness']:.3f} ence={s['ence']:.2f}")
    (OUT / "fig3_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nfigures written to {FIG_DIR}")
