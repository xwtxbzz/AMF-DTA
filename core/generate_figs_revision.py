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
# The manuscript lives in `paper/` and includes this figure as
# `../autoresearch/figures/fig2_gate_weights.pdf`, which resolves to the
# *repository* root's copy -- not this bundle's. The two were kept in sync by
# hand, which is exactly how a figure goes stale without anyone noticing, so
# every save now writes both.
MIRROR_DIR = Path(__file__).resolve().parents[4] / "autoresearch" / "figures"


def _save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / name)
    if MIRROR_DIR.is_dir() and MIRROR_DIR != FIG_DIR:
        fig.savefig(MIRROR_DIR / name)
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
    """Five-seed gate weights: bars are the cross-seed mean, error bars the
    cross-seed std, and the dots the five individual seeds.

    The previous version plotted seed 42 alone, so a reader could not tell a
    reproducible preference from one initialization's accident. Showing the
    seeds separately is what makes the claim checkable: a preference that
    reproduces lands its five dots in a tight band (Davis context gate,
    $0.82$--$0.94$), an unreproducible one scatters over most of $[0,1]$
    (Davis drug gate, whose feature-Morgan weight runs the full range).

    Read from `gate_weights_5seeds.json`, which summarises the per-run gate
    dumps of `analysis_gates.py`. Not re-extracted here: the release bundle
    carries no checkpoints, because `package_release.py` copies `outputs/`
    with `KEEP_SUFFIX = {".json", ".csv"}`. The checkpoints survive in the
    full working tree, so this figure is recomputable from there.
    """
    data = json.loads((OUT / "gate_weights_5seeds.json").read_text(encoding="utf-8"))
    summary, per_seed, seeds = data["summary"], data["per_seed"], data["seeds"]

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.6))
    for ax, (name, labels) in zip(axes, (("drug", VIEWS), ("context", CTX))):
        x = np.arange(len(DATASETS))
        for i in range(3):
            means = np.array([summary[ds][name]["cross_seed_mean"][i]
                              for ds in DATASETS])
            stds = np.array([summary[ds][name]["cross_seed_std"][i]
                             for ds in DATASETS])
            pos = x + (i - 1) * 0.27
            ax.bar(pos, means, width=0.25, yerr=stds, color=C[i],
                   label=labels[i], capsize=1.5, error_kw={"lw": 0.6})
            jitter = np.linspace(-0.075, 0.075, len(seeds))
            for j, ds in enumerate(DATASETS):
                vals = [per_seed[ds][name][str(s)]["mean_weights"][i]
                        for s in seeds]
                ax.scatter(pos[j] + jitter, vals, s=2.2, color="k",
                           zorder=3, linewidths=0)
        ax.axhline(1 / 3, color="0.45", ls=":", lw=0.7, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[d] for d in DATASETS])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("mean gate weight")
        ax.set_title("drug-view gate" if name == "drug" else "context gate",
                     fontsize=9)
    axes[0].legend(ncol=3, fontsize=6.5, loc="upper center", frameon=False)
    axes[1].legend(ncol=3, fontsize=6.5, loc="upper center", frameon=False)
    fig.tight_layout()
    _save(fig, "fig2_gate_weights.pdf")
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
    _save(fig, "fig3_uncertainty_calibration.pdf")
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
