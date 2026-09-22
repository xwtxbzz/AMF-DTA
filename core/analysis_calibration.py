"""Review-response analyses from saved predictions (no training needed).

1. Uncertainty calibration (Sec. III-F): Gaussian NLL, 90% PI coverage,
   sharpness, ENCE, std-error correlation, Q1/Q4 RMSE ratio — per dataset,
   seed 42 and averaged over the five seeds.
2. FingerDTA-dagger seed ensemble: equal-weight average of the 5 seed
   predictions (a 5-model ensemble of the strongest baseline).
3. Model parameter counts per interaction mode (for the matched-parameter
   discussion).
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "outputs"
DATASETS = ("davis", "kiba", "metz")
SEEDS = (42, 52, 62, 72, 82)
MEMBERS = ("single", "m1", "m2", "m3", "m4")


def read_preds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    yt, yp = [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            yt.append(float(row["y_true"]))
            yp.append(float(row["y_pred"]))
    return np.asarray(yt), np.asarray(yp)


def gaussian_nll(y, mu, var):
    return 0.5 * (math.log(2 * math.pi) + np.log(var) + (y - mu) ** 2 / var)


def ence(y, mu, sigma, nbins=10):
    """Expected normalized calibration error: mean |bin RMSE - mean bin std|
    over std-quantile bins, normalized by mean std."""
    order = np.argsort(sigma)
    per = len(y) // nbins
    diffs = []
    for i in range(nbins):
        idx = order[i * per:(i + 1) * per]
        rmse = np.sqrt(np.mean((y[idx] - mu[idx]) ** 2))
        diffs.append(abs(rmse - sigma[idx].mean()))
    return float(np.mean(diffs) / sigma.mean())


def main() -> None:
    results = {}
    for ds in DATASETS:
        # NOTE: each repetition seed fixes its own split, so per-seed
        # calibration uses that seed's own y_true; cross-seed metrics are
        # averaged over the five repetitions (never bagged).
        per_seed = {}
        for seed in SEEDS:
            y, _ = read_preds(OUT / f"rail_single_seed{seed}_200/{ds}/test_predictions.csv")
            preds = np.stack([
                read_preds(OUT / f"rail_{m}_seed{seed}_200/{ds}/test_predictions.csv")[1]
                for m in MEMBERS
            ])
            per_seed[seed] = (preds.mean(0), preds.std(0), y)
        # seed-42 detailed metrics
        mu, sigma, y = per_seed[42]
        var = sigma ** 2 + 1e-8
        nll42 = float(gaussian_nll(y, mu, var).mean())
        cov42 = float(np.mean(np.abs(y - mu) <= 1.645 * sigma))
        sharp42 = float(sigma.mean())
        ence42 = ence(y, mu, sigma)
        rho42 = float(np.corrcoef(sigma, np.abs(y - mu))[0, 1])
        q1 = sigma <= np.quantile(sigma, 0.25)
        q4 = sigma >= np.quantile(sigma, 0.75)
        rmse_q1 = np.sqrt(np.mean((y[q1] - mu[q1]) ** 2))
        rmse_q4 = np.sqrt(np.mean((y[q4] - mu[q4]) ** 2))
        # 5-seed averaged calibration (mean over seeds of each metric)
        nlls, covs, sharpness, rhos = [], [], [], []
        for mu_s, sig_s, y_s in per_seed.values():
            var_s = sig_s ** 2 + 1e-8
            nlls.append(float(gaussian_nll(y_s, mu_s, var_s).mean()))
            covs.append(float(np.mean(np.abs(y_s - mu_s) <= 1.645 * sig_s)))
            sharpness.append(float(sig_s.mean()))
            rhos.append(float(np.corrcoef(sig_s, np.abs(y_s - mu_s))[0, 1]))
        results[ds] = {
            "nll_seed42": nll42, "nll_5seed_avg": float(np.mean(nlls)),
            "coverage90_seed42": cov42, "coverage90_5seed_avg": float(np.mean(covs)),
            "sharpness_seed42": sharp42, "sharpness_5seed_avg": float(np.mean(sharpness)),
            "ence_seed42": ence42, "rho_seed42": rho42,
            "rho_5seed_avg": float(np.mean(rhos)),
            "q1_q4_rmse": (float(rmse_q1), float(rmse_q4), float(rmse_q4 / rmse_q1)),
        }
        print(f"{ds}: " + json.dumps({k: (round(v, 4) if not isinstance(v, tuple)
                                          else tuple(round(x, 4) for x in v))
                                      for k, v in results[ds].items()}, ensure_ascii=False))

    # --- parameter counts per interaction mode ------------------------------
    from framework.train import TrainConfig, build_model
    print("\nParameter counts (H=256, fc 768-256):")
    for mode in ("rail", "fixed", "mlp_interact", "poly_interact"):
        model = build_model(TrainConfig(interaction=mode), protein_vocab_size=1000)
        n = sum(p.numel() for p in model.parameters())
        print(f"  {mode}: {n:,}")
        if mode == "rail":
            rail_n = n
    print(f"  ratio mlp/rail: {sum(p.numel() for p in build_model(TrainConfig(interaction='mlp_interact'), 1000).parameters()) / rail_n:.2f}")

    out = {"uncertainty": results}
    (OUT / "review_analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved outputs/review_analysis.json")


if __name__ == "__main__":
    main()
