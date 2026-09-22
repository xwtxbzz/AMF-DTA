"""Self-check: (1) Ens beats every baseline in all 15 (seed, dataset) RMSE pairs;
(2) cold-start numbers in the paper vs autoresearch outputs."""
import json
from pathlib import Path

SEEDS = (42, 52, 62, 72, 82)
DATASETS = ("davis", "kiba", "metz")
# Relative to this file, not absolute. The three trees sit in the same
# arrangement in the release bundle as they do here, so `__file__` resolves
# both -- whereas `D:\try\...` resolves only on the machine that wrote it and
# turns the script into a dead link the moment anyone else unpacks it.
HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "outputs"
RAIL = HERE / "outputs"
COLD = HERE.parent / "autoresearch" / "outputs"

ens = json.loads((RAIL / "summary_raildta_5seeds_200.json").read_text(encoding="utf-8"))
ens_rmse = lambda s, d: ens["ensemble_per_seed"][str(s)][d]["rmse"]


def base_rmse(model, seed, ds):
    if model == "fingerdta_dagger":  # same-protocol re-implementation in rail_dta repo
        p = RAIL / f"fingerdta_seed{seed}_200" / ds / "test_metrics.json"
    else:
        p = MAIN / f"{ds}_{model}_all200" / "test_metrics.json" if seed == 42 else \
            MAIN / f"{ds}_{model}_seed{seed}_200" / "test_metrics.json"
        if seed == 42 and not p.exists():  # some models use seed42_200 naming
            p = MAIN / f"{ds}_{model}_seed42_200" / "test_metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))["rmse"]


print("=== 1. Ens vs each baseline: wins / 15 pairs ===")
models = ["fingerdta_dagger", "fingerdta", "widedta", "deepdta", "attentiondta",
          "drugmgr", "mambatransdta", "graphdta", "ckdta", "edta"]
missing = []
for m in models:
    wins, pairs = 0, 0
    for s in SEEDS:
        for d in DATASETS:
            try:
                b = base_rmse(m, s, d)
            except FileNotFoundError as e:
                missing.append(f"{m} seed{s} {d}: {e}")
                continue
            pairs += 1
            if ens_rmse(s, d) < b:
                wins += 1
    print(f"{m:18s}: {wins}/{pairs} pairs (Ens better)")
if missing:
    print("MISSING FILES:", *missing, sep="\n  ")

print("\n=== 2. cold-start (paper: drug 0.6477->0.6454, CI 0.7529->0.7560;")
print("   target 0.6678->0.6630, CI 0.7660->0.7696) ===")
for tag in ["icasp_railcold_fa3_cold_drug", "icasp_railcold_fa8_cold_drug",
            "icasp_railcold_fa8_cold_drug_v2", "icasp_railcold_fa3_cold_target",
            "icasp_railcold_fa8_cold_target", "icasp_railcold_fa8_cold_target_v2",
            "icasp_cold_drug_seed42", "icasp_cold_target_seed42"]:
    p = COLD / tag
    if not p.is_dir():
        print(f"{tag}: MISSING"); continue
    rmses, cis = [], []
    for d in DATASETS:
        m = json.loads((p / d / "test_metrics.json").read_text(encoding="utf-8"))
        rmses.append(m["rmse"]); cis.append(m["ci"])
    print(f"{tag:38s}: rmse {sum(rmses)/3:.4f} (per-ds {[round(x,4) for x in rmses]}) "
          f"ci {sum(cis)/3:.4f}")
