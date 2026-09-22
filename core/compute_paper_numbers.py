"""Compute final Table I / Table II numbers + body-text percentages for the paper.

- Table I rows read per-seed rmse/ci from the summary JSON (Ens/SWA/FingerDTA-dagger)
  and evaluate the Single (best_state.pt) row.
- Table II lower group reads the raw per-seed test_metrics.json files.
- avg +/- std = stats over the 5 seeds of each seed's 3-dataset mean RMSE,
  matching the baseline rows' convention; population std is also printed.
- Also verifies the FingerDTA baseline +/- convention (0.4727 +/- 0.0095).
"""
import json
import statistics
from pathlib import Path

import run_amfdta_5seeds as r5

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DATASETS = r5.DATASETS
SEEDS = r5.SEEDS


def stats(vals):
    return {
        "mean": statistics.mean(vals),
        "sample_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "pop_std": statistics.pstdev(vals),
    }


def summarize_rows(per_seed):
    """per_seed: dict[str(seed)] -> {ds: {"rmse": ..., "ci": ...}}"""
    out = {}
    for ds in DATASETS:
        out[ds] = {
            "rmse": stats([per_seed[str(s)][ds]["rmse"] for s in SEEDS]),
            "ci": stats([per_seed[str(s)][ds]["ci"] for s in SEEDS]),
        }
    per_seed_avg_rmse = [statistics.mean(per_seed[str(s)][ds]["rmse"] for ds in DATASETS)
                         for s in SEEDS]
    per_seed_avg_ci = [statistics.mean(per_seed[str(s)][ds]["ci"] for ds in DATASETS)
                       for s in SEEDS]
    out["avg"] = {"rmse": stats(per_seed_avg_rmse), "ci": stats(per_seed_avg_ci)}
    out["per_seed_avg_rmse"] = per_seed_avg_rmse
    return out


def load_test_metrics(root: Path, ds: str) -> dict:
    return json.loads((root / ds / "test_metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    summary = json.loads(
        (OUTPUT_ROOT / "summary_raildta_5seeds_200.json").read_text(encoding="utf-8"))
    rows = {
        "Ens": summarize_rows(summary["ensemble_per_seed"]),
        "SWA": summarize_rows(summary["single_swa_per_seed"]),
        "FingerDTA_dagger": summarize_rows(summary["fingerdta_dagger_per_seed"]),
    }

    # --- Single row: evaluate best_state.pt on the test split ---
    single_per_seed: dict[str, dict] = {}
    for seed in SEEDS:
        single_per_seed[str(seed)] = {}
        for ds in DATASETS:
            m = r5.evaluate_best_state("single", seed, ds)
            single_per_seed[str(seed)][ds] = {"rmse": m["rmse"], "ci": m["ci"]}
            print(f"single {ds} seed{seed}: rmse={m['rmse']:.4f} ci={m['ci']:.4f}", flush=True)
    rows["Single"] = summarize_rows(single_per_seed)

    # --- Table II lower group: raw per-seed files ---
    ablation = {}
    for variant in ("F-A3", "F-A7", "F-A8", "F-A10", "F-A9"):
        tag = variant.replace("-", "")
        if variant == "F-A9":
            probe = OUTPUT_ROOT / "rail_ab_fa9_seed42_200" / "diverged.json"
            ablation[variant] = {"probe": json.loads(probe.read_text(encoding="utf-8"))}
            continue
        per_seed = {}
        for seed in SEEDS:
            base = OUTPUT_ROOT / f"rail_ab_{tag}_seed{seed}_200"
            per_seed[str(seed)] = {
                ds: {"rmse": load_test_metrics(base, ds)["rmse"],
                     "ci": load_test_metrics(base, ds)["ci"]} for ds in DATASETS
            }
        ablation[variant] = summarize_rows(per_seed)

    # --- Baseline convention check (FingerDTA original, 5 seeds) ---
    base_dir = Path("D:/try/outputs")
    base_per_seed = {}
    for seed in SEEDS:
        folder = "all200" if seed == 42 else f"seed{seed}_200"
        base_per_seed[str(seed)] = {
            ds: json.loads((base_dir / f"{ds}_fingerdta_{folder}" / "test_metrics.json")
                           .read_text(encoding="utf-8"))["rmse"] for ds in DATASETS}
    base_avgs = [statistics.mean(base_per_seed[str(s)][ds] for ds in DATASETS) for s in SEEDS]
    print("\nFingerDTA baseline per-seed avg:", [f"{v:.4f}" for v in base_avgs])
    print(f"FingerDTA baseline avg: {statistics.mean(base_avgs):.4f} "
          f"sample={statistics.stdev(base_avgs):.4f} pop={statistics.pstdev(base_avgs):.4f}")

    result = {"rows": rows, "ablation": ablation}
    (OUTPUT_ROOT / "paper_numbers.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    def fmt(r, key="rmse", pop=True):
        s = r[key]
        sd = s["pop_std"] if pop else s["sample_std"]
        return f"{s['mean']:.4f}+-{sd:.4f}"

    print(f"\n=== Table I rows (population std) ===")
    for name, r in rows.items():
        print(f"{name:16s} davis {fmt(r['davis'])} | kiba {fmt(r['kiba'])} | "
              f"metz {fmt(r['metz'])} | avg {fmt(r['avg'])} | CI {fmt(r['avg'], 'ci')}")
    print(f"\n=== Table II lower group (population std) ===")
    for v, r in ablation.items():
        if v == "F-A9":
            print("F-A9:", r)
        else:
            print(f"{v:6s} davis {fmt(r['davis'])} | kiba {fmt(r['kiba'])} | "
                  f"metz {fmt(r['metz'])} | avg {fmt(r['avg'])} | CI {fmt(r['avg'], 'ci')}")

    base = 0.4727
    print(f"\n=== Percentages ===")
    for name, r in rows.items():
        print(f"{name:16s} vs FingerDTA {base:.4f}: "
              f"{(base - r['avg']['rmse']['mean']) / base * 100:.1f}%")
    fdt_avg = rows["FingerDTA_dagger"]["avg"]["rmse"]["mean"]
    ens_avg = rows["Ens"]["avg"]["rmse"]["mean"]
    swa_avg = rows["SWA"]["avg"]["rmse"]["mean"]
    single_avg = rows["Single"]["avg"]["rmse"]["mean"]
    print(f"Ens vs FingerDTA-dagger {fdt_avg:.4f}: {(fdt_avg - ens_avg) / fdt_avg * 100:.1f}%")
    print(f"Ens vs CKDTA 0.5742: {(0.5742 - ens_avg) / 0.5742 * 100:.1f}%")
    print(f"Ens vs EDTA 0.6218: {(0.6218 - ens_avg) / 0.6218 * 100:.1f}%")
    print(f"Ens metz vs FingerDTA metz 0.5380: "
          f"{(0.5380 - rows['Ens']['metz']['rmse']['mean']) / 0.5380 * 100:.1f}%")
    print(f"Ens vs SWA {swa_avg:.4f}: {(swa_avg - ens_avg) / swa_avg * 100:.1f}%")
    print(f"Ens vs Single {single_avg:.4f}: {(single_avg - ens_avg) / single_avg * 100:.1f}%")
    print(f"Ens CI avg: {rows['Ens']['avg']['ci']['mean']:.4f}")
    print(f"Ens per-seed avg rmse: {[f'{v:.4f}' for v in rows['Ens']['per_seed_avg_rmse']]} "
          f"CV={statistics.pstdev(rows['Ens']['per_seed_avg_rmse']) / statistics.mean(rows['Ens']['per_seed_avg_rmse']) * 100:.1f}%")
    # ablation deltas vs F-A3
    fa3 = ablation["F-A3"]["avg"]["rmse"]["mean"]
    print(f"\n=== Ablation deltas vs F-A3 {fa3:.4f} ===")
    for v in ("F-A7", "F-A8", "F-A10"):
        m = ablation[v]["avg"]["rmse"]["mean"]
        print(f"{v}: {m:.4f} ({100 * (m - fa3) / fa3:+.2f}%)")


if __name__ == "__main__":
    main()
