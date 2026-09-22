"""Cross-seed gate-weight summary (reviewer point: is Fig. 2 a seed-42 fluke?).

The paper's Fig. 2 paragraph was written against seed 42 alone, so the claim
"the context gate prefers the target fingerprint on Davis" rested on a single
initialization and could not be distinguished from a seed artifact. The
per-sample gate weights for all five seeds were already extracted by
`analysis_gates.py`; this script aggregates those artefacts and reports the
spread of the *dataset-level* mean weight ACROSS SEEDS, which is the quantity
a single-seed figure cannot show.

Two spreads, two different questions:

  * within a seed (`std_across_samples`, in the per-run JSONs) -- does the
    gate vary pair to pair at all?
  * across seeds (`cross_seed_std`, computed here) -- does the dataset-level
    preference reproduce across initializations?  A preference that held in
    one seed only would have a cross-seed range covering the uniform 1/3
    line, and the per-seed argmax would not be stable.

Provenance.  This script only READS the existing per-run JSONs; it does not
re-extract. That is a property of the *bundle*, not of the project: the
release packager copies `outputs/` with `KEEP_SUFFIX = {".json", ".csv"}`
(see `package_release.py`), so no `best_state.pt` reaches `core/outputs/`.
The checkpoints themselves are intact in the full working tree
(`rail_dta/outputs/`), so every number summarised here IS recomputable --
from there, by re-running the extraction. The packager's stated reason for
dropping them, "read by no audit", is wrong: `analysis_gates.py` and the
three `analysis_gate_intervention*` / `analysis_retrained_matched_state`
scripts all `torch.load` a `best_state.pt`. The per-run JSONs remain the
primary evidence `audit_paper_numbers.py` checks.

Usage: python analysis_gate_weights_5seeds.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATASETS = ("davis", "kiba", "metz")
SEEDS = (42, 52, 62, 72, 82)
VIEWS = {"drug": ("Morgan", "feature-Morgan", "MACCS"),
         "context": ("drug", "target-FP", "sequence")}
UNIFORM = 1.0 / 3.0


def source_path(seed: int) -> Path:
    """`analysis_gates.py` keeps the original filename for its default run."""
    if seed == 42:
        return OUT / "gate_analysis.json"
    return OUT / f"gate_analysis_rail_single_seed{seed}_200.json"


def main() -> None:
    missing = [s for s in SEEDS if not source_path(s).exists()]
    if missing:
        print(f"missing per-run gate JSONs for seeds {missing}", flush=True)
        sys.exit(1)

    per_seed: dict = {ds: {name: {} for name in VIEWS} for ds in DATASETS}
    for seed in SEEDS:
        g = json.loads(source_path(seed).read_text(encoding="utf-8"))
        for ds in DATASETS:
            for name in VIEWS:
                per_seed[ds][name][str(seed)] = g[ds][name]

    summary: dict = {}
    for ds in DATASETS:
        summary[ds] = {}
        for name, views in VIEWS.items():
            m = np.array([per_seed[ds][name][str(s)]["mean_weights"]
                          for s in SEEDS])                       # [5, 3]
            argmaxes = [int(np.argmax(m[i])) for i in range(len(SEEDS))]
            k = argmaxes[0]
            summary[ds][name] = {
                "views": list(views),
                "per_seed_mean_weights": m.tolist(),
                "cross_seed_mean": m.mean(axis=0).tolist(),
                "cross_seed_std": m.std(axis=0, ddof=1).tolist(),
                "cross_seed_min": m.min(axis=0).tolist(),
                "cross_seed_max": m.max(axis=0).tolist(),
                "argmax_per_seed": argmaxes,
                "argmax_view": views[k],
                "argmax_stable": len(set(argmaxes)) == 1,
                # The strict form of "reproduces": the argued view leads
                # uniform in every seed, not just on average.
                "argmax_beats_uniform_all_seeds":
                    bool(np.all(m[:, k] > UNIFORM)),
                # Sample-level variation, pooled over seeds (the paper's
                # "the gates do vary at the sample level").
                "max_std_across_samples": float(max(
                    np.max(per_seed[ds][name][str(s)]["std_across_samples"])
                    for s in SEEDS)),
            }
            s = summary[ds][name]
            print(f"{ds:6s} {name:8s} argmax={s['argmax_view']:13s} "
                  f"stable={str(s['argmax_stable']):5s} "
                  f"beats-uniform-all-seeds={str(s['argmax_beats_uniform_all_seeds']):5s} "
                  f"mean={np.round(s['cross_seed_mean'], 3).tolist()} "
                  f"sd={np.round(s['cross_seed_std'], 3).tolist()} "
                  f"range={[[round(a, 3), round(b, 3)] for a, b in
                            zip(s['cross_seed_min'], s['cross_seed_max'])]}")

    path = OUT / "gate_weights_5seeds.json"
    path.write_text(json.dumps(
        {"seeds": list(SEEDS), "source_files": [str(source_path(s).name)
                                                for s in SEEDS],
         "per_seed": per_seed, "summary": summary},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {path}", flush=True)


if __name__ == "__main__":
    main()
