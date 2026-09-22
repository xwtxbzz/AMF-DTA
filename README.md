<h1 align="center">AMF-DTA</h1>

<p align="center">
  <b>A Causal Analysis of Gate Interventions in Adaptive View Fusion<br>
     for Drug-Target Affinity Prediction</b><br>
  <sub>ICASSP 2027 submission &middot; code, data and result artefacts</sub>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c">
  <img alt="Benchmarks" src="https://img.shields.io/badge/benchmarks-Davis%20%7C%20KIBA%20%7C%20Metz-informational">
  <img alt="Audit" src="https://img.shields.io/badge/paper%20numbers-187%20checks%2C%200%20mismatched-success">
</p>

<p align="center">
  <img src="assets/architecture.png"
       alt="The AMF-DTA framework. Drug fingerprints and the protein sequence are encoded into five views; a drug-view gate learns per-sample view weights and a cross-modal context gate learns a context vector. A single interaction slot combines them, and a three-branch score mixer with a training-statistics bias prior produces the affinity."
       width="540">
</p>

---

## Key finding

**The usual way of measuring an adaptive fusion stage is biased. AMF-DTA is
the case study the corrected measurement is applied to.**

Freezing a trained model's gates to a constant `1/K` degrades RMSE in all 15
(seed, dataset) pairs — a median **+21.0%**. Retraining the *same*
architecture from scratch under those weights, which removes the gates from
training rather than from an already-trained model, loses a median **+0.9%**.
The two arms differ only in *when* the gates were removed, not in the state
they were evaluated at, so a post-hoc intervention of this kind overstates
gating's architectural value by **~24×** — and across the eighteen
estimator/trim variants we formed (three estimators × three trims, pooled or
per dataset) it never falls below **9.7×**, its weakest cell KIBA.

The comparison is **state-matched**: both arms are evaluated on
best-validation checkpoints, and the gated arm is required to reproduce the
post-hoc arm's own recorded baseline to within `1.4e-6` RMSE. Comparing a
post-hoc delta taken from one checkpoint against a retrained delta taken from
another is the specific error this repository exists to rule out — the
protocol gap between those states (2.4% for the no-gate arm, `F-A3` at 0.4469
best-validation against 0.4363 final-fit) is larger than the effect being
measured (~1%).

Applied to the case study, the corrected value is narrower than the design
assumes. A permutation placebo that keeps the learned gate weights but breaks
their pairing with the samples costs a median **+3.4%** against uniform's
+21%, so most of that penalty is the lost *dataset-level* weighting rather
than per-sample assignment — except on KIBA, where the placebo costs as much
as uniform (13.0% vs 12.7%). That condition is testable by training: `F-D1`
keeps both gate MLPs at their parameter count while feeding each a learned
constant, and costs a median **+1.0%** against the probe's +1.4% over the 13
pairs whose checkpoint exists (`p = 0.31`) — the magnitude holds, the
per-dataset ordering does not.

The paper also reports **negative results**, and they are kept visible here:

- The gates are worth **~1%**: removing both costs a median 0.9%, worse in
  13 of 15 pairs (`p = 3.4e-3`) — at or below the 1.7–2.1% per-seed spread of
  the protocol. Only the **product interaction** (+8.7% when removed) exceeds
  seed noise;
- RAIL, a DTA adaptation of rational (RAN) interactions, is
  **statistically indistinguishable** from fixed hand-designed terms (`F-A3`
  ahead on 10 of 15 pairs, difference bounded within ±1.3%, `p = 0.28`). The
  reported slot is a parameter-matched lightweight MLP that improves on it,
  but it does not differ from its own degree-2 polynomial basis (`p = 0.45`),
  so the choice is post hoc — and the paper reports the nominally *worse* of
  the two.

For balance, the positive result the caution sits on: under the matched
five-seed protocol the single model reaches **0.4327** average RMSE (CI
**0.8782**), **8.5%** below the strongest baseline we reproduced (FingerDTA,
`0.4727 ± 0.0106`), and the five-member ensemble reaches **0.4089**
(CI 0.8897). The ensemble gain is attributed to averaging rather than to the
design, and the lead is over from-scratch re-implementations only, so it is
not a claim against pretrained-embedding methods.

## Repository layout

The layout **mirrors the original working tree on purpose**: every analysis
and audit script resolves its inputs by relative path from its own location
(`__file__.parent.parent / "outputs"`), so a script moved to a different depth
would silently read the wrong directory. The shape is preserved for that
reason, not for tidiness.

Two directories are renamed, and that is the *only* difference from the tree
the numbers came from. `rail_dta/` becomes `core/` — nothing imports that name,
so the rename is free. The model package inside it becomes `framework/`, which
scripts *do* import, so its rename carries a matching import rewrite. The edit
is mechanical and was *checked* rather than asserted: the build script that
produced this bundle applied the same text transformation to each source file
and required the shipped bytes to match, so any edit beyond the package name
failed the build. That script is not shipped here, and it is now retired —
this repository, not its output, is the artefact of record, so the files here
are edited directly rather than regenerated. Documentation prose that names the
source tree's own paths (`rail_dta/outputs/`) is deliberately left alone — it
describes where artefacts land in the tree the numbers came from.

```
assets/                      rendered figures for this README
data/{davis,kiba,metz}/      benchmarks: smiles / sequence / affinity
outputs/                     result JSON+CSV, baseline and architecture rows
core/
  outputs/                   result JSON+CSV: the AMF-DTA runs, the F-A* slot
                             ablation, the F-D1 sweep, the cold-start pilot
  framework/                 model, modules, data pipeline, ensemble
  *.py                       training drivers, analyses and audits
autoresearch/
  outputs/                   JSON evidence: Table II upper group (a1–a6),
                             parameter-count ablations, and the *superseded*
                             id-keyed cold runs -- the paper's cold rows are
                             core/outputs/rail_cold_*, driven by
                             core/run_cold_split.py, which keys on the
                             entity string and so leaks nothing
  figures/                   figure sources
dta_experiment/              framework: data, training loop, metrics
models/                      model implementations (imported by the framework)
configs/                     YAML experiment configurations
```

## Installation

```bash
git clone https://github.com/xwtxbzz/AMF-DTA.git
cd AMF-DTA
pip install -r requirements.txt
```

`requirements.txt` merges the framework's and the rail tree's manifests.
`torch-geometric` and `rdkit` are needed only by the graph-input and
fingerprint-input models respectively; the audits need neither a GPU nor
either of them.

## Reproducing the paper's numbers

The audits recompute every claimed value from the recorded artefacts. **No
GPU and no checkpoints are required** — the JSON and CSV under `outputs/`
are the complete evidence. The manuscript itself is *not* part of this
repository, so the 16 checks that read the paper's own text report as a single
`[skip]` line and `audit_paper_numbers.py`'s own total is 151 rather than 167;
every check that reads the artefacts alone still runs. Run from the
repository root:

```bash
# PYTHONPATH is the repository root, not core/: audit_architecture.py
# imports the `dta_experiment` package, which lives at the root. Without it
# that audit does not fail -- it prints one `[skip]` line, exits 0, and the
# total silently drops from 187 to 183.
PYTHONPATH=. python core/audit_paper_numbers.py

# Which numeric literals in the paper are not pinned by any check.
PYTHONPATH=. python core/audit_coverage.py

# The headline ratio, recomputed from the raw per-pair rows.
PYTHONPATH=. python core/_verify_headline_arithmetic.py

# Regenerate the values quoted in the manuscript from the artefact.
PYTHONPATH=. python core/_emit_claim_from_artifact.py
```

`audit_paper_numbers.py` takes an optional `.tex` path, so the manuscript
checks can be pointed at a copy of the paper kept outside this repository:
`python core/audit_paper_numbers.py path/to/ICASSP2027_armvdta.tex` reports
167 rather than 151, and all 16 pass against the submission draft.
`audit_coverage.py` also reads the paper's text but takes no argument, so it
degrades to a single `[skip]` line here.

The scripts prefixed with `_` are deliberately *independent* re-derivations —
the one link an audit of that shape cannot cover, because
`audit_paper_numbers.py` checks the manuscript against the artefact while
these check the artefact against itself. `_verify_headline_arithmetic.py`
runs to completion from this bundle. `_emit_claim_from_artifact.py` reads the
eight quoted values back out of the artefact and re-finds each one in the
manuscript; it currently exits 1 after verifying seven of them, because
`dataset_gate` — the eighth kind, added to the audit but not to this emitter's
claim table — raises `KeyError: 'dataset_gate'`.

## Training from scratch

No checkpoint is committed: `*.pt` is in `.gitignore`, so the repository
carries none of the ~12 GB the full training tree holds. The working tree does
keep 20 of them (277 MB) — the F-D1 sweep's 14 and the two seed-42 cold-start
runs' 6 — left in place because they are the only copy. To regenerate any
single run:

```bash
python -m dta_experiment.train --config configs/{dataset}_{model}.yaml
```

The five-seed sweeps are driven by `core/run_*.py` and the root-level scripts
below, with `MAX_PARALLEL = 3` and resume support. Expect the full sweep to
take GPU-days.

## Datasets

| Benchmark | Drugs | Targets | Pairs | Label scale |
|---|---:|---:|---:|---|
| Davis | 68 | 442 | 30,056 | pKd |
| KIBA | 2,111 | 229 | 118,254 | KIBA score |
| Metz | 1,423 | 170 | 35,259 | pKd |

Each directory holds three CSVs — `smiles.csv` (drug id to SMILES),
`sequence.csv` (target id to sequence) and `affinity.csv` (drug, target,
value). A repetition seed sets both the random 8:1:1 partition and the
initialization, so every method sees the same split at a given seed and all
comparisons are **paired** over the 15 (seed, dataset) pairs at seeds
{42, 52, 62, 72, 82}.

## Verification

The three audits report **187 checks, 0 mismatched** in total —
`audit_paper_numbers.py` 151, `audit_baseline_rows.py` 32 and
`audit_architecture.py` 4 (151 + 32 + 4 = 187). `audit_paper_numbers.py` runs
the other two as subprocesses, so the single command above prints all three
tallies, and it is their sum that both this section and the retired build
script count. The count is the guard, not decoration: a repository missing one
input does not error, it silently *skips* checks, so an unchanged total is the
signal that every input arrived. The `PYTHONPATH` note above is that failure
mode in miniature — get it wrong and `audit_architecture.py`'s four checks
become one `[skip]` line, and 187 reads as 183.

The 16 manuscript-reading checks are the difference from the 203 the full
working tree produces. They come in pairs for each of the eight quoted values:
one that the value appears the expected number of times, and one that every
copy of it agrees.

## What is not included, and why

| Excluded | Size | Reason |
|---|---|---|
| `*.pt` checkpoints | ~12 GB | `.gitignore` excludes them, so the repository carries none, and they are regenerable by training. The working tree keeps 20 (277 MB): the F-D1 sweep and the seed-42 cold runs. Not because nothing reads checkpoints — seven scripts `torch.load` a `best_state.pt` — but because the ones they need belong to the *training* runs, which are absent. The consequence is that those seven cannot run from this bundle alone; the per-run JSON they produced is shipped and audited instead |
| training logs | — | Superseded by `history.csv` / `test_metrics.json` |
| `data/{chembl,bdb,ttd}_csv` | 3.6 GB | Used by no experiment in the paper |
| `autoresearch/outputs/**/*.csv` | 161 MB | Prediction dumps with no reader in the chain; the JSON summaries beside them carry every value the audits check |

One driver in `core/` is left in place but no longer has artefacts behind it:
`run_fingerdta_ensemble_seeds.py` trains the four remaining members of a
FingerDTA *ensemble* — a row the superseded draft reported and the manuscript
does not — so running it would only recreate directories no claim rests on.
The manuscript's FingerDTA row is instead the cited baseline, re-implemented
in the root tree and audited by `audit_baseline_rows.py`. The third driver of
that lineage, `run_amfdta_review_5seeds.py`, was *kept* for its `F-M1`/`F-M2`
variants (Table I's reported row; Table II's MLP and polynomial slots), with
its FingerDTA-ensemble tail removed, since leaving the tail would have
retrained the deleted members on the next run.

## Citation

Provisional — replace with the published entry once available:

```bibtex
@inproceedings{wang2027amfdta,
  title     = {A Causal Analysis of Gate Interventions in Adaptive View
               Fusion for Drug-Target Affinity Prediction},
  author    = {Wang, Zhensong and Sun, Yuwei and Wang, Juan},
  booktitle = {IEEE International Conference on Acoustics, Speech and
               Signal Processing (ICASSP)},
  year      = {2027},
  note      = {Under review}
}
```

## License

**No license has been chosen yet.** Until a `LICENSE` file is added, the
default applies: all rights reserved, and no permission is granted to reuse
this code. Add one before making the repository public.

## Authors

- **Zhensong Wang**, **Yuwei Sun**, **Juan Wang** (corresponding) —
  Inner Mongolia University, Hohhot, China

## Root-level drivers

Sweep drivers and utilities that live at the top level:

- `check_status.py`
- `compile_all_results.py`
- `generate_model_configs.py`
- `run_2024_models_5seeds.py`
- `run_2025_2026_models_5seeds.py`
- `run_all_experiments.py`
- `run_edta_resume.py`
- `run_graphdta_attentiondta_5seeds.py`
- `run_missing_5seeds_200.py`
