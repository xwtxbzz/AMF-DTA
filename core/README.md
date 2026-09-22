# AMF-DTA

**A Causal Analysis of Gate Interventions in Adaptive View Fusion for
Drug-Target Affinity Prediction** — code for the ICASSP 2027 submission.
AMF-DTA is the *case study* the paper's measurement protocol is applied to;
`../README.md` states what that protocol found and why the usual way of
measuring an adaptive fusion stage is biased.

AMF-DTA is a two-stage adaptive filterbank over views: a per-pair drug-view
reliability gate over three fingerprints, a cross-modal context gate merging
the drug with the two protein views, an interchangeable interaction slot, and
a multi-branch score mixer whose bias branch consumes training-set-only
statistics.

```
../data/
  davis/{smiles,sequence,affinity}.csv     (DRUGID,SMILES / TARGETID,SEQUENCE / DRUGID,TARGETID,VALUE)
  kiba/  ...
  metz/  ...
framework/
  modules.py   RationalUnit1D / RationalUnit2D / RailInteractionLayer
  model.py     AMFDTA (dual-gate fusion + swappable interaction slot + score mixer)
  data.py      fingerprints, tokenizer, fixed 8:1:1 split, bias statistics, metrics
  train.py     paper protocol: 200 epochs, final-fit + SWA, AMP, alpha 10x lr
  ensemble.py  5-member equal-weight ensemble (single + m1..m4)
run.py         CLI
analysis_*.py  gate interventions, matched-state arms, ratio sensitivity
audit_*.py     the paper-numbers audit (described in ../README.md)
run_*.py       the sweeps that produced the shipped artefacts
```

## Install & data

```bash
pip install -r ../requirements.txt
# place Davis / KIBA / Metz CSVs under ../data/<dataset>/ (see layout above)
```

## Train

```bash
# the `single` member's config (interaction="rail") -- the config behind
# Table II's F-A8 row; writes to outputs/rail_single/, not to the shipped
# rail_single_seed*_200/ (see the naming note below)
python run.py train --tag single

# all five ensemble members
python run.py train-all

# equal-weight ensemble from member test predictions
python run.py ensemble

# quick sanity check (2 epochs, davis only)
python run.py train --tag single --smoke

# parameter count
python run.py params --tag single
```

Outputs go to `outputs/<run_tag>/<dataset>/`:
`history.csv`, `best_state.pt` (best-validation-RMSE checkpoint),
`test_predictions.csv`, `test_metrics.json` (evaluated on the final-fit SWA
state), plus `summary.json` per run.

`run.py` names the directory after the config alone (`rail_single`,
`rail_m1`, …). The shipped five-seed artefacts instead append the seed and
epoch budget (`rail_single_seed42_200`), because the sweep drivers each pass
their own `run_tag`; the two naming schemes are the same layout at different
paths, not different results.

## Paper protocol (Sec. III-A) — fixed for all models

| Setting | Value |
|---|---|
| split | random 8:1:1 at the run's seed, tokenizers/statistics fit on train only |
| seeds | 42/52/62/72/82 for the reported rows; Table II's upper group is seed 42 only, as its caption says |
| epochs | 200 (best-validation-RMSE checkpoint) |
| final fit | 12 epochs on train+val, SWA from epoch 10 (the reported test numbers; members `m1`–`m4` use 13) |
| optimizer | Adam, lr 2e-4 (RAIL α-gates 10x = 2e-3, wd 0), wd 1e-5, clip 5.0 |
| batch / AMP | 192 / FP16 mixed precision |
| fingerprints | Morgan (2048, r=2), feature-Morgan (2048), MACCS (167), protein k-mer hash (k=3, 1024) |
| protein sequence | char tokenizer, max 1000 residues (embedding + mean pool, see note) |
| label norm | per-dataset z-score from train statistics; Davis is the identity, its labels already normalized |
| ensemble | 5 members (fusion width 512–768, dropout 0.2–0.25, hidden 256–384), equal weight |

## Mapping to the paper's numbers

| Table row | How it is produced |
|---|---|
| Table I AMF-DTA (M1) .4327±.0082, CI .8782 | `run_amfdta_review_5seeds.py` — `interaction="mlp_interact"`, five seeds |
| Table I AMF-DTA (Ens.) .4089±.0090, CI .8897 | `run_amfdta_5seeds.py` (members `single`, `m1`–`m4`), then `run.py ensemble` |
| Table I FingerDTA .4727±.0106 | **not this package's artefacts**: the root tree's `outputs/{ds}_fingerdta_seed{s}_200/`, audited by `audit_baseline_rows.py`. `run.py train --tag fingerdta` trains a FingerDTA baseline here too, but it writes to `outputs/fingerdta/<dataset>/`, which nothing in the paper quotes |
| Table II upper A1–A6 (seed 42) | `TrainConfig` switches, all on `interaction="fixed"`: A1 both gates off, A2 `use_drug_gate=False`, A3 the full config (the same one F-A3 runs over five seeds), A4 `interaction="no_abs"`, A5 `interaction="no_product"`, A6 `use_bias=False`. These are `TrainConfig` fields rather than `run.py` flags, and the driver that enqueued them belongs to the working tree's `autoresearch/`, not to this bundle — the per-run JSON under `autoresearch/outputs/a1..a6/` is what ships |
| Table II lower F-A3 / F-A7 / F-A8 / F-A10 | `run_amfdta_ablation_5seeds.py` — `fixed` / `rail_fixed_interact` / `rail` / `rail` with `residual_gate=False` |
| Table II F-M1 / F-M2 | `run_amfdta_review_5seeds.py` — `mlp_interact` / `poly_interact` |
| Table II F-A11 (retrained no-gate) | `run_no_gate_5seeds.py` |
| Sec. III-D F-D1 (dataset-level gate) | `run_dataset_gate_5seeds.py` — `gate_mode="dataset"` |
| Sec. III-D F-A9 (diverges) | `run_amfdta_ablation_5seeds.py`'s probe — davis, seed 42, `positive_denominator=False` |
| cold-start rows | `run_cold_split.py` — `cold_key="entity"`, both splits; a one-seed pilot, and its summary records `n_seeds` so it cannot be read as a five-seed result |

Two naming traps, both of which the artefacts settle rather than the prose:

* `rail_single_seed*_200` and `rail_ab_FA8_seed*_200` are **one run under two
  names** — byte-identical across all 15 (seed, dataset) pairs — so Table II's
  F-A8 row and the ensemble's `single` member are the same measurements.
* `rail_m1_seed*` / `rail_m2_seed*` (the ensemble members) are **different
  runs** from `rail_ab_FM1_seed*` / `rail_ab_FM2_seed*` (Table I's M1 row and
  Table II's F-M1/F-M2): same configs, but 0/15 pairs identical, so averaging
  the members and quoting the ablation row are not the same computation.

## Implementation notes

* **Pole-free by construction**: every rational unit has denominator
  `1 + softplus(q) + eps >= 1 + eps`; dropping this constraint (F-A9,
  `positive_denominator=False`) makes training diverge with NaNs.
* **Identity-safe residual gating**: α initializes at 0 and receives a 10x
  learning rate, so each unit starts as the identity map / product anchor and
  gradually turns on. F-A10 removes it without collapsing.
* **Sequence encoder**: the paper calls the sequence view the "sequence
  encoder"; the implementation is the character-embedding + masked-mean-
  pooling encoder (`ProteinSequenceEncoder`).
* **The interaction slot is swappable**: `interaction=` selects `fixed`
  (hand-designed product and difference terms), `rail`,
  `rail_fixed_interact` (rational main effects only), `mlp_interact`,
  `poly_interact`, or `no_abs`/`no_product` for the term-removal ablations.
  RAIL is one option among these, and the paper reports it as a negative
  result rather than as the contribution.
* Determinism: `seed_everything(seed)` at the start of every run; dataset
  split, tokenizers and bias statistics are all fixed at that seed.

## Citation

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
