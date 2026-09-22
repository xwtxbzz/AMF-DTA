# RAIL-DTA

**Rational-ANOVA Interaction Layers for Drug-Target Affinity Prediction** —
official PyTorch implementation (ICASSP 2027 submission).

RAIL-DTA casts DTA prediction as a functional-ANOVA decomposition of the
drug-target interaction: main effects plus budgeted pairwise effects, each a
pole-free Padé-style rational unit with identity-safe residual gating, fed by a
sample-level drug-view reliability gate and a cross-modal context gate, and
combined by a multi-branch score mixer whose bias branch consumes
training-set-only statistics.

```
data/
  davis/{smiles,sequence,affinity}.csv     (DRUGID,SMILES / TARGETID,SEQUENCE / DRUGID,TARGETID,VALUE)
  kiba/  ...
  metz/  ...
rail_dta/
  modules.py   RationalUnit1D / RationalUnit2D / RailInteractionLayer  (Sec. II-C)
  model.py     RailDTA (dual-gate fusion + RAIL + score mixer), FingerDTA baseline
  data.py      fingerprints, tokenizer, fixed 8:1:1 split, bias statistics, metrics
  train.py     paper protocol: 200 epochs, final-fit + SWA, AMP, alpha 10x lr
  ensemble.py  5-member equal-weight ensemble (single + m1..m4)
run.py         CLI
```

## Install & data

```bash
pip install -r requirements.txt
# place Davis / KIBA / Metz CSVs under ./data/<dataset>/ (see layout above)
```

## Train

```bash
# single model (the "SWA" row of Table I)
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

Outputs go to `outputs/rail_<config>/<dataset>/`:
`history.csv`, `best_state.pt` (best-validation-RMSE checkpoint),
`test_predictions.csv`, `test_metrics.json` (evaluated on the final-fit SWA
state), plus `summary.json` per run.

## Paper protocol (Sec. III-A) — fixed for all models

| Setting | Value |
|---|---|
| split | fixed random 8:1:1, seed 42, tokenizers/statistics fit on train only |
| epochs | 200 (best-validation-RMSE checkpoint) |
| final fit | 12 epochs on train+val, SWA from epoch 10 (reported test numbers) |
| optimizer | Adam, lr 2e-4 (RAIL α-gates 10x = 2e-3, wd 0), wd 1e-5, clip 5.0 |
| batch / AMP | 192 / FP16 mixed precision |
| fingerprints | Morgan (2048, r=2), feature-Morgan (2048), MACCS (167), protein k-mer hash (k=3, 1024) |
| protein sequence | char tokenizer, max 1000 residues (embedding + mean pool, see note) |
| label norm | per-dataset z-score (Davis identity) |
| ensemble | 5 members (fusion width 512–768, dropout 0.2–0.25, hidden 256–384), equal weight |

## Mapping to the paper's numbers

| Table I row | How to reproduce |
|---|---|
| RAIL-DTA (Single) 0.4363 | best-state eval of `single` member |
| RAIL-DTA (SWA) 0.4229 | test metrics of `single` (SWA state is the default test eval) |
| RAIL-DTA (Ens.) 0.3921 / CI 0.8925 | `python run.py ensemble` |
| FingerDTA† 0.4199 | `python run.py train --tag fingerdta` |
| Ablations A1–A6, F-A7/F-A10 | `TrainConfig` switches: `interaction="fixed"|"no_abs"|"no_product"`, `use_drug_gate`, `use_context_gate`, `use_bias`, `positive_denominator`, `residual_gate` |

## Implementation notes

* **Pole-free by construction**: every rational unit has denominator
  `1 + softplus(q) + eps >= 1 + eps`; dropping this constraint (F-A9,
  `positive_denominator=False`) makes training diverge with NaNs.
* **Identity-safe residual gating**: α initializes at 0 and receives a 10x
  learning rate, so each unit starts as the identity map / product anchor and
  gradually turns on.
* **Sequence encoder**: the paper calls the sequence view the "sequence
  encoder"; the implementation is the character-embedding + masked-mean-
  pooling encoder (`ProteinSequenceEncoder`).
* Determinism: `seed_everything(42)` at the start of every run; dataset
  split, tokenizers and bias statistics are all seed-42 fixed.

## Citation

```bibtex
@inproceedings{raildta2027,
  title={RAIL-DTA: Rational-ANOVA Interaction Layers for Drug-Target Affinity Prediction},
  author={},  % TODO: authors
  booktitle={Proc. IEEE ICASSP},
  year={2027}
}
```
