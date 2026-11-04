# H11 probe: findings log

A running log of results from the `h11_probe/` pipeline. Each finding states a claim, the exact numbers behind it and where they came from, the caveats, and the hypothesis in [hypothesis.md](../hypothesis.md) it bears on.

All probe numbers are macro-F1 (%) on the 3-class tweet sentiment test split (3,432 rows), from a logistic regression fitted on the full train split (26,732 rows). The probe uses the same settings as `evaluate.py`: `LogisticRegression(random_state=42, max_iter=1000)` with sklearn defaults otherwise and no feature scaling. Features are stored as float32.

---

## Finding 1. Untrained TinyBERT beats every trained model at every width

**Claim.** The frozen, untrained TinyBERT-6L `pooler_output` scores higher than the best of the four trained models (MRL, MRL+SIA, MRL+PIC, MIPIC) in the README at every nested width from 16 to 768. The gap is 3.86 to 5.61 macro-F1 points. On this evaluation, none of the training recipes improves on the starting backbone.

**Evidence.**

| Width | Untrained TinyBERT (pooler) | Best trained model | Best trained F1 | Gap (untrained minus best) |
|---:|---:|:---|---:|---:|
| 16 | 62.62 | MIPIC | 58.60 | +4.02 |
| 32 | 66.12 | MIPIC | 60.51 | +5.61 |
| 64 | 67.96 | MRL | 62.82 | +5.14 |
| 128 | 69.61 | MRL | 65.57 | +4.04 |
| 256 | 70.68 | MRL | 66.82 | +3.86 |
| 512 | 71.63 | MRL | 67.61 | +4.02 |
| 768 | 71.73 | MRL | 67.24 | +4.49 |

Sources:
- Untrained column: `h11_probe/results/base_grid.csv`, rows with `layer = pooler`. All 7 fits converged.
- Trained columns: the "Current controlled results" table in [README.md](../README.md), also stored in `h11_probe/expected_readme.json`.

Both sides use the same read-out (`pooler_output`, the vector `evaluate.py` probes), the same probe settings and the same train and test files.

**Caveats.**
- **Single training seed.** Each trained model comes from one run, with no seed variation. `run_experiments.py` itself notes that two identical MRL runs differed by about 3 F1 at 768D before seeding was added.
- **Not official TweetEval.** The evaluation files are the Kaggle Tweet Sentiment Extraction data, not the public TweetEval benchmark (see the README). The result may not carry over to TweetEval or to other tasks.
- **Different environment.** The README numbers were produced on Kaggle (2x T4), and our base numbers were produced locally (RTX 3060, sklearn 1.9.0). As a cross-check, our rerun reproduces the untrained-backbone numbers quoted in the `evaluate.py` comment (62.62 to 71.99) to within 0.26 points at every width. That is far smaller than the 3.86 point minimum gap here.
- **Epoch selection is not documented for every model.** MIPIC's README numbers come from the epoch the held-out probe selected (epoch 1 of 5). The README doesn't say which epoch the other three models' numbers come from.
- **Not yet significance-tested.** A paired bootstrap on the saved predictions is still to do. A gap of at least 3.86 points is large relative to the roughly 0.8 point standard error of macro-F1 on 3,432 test rows, but no formal test has been run.

**Relates to.**
- **H17** (MIPIC matches baselines at full capacity). The right baseline includes the untrained backbone, and on this evaluation every trained model is below it. MIPIC is 7.09 points below untrained TinyBERT at 768D (64.64 vs 71.73).
- **H14** (PIC makes small prefixes learn task information early in training). MIPIC's held-out probe preferred epoch 1 of 5 ([MIPIC_TRAINING_README.md](../MIPIC_TRAINING_README.md)). Together with this finding, that fits training degrading the encoder for this task rather than PIC speeding up learning. `run_experiments.py` makes a similar note: the contrastive loss is essentially solved within one epoch, and later epochs lowered MRL's 768D score to 62.52 against 70.55 for the untrained backbone.

---

## Finding 2. In base TinyBERT, layer 5 [CLS] beats layer 6 [CLS] at narrow widths

**Claim.** In untrained TinyBERT, a prefix of the layer 5 [CLS] vector is a better sentiment feature than the same-width prefix of the final layer 6 [CLS] at widths 16 to 128, by 0.77 to 1.70 points. At 256 and 512 the two layers are nearly tied, and at 768 layer 6 is ahead. The final layer is not the best place to read short prefixes, even before any training.

**Evidence.**

| Width | Layer 5 [CLS] | Layer 6 [CLS] | Layer 5 minus layer 6 |
|---:|---:|---:|---:|
| 16 | 63.86 | 62.16 | +1.70 |
| 32 | 67.07 | 66.13 | +0.94 |
| 64 | 69.57 | 68.46 | +1.11 |
| 128 | 71.63 | 70.86 | +0.77 |
| 256 | 72.27 | 72.19 | +0.09 |
| 512 | 73.07 | 72.79 | +0.28 |
| 768 | 72.71 | 73.51 | -0.80 |

Source: `h11_probe/results/base_grid.csv`, rows with `layer = 5` and `layer = 6`. The vectors are the raw `hidden_states[L][:, 0, :d]`, with no pooler applied. All fits at widths 16 to 128 converged quickly (at most 111 iterations).

**Caveats.**
- **Small gaps, not yet tested.** Only the 16D gap (+1.70) is clearly larger than the rough 0.8 point standard error. The 32D, 64D and 128D gaps are close to it. Because both layers are scored on the same test rows, a paired bootstrap on `h11_probe/results/base_preds.npz` (keys `L5_d16` vs `L6_d16` and so on) will give a sharper answer and still needs to be run.
- **Wide widths are affected by convergence.** Layer 5 at 512D and 768D hit the 1000-iteration cap, so the comparison at those widths, including the 768D reversal, partly reflects the solver.
- **Base model only.** This describes the starting point. Whether MRL, MRL+PIC or MIPIC change the ordering is the next comparison.

**Relates to.**
- **H11** (early-layer, low-dimensional prefixes become task-useful). This sets the baseline H11 must be judged against. Intermediate-layer prefixes are already competitive in the untrained model, so a MIPIC advantage at an early checkpoint has to be measured against base TinyBERT at the same `(layer, width)` cell, not against the final layer.
- Also relevant to **H8** (lower layers hold syntax, higher layers hold task signal). For sentiment in TinyBERT-6L, task signal at narrow widths peaks at layer 5, not at the last layer.
