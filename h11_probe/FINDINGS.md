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

**Update, 2026-11-04: against our own seed 42 retrain (PRELIMINARY).** We retrained the models locally with identical code and settings (see Finding 4), and they score 2 to 7 points higher than the README's models. Repeating the comparison against these retrained models, base TinyBERT **still wins at every width, but the gap shrinks to about 0.2 to 2.3 points** (from 3.86 to 5.61 above).

| Width | Untrained TinyBERT (pooler) | Best seed 42 model (preliminary) | Best seed 42 F1 | Gap (untrained minus best) |
|---:|---:|:---|---:|---:|
| 16 | 62.62 | MRL+PIC | 61.43 | +1.19 |
| 32 | 66.12 | MRL | 63.86 | +2.26 |
| 64 | 67.96 | MRL | 65.90 | +2.06 |
| 128 | 69.61 | MRL | 68.64 | +0.97 |
| 256 | 70.68 | MRL | 69.73 | +0.95 |
| 512 | 71.63 | MRL | 70.96 | +0.68 |
| 768 | 71.73 | MRL | 71.52 | +0.22 |

Source: `h11_probe/runs/seed42/tweeteval_results.json`, `evaluations` entries for `mrl`, `mrl_pic` and `mipic`. These are the epochs chosen by the held-out probe (`best.pt`). All probe fits converged. MRL+SIA was still training when this table was written, so it isn't included. It was the weakest model in the README at every width, so it's unlikely to change the "best" column.

The narrow gaps at 512D and 768D (0.68 and 0.22) are well within the rough 0.8 point standard error, so at full width base and retrained MRL are effectively tied on this evidence. MRL's last-epoch checkpoint (`mrl_last_epoch`) gets even closer at 768D (71.66, a gap of 0.07). **This is one seed. The finding will be finalised after seeds 1 and 2**, together with a paired bootstrap.

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

---

## Finding 3. On STS-B, base TinyBERT's best layer is layer 2, at every width

**Claim.** For semantic similarity on STS-B, untrained TinyBERT's layer 2 [CLS] is the best of all seven rows (layers 1 to 6 and `pooler_output`) at every width, ahead of the next best layer (layer 3) by 5.71 to 7.89 Spearman points. A 32-dimensional prefix of layer 2 (59.29) beats the full 768-dimensional `pooler_output` (49.56) by 9.73 points.

**Evidence.** Spearman correlation (x100) between the cosine similarity of the truncated vectors and the gold scores, on STS-B test pairs where neither sentence appears in the training corpus (n = 1,261).

| Width | Layer 2 [CLS] | Next best row | Next best Spearman | Margin |
|---:|---:|:---|---:|---:|
| 16 | 55.73 | layer 3 | 50.02 | +5.71 |
| 32 | 59.29 | layer 3 | 53.17 | +6.12 |
| 64 | 62.23 | layer 3 | 55.30 | +6.93 |
| 128 | 63.79 | layer 3 | 56.28 | +7.51 |
| 256 | 64.17 | layer 3 | 56.28 | +7.89 |
| 512 | 63.49 | layer 3 | 56.57 | +6.92 |
| 768 | 63.79 | layer 3 | 56.66 | +7.13 |

For comparison, the final-layer read-outs at full width are `pooler_output` 49.56 and layer 6 [CLS] 49.74.

Source: `h11_probe/results/base_sts_clean.csv` (produced by `sts_eval.py --drop-overlap`). Using all 1,379 test pairs (`base_sts.csv`) gives the same conclusion: layer 2 is best at every width by 7.15 to 9.13 points, and layer 2 at 32D (60.00) beats the pooler at 768D (48.87).

**Caveats.**
- **Raw [CLS] cosine is a weak baseline for the final layer.** The final-layer [CLS] of BERT-style models is known to work poorly for cosine similarity without fine-tuning, so the low layer 6 and pooler scores are expected. The result shows where similarity information is easiest to read with plain cosine. It doesn't show that early layers "understand" meaning better.
- **No learned component.** Unlike the sentiment probe, this is zero-shot cosine with no fitted classifier, so it rewards geometry (how the vectors are spread out) as well as content.
- **Base model only, one run.** No trained models yet, and no confidence intervals. With n = 1,261 the margins of 5.7 points or more are large, but a bootstrap is still pending.
- **Overlap removal.** 118 of 1,379 pairs were dropped because a sentence appears verbatim in `data/final_data.csv` (all from its STS-B rows). For base this can't be contamination, since base never trained on that file. It is the version that will be fair to the trained models.

**Relates to.**
- **H8** (lower layers hold syntax, higher layers hold task signal). For STS in base TinyBERT-6L, the most usable similarity signal sits at layer 2, which is earlier than for sentiment (layer 5 at narrow widths, Finding 2). "Task signal" peaks at different depths for different tasks, so the simple lower-is-syntax, higher-is-task picture doesn't describe this model on its own.
- **H11** (early-layer, low-dimensional prefixes become task-useful). On STS, the base model's PIC checkpoint at (layer 2, 32D) already scores 59.29. A MIPIC advantage at that checkpoint would need to beat an already strong cell, not just improve on a weak early layer.

**Prediction, recorded 2026-11-04 before any trained-model STS results.** SimCSE-trained models (MRL, MRL+PIC, MIPIC) will beat base TinyBERT on STS-B at the final layer, even though they lose to base on sentiment (Finding 1). Concretely, on the clean pairs, their `pooler_output` at 768D will exceed base's 49.56 and their layer 6 [CLS] at 768D will exceed base's 49.74. Reasoning: the SimCSE objective is contrastive on sentence pairs and is known to fix the poor cosine geometry of untrained final layers, which would help STS directly while doing nothing for, or even hurting, a linear sentiment probe.

---

## Finding 4. Retraining with identical code gives 2 to 7 points more than the README (PRELIMINARY, seed 42)

**Claim.** Retraining MRL, MRL+PIC and MIPIC with the unmodified repository code and its default settings (seed 42, 5 epochs, lr 2e-5, global batch 16, held-out epoch selection) on one local RTX 3060 gave TweetEval macro-F1 2.00 to 7.16 points higher than the README's Kaggle run. This held for all three models at every width. The run-to-run difference is as large as, or larger than, the differences between methods that the README reports.

**Evidence.** Macro-F1 (%) of the `pooler_output` probe. Local = seed 42 retrain (preliminary). README = "Current controlled results" table. Diff = local minus README.

| Width | MRL local | MRL README | Diff | MRL+PIC local | MRL+PIC README | Diff | MIPIC local | MIPIC README | Diff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 59.87 | 56.46 | +3.41 | 61.43 | 54.27 | +7.16 | 61.26 | 58.60 | +2.66 |
| 32 | 63.86 | 59.99 | +3.87 | 62.49 | 59.85 | +2.64 | 62.55 | 60.51 | +2.04 |
| 64 | 65.90 | 62.82 | +3.08 | 65.08 | 61.62 | +3.46 | 63.83 | 61.83 | +2.00 |
| 128 | 68.64 | 65.57 | +3.07 | 66.10 | 62.55 | +3.55 | 65.57 | 62.65 | +2.92 |
| 256 | 69.73 | 66.82 | +2.91 | 67.23 | 63.21 | +4.02 | 66.37 | 62.92 | +3.45 |
| 512 | 70.96 | 67.61 | +3.35 | 69.20 | 64.27 | +4.93 | 67.24 | 64.01 | +3.23 |
| 768 | 71.52 | 67.24 | +4.28 | 69.80 | 64.82 | +4.98 | 67.71 | 64.64 | +3.07 |

Sources:
- Local: `h11_probe/runs/seed42/tweeteval_results.json` (`evaluations.mrl`, `.mrl_pic`, `.mipic`), produced by `h11_probe/train_all.ps1 -Seed 42`. All 21 probe fits converged.
- README: [README.md](../README.md), also in `h11_probe/expected_readme.json`.

**Run-to-run differences are as big as the README's method gaps.** The README's headline MIPIC advantage over MRL at 16D is +2.1 (58.60 vs 56.46). That is smaller than 19 of the 21 local-minus-README differences above (mean +3.53), and about the same as the smallest two (+2.00 and +2.04). So a single run per model can't separate method effects of that size from run-to-run variation.

**Model ranking at narrow widths in seed 42 (preliminary).**
- At 16D, MRL+PIC (61.43) and MIPIC (61.26) are tied, 0.17 apart, and both are ahead of MRL (59.87) by 1.56 and 1.39 points.
- At 32D the order flips: MRL (63.86) is ahead of MIPIC (62.55) by 1.31 and of MRL+PIC (62.49) by 1.37.
- From 64D upward MRL leads, as in the README.

**Caveats.**
- **Different hardware and parallelism.** Local is one RTX 3060. The README used Kaggle with 2x T4 under DDP. The global batch is 16 in both, and the number of in-batch negatives is the same, but the numerical kernels and AMP behaviour differ.
- **Different data order and dropout masks.** One process versus two DDP ranks gives a different shuffling of the training corpus and different dropout masks, even with the same seed. This is ordinary run-to-run randomness, and it's what this finding suggests is large.
- **Epoch selection doesn't explain it.** Locally the held-out probe chose epoch 1 for MRL and MRL+PIC and epoch 3 for MIPIC, while the README's MIPIC used epoch 1. But the local last-epoch checkpoints (`*_last_epoch` in the same JSON) are also above the README at every width, by 0.55 to 4.42 points.
- **Only one local seed so far.** Seeds 1 and 2 are pending. They will show whether seed 42 is a lucky draw or whether local runs are systematically higher than Kaggle. Without them, "local vs README" mixes seed variance with any systematic environment effect.
- **Not significance-tested.** No bootstrap yet. The rough standard error of macro-F1 on 3,432 test rows is about 0.8 points, so the 16D tie between MRL+PIC and MIPIC (0.17) is a tie, and the 1.3 to 1.6 point gaps against MRL are marginal on their own.

**Relates to.**
- **H15** (SIA is essential: MIPIC without SIA, i.e. MRL+PIC, should be clearly worse than MIPIC at 16D and 32D). In seed 42 it isn't: MRL+PIC and MIPIC are tied at 16D (61.43 vs 61.26) and at 32D (62.49 vs 62.55). On this one seed, adding SIA on top of PIC gives nothing at small widths.
- **H16** (SIA and PIC are synergistic). PIC's gain over MRL at 16D (+1.56) is already as large as MIPIC's (+1.39), which leaves no room for a positive interaction unless MRL+SIA comes out below MRL at 16D. The interaction term can be computed once MRL+SIA finishes.
- **H18** (MIPIC's advantage is largest at 16 to 32D). It holds at 16D, but at 32D plain MRL beats MIPIC in seed 42.
- **H17** and Finding 1. The 768D gap between MRL and MIPIC grows from 2.6 in the README to 3.8 locally (71.52 vs 67.71).
- Every comparison between methods in the README that is smaller than about 3 points should be treated as unresolved until it is replicated over seeds.
