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

Both sides use the same read-out (`pooler_output`, the vector `evaluate.py` probes), the same probe settings and the same train and test files. Note that the README models were trained with an older code version (MIPIC_FINAL_V20, attention KL on), not the code currently in this repository (see Finding 5).

**Update, 2026-11-04: against our own seed 42 retrain (PRELIMINARY).** We retrained all four models locally with the current repository code and its default settings (see Findings 4 and 5), and they score 2 to 11 points higher than the README's models. Repeating the comparison against these retrained models, **the gap shrinks to between -0.57 and +1.33 points** (from 3.86 to 5.61 above). Base still wins at 32D to 768D, but **at 16D MRL+SIA is 0.57 points ahead of base**, so the headline claim of this finding doesn't hold at every width in seed 42.

| Width | Untrained TinyBERT (pooler) | Best seed 42 model (preliminary) | Best seed 42 F1 | Gap (untrained minus best) |
|---:|---:|:---|---:|---:|
| 16 | 62.62 | MRL+SIA | 63.19 | -0.57 |
| 32 | 66.12 | MRL+SIA | 65.76 | +0.35 |
| 64 | 67.96 | MRL+SIA | 66.63 | +1.33 |
| 128 | 69.61 | MRL | 68.64 | +0.97 |
| 256 | 70.68 | MRL | 69.73 | +0.95 |
| 512 | 71.63 | MRL | 70.96 | +0.68 |
| 768 | 71.73 | MRL | 71.52 | +0.22 |

Source: `h11_probe/runs/seed42/tweeteval_results.json`, `evaluations` entries for `mrl`, `mrl_sia`, `mrl_pic` and `mipic`. These are the epochs chosen by the held-out probe (`best.pt`). All probe fits converged.

Every gap in this table except 64D (1.33) is within the rough 0.8 point standard error, so on this evidence the best retrained model and base are effectively tied at most widths. Neither side clearly wins. MRL's last-epoch checkpoint (`mrl_last_epoch`) gets even closer at 768D (71.66, a gap of 0.07). **This is one seed. The finding will be finalised after seeds 1 and 2**, together with a paired bootstrap.

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

## Finding 4. The README numbers are not reproducible with the current code: retraining gives 2 to 7 points more (PRELIMINARY, seed 42)

**Claim.** Retraining MRL, MRL+PIC and MIPIC with the current, unmodified repository code and its default settings (seed 42, 5 epochs, lr 2e-5, global batch 16, held-out epoch selection) on one local RTX 3060 gave TweetEval macro-F1 2.00 to 7.16 points higher than the README's Kaggle run. This held for all three models at every width.

*Revised 2026-11-04.* An earlier version of this finding put the gap down to run-to-run variation. That was wrong. The README numbers came from an older code version (MIPIC_FINAL_V20) whose configuration differs from the current repository: most clearly, the attention KL term was on (see Finding 5). The README numbers therefore can't be reproduced with the current code, and this comparison mixes configuration, environment and seed effects. True seed-to-seed variance under the current code hasn't been measured yet. Seeds 1 and 2 will measure it.

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

**The README-vs-local gaps are as big as the README's method gaps.** The README's headline MIPIC advantage over MRL at 16D is +2.1 (58.60 vs 56.46). That is smaller than 19 of the 21 local-minus-README differences above (mean +3.53), and about the same as the smallest two (+2.00 and +2.04). These differences are not run-to-run noise: they combine a configuration change (Finding 5), different hardware and parallelism, and seed. Either way, a method gap of 2 points in the README can't be read as a method effect until the comparison is rerun under one configuration with several seeds.

**Model ranking at narrow widths in seed 42 (preliminary).**
- At 16D, MRL+PIC (61.43) and MIPIC (61.26) are tied, 0.17 apart, and both are ahead of MRL (59.87) by 1.56 and 1.39 points.
- At 32D the order flips: MRL (63.86) is ahead of MIPIC (62.55) by 1.31 and of MRL+PIC (62.49) by 1.37.
- From 64D upward MRL leads, as in the README.

**Caveats.**
- **Different code version and configuration.** The README run used MIPIC_FINAL_V20 with attention KL on. The current code has it hardcoded off. This matters only for the arms with SIA (MRL+SIA and MIPIC), but other unlogged changes between versions can't be ruled out for any arm. See Finding 5.
- **Different hardware and parallelism.** Local is one RTX 3060. The README used Kaggle with 2x T4 under DDP. The global batch is 16 in both, and the number of in-batch negatives is the same, but the numerical kernels and AMP behaviour differ.
- **Different data order and dropout masks.** One process versus two DDP ranks gives a different shuffling of the training corpus and different dropout masks, even with the same seed.
- **Epoch selection doesn't explain it.** Locally the held-out probe chose epoch 1 for MRL and MRL+PIC and epoch 3 for MIPIC, while the README's MIPIC used epoch 1. But the local last-epoch checkpoints (`*_last_epoch` in the same JSON) are also above the README at every width, by 0.55 to 4.42 points.
- **Only one local seed so far.** Seeds 1 and 2 are pending. They will give the first measurement of seed variance under the current code, which this finding can't provide.
- **Not significance-tested.** No bootstrap yet. The rough standard error of macro-F1 on 3,432 test rows is about 0.8 points, so the 16D tie between MRL+PIC and MIPIC (0.17) is a tie, and the 1.3 to 1.6 point gaps against MRL are marginal on their own.

**Relates to.**
- **H15** (SIA is essential: MIPIC without SIA, i.e. MRL+PIC, should be clearly worse than MIPIC at 16D and 32D). In seed 42 it isn't: MRL+PIC and MIPIC are tied at 16D (61.43 vs 61.26) and at 32D (62.49 vs 62.55). On this one seed, adding SIA on top of PIC gives nothing at small widths.
- **H15, second half** (MRL+SIA beats MRL). MRL+SIA finished after this finding was first written, at 63.19 / 65.76 / 66.63 at 16D / 32D / 64D. That's ahead of MRL by +3.32, +1.90 and +0.73, and behind it by 0.23 to 0.70 from 128D up. This reverses the README, where MRL+SIA was the worst arm at every width (see Finding 5 for why).
- **H16** (SIA and PIC are synergistic). With MRL+SIA in, the interaction term (MIPIC gain minus SIA gain minus PIC gain, each relative to MRL) is negative at every width in seed 42: -3.49 at 16D, -1.84 at 32D, -1.98 at 64D, and -0.17 to -1.57 from 128D up. On this seed, SIA and PIC interfere rather than combine.
- **H18** (MIPIC's advantage is largest at 16 to 32D). It holds at 16D, but at 32D plain MRL beats MIPIC in seed 42.
- **H17** and Finding 1. The 768D gap between MRL and MIPIC grows from 2.6 in the README to 3.8 locally (71.52 vs 67.71).
- Every comparison between methods in the README that is smaller than about 3 points should be treated as unresolved until it is replicated over seeds under one configuration.
- **Finding 5** explains the largest part of the README-vs-local gap for the SIA arms.

---

## Finding 5. The README results come from an older code version with attention KL on; the current code has it off

**Claim.** The numbers in [README.md](../README.md) and [MIPIC_TRAINING_README.md](../MIPIC_TRAINING_README.md) were produced by code version MIPIC_FINAL_V20, in which the SIA attention KL term was **on**. The current repository hardcodes `use_attention_kl=False` in [models/mrl_sia.py:119](../models/mrl_sia.py#L119) and [models/mrl_sia_pic.py:148](../models/mrl_sia_pic.py#L148). This configuration change plausibly accounts for most of the roughly 10 point MRL+SIA gap between the README and our seed 42 retrain. **Status: strong circumstantial evidence, pending a KL-on retrain** (`h11_probe/train_kl_on.py`).

**Evidence that KL was on in the Kaggle run.** The training summary for MRL+SIA in the saved output of notebook cell 29 (`mipic-a1-lexis.ipynb`) logs a nonzero attention loss in every epoch. Our local run logs exactly zero.

| Epoch | Kaggle `sia_attn_loss` | Kaggle `sia_cka_loss` | Local `sia_attn_loss` | Local `sia_cka_loss` |
|---:|---:|---:|---:|---:|
| 1 | 0.0987 | 0.0479 | 0.0000 | 0.0495 |
| 2 | 0.0413 | 0.0290 | 0.0000 | 0.0322 |
| 3 | 0.0608 | 0.0227 | 0.0000 | 0.0268 |
| 4 | 0.0627 | 0.0201 | 0.0000 | 0.0237 |
| 5 | 0.0827 | 0.0211 | 0.0000 | 0.0229 |

The CKA part is nearly the same in both runs, so the SIA difference is the KL term. For MIPIC, the log doesn't split the SIA loss, but Kaggle MIPIC's `sia_loss` was 0.1634 in epoch 1 against 0.0480 locally, the same size of jump. So KL was very likely on for MIPIC too (inferred, not logged directly).

**Settings table, MRL+SIA.**

| Setting | Kaggle (notebook cell 29 log) | Local seed 42 (`h11_probe/runs/seed42/logs/mrl_sia.log`) | Current source |
|---|---|---|---|
| **Attention KL** | **On** (`sia_attn_loss` nonzero, table above) | **Off** (`sia_attn_loss` 0.0000) | Off: `use_attention_kl=False` hardcoded. The `SIALoss` constructor default is True. |
| Code version | `VERSION.txt` = MIPIC_FINAL_V20, "120 passed" | No `VERSION.txt` in repo; `tests/` has 121 `def test_` functions | |
| Hardware and parallelism | 2x T4, DDP, 8 per rank, DistributedSampler, `static_graph=True`, OMP_NUM_THREADS=1 | 1 RTX 3060, batch 16, plain sampler | |
| Selected epoch | 2 of 5 | 1 of 5 | |
| Held-out probe, epochs 1 to 5 | 55.36, 56.04, 53.93, 54.74, 54.67 | 65.64, 63.76, 63.86, 61.56, 61.70 | |
| Final losses (epoch 5) | total 0.0664, mrl 0.0103, sia 0.1039 (attn 0.0827, cka 0.0211) | total 0.0152, mrl 0.0036, sia 0.0229 (attn 0, cka 0.0229) | |
| Training time | 34.3 min | 28.2 min | |

Settings that match:
- Seed 42, 5 epochs, lr 2e-5, max length 256, global batch 16, and the same 20,240-row corpus with identical per-source counts.
- alpha = 0.4 in both, checked from the logged losses. Kaggle epoch 1: 0.4 x 0.2212 + 0.6 x 0.1466 = 0.1765, as logged. Local: 0.4 x 0.1746 + 0.6 x 0.0495 = 0.0995, as logged.
- The MRL temperature, `base_k`, `align_layers` and `sia_temperature` aren't logged on Kaggle. The current defaults are 0.05, 64, (1 to 6) and 1.0, and the local metadata confirms 0.05 and 1.0. The near-identical CKA losses suggest these didn't change, but that is not proven.

**Evidence that the notebook ran a different code version from the current repository.**
- **Stale cell 14.** The source sets `RUN_MODE = "single"`, `OUTPUT_DIR = "/kaggle/working/mipic_results_no_kl"` and `FRESH = True`. The saved output reports `mode: ddp`, `output: /kaggle/working/mipic_results`, `fresh: False`, and prints a `CUDA_VISIBLE_DEVICES:` line that the current source doesn't produce. MIPIC_TRAINING_README.md already notes this cell's output is stale.
- **Unexecuted cell 3.** It asserts `use_attention_kl=False` in `models/mrl_sia_pic.py`, but it has no execution count and no output, so it was added after the recorded run.
- **Changing test counts.** The markdown in cell 9 says "Expect 100 passed", the executed cell 10 shows "120 passed", and the local `tests/` now has 121 test functions. The test suite changed at least twice.
- The name `mipic_results_no_kl` in the current source suggests the no-KL experiment came after the recorded run.

**Decomposition of the MRL+SIA gap.** Local minus README, macro-F1 points:

| Width | MRL+SIA gap | MRL gap (shared offset) | Specific to MRL+SIA |
|---:|---:|---:|---:|
| 16 | +9.93 | +3.41 | +6.52 |
| 32 | +10.60 | +3.87 | +6.73 |
| 64 | +10.47 | +3.08 | +7.39 |
| 128 | +11.11 | +3.07 | +8.04 |
| 256 | +10.67 | +2.91 | +7.76 |
| 512 | +11.26 | +3.35 | +7.92 |
| 768 | +11.47 | +4.28 | +7.19 |

1. **A shared offset of about 3 to 4 points.** MRL has no SIA at all and still gains +2.9 to +4.3. Its held-out probe at epoch 1 was 65.37 locally against 60.55 on Kaggle. This part comes from the hardware, DDP vs single GPU, data order, seed draw and any other unlogged changes between code versions.
2. **About 6.5 to 8 points specific to MRL+SIA, plausibly from the KL term.**
   - The KL gradient reaches the encoder. The teacher distribution is detached, but the student scores are computed from each layer's prefix `full[..., :dim]`, at all six aligned layers ([losses/sia_loss.py](../losses/sia_loss.py)).
   - It dominates SIA: 67% of `sia_loss` in epoch 1 and 80% in epoch 5.
   - It fights the other objectives. The attention loss doubles from epoch 2 to epoch 5 (0.041 to 0.083), and the total loss rises from 0.048 to 0.066, while the MRL loss stays flat.
   - The damage is visible early: the Kaggle held-out probe is already about 10 points lower at epoch 1.
   - In MIPIC the same term sits next to a PIC loss roughly ten times larger, which fits MIPIC's much smaller README-vs-local gap (+2.0 to +3.5, Finding 4).

**Caveats.**
- **Circumstantial, not causal.** The code version changed too, and other differences between V20 and the current code are not logged. The KL explanation is confirmed only if a local KL-on retrain (`h11_probe/train_kl_on.py --model mrl_sia`) lands near the README's MRL+SIA numbers, minus the shared offset.
- **One local seed.** The 6.5 to 8 point specific gap could shrink or grow once seeds 1 and 2 show how much MRL+SIA varies from seed to seed.
- **The MIPIC KL inference is indirect.** It rests on the size of `sia_loss`, not on a logged attention-loss value.

**Relates to.**
- **H6** (attention KL enforces token-ranking consistency, and that matters). The recorded run had KL **on**, which contradicts the current note in [hypothesis.md](../hypothesis.md) that the code ran without it. If the KL-on retrain confirms this finding, KL is not neutral: on this backbone and task it costs MRL+SIA around 7 points at every width.
- **H15** (SIA is essential). The README's conclusion that SIA alone hurts (MRL+SIA worst at every width) was measured with KL on. With KL off, in seed 42, MRL+SIA is the best arm at 16D to 64D (Finding 4). H15 has to be judged separately for the CKA part and the KL part of SIA.
