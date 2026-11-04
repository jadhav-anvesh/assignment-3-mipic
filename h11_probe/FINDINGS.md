# H11 probe: findings log

A running log of results from the `h11_probe/` pipeline. Each finding states a claim, the exact numbers behind it and where they came from, the caveats, and the hypothesis in [hypothesis.md](../hypothesis.md) it bears on.

All probe numbers are macro-F1 (%) on the 3-class tweet sentiment test split (3,432 rows), from a logistic regression fitted on the full train split (26,732 rows). The probe uses the same settings as `evaluate.py`: `LogisticRegression(random_state=42, max_iter=1000)` with sklearn defaults otherwise and no feature scaling. Features are stored as float32.

---

## Finding 1. Pretrained TinyBERT, read through its untrained random pooler, matches or beats every trained model on the sentiment probe (PRELIMINARY, seed 42)

**Claim.** On this tweet sentiment probe, pretrained TinyBERT-6L read through its untrained, random pooler (`pooler_output`, a fixed random projection of the layer 6 [CLS]; see Finding 9) matches or beats all four seed 42 trained models (MRL, MRL+SIA, MRL+PIC, MIPIC) at every width from 16 to 768. This holds with the default probe and with a scaled, CV-tuned probe, at probe sizes from 250 to 26,732 rows, and with training-free kNN. No trained model is significantly better than base in any cell; base is significantly better in 19 of 28 cells (paired bootstrap, default probe). An unscaled probe fitted on few rows reverses the result, because the trained models' pooler features are about twice base's scale, and the probe that `run_experiments.py` uses to choose the epoch shares that bias. PRELIMINARY: one training seed.

**Scope: sentiment only.** This finding is about the sentiment probe. On STS-B (clean pairs: 1,261 pairs after dropping those that appear in the training corpus), the same trained models clearly beat base at the final layer. Spearman x 100:

| Read-out | base | MRL | MRL+PIC | MIPIC | MRL+SIA |
|---|---:|---:|---:|---:|---:|
| Raw [CLS], layer 6, 768D (PIC checkpoint (6,768)) | 49.74 | 65.53 | 62.99 | 66.75 | 64.78 |
| Pooler, 16D | 42.00 | 55.38 | 53.66 | 59.57 | 57.74 |
| Pooler, 64D | 47.54 | 61.67 | 58.11 | 64.40 | 62.09 |
| Pooler, 768D | 49.56 | 65.45 | 62.54 | 66.52 | 65.06 |

Source: `h11_probe/results/{base,mrl_s42,mrl_pic_s42,mipic_s42,mrl_sia_s42}_sts_clean.csv`, from `sts_eval.py --drop-overlap`. The trained models lead by 13 to 17 points at (6,768) and by 10.6 to 17.6 points at every width of the pooler row; no bootstrap has been run on STS-B. So training is not useless: it moves the representation toward sentence similarity, at the expense of sentiment information.

### Verification (nine checks, seed 42)

| # | Check | Result | Effect on the claim | Files |
|---|---|---|---|---|
| 1 | Is base's pooler pretrained? | No. Its weight is indistinguishable from iid Gaussian noise (kurtosis 3.003; top singular value 1.001x the Marchenko-Pastur edge; row-norm CV 0.0252 against 0.0255 for iid), and its bias is exactly zero. All 36 trained encoder matrices in the same checkpoint have singular value ratios of 1.29 to 8.63 and row-norm CVs of 0.063 to 0.436. The tanh is near linear on real inputs (largest pre-activation 0.134, largest relative gap 0.59%). Training barely moves the pooler (cosine to base 0.995 to 0.999) | Corrects the description of base; the comparison is still like for like | `pooler_check.py` -> `results/pooler_check.csv`; `weight_drift.py` -> `results/weight_drift_s42.csv` |
| 2 | Same pipeline on both sides | Same loader, tokenizer, max_length 256, float32 features, read-out and files (26,732 / 3,432 rows, identical labels). Refitting base's pooler row with today's code gives identical predictions at all 7 widths | Confirms | `refit_check.py` -> `results/refit_check_base.csv`; `results/base_grid.csv`, `results/*_s42_grid.csv` |
| 3 | Overlap with the training corpus | No test row appears in `final_data.csv`. All 3,000 TWEET_EVAL_SENT corpus rows are in `tweet_train.csv` (11.2% of the probe-train split; labels 841 / 1,249 / 910). That exposure can only help the trained models | Confirms | `overlap_check.py` -> `results/overlap_check.txt` |
| 4 | Paired bootstrap, pooler row, full probe | No trained model is significantly above base at any width. Base is significantly ahead in 19 of 28 cells. The best trained cell is MRL+SIA at 16D: +0.57 [-1.03, +2.21] | Confirms | `results/bootstrap_{mrl,mrl_pic,mipic,mrl_sia}_s42_vs_base.csv`, rows with `layer = pooler` |
| 5 | Probe size, default (unscaled) probe | At 16D and 64D with 250 to 1,000 rows, the trained models lead by 4.2 to 31.4 points on average, and are ahead on every subset; base's probe is close to a one-class predictor there. Base catches up at about 10,000 rows. At 768D base ties or wins from 500 rows up | Weakens, then explained by check 6 | `probe_size.py` -> `results/probe_size_s42.csv`, `results/probe_size_s42_summary.csv` |
| 6 | Scaled probe, C tuned by 5-fold CV, same subsets as check 5 | Base recovers completely. It beats the trained model on every subset in 45 of 48 (model, size, width) cells. The exceptions are all MRL+SIA at 16D: -0.07 at 1,000 rows, +0.47 at 3,000, +0.53 at full size. At full size base leads at 768D by 1.48 to 4.87 points | Confirms; the check 5 gap was a feature-scale artifact | `probe_tuned.py` -> `results/probe_tuned_s42.csv`, `results/probe_tuned_s42_summary.csv` |
| 7 | kNN, no fitted classifier (k = 10 and 50; cosine and raw Euclidean) | Base is top in 11 of 12 cells. The exception is raw Euclidean, k = 10, 16D: MRL+SIA ahead by 0.15. L2-normalised Euclidean reproduces cosine exactly, as it must | Confirms | `probe_knn.py` -> `results/probe_knn_s42.csv` |
| 8 | Epoch selection probe (`evaluate.validation_probe_score`, as in `run_experiments.py`) | Base scores 63.16. Epoch 1 scores: MRL 65.37, MRL+PIC 64.56, MIPIC 62.42, MRL+SIA 65.64. Rescoring each `best.pt` reproduces its logged score | Weakens for the selection metric only. That probe is unscaled and fits on 4,800 rows, the regime where checks 5 and 6 show a scale bias (inferred, not rerun with scaling) | `selection_probe.py` -> `results/selection_probe_s42.csv`; `runs/seed42/logs/{mrl,mrl_pic,mipic,mrl_sia}.log` |
| 9 | What the paper and repo files say | `LEXIS_A1.pdf` reports no untrained baseline: Table 1 (p.3) lists SimCSE, MRL, ESE and MIPIC; Section 6.1 (p.6) evaluates "the four trained models"; Table 3 (p.7) lists those four only. `README.md` and `MIPIC_TRAINING_README.md` give no baseline numbers. Two code comments do: `evaluate.py` lines 28 to 32 (untrained TinyBERT-6L, 62.62 to 71.99 on 26,732 rows) and `run_experiments.py` lines 523 to 525 ("62.52 macro-F1 at d=768 against 70.55 for the untrained backbone"). `hypothesis.md` uses base TinyBERT only as a planned control | Neutral; the baseline is missing from the report | The files named |

Probe-size details (checks 5 and 6), mean macro-F1 over 5 subsets at 16D:

| Probe rows | Probe | base | MRL | MRL+PIC | MIPIC | MRL+SIA |
|---:|---|---:|---:|---:|---:|---:|
| 250 | default | 21.0 | 31.2 | 44.7 | 47.4 | 38.9 |
| 250 | scaled, tuned | **59.1** | 55.9 | 57.1 | 56.8 | 57.8 |
| 1,000 | default | 26.0 | 48.9 | 56.7 | 57.4 | 57.2 |
| 1,000 | scaled, tuned | **62.1** | 59.0 | 60.2 | 60.2 | 62.1 |
| 26,732 | default | 62.6 | 59.9 | 61.4 | 61.2 | **63.2** |
| 26,732 | scaled, tuned | 63.0 | 60.1 | 61.7 | 61.3 | **63.5** |

Mean per-coordinate SD of the pooler features (printed by `probe_size.py`): base 0.0129, MRL 0.0232, MRL+SIA 0.0230, MIPIC 0.0270, MRL+PIC 0.0280. Under the default probe's fixed C = 1, base's smaller features are regularised much harder, which is why its small-probe fits collapse (accuracy 39 to 43%, macro-F1 21 to 26, close to always predicting neutral). In check 6 the chosen C at 768D is mostly 0.01, the bottom of the grid, so a wider grid could move those cells slightly.

### What it means for the MIPIC paper

**It does show:** in this reproduction, training adds no sentiment information at any width. So the ranking of MRL, SIA, PIC and MIPIC here measures how much each one loses against the backbone, not how much each gains. The report should include the untrained baseline.

**It doesn't show:** that the paper's numbers are wrong. The paper reports no untrained baseline and uses different data, so we can't tell whether its trained models beat their own backbone. It also says nothing about the paper's other tasks or backbones.

**Caveats.**
- **One training seed.** Checks 1 (the pooler drift part), 4, 6, 7 and 8 should be repeated on seeds 1 and 2. Checks 2, 3 and 9 don't depend on the seed, and check 6 supersedes check 5.
- **Not official TweetEval.** The evaluation files are the Kaggle Tweet Sentiment Extraction data (see the README and `LEXIS_A1.pdf` Section 3.1).
- **STS-B has no CIs.** The STS-B gaps are large, but no bootstrap was run on them.

**Relates to.**
- **H17** (MIPIC matches baselines at full capacity). The right baseline includes the backbone. In seed 42, MIPIC's pooler at 768D is 3.96 [2.67, 5.29] points below base (`results/bootstrap_mipic_s42_vs_base.csv`).
- **Finding 9** (the random pooler), which explains what base's read-out is.
- **Finding 8**, whose epoch selection uses the probe that check 8 shows is biased by feature scale.

### History

The original claim and its evidence, kept as a record. They are superseded by the claim and verification above.

#### Original claim (against the README models)

The frozen, untrained TinyBERT-6L `pooler_output` scores higher than the best of the four trained models (MRL, MRL+SIA, MRL+PIC, MIPIC) in the README at every nested width from 16 to 768. The gap is 3.86 to 5.61 macro-F1 points. On this evaluation, none of the training recipes improves on the starting backbone.

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

#### Update, 2026-11-04: against our own seed 42 retrain

We retrained all four models locally with the current repository code and its default settings (see Findings 4 and 5), and they score 2 to 11 points higher than the README's models. Repeating the comparison against these retrained models, **the gap shrinks to between -0.57 and +1.33 points** (from 3.86 to 5.61 above). Base still wins at 32D to 768D, but **at 16D MRL+SIA is 0.57 points ahead of base**, so the headline claim of this finding doesn't hold at every width in seed 42.

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

The paired bootstrap has since been run (verification check 4): MRL+SIA's 0.57 point lead at 16D has a CI of [-1.03, +2.21], so it is not significant. The values in this table come from `tweeteval_results.json` and differ by up to 0.23 from the `*_s42_grid.csv` refits on re-extracted features used in the verification (MRL at 768D: 71.52 here, 71.29 in the grid; at 512D: 70.96 and 71.13). With the grid values, base leads MRL at 768D by 0.44 rather than 0.22.

#### Original caveats and links

Kept as written. "Not yet significance-tested" is addressed by check 4, and the description of the read-out is corrected by check 1.

- **Single training seed.** Each trained model comes from one run, with no seed variation. `run_experiments.py` itself notes that two identical MRL runs differed by about 3 F1 at 768D before seeding was added.
- **Not official TweetEval.** The evaluation files are the Kaggle Tweet Sentiment Extraction data, not the public TweetEval benchmark (see the README). The result may not carry over to TweetEval or to other tasks.
- **Different environment.** The README numbers were produced on Kaggle (2x T4), and our base numbers were produced locally (RTX 3060, sklearn 1.9.0). As a cross-check, our rerun reproduces the untrained-backbone numbers quoted in the `evaluate.py` comment (62.62 to 71.99) to within 0.26 points at every width. That is far smaller than the 3.86 point minimum gap here.
- **Epoch selection is not documented for every model.** MIPIC's README numbers come from the epoch the held-out probe selected (epoch 1 of 5). The README doesn't say which epoch the other three models' numbers come from.
- **Not yet significance-tested.** A paired bootstrap on the saved predictions is still to do. A gap of at least 3.86 points is large relative to the roughly 0.8 point standard error of macro-F1 on 3,432 test rows, but no formal test has been run.

Original links:
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
- **H16** (SIA and PIC are synergistic). With MRL+SIA in, the interaction term (MIPIC gain minus SIA gain minus PIC gain, each relative to MRL) is negative at every width in seed 42: -3.49 at 16D, -1.84 at 32D, -1.98 at 64D, and -0.17 to -1.57 from 128D up. On this seed, SIA and PIC interfere rather than combine at the pooler. On raw [CLS] at the early PIC checkpoints the interaction is strongly positive instead (Finding 7).
- **H18** (MIPIC's advantage is largest at 16 to 32D). It holds at 16D, but at 32D plain MRL beats MIPIC in seed 42.
- **H17** and Finding 1. The 768D gap between MRL and MIPIC grows from 2.6 in the README to 3.8 locally (71.52 vs 67.71).
- Every comparison between methods in the README that is smaller than about 3 points should be treated as unresolved until it is replicated over seeds under one configuration.
- **Finding 5** covers the README-vs-local gap for the SIA arms. *Revised 2026-11-04:* the KL setting explains only about 1 to 2.5 points of it; most of the MRL+SIA gap is still unexplained.

---

## Finding 5. The README results come from an older code version with attention KL on, but KL explains only a small part of the MRL+SIA gap (PRELIMINARY, seed 42)

*Replaced 2026-11-04.* The first version of this finding said the KL setting "plausibly accounts for most of" the MRL+SIA gap. The KL-on retrain doesn't support that, and this version replaces it.

**Claim.** The README numbers were produced by MIPIC_FINAL_V20 with the SIA attention KL term on; the current code hardcodes it off ([models/mrl_sia.py:119](../models/mrl_sia.py#L119), [models/mrl_sia_pic.py:148](../models/mrl_sia_pic.py#L148)). A local KL-on retrain (seed 42, `h11_probe/train_kl_on.py`) shows that turning KL on costs MRL+SIA 0.87 to 1.61 points and MIPIC 0.78 to 2.55 points of pooler macro-F1. That is roughly 11% to 24% of the 6.5 to 8 point MRL+SIA-specific gap, so **KL is not the main cause**. Every 95% CI for the KL cost lies between -3.93 and +0.79 points, so a cost of 6.5 points is ruled out at every width.

With KL matched, local MIPIC lands 0.4 to 2.3 points above the README, inside MRL's 2.9 to 4.3 point shared offset. Local MRL+SIA is still 8.9 to 10.4 points above the README (5.1 to 7.0 after removing that offset). Whatever produces the rest is specific to the MRL+SIA arm of the Kaggle run.

**Status: KL difference and code version difference confirmed; cause of the remaining gap unknown. PRELIMINARY: one training seed.**

### Confirmed: KL was on in the Kaggle run

The training summary for MRL+SIA in the saved output of notebook cell 29 (`mipic-a1-lexis.ipynb`) logs a nonzero attention loss in every epoch. Our default local run logs exactly zero; the KL-on retrain logs nonzero values from the first step (its built-in check found `sia_attn_loss` nonzero on 50 of the first 50 steps).

| Epoch | Kaggle attn | Kaggle CKA | Local KL off attn | Local KL off CKA | Local KL on attn | Local KL on CKA |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0987 | 0.0479 | 0.0000 | 0.0495 | 0.1013 | 0.0498 |
| 2 | 0.0413 | 0.0290 | 0.0000 | 0.0322 | 0.0257 | 0.0334 |
| 3 | 0.0608 | 0.0227 | 0.0000 | 0.0268 | 0.0204 | 0.0294 |
| 4 | 0.0627 | 0.0201 | 0.0000 | 0.0237 | 0.0148 | 0.0267 |
| 5 | 0.0827 | 0.0211 | 0.0000 | 0.0229 | 0.0136 | 0.0262 |

For MIPIC, neither log splits the SIA loss, but the combined `sia_loss` tells the same story:

| Epoch | Kaggle MIPIC | Local MIPIC KL off | Local MIPIC KL on |
|---:|---:|---:|---:|
| 1 | 0.1634 | 0.0480 | 0.1590 |
| 2 | 0.0703 | 0.0320 | 0.0684 |
| 3 | 0.0550 | 0.0278 | 0.0616 |
| 4 | 0.0506 | 0.0258 | 0.0600 |
| 5 | 0.0493 | 0.0250 | 0.0644 |

Local KL-on MIPIC matches Kaggle MIPIC closely at epochs 1 and 2, which confirms the earlier inference that KL was on for MIPIC too.

**Settings table, MRL+SIA.**

| Setting | Kaggle (notebook cell 29 log) | Local KL off (`runs/seed42`) | Local KL on (`runs_kl/seed42`) |
|---|---|---|---|
| **Attention KL** | **On** | **Off** (hardcoded) | **On** (`train_kl_on.py` sets `use_attention_kl=True`) |
| Code version | `VERSION.txt` = MIPIC_FINAL_V20, "120 passed" | Current repository, 121 test functions, no `VERSION.txt` | Same as KL off |
| Hardware and parallelism | 2x T4, DDP, 8 per rank, DistributedSampler, `static_graph=True` | 1 RTX 3060, batch 16, plain sampler | Same as KL off |
| Selected epoch | 2 of 5 | 1 of 5 | 1 of 5 |
| Held-out probe, epochs 1 to 5 | 55.36, 56.04, 53.93, 54.74, 54.67 | 65.64, 63.76, 63.86, 61.56, 61.70 | 64.57, 62.23, 62.47, 63.47, 63.27 |
| Final losses (epoch 5) | total 0.0664, mrl 0.0103, sia 0.1039 | total 0.0152, mrl 0.0036, sia 0.0229 | total 0.0258, mrl 0.0049, sia 0.0398 |

Settings that match in all three: seed 42, 5 epochs, lr 2e-5, max length 256, global batch 16, the same 20,240-row corpus with identical per-source counts, and alpha = 0.4 (checked from the logged losses). The MRL temperature, `base_k`, `align_layers` and `sia_temperature` aren't logged on Kaggle.

**Confirmed: the notebook ran a different code version from the current repository.**
- **Stale cell 14.** The source sets `RUN_MODE = "single"`, `OUTPUT_DIR = "/kaggle/working/mipic_results_no_kl"` and `FRESH = True`, but the saved output reports `mode: ddp`, `output: /kaggle/working/mipic_results` and `fresh: False`.
- **Unexecuted cell 3.** It asserts `use_attention_kl=False` but has no execution count and no output, so it was added after the recorded run.
- **Changing test counts.** Cell 9 says "Expect 100 passed", the executed cell 10 shows "120 passed", and the local `tests/` has 121 test functions.

### The KL cost, with paired bootstrap CIs

Pooler macro-F1, KL on minus KL off. Values come from `h11_probe/results/bootstrap_mrl_sia_kl_s42_vs_mrl_sia_s42.csv` and `bootstrap_mipic_kl_s42_vs_mipic_s42.csv`: 1,000 paired resamples of the 3,432 test rows, seed 0. `*` = the 95% CI excludes zero.

| Width | MRL+SIA: KL on minus off | MIPIC: KL on minus off |
|---:|---:|---:|
| 16 | -1.04 [-2.26, +0.24] | -0.84 [-2.40, +0.79] |
| 32 | -1.68 [-2.78, -0.55]* | -0.65 [-2.14, +0.77] |
| 64 | -0.93 [-1.99, +0.09] | -1.58 [-2.90, -0.25]* |
| 128 | -0.86 [-1.86, +0.12] | -2.60 [-3.93, -1.34]* |
| 256 | -1.18 [-2.12, -0.22]* | -1.35 [-2.67, -0.05]* |
| 512 | -1.01 [-1.81, -0.14]* | -1.58 [-2.85, -0.37]* |
| 768 | -1.38 [-2.24, -0.49]* | -0.83 [-2.04, +0.42] |

The bootstrap scores the probe-grid predictions, which are fitted single-threaded. They differ from the `evaluate.py` numbers in the training JSON by at most 0.23 points per cell, which is why the ranges in the claim (0.87 to 1.61 and 0.78 to 2.55, from the JSON) differ slightly from this table. In both arms the sign is negative at every width and 4 of 7 CIs exclude zero.

### Decomposition of the MRL+SIA gap

Pooler macro-F1 from the training JSON (`evaluate.py`), local minus README:

| Width | KL off minus README | KL on minus README | KL on minus off | MRL offset | Left after KL and offset |
|---:|---:|---:|---:|---:|---:|
| 16 | +9.93 | +8.94 | -0.99 | +3.41 | +5.53 |
| 32 | +10.60 | +8.99 | -1.61 | +3.87 | +5.12 |
| 64 | +10.47 | +9.48 | -0.99 | +3.08 | +6.40 |
| 128 | +11.11 | +10.04 | -1.07 | +3.07 | +6.97 |
| 256 | +10.67 | +9.42 | -1.25 | +2.91 | +6.51 |
| 512 | +11.26 | +10.39 | -0.87 | +3.35 | +7.04 |
| 768 | +11.47 | +10.19 | -1.28 | +4.28 | +5.91 |

For comparison, local KL-on MIPIC minus README MIPIC is +1.76, +1.20, +0.41, +0.37, +2.17, +1.95 and +2.29 at 16D to 768D: inside the MRL offset at every width.

### What KL on does not reproduce

- **The attention loss trajectory.** It starts at the same size (epoch 1: 0.1013 local, 0.0987 Kaggle). Locally it then falls every epoch, to 0.0136. On Kaggle it rises after epoch 2, to 0.0827.
- **The early damage.** Kaggle's held-out probe was 55.36 at epoch 1. Locally it was 64.57 with KL on and 65.64 with KL off.
- **The pattern is specific to Kaggle's MRL+SIA.** Kaggle MIPIC's `sia_loss` fell every epoch.

### Possible explanations for the rest. None is confirmed.

1. **An older SIA implementation.** The current source and tests show that earlier versions differed:
   - `token_scores` accepts and discards `proj`, `W_Q` and `W_K` "for earlier repository callers", and `d_att` is "retained only for compatibility with prior constructors", so an earlier SIA used learned query and key projections ([losses/sia_loss.py](../losses/sia_loss.py)).
   - [tests/test_kaggle_readiness.py](../tests/test_kaggle_readiness.py) asserts that `MIPIC_SIA_CONTEXTUAL_MASK`, `MIPIC_CKA_DTYPE`, `MIPIC_MRL_PROJECTION`, `MIPIC_PIC_LR_MULTIPLIER` and `MIPIC_MAX_GRAD_NORM` "should have been removed". The current code excludes [CLS] as an attention key and says including it "collapses a_D onto a one-hot distribution over [CLS]".
   - Which of these V20 used is not logged, and the notebook sets none of the removed switches. Against this explanation: MIPIC uses the same `SIALoss` and does reproduce. Also, a test says keeping [CLS] as a key makes the attention loss larger, yet Kaggle's epoch 1 attention loss is slightly smaller than ours.
2. **Code specific to the MRL+SIA arm in V20.** MIPIC, built on a different model class, reproduces.
3. **DDP on 2x T4 with 8 per rank.** MRL and MIPIC also ran under DDP and reproduce within the offset, so this would have to interact with the MRL+SIA arm specifically.
4. **Other unlogged changes between V20 and the current code.** The test suite went from 100 to 120 to 121 tests.
5. **Seed variance of MRL+SIA.** Seeds 1 and 2 will measure it.

**Caveats.**
- **One training seed.** The bootstrap CIs cover test-set sampling only, not variation between training runs.
- **The KL-on runs used the current code with one flag changed.** They are not a reproduction of V20.

**Relates to.**
- **H6** (attention KL matters). KL is not neutral, but at the pooler it costs about 1 to 2.5 points, not 7. Its largest effect is on early raw [CLS]: MIPIC at (layer 1, 16D) drops from 54.65 to 43.06, -11.59 [-13.64, -9.57] (Finding 6).
- **H15** (SIA is essential). With KL on, MRL+SIA still beats MRL at 16D (+2.33) and trails it from 128D up (-1.3 to -1.9). The README's "SIA hurts at every width" is not reproduced with KL either on or off.

---

## Finding 6. H11 is supported in seed 42: MIPIC's early checkpoints carry far more sentiment signal than MRL or base, at the cost of later layers (PRELIMINARY, seed 42)

**Claim.** At the first two PIC checkpoints, MIPIC's raw [CLS] prefix beats both MRL and base TinyBERT by a wide margin: about +20 points at (layer 1, 16D) and +6 points at (layer 2, 32D). MRL+PIC also beats MRL at both. All six CIs required by the decision rule exclude zero, so **H11 is supported for seed 42**. The gain is confined to the early checkpoints. From (layer 4, 256D) onward MIPIC is significantly below both MRL and base.

**Decision rule** (set before the results): H11 is supported if MIPIC beats both MRL and base at (1,16) and (2,32) with 95% CIs excluding zero, and MRL+PIC also beats MRL there.

| Comparison | (1,16) | (2,32) | Rule met |
|---|---:|---:|:---:|
| MIPIC minus MRL | +20.43 [+18.20, +22.44] | +5.98 [+4.18, +7.89] | Yes |
| MIPIC minus base | +20.77 [+18.58, +22.68] | +5.88 [+4.03, +7.80] | Yes |
| MRL+PIC minus MRL | +9.74 [+7.86, +11.55] | +2.48 [+0.73, +4.31] | Yes |

**Verdict for seed 42: supported.** No CI in the rule includes zero. The narrowest margin is MRL+PIC at (2,32), whose lower bound is +0.73.

### Sentiment macro-F1 at the six PIC checkpoints

Raw [CLS] `hidden_states[L][:, 0, :d]`, the vector PIC supervises. MRL+SIA KL on is in `h11_probe/results/mrl_sia_kl_s42_grid.csv` and is left out for space; it tracks MRL+SIA within 1.5 points.

| (layer, dim) | Base | MRL | MRL+PIC | MIPIC | MRL+SIA | MIPIC KL on |
|---|---:|---:|---:|---:|---:|---:|
| (1,16) | 33.88 | 34.22 | 43.95 | **54.65** | 34.37 | 43.06 |
| (2,32) | 57.20 | 57.11 | 59.59 | **63.09** | 56.10 | 61.18 |
| (3,64) | **68.24** | 67.38 | 66.78 | 66.65 | 67.76 | 65.34 |
| (4,256) | **72.04** | 71.89 | 69.85 | 68.52 | 71.71 | 67.92 |
| (5,512) | **73.07** | 71.36 | 70.27 | 68.82 | 71.81 | 68.08 |
| (6,768) | **73.51** | 71.10 | 70.25 | 68.32 | 71.49 | 68.36 |

### Paired bootstrap CIs at every PIC checkpoint

`*` = CI excludes zero. Sources: `h11_probe/results/bootstrap_{a}_vs_{b}.csv`.

| (layer, dim) | MIPIC minus MRL | MIPIC minus base | MRL+PIC minus MRL | MRL+SIA minus MRL |
|---|---:|---:|---:|---:|
| (1,16) | +20.43 [+18.20, +22.44]* | +20.77 [+18.58, +22.68]* | +9.74 [+7.86, +11.55]* | +0.15 [-0.56, +0.86] |
| (2,32) | +5.98 [+4.18, +7.89]* | +5.88 [+4.03, +7.80]* | +2.48 [+0.73, +4.31]* | -1.01 [-1.98, -0.16]* |
| (3,64) | -0.73 [-2.09, +0.58] | -1.59 [-2.93, -0.25]* | -0.60 [-1.90, +0.64] | +0.39 [-0.44, +1.16] |
| (4,256) | -3.37 [-4.69, -2.13]* | -3.52 [-4.95, -2.29]* | -2.04 [-3.26, -0.73]* | -0.18 [-1.02, +0.66] |
| (5,512) | -2.54 [-3.89, -1.26]* | -4.25 [-5.65, -2.86]* | -1.10 [-2.35, +0.14] | +0.45 [-0.48, +1.43] |
| (6,768) | -2.77 [-4.13, -1.36]* | -5.19 [-6.57, -3.82]* | -0.84 [-2.16, +0.43] | +0.39 [-0.67, +1.43] |

### What the pattern shows

- **Early gain, late loss.** MIPIC's advantage is very large at (1,16), smaller at (2,32), gone at (3,64), and reversed from (4,256) on, where it trails base by 3.5 to 5.2 points. PIC appears to move sentiment signal into the early layers at the expense of the later ones and the pooler (Finding 4: MIPIC is the weakest arm at the pooler from 64D up).
- **It is PIC, not SIA.** MRL+SIA is indistinguishable from MRL at (1,16) and slightly worse at (2,32). MRL+PIC gets about half of MIPIC's gain at (1,16), and MIPIC is 10.7 points above MRL+PIC there, so SIA adds a lot on top of PIC (see Finding 7).
- **Not only at the checkpoint width.** At layer 1, MIPIC beats base at every width, from 54.65 vs 33.88 at 16D to 67.82 vs 63.14 at 768D.
- **The prefix itself is special in MIPIC.** In the subset probe at (1,16), MIPIC's first 16 coordinates beat the mean of 20 random 16-coordinate subsets by +4.86 points (random SD 2.38). For every other model the prefix is within 1.4 points of random (base +0.49, MRL +0.87, MRL+PIC -0.24, MRL+SIA +0.99, MRL+SIA KL on -0.43, MIPIC KL on -1.34).

### With KL on, the paper's configuration

| Comparison | (1,16) | (2,32) |
|---|---:|---:|
| MIPIC KL on minus MIPIC KL off | -11.59 [-13.64, -9.57]* | -1.91 [-3.32, -0.59]* |
| MIPIC KL on minus MRL | +8.84 [+6.98, +10.61]* | +4.07 [+2.42, +6.00]* |
| MIPIC KL on minus base | +9.18 [+7.28, +10.91]* | +3.98 [+2.29, +5.82]* |

Sources: the first row is `h11_probe/results/bootstrap_mipic_kl_s42_vs_mipic_s42.csv` (`bootstrap.py`). The other two are `bootstrap_mipic_kl_s42_vs_mrl_s42.csv` and `bootstrap_mipic_kl_s42_vs_base.csv`, written by `h11_probe/bootstrap_extra.py`, which runs `bootstrap.py` itself (same seed 0 index matrix, 1,000 resamples). KL on cuts MIPIC's (1,16) advantage roughly in half, but the H11 rule still passes for KL-on MIPIC at both checkpoints. The KL-on MIPIC (1,16) prefix also loses its advantage over random subsets (-1.34 against random).

**STS-B at the same checkpoints** (Spearman x100, clean pairs, n = 1,261; no CIs):

| (layer, dim) | Base | MRL | MRL+PIC | MIPIC | MRL+SIA | MIPIC KL on |
|---|---:|---:|---:|---:|---:|---:|
| (1,16) | 25.19 | 25.10 | 16.26 | **39.97** | 26.89 | 21.81 |
| (2,32) | **59.29** | 56.28 | 41.96 | 52.37 | 55.98 | 47.54 |

On STS, MIPIC's early gain holds only at (1,16). At (2,32), base is best.

**Caveats.**
- **One training seed.** The CIs cover test-set sampling only, not variation between training runs. The (1,16) gaps are large enough that seed variance is unlikely to erase them, but the (2,32) gaps of 2.5 to 6 points need seeds 1 and 2.
- **MRL+PIC is weak on STS at the early checkpoints.** It scores 16.26 at (1,16) and 41.96 at (2,32), the lowest of all models and well below base (25.19 and 59.29). Its early sentiment gain does not carry over to similarity; PIC alone seems to make the early prefix sentiment-specific.
- **Unconverged probe fits at (5,512).** The probe fits for base and MRL+SIA at (layer 5, 512D) hit the 1,000-iteration cap. Comparisons involving those two cells at (5,512) partly reflect the solver. No other PIC-checkpoint cell is affected.
- **No sign of STS contamination at layer 2.** Dropping the 118 overlapping pairs changes the trained models' layer 2 Spearman by -0.61 to +1.51 points, against -0.96 to -0.46 for base, so the (2,32) STS comparison is not inflated by training overlap. At the pooler from 32D up, removing overlap lowers the trained models' Spearman by 0.25 to 2.72 points more than it lowers base's, a mild contamination effect that doesn't touch the H11 checkpoints.
- **KL off is not the paper's configuration.** The main table uses the current code's default (KL off). The KL-on rows above are the closer match to the paper.

**Relates to.**
- **H11** (early-layer, low-dimensional prefixes become task-useful). Supported in seed 42, by the pre-set rule, with and without KL.
- **H18** (MIPIC's advantage is largest at 16D to 32D). On raw [CLS] at the PIC checkpoints, yes: it is largest at (1,16) and reverses from (4,256). At the pooler it is not (Finding 4).
- **H17** (MIPIC matches baselines at full capacity). No: at (6,768) MIPIC is 2.77 below MRL and 5.19 below base, CIs excluding zero.
- **H15** (SIA is essential). SIA alone does nothing at the early checkpoints, but adding it to PIC roughly doubles PIC's gain at (1,16) (Finding 7).
- **Finding 2.** Base is already strong from layer 3 on, so MIPIC's gain is real only where base is weak, in layers 1 and 2.

---

## Finding 7. SIA and PIC are synergistic at the early PIC checkpoints and interfere at the late ones and at the pooler (PRELIMINARY, seed 42)

**Claim.** The SIA x PIC interaction (MIPIC minus MRL+SIA minus MRL+PIC plus MRL, equivalently MIPIC's gain over MRL minus the SIA and PIC gains) is strongly **positive** on raw [CLS] at (1,16) and (2,32): together the two objectives give more than the sum of their parts. It is near zero at (3,64) and (4,256) and **negative** at (5,512) and (6,768). At the pooler (the read-out in Finding 4), it is negative or near zero at every width. Whether SIA and PIC are synergistic depends on where you read the representation.

**Evidence.** Interaction in macro-F1 points, with 95% CIs from 1,000 paired resamples of the test rows (the same seed 0 index matrix as `bootstrap.py`, all four models scored on each resample). `*` = CI excludes zero. Source: `h11_probe/results/bootstrap_interaction_s42.csv`, written by `python h11_probe/bootstrap_extra.py` from `h11_probe/results/{mrl,mrl_pic,mrl_sia,mipic}_s42_preds.npz`. It holds all 49 (layer, width) cells, with the per-arm F1, the three gains, the interaction and its CI.

Raw [CLS] at the PIC checkpoints:

| (layer, dim) | SIA gain | PIC gain | MIPIC gain | Interaction |
|---|---:|---:|---:|---:|
| (1,16) | +0.15 | +9.74 | +20.43 | **+10.55 [+8.33, +12.78]*** |
| (2,32) | -1.01 | +2.48 | +5.98 | **+4.51 [+2.82, +6.25]*** |
| (3,64) | +0.39 | -0.60 | -0.73 | -0.51 [-1.99, +1.09] |
| (4,256) | -0.18 | -2.04 | -3.37 | -1.15 [-2.69, +0.38] |
| (5,512) | +0.45 | -1.10 | -2.54 | **-1.90 [-3.49, -0.40]*** |
| (6,768) | +0.39 | -0.84 | -2.77 | **-2.32 [-3.93, -0.52]*** |

Gains are relative to MRL at the same cell.

Pooler, for comparison with Finding 4 (which used the `evaluate.py` values: -3.49, -1.84, -1.98, then -0.17 to -1.57):

| Width | 16 | 32 | 64 | 128 | 256 | 512 | 768 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Interaction | -3.55* | -2.04 | -2.00* | -0.14 | +0.13 | -1.12 | -1.61* |
| 95% CI | [-5.78, -1.31] | [-4.13, +0.06] | [-3.81, -0.11] | [-1.87, +1.64] | [-1.45, +1.71] | [-2.58, +0.34] | [-3.15, -0.03] |

At the pooler, 3 of 7 CIs exclude zero (16D, 64D, 768D), all negative. The other four include zero, so "interference at the pooler" is clear only at those three widths.

**Caveats.**
- **One training seed; test-set CIs only.**
- **Unconverged cell.** The MRL+SIA fit at (5,512) hit the iteration cap, and it enters the (5,512) interaction.
- **KL off.** All four arms are the current-code default. The interaction with KL on would need KL-on MRL+SIA and MIPIC against KL-off MRL and MRL+PIC; not computed.
- **Additive scale.** The interaction is defined on macro-F1 points. Near the bottom of the scale (layer 1 base is 33.88, close to chance for 3 classes) an additive model may not be the right null, so the size of the (1,16) synergy should be read with care.

**Relates to.**
- **H16** (SIA and PIC are synergistic). Supported at the early PIC checkpoints, contradicted at the late ones and at the pooler. The pooler-only reading in Finding 4 ("SIA and PIC interfere") was incomplete.
- **H15** (SIA is essential). SIA on its own does nothing at (1,16), but it is what turns PIC's +9.7 into MIPIC's +20.4.

---

## Finding 8. Epoch 1 is the best epoch for almost every model, with or without PIC (PRELIMINARY, seed 42)

**Claim.** The held-out probe selects epoch 1 of 5 for five of the six local runs, including MRL, which has no PIC. The test set agrees: for MRL, MRL+PIC and MRL+SIA, the selected checkpoint beats the last-epoch checkpoint by up to 2.16, 3.94 and 3.67 points. A best epoch of 1 therefore says that further training hurts this probe in general. It is **not evidence that PIC speeds up learning**, which is how [hypothesis.md](../hypothesis.md) currently uses it for H14.

**Evidence.** Held-out probe macro-F1 per epoch (a single score on a held-out slice of the probe-train split, never the test set; `epoch_selection` in `tweeteval_results.json`) and the test-set difference between the selected and last epochs at the pooler, across the seven widths (`evaluations.{model}` minus `evaluations.{model}_last_epoch`).

| Model | Selected epoch | Held-out probe, epochs 1 to 5 | Test: selected minus last |
|---|---:|---|---:|
| MRL | 1 | 65.37, 65.04, 64.86, 64.04, 63.80 | -0.15 to +2.16 |
| MRL+PIC | 1 | 64.56, 62.57, 61.35, 61.63, 61.02 | +1.02 to +3.94 |
| MIPIC | 3 | 62.42, 61.67, 62.59, 62.18, 62.05 | +0.06 to +1.24 |
| MRL+SIA | 1 | 65.64, 63.76, 63.86, 61.56, 61.70 | +0.45 to +3.67 |
| MRL+SIA KL on | 1 | 64.57, 62.23, 62.47, 63.47, 63.27 | -0.62 to +0.15 |
| MIPIC KL on | 1 | 61.39, 60.09, 61.07, 60.44, 61.14 | -0.30 to +0.63 |

MIPIC (KL off) is the only exception, and its curve is nearly flat: epoch 3 beats epoch 1 by 0.17. This fits the note in `run_experiments.py`'s `_make_selector`: the contrastive objective is essentially solved within one epoch, and later epochs degrade the encoder.

**Observation.** Both KL-on runs are nearly flat after epoch 1 on the test set (within 0.63 points of the last epoch), while the KL-off runs lose up to 1.24 (MIPIC) to 3.94 (MRL+PIC) points. One seed, untested; it may hint that the KL term limits later drift.

**Caveats.**
- **Pooler only.** Selection and this comparison use the pooler probe. The raw [CLS] cells of Finding 6 were measured only on the selected checkpoint.
- **No per-epoch checkpoints at the early cells.** Only `best.pt` and the last epoch's scores are kept, so learning curves at (1,16) and (2,32) can't be drawn from these runs.
- **The selection metric is biased by feature scale.** `validation_probe_score` fits an unscaled logistic regression with C = 1 on 4,800 rows and averages over all seven widths, including 16D and 32D. Finding 1 (checks 5 and 6) shows that this setting favours the trained models simply because their pooler features are about twice base's scale. Base itself scores 63.16 on this metric, below the epoch 1 scores of MRL (65.37), MRL+PIC (64.56) and MRL+SIA (65.64), although base beats all of them on the full and the scaled probes. So the held-out curve in the table is not a clean measure of sentiment information, and it never registered that training was making the encoder worse than base. Whether the scale bias also shapes the choice between epochs (for example if feature scale grows with epochs) is untested; it would need the selection probe rerun with scaling on per-epoch checkpoints, which these runs don't keep. Source: `h11_probe/selection_probe.py` -> `h11_probe/results/selection_probe_s42.csv`.
- **One training seed.**

**Relates to.**
- **H14** (PIC makes small prefixes learn task information early in training). The current evidence in hypothesis.md ("MIPIC's best epoch was 1/5") does not distinguish H14 from general degradation after epoch 1, because MRL without PIC also peaks at epoch 1 and locally MIPIC is the one run that doesn't. A real test of H14 needs probe scores at 16D and 32D (raw [CLS] at layers 1 and 2) for every epoch of MRL and MIPIC, as the H14 test in hypothesis.md already proposes.
- **Finding 1**, H14 bullet, which already noted that epoch 1 selection fits training degrading the encoder.

---

## Finding 9. The sentence embedding is a random projection: TinyBERT's pooler was never trained, and training changes the embedding through the encoder (PRELIMINARY, seed 42)

**Claim.** `huawei-noah/TinyBERT_General_6L_768D` ships a pooler that was never trained. Its weight is indistinguishable from iid Gaussian noise, its bias is exactly zero, and on real inputs its tanh is within 0.6% of linear. So base's `pooler_output`, the z that the Matryoshka loss truncates, is about R h6: a fixed random projection of the layer 6 [CLS], shrunk about 9x. "The first d dimensions" of z means d random directions of h6, not d hidden coordinates.

Contrastive training changes the pooler's weights by more, relative to their size, than it changes the encoder's. But the pooler change has almost no effect on the probe: on base's layer 6 features, every trained pooler gives the same feature scale and the same sentiment F1 as base's own pooler. The trained models' larger feature scale (about 2x) and their lower sentiment F1 both come from the change in the layer 6 [CLS], that is, from the encoder, even though no encoder layer's weights move by more than 2.7%.

**Evidence: the pooler is untrained.** From `h11_probe/pooler_check.py` -> `h11_probe/results/pooler_check.csv`:

| Statistic | Pooler weight (768x768) | iid Gaussian expectation | 36 trained encoder matrices, same checkpoint |
|---|---:|---:|---:|
| Kurtosis (Gaussian = 3) | 3.003 | 3 | 3.13 to 514 (median 5.10) |
| Top singular value / Marchenko-Pastur edge | 1.001 | about 1 | 1.29 to 8.63 |
| Row-norm CV | 0.0252 | 0.0255 | 0.063 to 0.436 |

- The pooler bias is exactly zero, as are all four bias tensors of the pretraining heads in the hub checkpoint (`cls.*`).
- On base's test layer 6 [CLS], the largest pre-activation is 0.134 and the largest relative tanh gap is 0.59%. Layer 6 [CLS] has mean per-coordinate SD 0.123; the pooler output has 0.0130.
- Kurtosis alone would not settle it, since some trained attention matrices are also close to 3 (the lowest is 3.13). The singular value and row-norm statistics separate the pooler from every trained matrix.
- As a random projection, the pooler costs little: base's pooler prefix scores 62.62 at 16D, against 62.16 for the first 16 raw coordinates of layer 6 [CLS] (`results/base_grid.csv`). At 768D the raw layer 6 [CLS] is ahead (73.51 against 71.73).

**Evidence: where training acts.** Relative weight change, ||W_trained - W_base|| / ||W_base|| over each group, from `h11_probe/weight_drift.py` -> `h11_probe/results/weight_drift_s42.csv` (`best.pt` checkpoints):

| Group | MRL | MRL+SIA | MRL+PIC | MIPIC |
|---|---:|---:|---:|---:|
| Embeddings | 0.49% | 0.53% | 0.65% | 1.14% |
| Layers 1 to 6 | 0.30 to 0.52% | 0.41 to 0.57% | 0.67 to 1.22% | 1.16 to 2.63% |
| Whole encoder, excluding pooler | 0.45% | 0.50% | 0.88% | 1.62% |
| Pooler | 4.26% | 4.59% | 4.86% | 10.45% |
| Pooler cosine to base | 0.9991 | 0.9989 | 0.9988 | 0.9946 |

The PIC models move the early layers most (MRL+PIC: 1.16% and 1.22% at layers 1 and 2, down to 0.67% at layer 6; MIPIC: 2.63% at layer 1, down to 1.16% at layer 6), while MRL and MRL+SIA move the late layers slightly more than the early ones.

Pooler swap, from `h11_probe/pooler_swap.py` -> `h11_probe/results/pooler_swap_s42.csv`: base's layer 6 [CLS] features passed through each trained pooler, with the default probe:

| Pooler weights (on base h6) | Feature SD / base | Row cosine to base output | F1 16D | F1 64D | F1 768D |
|---|---:|---:|---:|---:|---:|
| base | 1.000 | 1.000 | 62.65 | 68.04 | 71.80 |
| MRL | 0.994 | 0.949 | 62.88 | 67.91 | 71.80 |
| MRL+SIA | 0.997 | 0.950 | 62.83 | 68.14 | 71.80 |
| MRL+PIC | 0.977 | 0.954 | 62.79 | 68.02 | 71.94 |
| MIPIC | 0.983 | 0.927 | 63.60 | 67.95 | 72.01 |

The trained pooler weights rotate the output a little (row cosine 0.93 to 0.95) but keep its scale and its sentiment F1: every swapped cell is within -0.13 to +0.95 of base. By contrast, the trained models' own pooler features (trained pooler on trained h6) have mean SD 0.0230 to 0.0280 against base's 0.0129, and score 0.44 to 3.96 points below base at 768D (`results/*_s42_grid.csv`; CIs in Finding 1, check 4). So the scale change and the sentiment loss come from h6. (The base row here is recomputed in float64 on CPU and differs from `base_grid.csv` by up to 0.08.)

**What it means for where the Matryoshka loss acts.** The loss is applied to prefixes of z = tanh(W h6 + b), with W random and nearly frozen in effect. To make the first 16 or 64 coordinates of z useful, training has to reshape h6 so that the contrastive signal falls along the fixed random directions picked out by the first rows of W. That happens through small changes spread over the encoder (well under 3% per layer), which double h6's scale and, on this task, remove some sentiment information. Any "Matryoshka ordering" lives in this near-random basis, not in the hidden coordinates of h6. This supports the pooling concerns in [hypothesis.md](../hypothesis.md) (lines 68 and 372).

**Caveats.**
- **One direction of the swap only.** The swap holds h6 at base and changes the pooler. The reverse (trained h6 through base's pooler) needs the trained models' layer 6 features, which were deleted to free disk space; re-extracting them takes a GPU pass. The two effects need not add up exactly.
- **Weight change is not effect size.** Relative Frobenius change says how far parameters moved, not how much the function changed; the swap is the functional evidence.
- **One training seed.**

**Relates to.**
- **Finding 1**: base's read-out is this random projection, and the trained models' scale change is what biases the unscaled small probes.
- **H3** and the pooling notes in hypothesis.md: MRL on `pooler_output` constrains random mixtures of hidden coordinates, not the coordinates themselves.
- **H11 and H14** (PIC): the PIC models move the early layers most, consistent with PIC acting on the early checkpoints (Finding 6).
