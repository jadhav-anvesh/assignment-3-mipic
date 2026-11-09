# H1 test plan: coarse-to-fine prefixes (sentence level)

Status: plan. No results yet. Written 2026-11-05.

## 1. The claim

The MIPIC paper makes these statements:

- §2.1: MRL learns "nested coarse-to-fine prefixes within a single embedding".
- §3.2.2: "low-dimensional cores capture a coarse semantic skeleton, while higher-dimensional spaces progressively incorporate finer details".
- §5: SIA and PIC "enable a coarse-to-fine refinement that maintains high quality at every dimension".

The paper does not define "coarse" or "fine". The paper does not measure them. It shows only the score at each width.

This plan tests the claim on the sentence embedding only. It does not test token vectors.

## 2. The three parts of H1

| ID | Statement | Status |
|---|---|---|
| H1-a | The prefix `z[0:d]` holds more task information than other sets of d coordinates. | Precondition. MRL causes this by design. |
| H1-b | The prefix holds coarse information first. Fine information comes later. | Main claim. |
| H1-c | The block `z[d:2d]` adds fine detail inside each coarse class. It does not add more coarse information. | Refinement claim. |

"Coarse" and "fine" have one meaning in this plan: two levels of the same label hierarchy on the same sentences. Each fine label is a member of one coarse label.

## 3. Setup

Use the trained models from the `h11_probe` runs. Do not train new models. Do not change the configuration.

| Item | Value |
|---|---|
| Code | This repository, as `h11_probe/train_all.ps1` runs it |
| Checkpoints | `MRL_LEXIS_H11_checkpoints/h11_probe/runs/seed{42,1,2}/checkpoints/{mrl,mrl_sia,mrl_pic,mipic}/best.pt`. Load them with `h11_probe/load_encoder.py`. |
| Sentence embedding z | `pooler_output` (one 768-value vector for each sentence) |
| Attention KL | Off (`MIPIC_ATTENTION_KL=0`, set by the `h11_probe` launchers) |
| Training | 5 epochs, lr 2e-5, max length 256, batch 16, `data/final_data.csv` |
| Seeds | 42, 1, 2 |
| Widths d | 16, 32, 64, 128, 256 (768 is the reference) |

Models:

| Model | Purpose |
|---|---|
| Base TinyBERT-6L (no training) | Null model. Its pooler is random, so its coordinates have no order. |
| MRL | Shows the effect of MRL alone. |
| MRL+SIA, MRL+PIC | Show the effect of each part. |
| MIPIC | The model that the claim is about. |

Read-outs:

- Main: `pooler_output`. The MRL loss and the paper's evaluation use this vector.
- Secondary: the raw layer 6 [CLS] vector. SIA and PIC act on these coordinates. Report this read-out, but do not apply the decision rules to it.

## 4. Data

Use datasets with an official label hierarchy. The coarse and fine labels must be on the same sentences.

| Dataset | Coarse level | Fine level | Notes |
|---|---|---|---|
| TREC | 6 classes | 50 classes | Labels are part of the dataset. The test set has 500 rows. |
| GoEmotions | 4 (sentiment) and 7 (Ekman) | 28 classes | Use the official mapping. Keep only rows with one label. |

Use the official train, validation and test splits. TREC has no validation split. Take 10% of TREC train as validation, with a fixed seed.

## 5. Coordinate orders

For each model, seed, read-out and dataset, compare these orders of the 768 coordinates:

1. **Prefix**: coordinates 0, 1, 2, ... (the MRL order).
2. **Random**: 10 random permutations with fixed seeds (permutation k uses seed 1000 + k). The first d coordinates of each permutation are one random set. These sets are nested, the same as the prefix.
3. **Reverse**: coordinates 767, 766, ... (the last d coordinates).
4. **PCA**: the top d principal components of the model's own z, fitted on the train split. This shows a "highest variance first" order.

At width d, each order gives one feature set of d values.

## 6. Probes

- Use logistic regression.
- Standardize each feature with the train-split mean and SD.
- Select C by 5-fold CV on train at full width (768) for each model and label level. Use that C for all orders and widths.
- Report macro-F1 on the test split. Macro-F1 calculates the F1 of each class, then takes the plain mean. Thus each class has the same weight, and a probe cannot get a high score if it ignores the small classes.
- Run a kNN probe (k = 10, cosine) on the prefix and random orders at d = 16 and 32. This is a check without a fitted classifier.
  - Reason: a logistic-regression probe learns its own weights for the d coordinates. A good score can come from the probe, not from the arrangement of z. kNN learns no weights. It finds the 10 nearest train sentences with cosine distance and uses the majority label. Thus kNN measures only the geometry of z: are sentences with the same label near each other in these d coordinates?
  - If kNN also shows "prefix better than random", the result does not depend on the fitted probe. If kNN does not show it, examine the logistic-regression result before you use it.
  - Use the raw coordinates of z (no standardization), because the check is about the geometry of z itself.

## 7. Metrics

Let S(d) be the test macro-F1 at width d. Let S_chance be the macro-F1 of a stratified random classifier. This classifier guesses each label at its train frequency.

Summary of the metrics:

| Metric | Question | Value that supports H1 |
|---|---|---|
| R(d) | Which fraction of the full-width information do the d coordinates keep? | (Building block) |
| A(d) | Are the first d coordinates better than d random coordinates? | A > 0 (H1-a) |
| ΔCFI(d) | Does the prefix favour coarse labels more than a random order does? | ΔCFI > 0 (H1-b) |
| ΔRI(d) | Does the block [d:2d] add fine detail inside each coarse class, more than with a random order? | ΔRI > 0 (H1-c) |

**Retention.**
R(d) = (S(d) − S_chance) / (S(768) − S_chance)

R puts each task on the same scale. R = 0 means the d coordinates are not better than a guess. R = 1 means they are as good as all 768 coordinates. Raw F1 cannot compare a 6-class task with a 50-class task, because chance and maximum scores are different. R can.

**Prefix advantage (H1-a).**
A(d) = S_prefix(d) − mean of S_random(d).
Also report the percentile of S_prefix(d) in the 10 random values.

A is the gain from the MRL order over a random choice of the same number of coordinates. The percentile shows if the gain is larger than the spread of the random sets.

**Coarse-to-fine index (H1-b).**
CFI(d) = R_coarse(d) − R_fine(d)
ΔCFI(d) = CFI_prefix(d) − mean of CFI_random(d)

CFI > 0 means the d coordinates keep more of the coarse information than of the fine information. A coarse task gets high retention at small d with any order, including a random order. Thus CFI > 0 alone does not support H1. ΔCFI removes the part that a random order also gives. Only ΔCFI > 0 supports H1-b.

Example: the prefix has CFI = 0.40 and the random orders have a mean CFI = 0.30. Then ΔCFI = 0.10, and the prefix favours coarse information more than chance does. If the prefix is better by the same amount on coarse and fine, ΔCFI = 0. Then the prefix is better, but it is not "coarse first".

**Refinement index (H1-c).**
W(d) is the within-coarse fine score. To calculate it, fit one fine probe for each coarse class, on the rows of that class only. Then take the mean of the scores, weighted by class size. Normalize W like R.
RI(d) = [W(2d) − W(d)] − [R_coarse(2d) − R_coarse(d)]
ΔRI(d) = RI_prefix(d) − mean of RI_random(d)

W asks: if the coarse class is known, can the d coordinates find the fine class? For example, inside TREC `LOC`, can they separate `city` from `country`? RI compares two gains when the width goes from d to 2d: the gain in W and the gain in coarse retention. RI > 0 means the new block [d:2d] helps fine-within-coarse more than coarse. That is "refinement". ΔRI removes the part that a random order also gives.

**Uncertainty.**
- Use a paired bootstrap over test rows: 1,000 resamples, seed 0. Each resample takes the test rows at random, with replacement, and calculates all metrics again. The middle 95% of the 1,000 values is the 95% CI.
- Use the same resampled rows for all orders and models. Thus a difficult resample affects all orders and models in the same way, and the differences between them are fair.
- Report the mean, SD and sign agreement over the 3 seeds. Sign agreement shows if the effect has the same direction in all 3 seeds.

## 8. Decision rules

Fix these rules before you look at results. Apply them to the `pooler_output` read-out. Use d = 16 and 32. The tests must pass on both datasets and in all 3 seeds.

| Part | Supported if | Not supported if |
|---|---|---|
| H1-a | z(d) > 2 for MIPIC, with z(d) = A(d) / SD of the 10 random scores. Base has z(d) ≤ 2. | z(d) ≤ 2 for MIPIC, or z(d) > 2 for base. |
| H1-b | ΔCFI(d) > 0 for MIPIC, and the 95% CI does not include 0. | The CI of ΔCFI includes 0, or ΔCFI < 0. |
| H1-b, caused by MIPIC | ΔCFI(MIPIC) − ΔCFI(MRL) > 0, with a paired CI that does not include 0. | The difference has a CI that includes 0. Then the order comes from MRL, not from SIA or PIC. |
| H1-c | ΔRI(d) > 0 for MIPIC, and the 95% CI does not include 0. | The CI of ΔRI includes 0, or ΔRI < 0. |

Note on the H1-a rule (changed 2026-11-07): the first rule was "above the 95th percentile of the random sets". With 10 random sets, the highest possible result is "above all 10", which is about the 91st percentile. Thus that rule could not pass. "z > 2" (more than 2 SD above the random mean) is the nearest equivalent of the 95% level. We made this change because of the number of random sets, not because of the results. The result of H1-a is the same with both rules.

Interpretation:

- H1-a supported, but H1-b not supported: the prefix is ordered by usefulness for the training objective, not by granularity.
- Base must show A(d) ≈ 0 and ΔCFI(d) ≈ 0. If it does not, the pipeline has a fault. Stop and find the fault.

## 9. Procedure

1. Load the `best.pt` checkpoints of the 4 trained arms and 3 seeds from the `h11_probe` runs (see section 3).
2. Download TREC and GoEmotions. Apply the GoEmotions mapping. Save the label levels.
3. Extract `pooler_output` and layer 6 [CLS] for each split, model and seed. Store as float32.
4. Make the 10 random permutations and the PCA orders. Save them.
5. Select C for each model and label level.
6. Fit the probes for each order and width. Save the test predictions.
7. Run the kNN check.
8. Calculate R, A, CFI, ΔCFI, W, RI and ΔRI.
9. Run the paired bootstrap.
10. Apply the decision rules in section 8.
11. Write the results in `H1_results.md`.

## 10. Outputs

- One table for each metric: model × width × dataset × seed.
- One plot for each dataset: R(d) for coarse and fine, prefix against the random band, one panel for each model.
- `H1_results.md` with the verdict for H1-a, H1-b and H1-c, and all CIs.

## 11. Limits

- The TREC test set has 500 rows. CIs on TREC will be wide.
- This plan uses one meaning of "coarse" and "fine". A different hierarchy can give a different result.
- The setup uses the `h11_probe` models: current code, attention KL off. It is not the original V20 code, which had KL on. Thus the README numbers do not apply to these models.
- `best.pt` is the epoch (1 to 3) that the `h11_probe` held-out probe selected. It is not the epoch-5 model.
- These results apply to TinyBERT-6L only.
