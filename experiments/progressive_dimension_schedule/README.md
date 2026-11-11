# H13 — Progressive PIC layer-dimension schedule

## 1. Hypothesis

The progressive PIC layer-dimension schedule improves the utility of small embedding prefixes.

PIC applies a layer-specific representation dimension during training. This experiment tests whether growing that dimension across encoder layers is more useful at evaluation widths of 16 and 32 than using a constant, reversed, or irregular schedule.

## 2. Experiment design

Train four fresh MIPIC models per seed. Keep the training corpus, model, optimizer, training duration, and evaluation protocol fixed; vary only the PIC layer-dimension schedule.

| Schedule | Dimensions at encoder layers 1–6 |
|---|---|
| `progressive` | 16, 32, 64, 256, 512, 768 |
| `constant_small` | 16, 16, 16, 16, 16, 16 |
| `reversed` | 768, 512, 256, 64, 32, 16 |
| `random_order` | 16, 512, 32, 256, 64, 768 |

The irregular schedule is a fixed control, not a random sample. It uses the same dimensions as the progressive schedule in a different order.

## 3. Training and evaluation

| Item | Setting |
|---|---|
| Backbone | TinyBERT-6L (768-dimensional) |
| Training data | `data/final_data.csv` |
| Probe-training data | `data/tweet_train_3k.csv` |
| Probe-test data | `data/tweet_test.csv` |
| Seeds | 1, 2, 42 |
| Epochs | 5 |
| Batch size | 16 |
| Maximum sequence length | 256 |
| Training batches | Full corpus; no batch limit |
| Probe dimensions | 16, 32, 64, 128, 256, 512, 768 |
| Primary metric | Test macro-F1 |

Each schedule starts from a newly initialized, seed-controlled model. Evaluation uses the repository's frozen-encoder Tweet sentiment probe protocol. The hypothesis metric is the mean macro-F1 at 16D and 32D, averaged over the three seeds. The notebook marks the hypothesis supported only if `progressive` has a higher mean than **each** of the other three schedules.

## 4. Results

Mean test macro-F1 across seeds:

| Schedule | 16D | 32D | Mean at 16D/32D |
|---|---:|---:|---:|
| `progressive` | 0.5487 | 0.5792 | 0.5640 |
| `constant_small` | 0.4919 | 0.5176 | 0.5047 |
| `reversed` | 0.4890 | 0.5193 | 0.5041 |
| `random_order` | **0.5675** | **0.5905** | **0.5790** |

Macro-F1 across all evaluated widths:

| Schedule | 16D | 32D | 64D | 128D | 256D | 512D | 768D |
|---|---:|---:|---:|---:|---:|---:|---:|
| `progressive` | 0.5487 | 0.5792 | 0.5900 | 0.6020 | 0.6131 | 0.6208 | 0.6266 |
| `constant_small` | 0.4919 | 0.5176 | 0.5343 | 0.5545 | 0.5699 | 0.5839 | 0.5909 |
| `reversed` | 0.4890 | 0.5193 | 0.5330 | 0.5503 | 0.5662 | 0.5917 | 0.5987 |
| `random_order` | **0.5675** | **0.5905** | **0.5984** | **0.6058** | **0.6162** | **0.6286** | **0.6342** |

### Verdict: not supported under this experimental setting

Under this experiment's predeclared decision rule, the hypothesis is **not supported**: `progressive` must outperform every alternative, and `random_order` has the higher mean at 16D/32D (0.5790 versus 0.5640). This verdict is specific to the tested model, data, training configuration, probe, and three seeds; it does not establish that the progressive schedule is ineffective in other settings.

The observed gap is small: about 0.015 macro-F1, or 1.5 percentage points, on the 16D/32D average. The per-seed difference (`progressive` minus `random_order`) is -0.0326 (seed 1), -0.0142 (seed 2), and +0.0016 (seed 42). Given this limited experiment and the modest average gap, the progressive schedule remains a valid option to consider; these results alone do not show that it is meaningfully worse. No confidence intervals or significance test were computed, so the comparison should not be interpreted as evidence that the schedules are equivalent.

## 5. Reproduction

Open and run [`code/progressive_dimension_schedule.ipynb`](code/progressive_dimension_schedule.ipynb) in a Kaggle notebook with a GPU enabled. The notebook sets the three seeds, trains all four schedules for five epochs, evaluates all seven dimensions, and writes the result files below. It expects access to the repository data and installs the notebook dependencies during setup.

## 6. Files

- [`code/progressive_dimension_schedule.ipynb`](code/progressive_dimension_schedule.ipynb) — experiment runner and analysis.
- [`results/h13_probe_results.csv`](results/h13_probe_results.csv) — per-seed test metrics for each schedule and dimension.
- [`results/h13_training_history.csv`](results/h13_training_history.csv) — per-seed, per-epoch training loss components.
- [`results/h13_conclusion.json`](results/h13_conclusion.json) — hypothesis, decision, schedule definitions, and aggregate small-prefix scores.
- `results/final_data_seed{1,2,42}.csv` — seed-specific training-corpus copies used by the run.

## 7. Limitations

- The conclusion uses three seeds and does not include confidence intervals or a statistical test.
- The result depends on the supplied TinyBERT checkpoint, training corpus, Tweet sentiment splits, and probe protocol; it is not a direct reproduction of the paper's multi-task results.
- `random_order` is one hand-selected alternative schedule. It does not establish how other permutations would perform.
