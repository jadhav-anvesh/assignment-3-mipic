# SIA/PIC experiment

This folder contains the execution artifacts for the three-seed SIA/PIC
complement experiment. Training and model code remain shared at the repository
root (`run_experiments.py`, `train.py`, `evaluate.py`, `models/`, `losses/`,
and `data/`).

## Execution

- [`run_kl_on_kaggle.ipynb`](./run_kl_on_kaggle.ipynb) is the completed Kaggle run. It trained
  the six new attention-KL-on conditions: `MRL+SIA` and `MIPIC` for seeds
  `1, 2, 42`.
- [`evaluate_existing_checkpoints.ipynb`](./evaluate_existing_checkpoints.ipynb)
  evaluates the existing KL-off checkpoints locally. It does not retrain
  models or upload checkpoints.
- These are the exact completed notebook records.
- [`experiment.py`](./experiment.py) provides the shared orchestration and
  supports `--kl-on-only`, `--kl-off-only`, and `--kl-off-dir`.

Both runs use five epochs, learning rate `2e-5`, global batch size `16`,
`alpha=0.4`, maximum sequence length `256`, and widths
`16, 32, 64, 128, 256, 512, 768`. Evaluation covers Tweet sentiment,
Banking77, and STS-B.

## Results

[`results/kl_off_scores.json`](./results/kl_off_scores.json) contains the
completed local KL-off matrix: 252 rows from 3 seeds, 4 arms, 3 datasets, and
7 widths. [`results/kl_off_contrasts.csv`](./results/kl_off_contrasts.csv)
contains the corresponding seed-level contrasts and 95% t intervals.

Selected KL-off results (mean with 95% interval):

| Dataset | Quantity | Width | Result |
|---|---|---:|---:|
| Tweet | SIA effect | 16 | -1.19 [-4.61, 2.24] |
| Tweet | Interaction | 16 | -1.88 [-6.12, 2.35] |
| Tweet | Full-width gap | 768 | -2.66 [-4.77, -0.55] |
| Banking77 | SIA effect | 16 | -0.00 [-37.77, 37.76] |
| STS-B | Interaction | 768 | 3.65 [1.32, 5.98] |

The Kaggle KL-on run completed successfully and produced 126 score rows
(3 seeds × 2 arms × 3 datasets × 7 widths). Its raw Kaggle output is not
stored here; [`results/kl_on_run_status.json`](./results/kl_on_run_status.json)
records the completed run and matrix shape. The final KL-off/KL-on comparison
requires downloading that Kaggle `scores.json` and combining it with the local
KL-off scores.

These are exploratory three-seed results, not the originally planned
five-seed confirmatory experiment.
