# MIPIC Reproduction

A controlled reimplementation of **MIPIC: Matryoshka Representation Learning via Self-Distilled Intra-Relational and Progressive Information Chaining** on **TinyBERT-6L (768D)**.

The repository trains and evaluates four configurations under the same protocol:

- **MRL**
- **MRL + SIA**
- **MRL + PIC**
- **MIPIC (MRL + SIA + PIC)**

Representations are evaluated at seven nested dimensions: `16, 32, 64, 128, 256, 512, 768`.

## Evaluation setup

The encoder is trained on `data/final_data.csv` (20,240 sentences from seven sources). Downstream evaluation uses the supplied three-class tweet sentiment files:

- `data/tweet_train.csv` — 26,732 probe-training examples
- `data/tweet_test.csv` — 3,432 test examples

A Logistic Regression probe is fitted on the frozen encoder representations at each nested dimension. The evaluation files are not the public TweetEval benchmark; they correspond to the supplied Kaggle Tweet Sentiment Extraction data. Consequently, the reported values are intended for controlled comparison between the four implementations rather than direct numerical reproduction of the paper's multi-task TweetEval results.

## Repository structure

The repository has two parts. The MIPIC implementation (models, losses, training, evaluation and tests) matches [MIPIC_hypotheses](https://github.com/maverick-0215/MIPIC_hypotheses). `experiments/layerwise_probe/` holds the probing experiments that test the paper's claims. The POS-probe, Indic and routing experiments from the original course repository, [MIPIC_LEXIS_A1](https://github.com/ManojAnaparthi/MIPIC_LEXIS_A1), are not included.

```text
.
├── README.md                     # This file
│
├── data/                         # Training corpus and tweet sentiment probe splits
├── models/                       # TinyBERT wrappers: MRL, MRL+SIA, MRL+PIC, MIPIC
├── losses/                       # MRL, SIA and PIC losses, distributed helpers
├── train.py                      # Training routines
├── evaluate.py                   # Logistic-regression probe evaluation
├── run_experiments.py            # Main experiment runner (MIPIC_* environment variables)
├── prepare_training_corpus.py    # Builds data/final_data.csv
├── tests/                        # Unit, integration, protocol and attention-KL switch tests
│
├── kaggle/
│   ├── README.md                 # Kaggle (2x T4) setup and run instructions
│   └── kaggle_bootstrap.py
│
├── experiments/layerwise_probe/  # Probing experiments
│   ├── FINDINGS.md               # Findings 1 to 9 with numbers, CIs and caveats
│   ├── CROSS_SEED.md             # Cross-seed tables (seeds 42, 1, 2)
│   ├── train_all.ps1 / .sh       # Train the four arms for one seed (attention KL pinned off)
│   ├── train_kl_on.py            # Train MRL+SIA or MIPIC with attention KL forced on
│   ├── analyze_seed.ps1 / .sh    # Feature extraction, probe grid, STS-B and subset probes per seed
│   ├── load_encoder.py           # Load a bare encoder from any checkpoint format
│   ├── extract_cls.py            # [CLS] per layer and pooler output, float32
│   ├── probe_grid.py             # Probe on every (layer, width) cell
│   ├── sts_eval.py               # Zero-shot STS-B cosine Spearman per cell
│   ├── subset_probe.py           # Prefix versus other and random coordinate subsets
│   ├── bootstrap*.py             # Paired bootstrap CIs between models, KL-on pairs, SIA x PIC
│   ├── probe_size.py, probe_tuned*.py, probe_knn.py, selection_probe.py,
│   │   pooler_*.py, refit_check.py, overlap_check.py, check_padding.py,
│   │   weight_drift.py           # Checks behind Findings 1 and 9
│   ├── aggregate_seeds.py        # Builds CROSS_SEED.md and results/cross_seed_*.csv
│   ├── test_load_encoder.py      # Tests for load_encoder
│   ├── results/                  # Per-model result tables (tracked)
│   ├── runs/seed{42,1,2}/        # KL-off runs: result tables tracked, checkpoints and logs local
│   └── runs_kl/seed42/           # KL-on runs of MRL+SIA and MIPIC
│
└── requirements.txt              # Canonical dependencies for all environments
```

**Attention KL.** `MIPIC_ATTENTION_KL` switches the SIA attention-KL term (Eq. 6) on or off. It is on by default, as in the paper. The `layerwise_probe` runs were made with it off, so `experiments/layerwise_probe/train_all.*` set `MIPIC_ATTENTION_KL=0`; pass `-AttentionKL` / `--attention-kl` to train the SIA arms with it on.

## Requirements

Install the canonical dependencies in any environment:

```bash
pip install -r requirements.txt
```

The backbone is downloaded from Hugging Face on first use:
`huawei-noah/TinyBERT_General_6L_768D`.

## Run

The main entry point is:

```bash
python run_experiments.py --model all
```

Run an individual configuration with:

```bash
python run_experiments.py --model mrl
python run_experiments.py --model mrl_sia
python run_experiments.py --model mrl_pic
python run_experiments.py --model mipic
```

The default experimental configuration uses 5 epochs, learning rate `2e-5`, maximum sequence length `256`, global batch size `16`, and nested dimensions `16--768`.

For a quick smoke test:

```bash
MIPIC_MAX_BATCHES=3 python run_experiments.py --model mipic
```

For Kaggle / 2-GPU execution, see [`kaggle/README.md`](kaggle/README.md).

## Outputs

Results are written to `results/` by default:

```text
results/
├── tweeteval_results.json
├── tweeteval_results.csv
└── checkpoints/
    └── <model>/
        ├── model.pt
        └── training_state.pt
```

Training checkpoints contain the model and resumable training state. An incomplete run can be resumed by rerunning the same command; use `MIPIC_FRESH=1` for a fresh run.

## Tests

Run the full test suite with:

```bash
pytest -q
```

The tests cover the individual losses, model integration, evaluation protocol, training corpus, distributed execution, checkpoint handling, and Kaggle readiness.

## Current controlled results

Macro-F1 on the supplied three-class tweet sentiment evaluation:

| Dim. | MRL | MRL + SIA | MRL + PIC | MIPIC |
|---:|---:|---:|---:|---:|
| 16  | 56.46 | 53.26 | 54.27 | **58.60** |
| 32  | 59.99 | 55.16 | 59.85 | **60.51** |
| 64  | **62.82** | 56.16 | 61.62 | 61.83 |
| 128 | **65.57** | 57.30 | 62.55 | 62.65 |
| 256 | **66.82** | 58.37 | 63.21 | 62.92 |
| 512 | **67.61** | 58.99 | 64.27 | 64.01 |
| 768 | **67.24** | 59.53 | 64.82 | 64.64 |

MIPIC is strongest at 16 and 32 dimensions in this evaluation. MRL is stronger from 64 dimensions onward. The full report documents the experimental limitations and comparison with the paper.

## Reference

Phung Gia Huy et al. *MIPIC: Matryoshka Representation Learning via Self-Distilled Intra-Relational and Progressive Information Chaining*. Findings of ACL, 2026.

## Project report

The accompanying project report contains the reproduction methodology, dataset description, implementation comparison, ablation results, and discussion of the observed behaviour.
