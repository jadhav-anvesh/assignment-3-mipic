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

The repository contains the shared MIPIC implementation and a collection of controlled experiments that test the paper's claims and extensions. The implementation (models, losses, training, evaluation and tests) matches [MIPIC_hypotheses](https://github.com/maverick-0215/MIPIC_hypotheses). The experiment folders contain the additional analyses, ablations and cross-seed evaluations described in the project report.

```text
.
├── data/                         # Training corpus and tweet sentiment probe splits
├── experiments/
│   ├── coarse_to_fine/           # Coarse-to-fine hierarchy experiment
│   ├── dimension_routing/        # MRL dimension-routing experiments
│   ├── indic_experiment/         # MRL transfer to Hindi, Bengali, Gujarati, Tamil and Telugu
│   ├── layerwise_probe/          # Layer-wise and width-wise probing experiments
│   ├── progressive_dimension_schedule/ # Progressive PIC layer-dimension schedules
│   ├── sia_pic/                  # Five-seed SIA, PIC and MIPIC ablations
│   └── syntax_preservation/      # POS and sentiment probes for syntax preservation
├── losses/                       # MRL, SIA and PIC losses, distributed helpers
├── models/                       # TinyBERT wrappers: MRL, MRL+SIA, MRL+PIC, MIPIC
├── report/                       # Project report and figures
├── tests/                        # Unit, integration, protocol and attention-KL switch tests
├── .gitignore
├── evaluate.py                   # Logistic-regression probe evaluation
├── prepare_training_corpus.py    # Builds data/final_data.csv
├── README.md                     # This file
├── requirements.txt              # Canonical dependencies for all environments
├── run_experiments.py            # Main experiment runner (MIPIC_* environment variables)
└── train.py                      # Training routines
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

## Contributions

The experimental contributions are organized by experiment folder:

| Experiment | Contributors |
|---|---|
| `experiments/coarse_to_fine/` | Nakka Eshwar Karthikeya and Mamilla Aniruddh Reddy |
| `experiments/dimension_routing/` | Thoutam Dhruthika |
| `experiments/indic_experiment/` | Burra Saharsh |
| `experiments/layerwise_probe/` | Jadhav Anvesh |
| `experiments/progressive_dimension_schedule/` | Chinthala Shivamani |
| `experiments/sia_pic/` | A V S Manoj |
| `experiments/syntax_preservation/` | Ogiboyina Akash |
