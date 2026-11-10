# SIA/PIC/MIPIC five-seed experiment

This experiment tests three claims about semantic information alignment (SIA),
positional information compression (PIC), and their combination in MIPIC:

1. SIA is beneficial under severe dimensional truncation.
2. SIA and PIC have a positive interaction.
3. MIPIC does not materially differ from MRL at full width.

The original claims were based on single runs. This experiment evaluates the
claims across five seeds, seven nested embedding widths, and three datasets,
with and without the attention-KL term.

## Repository layout

The experiment reuses the shared implementation at the repository root. The
experiment-specific files are:

- [`experiment.py`](./experiment.py): multi-seed training, evaluation, and
  contrast computation.
- [`run_kl_on_kaggle.ipynb`](./run_kl_on_kaggle.ipynb): executed Kaggle run for
  the initial KL-on SIA/MIPIC conditions.
- [`mipic-a3-rem-seeds.ipynb`](./mipic-a3-rem-seeds.ipynb): executed Kaggle run
  for the remaining seeds and conditions.
- [`evaluate_existing_checkpoints.ipynb`](./evaluate_existing_checkpoints.ipynb):
  local evaluation of existing KL-off checkpoints.
- [`merged_results/`](./merged_results): complete five-seed score matrix and
  recomputed contrasts.

The root-level [`kaggle_dataset.zip`](../../kaggle_dataset.zip) contains the
source, tests, and datasets needed by the Kaggle wrapper. Existing checkpoints
are not included; they remain local.

## Notebook execution matrix

The notebooks are complementary; they do not each run the entire five-seed
matrix.

| Notebook | Environment | Seeds | Attention-KL | Arms run | Training |
|---|---|---|---|---|---|
| [`run_kl_on_kaggle.ipynb`](./run_kl_on_kaggle.ipynb) | Kaggle, 2× Tesla T4, DDP | `1, 2, 42` | on | MRL+SIA, MIPIC | 6 runs |
| [`mipic-a3-rem-seeds.ipynb`](./mipic-a3-rem-seeds.ipynb) | Kaggle, 2× Tesla T4, DDP | `3, 5` | off and on | KL-off: MRL, MRL+SIA, MRL+PIC, MIPIC; KL-on: MRL+SIA, MIPIC | 12 runs |
| [`evaluate_existing_checkpoints.ipynb`](./evaluate_existing_checkpoints.ipynb) | Local | `1, 2, 42` | off | MRL, MRL+SIA, MRL+PIC, MIPIC | 0 runs; evaluation only |

Both Kaggle notebooks use the same explicit training settings:

- `epochs=5`
- learning rate `2e-5`
- global batch size `16` (`8` per GPU)
- `alpha=0.4`
- maximum sequence length `256`
- cosine scheduler
- `MIPIC_RUN_MODE=ddp`
- `MIPIC_FRESH=0` / `FRESH=False`, allowing completed checkpoints to resume
- output directories: `/kaggle/working/kl_on_experiment` for the first run and
  `/kaggle/working/sia_pic_remaining_seeds` for the remaining-seed run

The local notebook does not train or upload checkpoints. It loads the existing
KL-off checkpoints from `h11_probe/runs/seed{1,2,42}` and evaluates all four
arms on TweetEval, Banking77, and STS-B at all seven widths. The Kaggle
notebooks train new conditions only; they never depend on those large local
checkpoints.

Each notebook also performs repository bootstrap, dependency installation,
two-GPU validation, dataset/schema checks, compilation and targeted tests,
training, probing, and result display. If an STS-B CSV is not present, the
notebook downloads and parses the official GLUE STS-B `dev.tsv` fallback.

## Experimental design

### Models

Each condition starts from a fresh TinyBERT-6L encoder with a 768-dimensional
hidden size:

- **MRL**: Matryoshka Representation Learning baseline.
- **MRL+SIA**: MRL plus semantic information alignment.
- **MRL+PIC**: MRL plus positional information compression.
- **MIPIC**: MRL+SIA+PIC.

The attention-KL term is relevant only to MRL+SIA and MIPIC. Therefore, the
KL-on matrix contains newly trained KL-on scores for those two arms, while the
MRL and MRL+PIC KL-on entries use their corresponding KL-off scores. This is
also how [`experiment.py`](./experiment.py) selects checkpoints.

### Training

All training uses:

- TinyBERT-6L, 768-dimensional backbone
- 5 epochs
- learning rate `2e-5`
- global batch size `16`
- per-GPU batch size `8` with two-GPU DDP
- MRL loss weight `alpha=0.4`
- maximum sequence length `256`
- cosine learning-rate schedule
- nested widths `16, 32, 64, 128, 256, 512, 768`
- seeds `1, 2, 3, 5, 42`

The contrastive training corpus is `data/final_data.csv` with 20,240 rows.
The corpus source counts observed during execution were:

| Source | Rows |
|---|---:|
| BANKING77 | 3,000 |
| TWEET_EVAL_SENT | 3,000 |
| DEEHAN1866/WIC | 3,000 |
| MRPC | 3,000 |
| DGEM_FORMAT | 3,000 |
| STSB | 2,893 |
| SICK | 2,347 |

### Evaluation

Evaluation uses:

- Tweet sentiment: macro-F1
- Banking77: macro-F1
- STS-B: cosine-similarity Spearman correlation, multiplied by 100

The classification probe uses the full available train and test splits:

- TweetEval: 26,732 train / 3,432 test
- Banking77: 10,003 train / 3,076 test
- STS-B: official GLUE `dev.tsv`, parsed into 1,468 examples

## Hypotheses and contrasts

For each dataset, width, KL setting, and seed, the following quantities are
computed:

```text
SIA effect  = MIPIC - (MRL+PIC)

Interaction = (MIPIC - MRL)
              - [(MRL+SIA - MRL) + (MRL+PIC - MRL)]
            = MIPIC - MRL+SIA - MRL+PIC + MRL

Full-width gap = MIPIC - MRL, at width 768
```

Confidence intervals are two-sided 95% Student-t intervals over seeds. The
five-seed intervals have four degrees of freedom.

The predicted claims would be supported if:

- the SIA-effect interval is above zero at widths 16 and 32;
- the interaction interval is above zero; and
- the full-width-gap interval contains zero.

## Results

The complete merged matrix is in [`merged_results/`](./merged_results):

- [`scores.json`](./merged_results/scores.json): 840 unique rows
  (`5 seeds × 2 KL settings × 4 model entries × 3 datasets × 7 widths`).
- [`contrasts.csv`](./merged_results/contrasts.csv): 90 contrast rows with
  seed values, means, and confidence intervals.
- [`manifest.json`](./merged_results/manifest.json): source files and merge
  construction details.

Selected five-seed results are shown below. Values are mean
`[95% confidence interval]`.

| Dataset | KL | SIA effect, 16D | Interaction, 16D | SIA effect, 32D | Interaction, 32D | Full-width gap |
|---|---|---:|---:|---:|---:|---:|
| Tweet | off | -1.05 [-2.67, 0.57] | -0.77 [-3.20, 1.67] | -0.33 [-2.08, 1.42] | -0.12 [-2.38, 2.15] | -3.01 [-4.45, -1.57] |
| Tweet | on | -1.95 [-4.67, 0.76] | -0.78 [-3.85, 2.30] | -0.72 [-2.53, 1.08] | 0.93 [-0.35, 2.20] | -3.10 [-3.79, -2.41] |
| Banking77 | off | -0.79 [-14.73, 13.16] | 1.94 [-10.30, 14.17] | 0.70 [-14.05, 15.46] | 4.51 [-10.53, 19.56] | -2.72 [-7.98, 2.53] |
| Banking77 | on | -1.11 [-11.26, 9.04] | -2.86 [-9.41, 3.69] | 0.12 [-9.34, 9.57] | -1.79 [-8.30, 4.72] | -2.84 [-5.87, 0.20] |
| STS-B | off | 0.85 [-1.15, 2.84] | **2.71 [0.24, 5.19]** | 0.85 [-0.96, 2.65] | **3.08 [0.58, 5.59]** | -1.58 [-6.14, 2.98] |
| STS-B | on | 1.84 [-0.72, 4.41] | 0.27 [-1.43, 1.96] | 1.30 [-1.83, 4.42] | -0.10 [-3.27, 3.06] | -2.01 [-3.56, -0.47] |

## Conclusion

The five-seed results do not provide general support for all three claims:

- **SIA benefit:** not established. The 16D and 32D SIA-effect intervals
  include zero for every dataset/KL combination.
- **SIA × PIC synergy:** not established as a general effect. Positive
  interactions are statistically distinguishable from zero only for STS-B
  with KL off at 16D and 32D.
- **Full-width parity:** not supported. The full-width gap is consistently
  negative for Tweet and is significantly negative for Tweet under both KL
  settings. STS-B KL-on is also significantly negative. Banking77 intervals
  include zero.

Thus, the evidence is dataset- and KL-dependent. It falsifies the broad
version of the paper's claims, while leaving open narrower claims about
specific datasets or settings. The negative full-width gaps also show that
MIPIC does not maintain parity with MRL uniformly at 768 dimensions.
