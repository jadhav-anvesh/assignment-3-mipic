# Running MIPIC on Kaggle

The notebook `MIPIC_Kaggle_FINAL.ipynb` runs these steps in order. Run every command from the repository root.

## 1. Setup

Attach the repository and the dataset containing `final_data.csv`, enable a GPU, then install the extra dependencies (Kaggle's CUDA build of PyTorch is kept):

```bash
pip install -q -r requirements-kaggle.txt
python -m pytest -q
```

## 2. Smoke run

Checks the training plumbing on 3 batches. With `MIPIC_SMOKE_FULL_PIPELINE=1` it also saves a checkpoint and runs a reduced evaluation. Results are labelled as truncated and are not reportable.

```bash
MIPIC_MAX_BATCHES=3 MIPIC_SMOKE_FULL_PIPELINE=1 MIPIC_EPOCHS=1 \
MIPIC_EVAL_TRAIN_SAMPLES=300 MIPIC_EVAL_TEST_SAMPLES=300 MIPIC_FRESH=1 \
MIPIC_OUTPUT_DIR=/kaggle/working/mipic_pipeline_check \
python run_experiments.py --model mipic
```

## 3. Full run

The default configuration is a 5-epoch run of each model. `--skip-existing` skips models that are already trained and evaluated, so an interrupted session can be resumed with the same command.

```bash
MIPIC_OUTPUT_DIR=/kaggle/working/mipic_results_all \
python run_experiments.py --model all --skip-existing
```

For two GPUs, replace `python` with `torchrun --standalone --nproc_per_node=2`.

## 4. Evaluate an existing checkpoint

```bash
MIPIC_CHECKPOINT=/kaggle/working/mipic_results/checkpoints/mipic/model.pt \
python run_experiments.py --model mipic --eval-only
```

Results are written to `tweeteval_results.json` and `tweeteval_results.csv` in `MIPIC_OUTPUT_DIR`. Checkpoints go to `checkpoints/<model>/model.pt`.

## Settings for the hypothesis experiments

- `MIPIC_SEED` (default `42`) sets the training seed. Use a separate `MIPIC_OUTPUT_DIR` per seed.
- `MIPIC_ATTENTION_KL` (default `1`) turns the Eq. (6) attention-KL term of SIA on or off for `mrl_sia` and `mipic`. With `0`, SIA keeps only the top-k CKA term. The value is recorded as `attention_kl` in the results metadata.

With `MIPIC_ATTENTION_KL=0` the SIA projections P_i receive no gradient. Under DDP this requires `MIPIC_DDP_STATIC_GRAPH=1`, which the notebook's `ddp` mode already sets. Single-GPU runs are unaffected.

```bash
MIPIC_SEED=13 MIPIC_ATTENTION_KL=0 \
MIPIC_OUTPUT_DIR=/kaggle/working/mipic_seed13_no_kl \
python run_experiments.py --model all --skip-existing
```
