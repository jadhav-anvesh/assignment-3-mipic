# Handoff: MIPIC reproduction and the H11 investigation

This note is for teammates picking up the work. It explains what the repository does, what we added, what we found, how to reproduce it and what is worth doing next. Details live in [h11_probe/FINDINGS.md](h11_probe/FINDINGS.md) (the findings log), [h11_probe/CROSS_SEED.md](h11_probe/CROSS_SEED.md) (all cross-seed tables) and [hypotheses.md](hypotheses.md) (the five hypotheses we test).

## a. Summary

MIPIC (Huy et al., 2026) is a Matryoshka representation learning method: it trains a sentence encoder so that the first d coordinates of the embedding are useful on their own, for several nested widths d, and adds two extra losses. SIA aligns the relational structure of small prefixes with the full representation, and PIC chains information between intermediate layers at fixed (layer, width) checkpoints. This repository reimplements MRL, MRL+SIA, MRL+PIC and MIPIC on TinyBERT-6L (768D), trains them on a 20,240-sentence corpus and evaluates them with a logistic-regression probe on a three-class tweet sentiment task at widths 16 to 768. Our part, in `h11_probe/`, tested hypothesis H11 (that PIC makes early-layer, low-width [CLS] prefixes task-useful) with a full layer-by-width probe grid, then followed up on what we found: an untrained-backbone baseline that matches or beats every trained model, the reason the README numbers can't be reproduced, a random pooler that was never trained, and three-seed replication of every result with paired bootstrap CIs.

## b. Quick start

**Setup** (Python 3.13 was used; 3.11 or newer should work):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-lock.txt
```

[requirements-lock.txt](requirements-lock.txt) pins the exact versions we used and records the machine (Windows 11, RTX 3060 Laptop GPU 6 GB, CUDA 12.6 build of PyTorch). The backbone `huawei-noah/TinyBERT_General_6L_768D` and the STS-B test set (`sentence-transformers/stsb`) download from Hugging Face on first use.

**Verify the install** (from the repository root):

```bash
python -m pytest h11_probe/test_load_encoder.py -q
```

Expected: 7 passed (the CUDA test is skipped on a machine without a GPU). It checks that `load_encoder` recovers the exact encoder weights from every checkpoint format. The repository's own suite in `tests/` also runs with `python -m pytest tests -q` (130 passed, 4 skipped on Windows without a GPU; the gloo DDP tests run on Linux and Kaggle).

**Rebuild base features** (about 1 minute on the GPU, 619 MB on disk). The feature folders are not shipped; most `h11_probe` scripts that refit probes need them:

```bash
python h11_probe/extract_cls.py --name base
```

For a trained model, add its checkpoint, for example `python h11_probe/extract_cls.py --name mrl_s42 --checkpoint h11_probe/runs/seed42/checkpoints/mrl/best.pt`. Features land in `h11_probe/features/<name>/` (train and test, layers 1 to 6 [CLS] and the pooler, float32).

## c. What we did, in order

1. **Base probes.** Extracted [CLS] at every layer plus `pooler_output` from untrained TinyBERT and fitted the `evaluate.py` probe on every (layer, width) cell (`extract_cls.py`, `probe_grid.py`). Checked that dynamic padding doesn't change the vectors (`check_padding.py`) and that float32 features reproduce `evaluate.py` exactly.
2. **STS-B.** Zero-shot cosine Spearman for every cell, with and without the 118 test pairs whose sentences appear in the training corpus (`sts_eval.py`).
3. **Subset probe.** Whether the first d coordinates beat other d-coordinate subsets (`subset_probe.py`).
4. **Retraining on one RTX 3060.** Confirmed the current code trains locally with the Kaggle settings (5 epochs, lr 2e-5, batch 16, max length 256) and wrote a launcher that always trains fresh (`train_all.ps1`).
5. **Three seeds.** Trained all four models for seeds 42, 1 and 2 and ran the analysis pipeline on each (`analyze_seed.ps1`), then paired bootstraps between models (`bootstrap.py`, `bootstrap_extra.py`).
6. **KL-on retrains, seed 42 only.** The README run had the SIA attention-KL term on; the code these runs used hardcoded it off (it is now the `MIPIC_ATTENTION_KL` switch, see section h). `train_kl_on.py` retrains MRL+SIA and MIPIC with it on.
7. **The nine Finding 1 checks.** Verified that untrained TinyBERT matching the trained models is not an artifact: pooler randomness, pipeline parity, data overlap, bootstrap, probe size, a scaled and CV-tuned probe, kNN, the epoch-selection probe, and what the paper and report say.
8. **Kaggle versus local.** Compared the README (Kaggle) run with local retrains: logs, settings, code version and the KL term (Finding 5).
9. **Cross-seed aggregation.** `aggregate_seeds.py` turns every per-seed result into mean, sample std, min, max and sign agreement across seeds, and applies the pre-registered decision rules ([CROSS_SEED.md](h11_probe/CROSS_SEED.md)).

## d. Results

Status is across seeds 42, 1 and 2. "Base" means pretrained TinyBERT-6L with no MIPIC training. Full evidence and CIs are in [FINDINGS.md](h11_probe/FINDINGS.md) and [CROSS_SEED.md](h11_probe/CROSS_SEED.md).

| # | Claim (one line) | Status | Seeds | Hypotheses |
|---|---|---|---|---|
| 1 | Base, read through its untrained random pooler, matches or beats every trained model on the sentiment probe, at every width, probe size and probe protocol tested | Holds on every bootstrap test (0 of 228 cells favour a trained model). The literal pre-registered rule says overturned, because of a loose subset criterion; both verdicts are reported | 3 | H17, H11 baseline |
| 2 | In base, layer 5 [CLS] beats layer 6 [CLS] at widths 16 to 128 | Holds (no CIs yet) | base only | H8, H11 |
| 3 | On STS-B, base's best layer is layer 2 at every width; trained models beat base at the final layer | Holds; the trained-model prediction is confirmed in all 3 seeds | base; 3 for the prediction | H8, H11 |
| 4 | The README numbers can't be reproduced with the current code: local runs are 1.8 to 9.0 points higher, in every seed | Holds (strengthened: seed std is 0.17 to 1.72) | 3 | H15 to H18 (reporting) |
| 5 | The README run had attention KL on, but KL explains only about 1 to 2.5 points of the MRL+SIA gap | Holds; the KL part is single-seed, the gap itself replicates | 1 (KL), 3 (gap) | H6, H15 |
| 6 | MIPIC's early PIC checkpoints carry more sentiment signal than MRL or base, at the cost of later layers | Holds (H11 supported) | 3 (KL-on part: 1) | H11, H17, H18 |
| 7 | SIA and PIC are synergistic at the early checkpoints | Overturned: positive only in seed 42, significantly negative in seed 2 | 3 | H16 |
| 8 | An early epoch (1 to 3 of 5) is selected for every model, with or without PIC | Holds, reworded (the seed 42 "epoch 1 for almost every model" is weakened) | 3 | H14 |
| 9 | TinyBERT's pooler was never trained (random Gaussian, zero bias); training changes the embedding through the encoder, not the pooler | Holds; both swap directions agree in all 12 runs | 3 | H1, H3 |

**H11 decision rule verdict: SUPPORTED.** The rule (set before the results): MIPIC beats both MRL and base, and MRL+PIC beats MRL, at (layer 1, 16D) and (layer 2, 32D), with paired bootstrap 95% CIs excluding zero, in all three seeds. All six comparisons pass in seeds 42, 1 and 2. The size varies a lot: MIPIC minus MRL at (1,16) is +20.43, +4.69 and +11.54.

## e. Hypothesis status

| ID | Hypothesis (short) | Status | One line |
|---|---|---|---|
| H1 | Prefixes are coarse-to-fine | Partly tested | Original POS probes found no prefix preference in token vectors; our subset probe finds the pooler prefix 5 to 6 points above random subsets at 16D after training (base +2.3), but in a random basis (Finding 9) |
| H2 | Larger prefixes refine the smaller one | Untested | |
| H3 | SIA concentrates information in the token prefix | Partly tested | Original POS probes: no prefix preference; MRL acts on random mixtures of coordinates (Finding 9) |
| H4 | Top-k linear CKA is an informative signal | Untested | |
| H5 | CLS-token similarity measures token importance | Untested | |
| H6 | Attention KL matters | Partly tested | Seed 42 KL-on retrain: costs 1 to 2.5 points at the pooler, -11.6 for MIPIC at (1,16) (Findings 5 and 6) |
| H7 | Top-k alignment denoises small prefixes | Untested | |
| H8 | Lower layers hold syntax, higher layers task signal | Partly tested | Base only: sentiment peaks at layer 5 for narrow widths, STS at layer 2 (Findings 2 and 3); no POS by layer |
| H9 | PIC leaves low-level information in the other dimensions | Untested | |
| H10 | PIC transfers information from deeper to earlier layers | Untested | The code detaches the deeper target |
| H11 | Early-layer, low-width prefixes become task-useful | Tested: supported | 3 seeds, pre-registered rule (Finding 6) |
| H12 | PIC maximises mutual information between checkpoints | Untested | |
| H13 | The progressive schedule is necessary | Untested | |
| H14 | PIC makes small prefixes learn early in training | Partly tested | The "best epoch 1" evidence is not specific to PIC (Finding 8); the per-epoch probe test is still needed |
| H15 | SIA is essential | Tested: not supported | SIA alone does nothing at the early checkpoints in any seed; the seed 42 MRL+SIA lead at 16D to 64D did not replicate |
| H16 | SIA and PIC are synergistic | Tested: not supported | No consistent interaction across seeds (Finding 7) |
| H17 | MIPIC matches baselines at full capacity | Tested: not supported | At 768D MIPIC is significantly below MRL and base in every seed |
| H18 | MIPIC's advantage is largest at 16 to 32D | Partly tested | True on raw [CLS] at the PIC checkpoints, false at the pooler |
| H19 | Better out-of-domain generalisation | Untested | |
| H20 | Works regardless of backbone depth and size | Untested | Only TinyBERT-6L used |

## f. How to reproduce

All commands run from the repository root. Times are on one RTX 3060 Laptop GPU (6 GB) with a 16-thread CPU. Bash equivalents of the two PowerShell launchers are in `h11_probe/train_all.sh` and `h11_probe/analyze_seed.sh` (same logic and flags; tested with dry runs only).

**Training** (about 90 minutes per seed: MRL 11, MRL+PIC 13, MIPIC 37, MRL+SIA 29 minutes):

```powershell
powershell -ExecutionPolicy Bypass -File h11_probe\train_all.ps1 -Seed 42      # also -Seed 1, -Seed 2; add -DryRun to preview; KL is pinned off
python h11_probe/train_kl_on.py --model mrl_sia --seed 42 --output-dir h11_probe/runs_kl/seed42
python h11_probe/train_kl_on.py --model mipic   --seed 42 --output-dir h11_probe/runs_kl/seed42
```

```bash
bash h11_probe/train_all.sh --seed 42          # Linux or Kaggle; --dry-run to preview; KL is pinned off
```

Each KL-on arm takes about as long as its KL-off counterpart (roughly 30 and 37 minutes; not separately timed). `train_kl_on.py --dry-run` checks that the KL term is really active without training.

**Analysis per seed** (about 4.5 minutes per model; writes `h11_probe/results/<model>_s<seed>_*`). Leave out `-DeleteFeatures` if you will run the Finding 1 or Finding 9 scripts next, since they need the features:

```powershell
powershell -ExecutionPolicy Bypass -File h11_probe\analyze_seed.ps1 -Seed 42
powershell -ExecutionPolicy Bypass -File h11_probe\analyze_seed.ps1 -Seed 42 -RunsDir h11_probe\runs_kl
```

```bash
bash h11_probe/analyze_seed.sh --seed 42 [--runs-dir h11_probe/runs_kl] [--dry-run] [--delete-features]
```

Base needs its own pass once: `python h11_probe/extract_cls.py --name base`, `python h11_probe/probe_grid.py --name base`, `python h11_probe/sts_eval.py --name base --drop-overlap` (and without `--drop-overlap`), `python h11_probe/subset_probe.py --name base --n-random 20`.

**Bootstraps** (seconds each; 1,000 paired resamples, seed 0):

```bash
python h11_probe/bootstrap.py --a mipic_s42 --b mrl_s42        # also mipic vs base, mrl_pic vs mrl, mrl_sia vs mrl, each model vs base
python h11_probe/bootstrap_extra.py --suffix _s42              # KL-on pairs plus the SIA x PIC interaction
python h11_probe/bootstrap_extra.py --suffix _s1 --skip-pairwise   # seeds without KL-on runs
python h11_probe/bootstrap_sts.py --suffix _s42 --drop-overlap # STS-B, model minus base (GPU, about 3 minutes per seed)
```

**Finding 1 checks** (need the features of base and the seed's four models; replace `s42` for other seeds):

```bash
M="base mrl_s42 mrl_pic_s42 mipic_s42 mrl_sia_s42"
python h11_probe/pooler_check.py                      # check 1, about 10 seconds
python h11_probe/refit_check.py --name base           # check 2
python h11_probe/overlap_check.py                     # check 3
python h11_probe/probe_size.py --models $M            # check 5, about 30 seconds
python h11_probe/probe_tuned.py --tag s42 --models $M # check 6, about 22 minutes (CPU)
python h11_probe/probe_tuned_bootstrap.py --tag s42 --models $M   # check 6 small sizes, about 8 minutes
python h11_probe/probe_knn.py --tag s42 --models $M   # check 7, about 3 minutes
python h11_probe/selection_probe.py --out h11_probe/results/selection_probe_s42.csv \
  --checkpoint mrl_s42=h11_probe/runs/seed42/checkpoints/mrl/best.pt   # check 8 (GPU, about 3 minutes); repeat --checkpoint per model
```

Check 4 is the model-vs-base bootstrap above; check 9 is a reading of the paper and repo files.

**Finding 9** (CPU, about a minute each):

```bash
python h11_probe/weight_drift.py --runs-dir h11_probe/runs/seed42/checkpoints --tag s42
python h11_probe/pooler_swap.py --runs-dir h11_probe/runs/seed42/checkpoints --tag s42      # needs features/base
python h11_probe/pooler_swap_reverse.py --tag s42                                          # needs the seed's features
```

**Cross-seed summary** (seconds):

```bash
python h11_probe/aggregate_seeds.py      # writes h11_probe/results/cross_seed_*.csv and h11_probe/CROSS_SEED.md
```

## g. File map (`h11_probe/`)

Paths are relative to `h11_probe/`. "features/N" means `features/N/{train,test}_{layer1..6,pooler,labels}.npy`.

| File | What it does | Reads | Writes |
|---|---|---|---|
| `load_encoder.py` | Loads a bare TinyBERT encoder from any checkpoint format `run_experiments.py` writes, strictly | checkpoint | |
| `test_load_encoder.py` | Tests for `load_encoder` | hub backbone | |
| `extract_cls.py` | Extracts [CLS] per layer and `pooler_output`, float32 | checkpoint, `data/tweet_*.csv` | features/N |
| `check_padding.py` | Checks dynamic padding doesn't change the saved vectors | features/N, checkpoint | stdout |
| `probe_grid.py` | Default `evaluate.py` probe on every (layer, width) cell, single-thread BLAS; optional README check | features/N, `expected_readme.json` | `results/N_grid.csv`, `N_preds.npz` |
| `sts_eval.py` | STS-B Spearman for every cell, optional overlap removal | checkpoint, hub STS-B, `data/final_data.csv` | `results/N_sts.csv`, `N_sts_clean.csv`, `sts_overlap.txt` |
| `subset_probe.py` | Prefix versus other and random coordinate subsets | features/N, `results/N_grid.csv` | `results/N_subsets*.csv` |
| `train_all.ps1`, `train_all.sh` | Trains the four arms for one seed, always fresh, attention KL pinned off; `-AttentionKL` / `--attention-kl` trains only the SIA arms with KL on | | `runs/seed<N>/` (KL on: `runs_kl/seed<N>/`) |
| `analyze_seed.ps1`, `analyze_seed.sh` | Steps 1 to 4 above for every finished model of one seed | `runs/seed<N>/` | features, results, `logs/analysis_*.log` |
| `train_kl_on.py` | Trains MRL+SIA or MIPIC with attention KL on | | `runs_kl/seed<N>/` |
| `bootstrap.py` | Paired bootstrap of macro-F1 difference per cell | `results/*_preds.npz` | `results/bootstrap_A_vs_B.csv` |
| `bootstrap_extra.py` | KL-on pairs and the SIA x PIC interaction | `results/*_preds.npz` | `results/bootstrap_interaction_<suffix>.csv`, KL-on pair CSVs |
| `bootstrap_sts.py` | Paired bootstrap of STS-B Spearman, model minus base | checkpoints, hub STS-B | `results/bootstrap_sts_*.csv`, `sts_cosines_*.npz` |
| `pooler_check.py` | Randomness statistics of base's pooler (check 1) | hub checkpoint, features/base | `results/pooler_check.csv` |
| `refit_check.py` | Refits the pooler row, compares with saved predictions (check 2) | features/N, `results/N_preds.npz` | `results/refit_check_N.csv` |
| `overlap_check.py` | Probe splits versus training corpus overlap (check 3) | `data/*.csv` | `results/overlap_check.txt` |
| `probe_size.py` | Default probe at several probe sizes (check 5) | features | `results/probe_size_<tag>*.csv` |
| `probe_tuned.py` | Scaled, CV-tuned probe at several sizes, full-size bootstrap (check 6) | features | `results/probe_tuned_<tag>*.csv`, `_preds.npz` |
| `probe_tuned_bootstrap.py` | Bootstrap of the small-size tuned cells (check 6) | features, `results/probe_tuned_<tag>.csv` | `results/probe_tuned_<tag>_small_*` |
| `probe_knn.py` | kNN majority vote, no fitted classifier (check 7) | features | `results/probe_knn_<tag>.csv` |
| `selection_probe.py` | The epoch-selection probe score of base and checkpoints (check 8) | checkpoints, `data/tweet_train.csv` | `results/selection_probe_*.csv` |
| `weight_drift.py` | Relative weight change per layer from base (Finding 9) | checkpoints | `results/weight_drift_<tag>.csv` |
| `pooler_swap.py` | Trained pooler on base layer 6 features (Finding 9) | features/base, checkpoints | `results/pooler_swap_<tag>.csv` |
| `pooler_swap_reverse.py` | Base pooler on trained layer 6 features (Finding 9) | features, checkpoints | `results/pooler_swap_reverse_<tag>.csv` |
| `aggregate_seeds.py` | Cross-seed tables and decision rules | everything in `results/`, `runs/*/tweeteval_results.json` | `results/cross_seed_*.csv`, `CROSS_SEED.md` |
| `expected_readme.json` | The README results table, for `probe_grid.py --check-readme` | | |
| `FINDINGS.md`, `CROSS_SEED.md` | Findings log and cross-seed tables | | |
| `runs/seed{42,1,2}/`, `runs_kl/seed42/` | Training output: `tweeteval_results.json/.csv` (tracked in git), `logs/` and `checkpoints/` (not tracked) | | |

## h. Known issues and gotchas

- **Attention KL is now a switch, on by default.** The model code comes from MIPIC_hypotheses: `MIPIC_ATTENTION_KL` (default `1`, as in the paper) turns the SIA attention-KL term (Eq. 6) on or off, and `tests/test_attention_kl_switch.py` covers it. Every run under `h11_probe/runs/` was trained with KL hardcoded off, so `train_all.ps1` and `train_all.sh` set `MIPIC_ATTENTION_KL=0` to stay reproducible. Calling `run_experiments.py` directly now trains with KL on. `train_kl_on.py` still works and forces KL on regardless of the switch. Line references to `models/mrl_sia.py` and `models/mrl_sia_pic.py` in FINDINGS.md point at the older code.

- **The folder is `h11_probe`, not `h11`.** A top-level folder named `h11` shadows the PyPI `h11` HTTP library that `httpx` and therefore `transformers` and `huggingface_hub` import, which breaks model loading.
- **Single-thread BLAS for reproducible probes.** `probe_grid.fit_one` fits under `threadpool_limits(1)` with contiguous arrays. With multithreaded BLAS, lbfgs can stop one iteration earlier or later depending on the thread count and flip a few test predictions. Reuse `fit_one` rather than calling `LogisticRegression` directly if you want bit-identical numbers.
- **PowerShell 5.1 and stderr.** With `$ErrorActionPreference = "Stop"`, PowerShell 5.1 turns any stderr line from a native program into a terminating error, and Python writes warnings to stderr. The launchers therefore run Python through `cmd.exe` with `>> log 2>&1` and read the real exit code. They also set `PYTHONUTF8=1`, because Python writing to a redirected file otherwise uses cp1252 and can crash on non-ASCII output.
- **Never resume a training run.** `run_experiments.py` saves a resumable `training_state.pt`, but the best-epoch tracker isn't in it, so a resumed run can keep the wrong `best.pt`. The launchers always train fresh (`MIPIC_FRESH=1`) and delete a stale `best.pt` first. Also, `run_experiments.py` exits 0 even when evaluation fails, so "finished" means `best.pt` exists and the model has an entry in `tweeteval_results.json`.
- **Stale Kaggle notebook outputs.** In `mipic-a1-lexis.ipynb`, cell 14's source doesn't match its saved output (mode, output folder and fresh flag differ), and cell 3 was added after the recorded run. Trust the logged outputs, not the cell sources, when reconstructing the Kaggle run (Finding 5).
- **`tests/test_kaggle_readiness.py` and the reference notebook.** These tests used to fail because `MIPIC_Kaggle_FINAL.ipynb` and `kaggle/README.md` were missing; both now come from MIPIC_hypotheses. `tests/test_mrl_baseline_integration.py` also needs the reference notebook `mipic.ipynb`.
- **The pooler is untrained.** `pooler_output` in `huawei-noah/TinyBERT_General_6L_768D` is a random Gaussian 768x768 layer with zero bias, and training barely changes its effect. So "the first d dimensions" of the embedding are d random projections of the layer 6 [CLS] (Finding 9).
- **Features are float32.** float16 changed probe F1 by up to 0.19 points, so all features are stored and probed in float32.
- **The README numbers come from an older code version** (MIPIC_FINAL_V20, attention KL on). They can't be reproduced with the current code; local runs are higher in every seed (Findings 4 and 5).
- **Unscaled probes and feature scale.** Trained models' pooler features are about twice base's scale. With the default unscaled probe and few training rows this alone favours the trained models; use the scaled, tuned probe for small probe sizes (Finding 1, checks 5 and 6). The epoch-selection probe in `run_experiments.py` has the same bias.
- **Logs contain absolute paths from the original machine.** The files under `h11_probe/runs*/seed*/logs/` record command lines and warnings with paths like `C:\ANVESH\...` and `C:\Users\ASUS\...`. They are kept as the original record; ignore those paths.
- **Unconverged probe fits.** Some raw [CLS] fits at layers 1 to 4 and 512D or 768D hit `max_iter=1000`, as does the (5,512) PIC checkpoint for MRL+SIA in every seed. They are flagged in the logs and in FINDINGS.md.

## i. What can be done next

Ordered by value for cost, highest first.

1. **Add the untrained baseline to the report.** Minutes. The report's tables compare trained models only; base matches or beats all of them on the sentiment probe (Finding 1).
2. **KL-on runs for seeds 1 and 2.** About 70 minutes of GPU per seed with `train_kl_on.py`, then `analyze_seed.ps1 -RunsDir h11_probe\runs_kl`. This removes the last single-seed statements (Findings 5, 6 and 8).
3. **Fix the resume bug.** Small code change: save the best-epoch tracker in `training_state.pt` and restore it on resume.
4. **Per-epoch probes for H14.** Save a checkpoint every epoch, then probe (1,16) and (2,32) per epoch for MRL and MIPIC. One extra training pass per model.
5. **More seeds.** 90 minutes per seed. The (1,16) effect sizes vary from 4.7 to 20.4, so more seeds would tighten every estimate.
6. **Evaluate on official TweetEval.** The current evaluation files are Kaggle Tweet Sentiment Extraction data, not the benchmark the paper uses.
7. **Get the MIPIC_FINAL_V20 code** to explain the remaining MRL+SIA gap in Finding 5.
8. **Layer-wise POS probes for H8 and H9.** The POS probes were removed from this repository; start from `pos_probe/` in the original course repository, [MIPIC_LEXIS_A1](https://github.com/ManojAnaparthi/MIPIC_LEXIS_A1) and extend it to read every layer and the non-prefix dimensions.
9. **Block and granularity probes for H1 and H2.** Probe coordinate blocks of the pooled output on tasks of different granularity (3-class sentiment, 6-class emotion, 77-class Banking77).
10. **H4 CKA signal test.** Compare CKA of the trained prefix against random and untrained baselines.
11. **H5 token importance.** Check whether CLS-token similarity tracks token importance (for example via deletion tests).
12. **H10 detach direction.** Train PIC without detaching the deeper target and compare.
13. **H13 schedule ablations.** Alternative (layer, width) schedules; the most expensive item, since each schedule needs full training runs.
