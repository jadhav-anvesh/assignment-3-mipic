# H2: Lower layers hold syntax, and PIC leaves it intact

This folder tests hypothesis 2 of the five hypotheses we selected for testing. The work is done in steps, one commit per step. This README records what each step does and why, and is updated as each step is finished.

## The claim being tested

The MIPIC paper (arXiv:2604.24374v1) motivates PIC with three statements it never measures:

- §1, p.2: "lower Transformer layers predominantly encode low level linguistic features such as syntax and basic semantics, while higher layers increasingly capture task specific and discriminative signals (Jawahar et al., 2019; Liu et al., 2019)."
- §1, p.2: "PIC allows the remaining dimensions to continue modeling low level and general linguistic features."
- §3.3, p.6: supervision is restricted to the prefix "while preserving low-level features and representational flexibility in the remaining space."

PIC supervises only the [CLS] prefix at fixed (layer, width) checkpoints: `hidden_states[l][:, 0, :d]` for (1,16), (2,32), (3,64), (4,256), (5,512), (6,768) ([losses/pic_loss.py](../losses/pic_loss.py)). Because the encoder weights are shared, this also changes every token's representation, and the paper doesn't check what it does to them.

**Hypothesis.**
- **(a)** In TinyBERT-6L, POS information peaks at a lower layer than sentiment information.
- **(b)** MIPIC's early layers keep their POS information.
- **(c)** That information sits in the coordinates outside the PIC prefix.

## Test design

**Models.** Base TinyBERT-6L (`huawei-noah/TinyBERT_General_6L_768D`, no training), MRL, MRL+PIC and MIPIC. MRL+SIA is probed too when its checkpoint is available. The trained models are the `best.pt` checkpoints from `run_experiments.py` for training seeds 42, 1 and 2, the same models probed for sentiment in [h11_probe/](../h11_probe/). Using the same models links the two results: h11 found that PIC adds sentiment signal at (1,16) and (2,32), and this test asks whether that gain coexists with intact POS information.

**Tasks.**

| Task | Data | Vector probed | Metric |
|---|---|---|---|
| POS (syntax) | UD English EWT r2.13, UPOS (17 tags) | First-subword vector of each word, `hidden_states[l][:, i, :]` | Accuracy |
| Sentiment (task signal) | `data/tweet_train.csv`, `data/tweet_test.csv` (3 classes) | [CLS] vector, `hidden_states[l][:, 0, :]` | Macro-F1 |

Layer l = 0 is the embedding output and l = 1…6 are the transformer layers. The sentiment files are the Kaggle Tweet Sentiment Extraction data used throughout this repository, not the official TweetEval benchmark (see the main [README](../README.md)); see decision 1 below for why.

**Splits.**

| Task | Fit on | Choose C on | Report on |
|---|---|---|---|
| POS | UD EWT train (204,577 words) | UD EWT dev (25,149 words) | UD EWT test (25,094 words) |
| Sentiment | 90% of `tweet_train.csv` (stratified) | the other 10% (about 2,670 rows) | `tweet_test.csv` (3,432 rows), unchanged |

**Probes.** Logistic regression on frozen features, standardized with the statistics of the fitting split. The regularization strength C is chosen on the validation split from a fixed grid, and the chosen model is scored on the test split. The standardization and the tuned C fix a bias found in h11 (Finding 1, checks 5 and 6): `evaluate.py`'s fixed, unscaled probe (C = 1) favours models whose features are larger in scale, and the trained models' features are about twice base's.

**Probe seeds.** Logistic regression is convex and the solver is deterministic, so refitting the same POS probe gives the same answer; POS gets one fit per cell. For sentiment, the probe seed sets the 90/10 validation split: 3 probe seeds (0, 1, 2) per cell, with the score averaged over them.

**Cells probed.**
1. **Layer profiles.** Layers 0–6 at full width (768D), for POS and for sentiment.
2. **Prefix vs suffix at the PIC checkpoints** (1,16), (2,32) and (3,64): the prefix [0:d] and the suffix [d:768], for POS (token vectors) and for sentiment ([CLS]).

**Decision rules.**
- **(a) supported** if, in base TinyBERT, the POS peak layer is earlier than the sentiment peak layer.
- **(b) supported** if MIPIC's POS accuracy at layers 1–3 is within seed noise of MRL's (defined below).
- **(c) supported** if, in MIPIC, POS(suffix) ≥ POS(prefix), while MIPIC's sentiment(prefix) is higher than MRL's at the same layer and width.
- **Falsified** if MIPIC's POS accuracy at layers 1–3 drops below MRL's by more than seed noise (PIC damages low-level features), or if MRL+PIC and MRL give the same profiles (PIC has no measurable effect on early layers).

**Seed noise: the operational rule.** Applied separately at each layer l = 1, 2, 3, to POS accuracy at full width (768D):

1. For each training seed s, d_s = MIPIC_s − MRL_s (accuracy points), with its 95% paired bootstrap CI on the test words.
2. σ = sample standard deviation of MRL's accuracy across the 3 seeds.
3. Threshold T = max(2σ, 0.5 points).

| Verdict | Condition | Meaning |
|---|---|---|
| Damage | mean d < −T **and** the CI lies below zero in all 3 seeds | Falsifies (b) at that layer |
| Within noise | \|mean d\| ≤ T | Supports (b) at that layer |
| Inconclusive | anything else, e.g. a large drop in one seed only, or a drop that is significant in every seed but smaller than T | Reported as inconclusive, counted as neither support nor falsification |

(b) is supported if every layer 1–3 is "within noise", and falsified if any layer shows "damage".

**PIC has a measurable effect** (second falsification check) if, at any layer 1–3, for POS accuracy or sentiment macro-F1, MRL+PIC differs from MRL by more than T (computed the same way, with σ from MRL) with the CI excluding zero in the same direction in all 3 seeds. If no early cell meets this, MRL+PIC and MRL count as giving the same profiles.

Why this form:
- **Two kinds of noise.** The bootstrap covers test-set noise, which is tiny with 25k test words (about ±0.2 points) and would flag trivial gaps. σ covers training-seed noise, which is what "seed noise" means. The rule needs both: a gap must be larger than normal seed variation and consistent across seeds.
- **2σ** is the conventional "outside normal variation" threshold. With 3 seeds σ is itself uncertain, so 1σ would label ordinary run-to-run variation as damage.
- **The 0.5-point floor** (about 125 of 25,094 test words) is the smallest POS loss treated as meaningful. It stops a very small σ from turning a trivial gap into damage.
- **σ from MRL alone**, because MRL is the reference. Including MIPIC's own variation would let an unstable MIPIC raise the threshold and hide its own damage.
- **The "inconclusive" outcome** stops a weak test from being reported as support: (b) claims there is *no* difference, and with 3 seeds a plain t-test would almost never reject it.

**Statistics.** Mean ± std over training seeds, paired differences between models within each seed, and 95% paired bootstrap confidence intervals on the test set (1,000 resamples).

## Decisions

Recorded before any trained-model results, so they can't be fitted to the outcome.

1. **Sentiment data: the repository's tweet files, with a validation split carved from train**. The official TweetEval benchmark (`cardiffnlp/tweet_eval`) matches the paper's wording and has a real validation split, but it was not chosen:
   - **Link to h11.** Rule (c) compares MIPIC's and MRL's sentiment prefixes. h11 found that PIC raises sentiment at (1,16) and (2,32) on exactly this data and these checkpoints (Finding 6). Keeping the data fixed means H2 can say whether that gain coexists with intact POS information, instead of comparing results from two datasets.
   - **Known overlap.** h11 already checked it: no test row is in the training corpus, and 3,000 training-corpus rows are in `tweet_train.csv`, which can only help the trained models. TweetEval would need its own overlap check, because all 3,000 rows of the corpus source labelled `TWEET_EVAL_SENT` are in the Kaggle `tweet_train.csv` (h11 `overlap_check.txt`), so the label doesn't say where they came from.
   - **Cost.** Sentiment numbers are not comparable with the paper's. This is true of the whole repository already.
   - Rules (a) and (b) are expected to come out the same on either dataset; only (c) depends on this choice.
2. **Seed noise: T = max(2σ, 0.5 points), σ from MRL, consistent CIs in all 3 seeds**. The full rule and its reasons are under "Seed noise: the operational rule" above.
3. **Trained models: the existing `best.pt` checkpoints for seeds 42, 1 and 2**, the same ones h11 probed, rather than new training runs. They come from the unmodified `run_experiments.py`, so they are ordinary trained models; reusing them saves about 4.5 GPU hours and keeps the link to h11's results.
   - **Location:** `MRL_LEXIS_H11_checkpoints/h11_probe/runs/seed{42,1,2}/checkpoints/{mrl,mrl_pic,mipic,mrl_sia}/best.pt` (not tracked in git). `best.pt` is the epoch chosen during training; `model.pt` beside it is the last epoch and is not used.
   - **Verified**: all 12 load strictly and differ from base. MIPIC seed 42, re-extracted here, reproduces h11's saved test predictions at (layer 1, 16D) and the pooler at 768D, with 18 and 15 of 3,432 predictions different (54.50 vs 54.65 and 67.72 vs 67.77 macro-F1). The small differences come from CPU vs GPU arithmetic and scikit-learn 1.5.1 vs 1.9.0; the other seeds' MIPIC and seed 42's MRL and MRL+PIC score 10 to 20 points away at (1,16).

**Expected sanity check.** At layer 0, [CLS] is the same vector for every sentence (the [CLS] token embedding plus position 0, then LayerNorm), so sentiment at layer 0 should be at chance. POS at layer 0 is a non-contextual word-embedding probe and should already be high.

## Steps

| Step | What it adds | Status |
|---|---|---|
| 1 | UD EWT loader: download, verify, parse; tests | Done |
| 2 | Feature extraction, layers 0–6: token first-subword vectors (UD EWT) and [CLS] vectors (tweets) | Done |
| 3 | Probes: layer profiles and prefix/suffix cells, for POS and sentiment | To do |
| 4 | Launcher scripts: extract → probe → delete features, per model and seed; full run on base | To do |
| 5 | Paired bootstrap, cross-seed aggregation, decision rules (a), (b), (c) and the falsification checks | To do |
| 6 | Plots of the layer profiles | To do |
| 7 | Run on the trained checkpoints; write up the findings | To do |

### Step 1. UD English EWT loader

**Files.** [ud_data.py](ud_data.py), [test_ud_data.py](test_ud_data.py).

**What it does.**
- Downloads the three CoNLL-U files of UD English EWT, release **r2.13**, from the official GitHub repository into `H2/data/ud_ewt/`, which is not tracked in git.
- Checks each file against a pinned SHA-256 hash, so every model and seed is probed on byte-identical data. A download is written to a temporary file and only renamed into place once its hash matches.
- Parses each sentence into words with their UPOS and XPOS tags. It keeps one entry per syntactic word and skips the two kinds of CoNLL-U line that are not words:
  - multiword-token ranges such as `1-2 don't`, whose parts (`do`, `n't`) are listed as separate words;
  - empty nodes such as `8.1`, which belong to enhanced dependencies and have no surface form.
- Checks the sentence and word counts of the release and that every UPOS tag is one of the 17 UD tags. UPOS labels use a fixed tag order, so label ids are identical everywhere. XPOS (Penn Treebank tags) is parsed as well, so hypothesis 1 can reuse the loader.

**Data.**

| Split | Sentences | Words | UPOS tags | XPOS tags |
|---|---:|---:|---:|---:|
| train | 12,544 | 204,577 | 17 | 49 |
| dev | 2,001 | 25,149 | 17 | 49 |
| test | 2,077 | 25,094 | 17 | 48 |

The most frequent tag, NOUN, covers 16.5% of test words; that is the majority-class baseline for POS accuracy.

**Run** (from the repository root):

```bash
python H2/ud_data.py                    # download if needed, verify, print the table above
python -m pytest H2/test_ud_data.py -q  # 6 passed (the real-data test is skipped until the download)
```

### Step 2. Feature extraction

**Files.** [extract_features.py](extract_features.py), [test_extract_features.py](test_extract_features.py).

**What it does.** For one encoder (base TinyBERT, or a trained checkpoint loaded with `h11_probe/load_encoder.py`), it runs every sentence through the frozen model once and saves, for every layer l = 0…6:

| Part | Rows | Vector saved per row | Labels saved |
|---|---:|---|---|
| `ud_train`, `ud_dev`, `ud_test` | one per UD word (204,577 / 25,149 / 25,094) | first subword of the word, `hidden_states[l][:, i, :]` | UPOS id, XPOS id, sentence index |
| `tweet_train`, `tweet_test` | one per tweet (26,732 / 3,432) | [CLS], `hidden_states[l][:, 0, :]` | sentiment label |

Files go to `H2/features/<name>/` as `{part}_layer{l}.npy` (float32, 768 columns), plus the label arrays and a `meta.json` with the settings, the checkpoint's SHA-256, row counts and timings. Rows are in the original data order.

**Choices and why.**
- **First subword.** A word split into several subwords (e.g. "unbelievably" → `un`, `##bel`, …) is represented by its first subword, the usual convention for token-level probes. UD words are passed to the tokenizer pre-split (`is_split_into_words=True`), so word boundaries come from UD, not from the tokenizer.
- **No truncation of UD sentences.** The longest one is 337 tokens, below the model's 512 positions; the script stops rather than truncate if that ever changes. Tweets keep h11's limit of 256 tokens (the longest tweet is 110), so tweet [CLS] vectors are the ones h11 probed.
- **Words with no subword** (e.g. a lone zero-width space) are dropped together with their labels and counted in `meta.json`. UD EWT r2.13 has none.
- **Length-sorted batches.** Sentences are grouped by token length, so little padding is computed: 1.8× faster than h11's data-order batching on the same 3,200 tweets (22 vs 12 tweets/s on CPU). Padding is masked in attention, and the tests check that every vector matches encoding its sentence alone (to 1e-4).
- **Flat memory.** Vectors are copied straight into memory-mapped `.npy` files. This matters on CPU: there, `hidden_states[l][:, 0, :].cpu().numpy()` is a *view* that keeps the whole batch tensor alive, so code that keeps such views (as `h11_probe/extract_cls.py` does) grows by about 50 MB per batch until the machine runs out of memory and torch crashes. It is harmless on a GPU, where `.cpu()` copies.
- **Resumable.** Each part is written under `*.partial.npy` names and renamed when complete, and `meta.json` lists completed parts. A rerun skips them, so an interruption costs at most the part in progress. A rerun with different settings into the same folder is refused.

**Cost per model** (CPU, 16-thread laptop): about 6.1 GB of features; about 20–35 minutes (tweets 22–40/s, UD 29–50 sentences/s, depending on machine load). The features can be deleted once the model is probed (step 4).

**Run** (from the repository root):

```bash
python H2/extract_features.py --name base --limit 50     # smoke test, seconds
python H2/extract_features.py --name base                # full run
python H2/extract_features.py --name mipic_s42 \
    --checkpoint MRL_LEXIS_H11_checkpoints/h11_probe/runs/seed42/checkpoints/mipic/best.pt
python -m pytest H2/ -q                                  # 14 passed, about a minute on CPU
```

## Layout

```text
H2/
├── README.md                 # This file
├── ud_data.py                # Step 1: UD EWT download, verification and parsing
├── test_ud_data.py           # Step 1: tests
├── extract_features.py       # Step 2: layer 0-6 UD word and tweet [CLS] features
├── test_extract_features.py  # Step 2: tests
├── data/                     # Downloaded UD EWT files (not tracked)
├── features/<name>/          # Extracted features (not tracked)
└── results/                  # Result tables (tracked; added in step 3)
```
