# MIPIC: Five Hypotheses to Test

Selected from [hypothesis.md](hypothesis.md) (IDs in brackets). Quotes are from [MIPIC.pdf](MIPIC.pdf) (arXiv:2604.24374v1); page numbers refer to the PDF.

Each hypothesis was chosen because it is central to the paper's argument, has a clear outcome that would falsify it, and can be run with this repository's code and checkpoints.

## Shared setup

- **Models.** Base TinyBERT-6L (`huawei-noah/TinyBERT_General_6L_768D`), MRL, MRL+SIA, MRL+PIC and MIPIC, all trained with the same corpus, settings and seeds.
- **Seeds.** At least 3 training seeds per model (5 for H4), and 3 probe seeds per configuration.
- **Probes.** Linear classifiers on frozen features, standardized with training-split statistics. Select on validation, report on test. Sentence tasks use the logistic-regression protocol in `evaluate.py`. Token tasks use UD English EWT with first-subword vectors, as in `pos_probe/`.
- **Statistics.** Report mean ± std over seeds and a 95% bootstrap confidence interval on the test set. Compare models with paired differences over the same seeds.
- **Layer indexing.** `hidden_states[l]` with output from layer l = 1…6; `hidden_states[0]` is the embedding layer.

---

## 1. Early dimensions hold the coarse information  [H1, H3]

**Paper says.**
- §2.1, p.3: MRL works by "learning nested coarse-to-fine prefixes within a single embedding".
- §3.2.2, p.5: "low-dimensional cores capture a coarse semantic skeleton, while higher-dimensional spaces progressively incorporate finer details."
- §1, p.2: SIA forces "critical semantic structures to be prioritized and concentrated within the low-dimensional prefix dimensions."

**Hypothesis.** In MIPIC, the first d coordinates carry more task information than any other d coordinates, and coarse labels saturate at a smaller d than fine labels. Both effects should be stronger than in base TinyBERT.

**Already known.** For POS tags on final-layer token vectors, the prefix was not privileged:
- a learned selector beat the first-k prefix by 2–5 macro-F1 at k = 64–256;
- dense L2 gates showed no preference by coordinate index.

There was no baseline model, so this can't yet be attributed to MIPIC.

**Test.**
1. **Representations.**
   - The pooled sentence embedding, which is where the MRL loss acts.
   - Token vectors at layers 1, 3 and 6, which are where SIA acts.
2. **Coordinate sets.** For d ∈ {16, 32, 64, 128}, probe four kinds of coordinate set:
   - the prefix [0:d];
   - the next block [d:2d];
   - the last block [768−d:768];
   - 20 random d-subsets, which give a null distribution.
3. **Coarse/fine task pairs.**
   - Token level: UPOS (17 tags) vs XPOS (about 45 Penn Treebank tags), both on UD EWT.
   - Sentence level: TweetEval sentiment (3 classes) vs Emotion (6 classes).
4. **Metrics.**
   - Prefix advantage = prefix score − mean random-subset score, plus the prefix's percentile in the random-subset distribution.
   - d₉₀ = the smallest d whose prefix reaches 90% of full-width performance.
5. **Supported if:**
   - MIPIC's prefix is above the 95th percentile of random subsets;
   - its prefix advantage is larger than base TinyBERT's;
   - d₉₀(UPOS) < d₉₀(XPOS) and d₉₀(sentiment) < d₉₀(emotion).

   **Falsified** if the MIPIC prefix sits inside the random-subset distribution, as it does for base TinyBERT.

---

## 2. Lower layers hold syntax, and PIC leaves it intact  [H8, H9]

**Paper says.**
- §1, p.2: "lower Transformer layers predominantly encode low level linguistic features such as syntax and basic semantics, while higher layers increasingly capture task specific and discriminative signals (Jawahar et al., 2019; Liu et al., 2019)."
- §1, p.2: "PIC allows the remaining dimensions to continue modeling low level and general linguistic features."
- §3.3, p.6: supervision is restricted to the prefix "while preserving low-level features and representational flexibility in the remaining space."

**Hypothesis.**
- (a) In TinyBERT-6L, POS information peaks at a lower layer than sentiment information.
- (b) MIPIC's early layers keep their POS information.
- (c) That information sits in the coordinates outside the PIC prefix.

**Test.**
1. **Layer profiles.** For layers 0–6, probe UPOS on full 768-dimensional token vectors and TweetEval sentiment on the full [CLS] vector. Compare base TinyBERT, MRL, MRL+PIC and MIPIC.
2. **Prefix vs suffix at the PIC checkpoints.** The TinyBERT checkpoints are (16, 1), (32, 2) and (64, 3). At each one, probe UPOS on the token prefix [0:d] and on the suffix [d:768]. Probe sentiment on the [CLS] prefix and suffix.
3. **Supported if:**
   - (a) the POS peak layer is earlier than the sentiment peak layer in base TinyBERT;
   - (b) MIPIC's POS score at layers 1–3 is within seed noise of MRL's;
   - (c) in MIPIC, POS(suffix) ≥ POS(prefix) while sentiment(prefix) is higher than MRL's at the same layer and d.

   **Falsified** if MIPIC's POS score at layers 1–3 drops below MRL's by more than seed noise (PIC damages low-level features), or if MRL+PIC and MRL give the same profiles (PIC has no measurable effect on early layers).

---

## 3. Early, small checkpoints become usable embeddings  [H11]

**Paper says.**
- §1, p.2: PIC "performs a controlled early intervention that gently biases the formation of low level representations toward task relevant subspaces, ensuring that useful discriminative structure is available from the beginning of the network processing."
- §3.3, p.6: PIC allows "early, low-capacity representations to serve as reliable building blocks for deeper ones."

The paper evaluates only the final layer, so this central depth claim is never measured.

**Hypothesis.** The truncated [CLS] vector at each PIC checkpoint is a better embedding under PIC-trained models than under models trained without PIC. This is the vector the code supervises: `hidden_states[l][:, 0, :d]` (see [losses/pic_loss.py](losses/pic_loss.py)).

**Test.**
1. **Checkpoints.** For each (d, l) in {(16,1), (32,2), (64,3), (256,4), (512,5), (768,6)}, extract z = `hidden_states[l][:, 0, :d]`.
2. **Evaluation.**
   - TweetEval: logistic-regression macro-F1.
   - STS-B: Spearman correlation of cosine similarity, with no training.
3. **Models.** All five. The contrast that matters is MRL+PIC and MIPIC against MRL and MRL+SIA.
4. **Controls.**
   - The same (l, d) with 20 random d-coordinate subsets.
   - The final-layer prefix of the same d, which shows how far below the normal output an early exit sits.
5. **Supported if** PIC-trained models beat MRL at layers 1–3 by more than seed noise on both tasks.

   **Falsified** if the early checkpoints score the same with and without PIC.

---

## 4. SIA is essential, SIA and PIC are synergistic, and full-width performance is unaffected  [H15, H16, H17]

**Paper says.**
- §5, p.7–8: "SIA is vital for structural organization; its removal causes sharp drops at small sizes, proving geometric order is key for information density."
- §5, p.7: "Table 3 highlights the synergy between SIA and PIC."
- §4.2, p.7: "while MIPIC maintains parity with baselines in full-capacity regimes, it significantly excels in retaining semantic density under extreme truncation."

All of this rests on single runs. The ablation (Table 3) covers two datasets and does not name its backbone; its numbers match the BERT columns of Tables 1–2.

**Already known.** This contradicts our single TinyBERT run:
- MRL+SIA had the lowest F1 of the four at every width (53.26 vs MRL's 56.46 at 16D);
- MRL beat MIPIC from 64D upward (67.24 vs 64.64 at 768D).

**Hypothesis.** Over seeds:
- (a) MIPIC > MRL+PIC at 16D and 32D, i.e. removing SIA hurts;
- (b) the SIA × PIC interaction is positive;
- (c) |MIPIC − MRL| at 768D is within seed noise.

**Test.**
1. **Runs.** Train MRL, MRL+SIA, MRL+PIC and MIPIC with 5 seeds each, at the paper's settings (Table 7: 5 epochs, learning rate 2e-5, batch size 16, α = 0.4). Run MRL+SIA and MIPIC both with and without the attention-KL term, since the current code disables it.
2. **Evaluation.** All 7 widths, on TweetEval, Banking77 and STS-B.
3. **Quantities, per width.**
   - SIA effect = MIPIC − (MRL+PIC).
   - Interaction = (MIPIC − MRL) − [(MRL+SIA − MRL) + (MRL+PIC − MRL)].
   - Full-width gap = MIPIC − MRL at 768D.
4. **Supported if** the 95% confidence interval of the SIA effect is above zero at 16D and 32D, the interaction is above zero, and the full-width gap's interval contains zero.

   **Falsified** if any of those intervals excludes the predicted direction. In particular, if MRL+SIA < MRL again, SIA alone hurts.

---

## 5. The top-k CKA loss is an informative training signal  [H4]

**Paper says.**
- §3.2.2, p.6: "By maximizing CKA on these nested subsets, we ensure that the geometric relationships and structural organization within the semantic core are preserved robustly across all granularities."
- A.5, p.13: the subsets are the top kᵢ = max(8, ⌈γᵢ·m⌉) tokens of one sentence, with γ = 0.2 at d = 16.

**Hypothesis.** Linear CKA between the student prefix (kᵢ × dᵢ) and the teacher (kᵢ × 768) separates aligned from unaligned representations, so lowering 1 − CKA reflects real alignment.

**Why it may fail.** Each CKA uses only 8–20 tokens as samples but 16–768 features. With so few samples relative to features, linear CKA is biased upward even for unrelated representations. The loss may therefore be close to its minimum from the start and give little gradient. This could also explain why SIA did not help in our run (hypothesis 4).

**Test.**
1. **Measurement, no training needed.** On held-out sentences, using each layer's actual top-kᵢ tokens, compute CKA(student prefix, teacher) for:
   - (a) the trained MIPIC prefix;
   - (b) the base TinyBERT prefix;
   - (c) a random Gaussian matrix of the same shape;
   - (d) the prefix of a *different* sentence's tokens with the same k, which is an unrelated pairing.
2. **Debiased estimate.** Repeat the measurement with debiased (unbiased-HSIC) CKA.
3. **Training trace.** Log L_CKA per (layer, d) for every step of one MIPIC run.
4. **Intervention.** Retrain MIPIC (3 seeds) with (i) debiased CKA, and (ii) CKA computed over the selected tokens pooled across the whole batch, so there are many more samples. Compare 16D and 32D scores with standard MIPIC.
5. **Supported if** trained CKA is clearly higher than (c) and (d), the gap survives debiasing, and L_CKA falls during training.

   **Falsified** if (c) and (d) already reach CKA close to the trained value. In that case the loss carries little information, and a change in step 4 that improves scores would locate the problem.
