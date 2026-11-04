# MIPIC: Assumptions and Hypotheses to Test

Source: Huy et al., *MIPIC: Matryoshka Representation Learning via Self-Distilled Intra-Relational and Progressive Information Chaining*, arXiv:2604.24374v1 ([MIPIC.pdf](MIPIC.pdf)). Page numbers refer to that PDF.

This document lists the assumptions MIPIC relies on, whether stated outright, implied by the method, or taken from cited work, and turns each into a hypothesis that can be tested. Each entry gives:

- **Paper:** the claim, quoted, with its location.
- **Hypothesis:** a falsifiable version of the claim.
- **Why test it:** what makes the claim uncertain.
- **Evidence so far:** results already in this repository, if any.
- **Test:** a concrete experiment, using this repository's code where possible.

The motivating case is H1. The POS probes in [pos_probe/](pos_probe/POS_EXPERIMENTS_README.md) found that the leading coordinates of MIPIC's token vectors are not privileged for POS tagging.

## Summary

| ID | Assumption | Kind | Evidence so far | Priority |
|---|---|---|---|---|
| H1 | Prefixes are coarse-to-fine; early coordinates hold coarse information | Stated | POS probes: against, for token vectors | High |
| H2 | Larger prefixes refine the smaller prefix's content rather than relearn it | Stated | Paper's own tables are non-monotone in places | High |
| H3 | SIA concentrates critical information in the prefix of token hidden states | Stated | POS probes: no prefix preference | High |
| H4 | Top-k linear CKA is an informative alignment signal | Implied | None | High |
| H5 | CLS–token similarity measures token importance | Stated | None | Medium |
| H6 | Attention KL enforces a token-ranking consistency that matters | Stated | Current code disables KL | Medium |
| H7 | Aligning all tokens saturates small prefixes; top-k denoises | Stated | None | Medium |
| H8 | Lower layers hold syntax, higher layers hold task signal (in TinyBERT and after MIPIC) | Cited | None | High |
| H9 | PIC leaves low-level linguistic information intact in the non-prefix dimensions | Stated | None | High |
| H10 | PIC transfers information from deeper layers to earlier ones | Stated | Code detaches the deeper target | Medium |
| H11 | Early-layer, low-dimensional CLS prefixes become task-useful | Stated | None | High |
| H12 | PIC maximizes mutual information between checkpoints | Stated | None | Low |
| H13 | The progressive (layer, dimension) schedule is necessary | Stated | Paper: ≤1.75 pt, one run | Medium |
| H14 | PIC makes small prefixes learn task information early in training | Stated | MIPIC's best epoch was 1/5 (not specific to PIC; see correction) | Medium |
| H15 | SIA is essential; removing it causes sharp drops at small sizes | Stated | Our run: MRL+SIA is worst at every width (KL on; see correction) | High |
| H16 | SIA and PIC are synergistic | Stated | Our run: gain only at 16–32D | Medium |
| H17 | MIPIC matches baselines at full capacity | Stated | Our run: MRL is 2.6 F1 higher at 768D | High |
| H18 | MIPIC's advantage is largest at 16–32 dimensions | Stated | Our run: consistent | Medium |
| H19 | MIPIC generalizes better out of domain | Stated | Exceptions in the paper's own tables | Low |
| H20 | The method works regardless of backbone depth and size | Stated | Only hand-tuned schedules reported | Low |

Reporting and reproduction issues found while reading are listed at the end.

---

## A. How information is ordered across dimensions

### H1. Coarse-to-fine prefixes

**Paper.**
- §2.1, p.3: MRL learns "nested coarse-to-fine prefixes within a single embedding".
- §3.2.2, p.5: "low-dimensional cores capture a coarse semantic skeleton, while higher-dimensional spaces progressively incorporate finer details".
- §5, p.8: SIA and PIC "enable a coarse-to-fine refinement that maintains high quality at every dimension".

**Hypothesis.** The first d coordinates carry the coarsest information: the information that matters most and is shared most widely. Later coordinates add progressively finer distinctions. Two consequences follow:

- (a) For a coarse task, a prefix should outperform any other block of d coordinates, and the gap should be largest at small d.
- (b) Coarse tasks should saturate at a smaller d than fine-grained tasks.

**Why test it.** The paper never defines "coarse" or "fine", and never measures them. The supporting evidence is aggregate performance at each width, which shows only that a prefix is *usable*, not *what* it contains. The paper also says something that cuts the other way. Low-level linguistic features, which a POS tag is, are said to live in the dimensions *outside* the PIC-supervised prefix (see H9). The question "do early coordinates hold syntax?" therefore mixes the paper's *layer* claim (lower layers hold syntax; H8) with its *dimension* claim (low dimensions hold a coarse *semantic* skeleton). Taken literally, the paper predicts that POS information is *not* concentrated in the prefix.

**Evidence so far.** Against, for final-layer token vectors. On UD English EWT, with frozen final-layer first-subword vectors:

- A learned k-coordinate selector beat the first-k prefix by 2.1–5.2 macro-F1 for k = 64–256. It lost at k = 16.
- The selected sets were close to random overlap across seeds (Jaccard 0.016–0.218).
- Dense L2 gates showed no index preference (Spearman ρ between index and gate: 0.02–0.05).

Caveats:

- The MRL loss is applied to `pooler_output`, a 768×768 dense layer plus tanh over [CLS]. This mixes every hidden coordinate, so MRL itself does not order the token-level coordinates. Only SIA constrains them (H3).
- No base-TinyBERT or MRL control was run, so the result cannot yet be attributed to MIPIC.

**Test.**
1. **Controls:** rerun the three POS probes on base TinyBERT and on the MRL, MRL+SIA and MRL+PIC checkpoints. If none of them shows a prefix advantage, MIPIC changes nothing here. If MRL+SIA shows one and MIPIC doesn't, PIC removes it.
2. **Block probes on sentence embeddings, where MRL actually acts.** For d ∈ {16, 32, 64}, compare a logistic-regression probe on coordinates [0:d] against probes on [d:2d], on a random block, and on a random d-subset of the pooled output. Use TweetEval (3 classes, coarse), Emotion (6 classes) and Banking77 (77 intents, fine). Prediction (a) needs prefix > other blocks. Prediction (b) needs the 3-class task to reach, say, 95% of its full-width F1 at a smaller d than Banking77 does.
3. **Granularity within one label set.** Merge Banking77's 77 intents into a few coarse groups and probe both label sets at each prefix width. Coarse-to-fine predicts the coarse labels saturate first.

### H2. Larger prefixes refine, rather than relearn

**Paper.** §3.2.2, p.5: nested token sets ensure "higher-capacity embeddings refine and enrich the semantic core learned by lower-capacity ones, rather than relearning it".

**Hypothesis.** The geometry of z(d) is largely preserved inside z(2d). Neighbours and decisions at d are mostly kept at 2d, and the extra coordinates are complementary to the prefix, not redundant with it. Performance should rise monotonically with d.

**Why test it.** The claim is a description of a mechanism, but no geometry is reported. The paper's own tables are not monotone:

- BERT TweetEval: 71.58 at 512D, 71.26 at 768D.
- TinyBERT SciTail: 75.97 → 75.34 → 75.61 from 256D to 768D.
- TinyBERT SICK: 67.38 at 32D, 67.32 at 64D; 70.80 at 512D, 70.65 at 768D.

In our run, MRL scores 67.61 at 512D and 67.24 at 768D.

**Test.**
- For each consecutive pair (d, 2d) of the pooled embedding, measure:
  - CKA between z(d) and z(2d);
  - k-nearest-neighbour overlap (k = 10);
  - prediction agreement between linear probes trained on each.
- Probe the suffix [d:2d] on its own. If the suffix alone nearly matches the prefix, the information is redundant rather than refining.
- Compare MIPIC with MRL. The claim is that MIPIC's nesting is *stronger* than plain MRL's.

### H3. SIA concentrates critical information in the token prefix

**Paper.**
- §1, p.2: SIA "reorganizes the internal feature hierarchy, forcing critical semantic structures to be prioritized and concentrated within the low-dimensional prefix dimensions".
- §3, p.4: SIA "reorganizes the internal representation hierarchy to prioritize core structural features within low-dimensional subspaces".

**Hypothesis.** After SIA, the prefix h_j[1:d] of token hidden states carries a disproportionate share of the information in h_j, at every aligned layer. A prefix should beat random coordinate subsets of the same size on token-level probes, by more than it does in models trained without SIA.

**Why test it.** SIA is the only loss that acts on token-level prefixes, at all six TinyBERT layers. Yet the POS probes found no prefix preference at the final layer. Either SIA doesn't concentrate information in the prefix, or it concentrates only what CKA rewards, which is sentence-internal geometry rather than features like POS. CKA is invariant to orthogonal transformations (§3.2.2, p.6). Matching Gram matrices therefore says nothing about which *features* sit in the prefix, only that their pairwise geometry is preserved.

**Evidence so far.** Final-layer POS probes found no prefix advantage (H1).

**Test.**
- At every layer, compare prefix-vs-random-subset probe gaps for MRL and MRL+SIA, on:
  - token-level tasks: POS, NER, word-level sentiment lexicon;
  - a geometric target: reconstructing the full-dimension token Gram matrix from k coordinates.
- If SIA helps only the geometric target, the "critical semantic structure" wording overstates what CKA enforces.

### H4. Top-k linear CKA is an informative alignment signal

**Paper.**
- §3.2.2, p.5–6: for each prefix d_i, CKA is computed between the student prefix h_i ∈ ℝ^{k_i×d_i} and the teacher H_i ∈ ℝ^{k_i×D}. The rows are the k_i selected tokens of *one* sentence.
- A.5, p.13: k_i = max(8, ⌈γ_i·m⌉) with γ = 0.2 for d = 16.

The implicit assumption is that 1 − CKA on these matrices is a meaningful training signal.

**Hypothesis.** The CKA loss distinguishes good from bad prefix geometry, and its value falls during training because the prefix has become aligned.

**Why test it.** With 8–20 tokens as samples and 16–768 features, the number of samples is far smaller than the feature dimension. In this regime, linear CKA is biased upward even between unrelated representations; debiased CKA estimators exist for this reason. A loss that is already near zero for random features gives little gradient. The CKA-based gains claimed in Table 3 could then come from something else.

**Test.**
1. For the actual (k_i, d_i) pairs, compute CKA between the teacher H_i and (a) the trained prefix, (b) a random Gaussian matrix, (c) the prefix of an untrained model. If (b) and (c) are already close to (a), the signal is weak.
2. Log L_CKA per (layer, dimension) over training.
3. Replace linear CKA with debiased CKA, or compute it over tokens pooled across the batch, and compare the final scores.

---

## B. Token importance and the design of SIA

### H5. CLS–token similarity measures token importance

**Paper.**
- §3.2.1, p.4: "We use h_CLS as the query anchor since it encodes global sequence semantics".
- §3.2.1, p.4: "The similarity between each token and this [CLS] vector serves as an estimate of the token importance".
- §3.2.2, p.5: the top tokens are "the most informative tokens".

**Hypothesis.** Ranking tokens by softmax(h_CLS · h_j / √D / τ) agrees with independent measures of importance for the sentence embedding or the downstream label.

**Why test it.** This is an untested modelling choice. It is not the model's attention: it is a dot product between hidden states, despite being called "attention-driven". In BERT-family models, [CLS] is often most similar to special tokens and punctuation. With τ = 0.05 the distribution is very sharp, so the top-k set may be dominated by one or two tokens.

**Test.** On held-out sentences, compute the rank correlation between this score and:
- the model's real [CLS]-row attention, averaged over heads;
- gradient×input attribution for the SimCSE embedding;
- occlusion: the drop in cosine similarity to the original embedding when a token is removed;
- occlusion for the sentiment probe's logit.

Also report the part-of-speech distribution of the selected tokens.

### H6. Attention KL enforces ranking consistency, and that matters

**Paper.** §3.2.1, p.4–5: minimizing KL(a_i ‖ a_D) enforces "a ranking consistency constraint", so "the most informative tokens remain emphasized even after compression".

**Hypothesis.** (a) The KL term improves agreement between the student's and the teacher's token *rankings*. (b) It improves downstream scores at small d.

**Why test it.** KL on softmax distributions is not a ranking loss. A student can lower the KL while reordering tokens in the tail. In addition, the projection P_i is learned jointly, so it can absorb the mismatch without changing the backbone.

**Evidence so far.** The current code turns the term off (`use_attention_kl=False` in [models/mrl_sia_pic.py](models/mrl_sia_pic.py)), and the local checkpoint folder is named `results_no_kl`. Check the recorded run's log to confirm whether KL was on.

**Correction, 2026-11-04.** The recorded run's log has now been checked: KL was **on** in the run behind the README numbers. The MRL+SIA training summary in notebook cell 29 logs `sia_attn_loss` of 0.0987, 0.0413, 0.0608, 0.0627 and 0.0827 for epochs 1 to 5. A local rerun of the current code logs 0.0000. The README run used an older code version (MIPIC_FINAL_V20); `use_attention_kl=False` was hardcoded later. KL was very likely on for MIPIC too (its epoch 1 `sia_loss` was 0.1634 against 0.0480 with KL off). The sentence above is still true of the current code, but not of the reported results. Details in [h11_probe/FINDINGS.md](h11_probe/FINDINGS.md), Finding 5.

**Test.** Train MIPIC with and without KL, using several seeds. Measure the Kendall τ between the rankings from a_i and a_D, and the top-k overlap. Compare TweetEval and STS scores at 16D and 32D.

### H7. Aligning the whole sequence saturates small prefixes; top-k denoises

**Paper.**
- §3.2.2, p.5: "forcing a small bottleneck to align with the entire sequence leads to information saturation"; top-k "acts as a denoising filter, preventing low-capacity prefixes from being saturated by irrelevant or redundant context".
- A.5, p.13: "a minimum floor of k_min = 8 tokens to preserve basic sentence structure".

**Hypothesis.** With everything else fixed, the γ = [0.2 … 0.7] schedule beats aligning all tokens (γ = 1) at small d. k_min = 8 beats smaller floors on short inputs.

**Why test it.** Neither the saturation claim nor the floor value is ablated. "Basic sentence structure" is not defined.

**Test.** Ablate:
- γ: the schedule, all 1.0, all 0.2, reversed, and random token subsets of the same size;
- k_min ∈ {1, 4, 8, 16}.

Report 16D and 32D scores, split by sentence length.

**Related claim (§3.2.2, p.5).** Nested token sets create a "dual Matryoshka structure". Nesting is guaranteed by construction, since all sets are top-k by the same scores. The testable part is whether nesting *helps*: compare against independent random subsets of the same sizes.

---

## C. Depth, layers and PIC

### H8. Lower layers hold syntax, higher layers hold task signal

**Paper.** §1, p.2: PIC "is motivated by empirical findings that lower Transformer layers predominantly encode low level linguistic features such as syntax and basic semantics, while higher layers increasingly capture task specific and discriminative signals (Jawahar et al., 2019; Liu et al., 2019)".

**Hypothesis.** Assume a layer-wise POS probe on TinyBERT-6L peaks at a lower layer than a layer-wise sentiment probe. Then (a) this already holds in base TinyBERT, and (b) it still holds after MRL and MIPIC training.

**Why test it.** The cited findings are for BERT-base and ELMo-style models. The paper never checks them for a 6-layer distilled model, nor after its own fine-tuning, which is designed to change early layers (H9).

**Test.** Extend the POS probe to read `hidden_states[l]` for l = 1…6 (a small change to [pos_probe/pos_probe.py](pos_probe/pos_probe.py)). Fit a TweetEval probe on [CLS] at each layer. Plot both curves for base TinyBERT, MRL and MIPIC.

### H9. PIC leaves low-level information intact in the remaining dimensions

**Paper.**
- §1, p.2: "PIC allows the remaining dimensions to continue modeling low level and general linguistic features".
- §3.3, p.6: supervision on the prefix preserves "low-level features and representational flexibility in the remaining space".
- §1, p.2: PIC "gently biases the formation of low level representations toward task relevant subspaces".

**Hypothesis.** At each PIC checkpoint (l_i, d_i), task information becomes concentrated in [1:d_i], while linguistic information (POS, syntax) is retained in [d_i+1:D]. Compared with MRL, MIPIC's early layers lose little or no POS probe accuracy.

**Why test it.** "Gently" and "preserves" are asserted, never measured. PIC acts only on the [CLS] vector, but through shared weights it changes every token's representation. This is the only place where the paper says where low-level features live in the dimension axis, and it puts them *outside* the prefix.

**Test.**
- At layers 1–3, run POS probes on the prefix [1:d_i] and on the suffix [d_i+1:D], for MIPIC, MRL and base TinyBERT.
- Run sentiment probes on the [CLS] prefix and suffix at the same layers.
- Prediction: in MIPIC, POS(suffix) ≥ POS(prefix) and sentiment(prefix) > sentiment(suffix), with total POS accuracy close to MRL's.

### H10. PIC transfers information from deeper layers to earlier ones

**Paper.**
- Abstract, p.1: PIC "incrementally transfers mature task semantics from deeper layers into earlier layers".
- §1, p.2: PIC propagates "task aware cues from upper layers into earlier ones".

**Hypothesis.** The useful effect of PIC requires the deeper checkpoint to act as teacher, i.e. a stop-gradient on z_{i+1}. Training symmetrically, or with the earlier checkpoint as teacher, is worse.

**Why test it.** Eq. 16 is a symmetric InfoNCE between φ_i(z_i) and z_{i+1}, and the paper does not say whether gradients reach z_{i+1}. Without a stop-gradient, information can flow either way. This repository detaches the target by default ([losses/pic_loss.py](losses/pic_loss.py), following the official notebook), but the direction was not ablated.

**Test.** Compare three variants with several seeds, at 16D, 32D and full width, plus the early-exit probe from H11:
- `detach_target=True`;
- `detach_target=False`;
- detaching the source instead of the target.

### H11. Early-layer, low-dimensional prefixes become task-useful

**Paper.**
- §1, p.2: PIC ensures "that useful discriminative structure is available from the beginning of the network processing".
- §3.3, p.6: "allowing early, low-capacity representations to serve as reliable building blocks for deeper ones".

**Hypothesis.** The truncated [CLS] at each checkpoint (e.g. layer 1 at 16D, layer 2 at 32D) gives better downstream probe scores under MIPIC than under MRL or base TinyBERT.

**Why test it.** All evaluation uses the final layer only. The paper's central depth claim is never measured directly, although early-exit embeddings are a natural use of it.

**Test.** For each checkpoint (l_i, d_i) in C, fit the same logistic-regression probe on TweetEval and run STS-B Spearman on z_i. Compare MIPIC, MRL+PIC, MRL and base TinyBERT. This needs only a forward pass with `output_hidden_states=True` on the saved checkpoints.

### H12. PIC maximizes mutual information between checkpoints

**Paper.** §3.3, p.6: PIC encourages "high mutual information (MI) between adjacent checkpoints", using I(z_i; z_{i+1}) ≥ log N − L_InfoNCE (Eq. 15).

**Hypothesis.** PIC's benefit grows with the number of in-batch negatives, as the MI framing implies.

**Why test it.** With batch size 16 (Table 7), the InfoNCE bound cannot exceed log 16 ≈ 2.77 nats. That is a very loose bound on MI between 16- and 32-dimensional continuous vectors. The benefit may come from the contrastive geometry rather than from MI.

**Test.** Log L_chain against log N during training. Vary the batch size (8, 16, 32, 64, using gradient accumulation only for the MRL term) and check whether PIC's gain tracks the size of the bound.

### H13. The progressive (layer, dimension) schedule is necessary

**Paper.**
- §5, p.8–9: "Using full dimensions everywhere removes the bottleneck needed for Matryoshka learning".
- §5, p.8–9: "Our design forces early layers to pack the most important information into small prefixes".
- Table 4.

**Hypothesis.** Pairing small dimensions with shallow layers beats (a) full dimensions at every checkpoint, (b) a reversed or shuffled pairing, and (c) a different choice of layers, by more than seed variance.

**Why test it.**
- In Table 4 the gaps are 0.1–1.75 points from a single run. At STS12-16 the gap is 0.10 points.
- The ablation covers only two datasets, and the backbone isn't stated. The numbers match the BERT column of Tables 1–2.
- The TinyBERT schedule skips 128: C = {(16,1), (32,2), (64,3), (256,4), (512,5), (768,6)} (A.5, p.13). This is not explained.

**Test.** Run the variants above plus a schedule that includes 128, using several seeds on TinyBERT.

### H14. PIC makes small prefixes learn task information early in training

**Paper.** §1, contributions, p.2: PIC "enables lower-dimensional representations to acquire task-relevant information early in training".

**Hypothesis.** Probe scores at 16D and 32D rise faster per epoch under MIPIC than under MRL.

**Why test it.** This is a claim about the training trajectory, separate from the depth claim (H11). No learning curves are reported.

**Evidence so far.** In our run, MIPIC's held-out probe chose epoch 1 of 5 ([MIPIC_TRAINING_README.md](MIPIC_TRAINING_README.md)). That fits fast early learning, but it is also consistent with later epochs degrading the representation.

**Correction, 2026-11-04.** The epoch 1 result above can't separate H14 from general degradation, so it is not evidence for H14. In local seed 42 retrains with the current code, the held-out probe also chose epoch 1 for MRL, which has no PIC, and for MRL+PIC and MRL+SIA. KL-off MIPIC is the only run that chose a later epoch (epoch 3, 0.17 above epoch 1 on a nearly flat curve). On the test set, the selected checkpoint beats the last epoch by up to 2.16 (MRL), 3.94 (MRL+PIC) and 3.67 (MRL+SIA) points. So a best epoch of 1 means further training hurts this probe in general, with or without PIC. Epoch selection also uses the pooler probe, not the 16D and 32D prefixes H14 is about. H14 remains untested; the per-epoch test below is still needed. This is preliminary (one seed). Details in [h11_probe/FINDINGS.md](h11_probe/FINDINGS.md), Finding 8.

**Test.** Evaluate each epoch's checkpoint at every width for MRL and MIPIC. Report the area under the learning curve and the best epoch.

---

## D. Empirical claims

### H15. SIA is essential

**Paper.** §5, p.7–8: "SIA is vital for structural organization; its removal causes sharp drops at small sizes, proving geometric order is key for information density."

**Hypothesis.** Over several seeds, MIPIC without SIA (= MRL+PIC) is reliably worse than MIPIC at 16D and 32D, and MRL+SIA is better than MRL.

**Why test it.** Table 3 shows drops of 2.0–3.3 points at 16D and 32D, from one run on two datasets with an unnamed backbone. "Proving" is too strong for that evidence.

**Evidence so far.** Against, in our TinyBERT run: MRL+SIA had the *lowest* F1 of the four variants at every width (53.26 at 16D against MRL's 56.46; 59.53 at 768D against 67.24). SIA alone hurt. The code ran without the attention-KL term (H6), so this tests the CKA part of SIA.

**Correction, 2026-11-04.** The last sentence above is wrong. The run behind these numbers had the attention KL term **on** (see the H6 correction), so it tested CKA plus KL, not CKA alone. A local seed 42 retrain with the current code (KL off) gives MRL+SIA 63.19 at 16D and 71.00 at 768D, ahead of MRL at 16D to 64D (+3.32, +1.90, +0.73) and 0.2 to 0.7 behind from 128D up. So "SIA alone hurt" holds only for SIA with KL on. This is preliminary (one seed, pending seeds 1 and 2 and a KL-on retrain). Details in [h11_probe/FINDINGS.md](h11_probe/FINDINGS.md), Findings 4 and 5.

**Test.** Run MRL, MRL+SIA, MRL+PIC and MIPIC with at least 3 seeds, crossed with KL on/off and α ∈ {0.4, 0.7}, at the paper's settings. Report mean ± std.

### H16. SIA and PIC are synergistic

**Paper.** §5, p.7: Table 3 "highlights the synergy between SIA and PIC".

**Hypothesis.** The gain from MIPIC over MRL is larger than the sum of the gains from MRL+SIA and MRL+PIC.

**Why test it.** The paper doesn't report MRL+SIA or MRL+PIC against plain MRL, so synergy can't be computed from its tables.

**Evidence so far.** In our run, MIPIC beats MRL+PIC clearly only at 16D (58.60 vs 54.27) and slightly at 32D (60.51 vs 59.85). At larger widths they are within 0.3 points.

**Test.** Use the runs from H15 to compute an interaction term per width.

### H17. MIPIC matches baselines at full capacity

**Paper.**
- Abstract, p.1, and §4.2, p.7: "MIPIC maintains parity with baselines in full-capacity regimes".
- §1, p.2: MIPIC "remains competitive at full capacity".

**Hypothesis.** At 512D and 768D, MIPIC is within seed variance of MRL.

**Why test it.** The SIA/PIC terms take weight away from the MRL loss (α = 0.4 in Table 7), so a cost at full width is plausible.

**Evidence so far.** Against, in our run: MRL is ahead from 64D upward, by 2.6 F1 at 768D (67.24 vs 64.64).

**Test.** The H15 runs, plus MRL trained with the same effective MRL loss weight, to separate a weighting effect from an SIA/PIC effect.

### H18. MIPIC's advantage is largest at 16–32 dimensions

**Paper.** §4.2, p.7: the gap "becomes particularly distinct at extremely low dimensions (16 and 32)". Appendix B, p.14, says the same.

**Hypothesis.** MIPIC − MRL is largest at 16D and decreases with width.

**Evidence so far.** Consistent: in our run MIPIC leads only at 16D (+2.1) and 32D (+0.5).

**Why test it.** The pattern has an alternative explanation. Lowering the MRL weight and adding prefix-focused losses could trade high-width quality for low-width quality, rather than improving compression. Measure the trade-off with seeds, and check whether an MRL variant that weights small prefixes more heavily (the weighted variant of Eq. 1) gives the same curve.

### H19. Better out-of-domain generalization

**Paper.** Appendix B, p.14: "on out-of-domain (OOD) datasets like STS12–16 and SciTail, MIPIC maintains a consistent accuracy lead".

**Hypothesis.** Averaged over seeds, MIPIC leads on OOD sets at most widths.

**Why test it.** The paper's own tables have exceptions:
- Table 9: BGE-M3 STS14 at 512D (72.28 vs MRL 74.93); Qwen Emotion at 512D (61.52 vs ESE 64.79); Qwen STS12 at 1024D (70.31 vs 70.74).
- Table 2: TinyBERT STS14 at 512D, where unsupervised SimCSE wins (63.93 vs 63.22).

**Test.** Evaluate on STS12–16 and SickR. Leave SciTail out unless it has been removed from the training corpus (see the reproduction notes below).

### H20. Works regardless of backbone depth and size

**Paper.** A.1, p.12: MIPIC "consistently optimizes the internal organization of semantic information regardless of the model's depth or total parameter count".

**Hypothesis.** MIPIC's gains do not depend on a hand-tuned layer schedule.

**Why test it.** Each backbone uses a different, hand-picked layer and checkpoint schedule (A.5). §7 (p.9) itself says performance "relies on the careful tuning of hyperparameters and the strategic selection of layer checkpoints".

**Test.** On TinyBERT, compare the paper's schedule with two other reasonable schedules, e.g. evenly spaced layers or a schedule that includes 128D.

---

## Reporting and reproduction notes

These issues affect how much the evidence above can be trusted, and how closely this repository can reproduce the paper.

1. **No variance.** Every table reports a single run with no seeds or confidence intervals. Many reported differences are below 1 point.
2. **Ablation backbone not stated.** The MIPIC column of Tables 3 and 4 equals the BERT MIPIC column of Tables 1 and 2 (e.g. STS12 at 16D = 62.08, TweetEval at 16D = 60.38), so the ablations appear to be BERT-only and cover two datasets.
3. **Training corpus size doesn't add up.** A.2 (p.12) gives 6,000 classification sentences, 3,000 STS pairs and 3,000 pair-classification pairs. That is 18,000 sentences after flattening, but the text says "24,000 unique sentences". This repository's corpus has 20,240 rows.
4. **Possible OOD contamination here.** Our corpus includes 3,000 `DGEM_FORMAT` rows. `dgem_format` is one of the formats in the SciTail release, so these are probably SciTail sentences, whereas the paper treats SciTail as out-of-domain with no training sample (Table 6). Check the source before using SciTail as an OOD result.
5. **"Task" means the SimCSE objective.** The MRL task loss is unsupervised SimCSE (A.3, p.12), so "task-relevant" and "task semantics" in the PIC description refer to what that contrastive objective rewards, not to any labelled task.
6. **Pooling is not specified.** The paper's z = f(x) does not say how the embedding is pooled. This repository applies MRL to `pooler_output`, which mixes all hidden coordinates. Prefix structure in the pooled output therefore does not imply prefix structure in the hidden-state coordinates (relevant to H1 and H3).
7. **Training cost explanation.** §5 (p.9) attributes the slowdown to "additional forward passes", but SIA and PIC can reuse the SimCSE forward passes' hidden states. Profiling would show whether the cost comes from extra passes or from the loss computations.
