# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Code Commanders
**Team Members:** Suryadevara Aditya,Ankur Sai Teja,Thota Veeraj
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We built an end-to-end entity resolution pipeline that matches Source 1 business records against Source 2 and Source 3 using a two-stage architecture: a fast, token-based inverted-index blocking stage to narrow ~12.5 million records down to a tractable candidate set, followed by a LightGBM classifier trained on similarity features and tuned specifically to maximize the precision-heavy F₀.₅ metric. Given the dataset scale, we prioritized vectorized, memory-bounded implementations at every stage over more semantically powerful but computationally infeasible approaches (e.g., transformer-embedding-based blocking), and deliberately tuned our decision threshold to be conservative, since F₀.₅ penalizes false merges twice as heavily as missed matches.

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory analysis of the raw data confirmed every noise pattern the problem statement warned about, and surfaced a few additional ones:

- **Legal suffix inconsistency**: "Corp" / "Corporation" / "Corp.", "Private" / "Pvt" / "Pvt.", and — only present in the test set's France records, not training — French forms like "SARL" and "SASU".
- **Punctuation and apostrophes**: names like "Orelee's Barbershop" split into separate tokens ("orelee", "s", "barbershop") under naive punctuation stripping unless apostrophes were removed outright rather than replaced with whitespace.
- **Address abbreviation inconsistency**: "Road" vs "Rd", "Drive" vs "Dr", "Avenue" vs "Ave" — inconsistent even within a single source.
- **Landmark-based addresses**: references like "Near Fortis Hospital" or "(Next To Iter College)" that carry no standardized matching signal.
- **Word-order transpositions**: business names where token order differs across sources (e.g., "Prime Money" vs "Money Prime"), which broke an early single-representative-token blocking scheme entirely.
- **Cross-script name variants**: some Source 2/Source 3 Indian business names appeared in Devanagari or Malayalam script rather than Latin script, meaning name-token overlap with an English-script Source 1 name is structurally zero — a real, unresolved limitation of our final approach.
- **Open country set**: training data covered only US and India; the test set added France with no training examples, confirming country could not be hardcoded or one-hot encoded as a fixed set.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (two-stage pipeline)

**Core Innovation:** Given the dataset's scale (~2.2M Source 1, ~5.0M Source 2, ~5.3M Source 3 records), our main technical contribution was not a single novel algorithm but a systematic, iterative approach to making every pipeline stage computationally tractable at this scale on commodity hardware (a single laptop, CPU-only, 16GB RAM) — replacing every per-row Python operation with a vectorized pandas/numpy/scipy equivalent, and streaming/chunking every stage to stay within memory bounds. We also deliberately tuned our matching threshold against the actual competition metric (macro F₀.₅), rather than a proxy like accuracy or standard F1, since F₀.₅'s 2x precision weighting materially changes the optimal operating point.

---

## 3. Candidate Generation (Blocking)

We evaluated and rejected a transformer-embedding-based blocking approach (sentence-transformers + FAISS nearest-neighbor search) early on: at 12.5M records, CPU-only encoding was projected to take 18–20+ hours, which was not viable given the time constraint. We instead used a **token-based inverted-index blocking scheme**:

- **Blocking keys used:** normalized name tokens and normalized address tokens, each combined with `country` to keep blocks source-appropriate (e.g., `india|prabhav`). Tokens were selected by **document frequency** across Source 2/Source 3 — only the rarest (most distinctive) tokens per entity were used for indexing, since common words (e.g., "trading", "services", "delhi") match too broadly to carry real signal and only bloat candidate sets.
- **Candidate scoring and capping:** each candidate was scored by weighted token overlap (name-token matches weighted 2x over address-token matches), and only the top-N highest-scoring candidates were retained per Source 1 entity — a hard cap, so total compute stays predictable and bounded regardless of how common any individual token turns out to be. Oversized blocks (any single token matching an unreasonably large number of records) were dropped outright during index construction as too generic to be useful.
- **Candidate pairs generated:** ~1.6M non-empty candidate rows in the final test-set `candidate_pairs.tsv`, averaging ~71–74 candidate entity_ids per Source 1 entity (full-scale train run: ~71.0 avg candidates/entity).
- **How we ensured true matches were not lost:** we measured recall directly against `train_ground_truth.tsv` at every blocking configuration change, iterating through roughly six distinct schemes (single-representative-token → multi-token → rare-token-weighted-and-capped) and tracking the recall / avg-candidates-per-entity tradeoff explicitly at each step, rather than tuning blindly. Our final full-scale blocking configuration achieved **~61.25% recall** (4,678,381 / 7,638,365 true matches recoverable) at ~71.0 avg candidates/entity — a deliberate tradeoff point chosen because further loosening the candidate cap made downstream feature computation infeasible in the remaining time, while further tightening it discarded meaningfully more true matches.

---

## 4. Matching Model

**Features used:**
- **Name features:** `name_jaccard` (token-set Jaccard similarity on normalized business names, computed via hashed sparse vectors and sparse matrix overlap rather than a per-pair Python loop, for speed at scale), `shared_name_tokens` (raw overlap count), `name_len_diff` (absolute character-length difference)
- **Address features:** `addr_jaccard` (token-set Jaccard similarity on normalized addresses, same vectorized approach), `shared_addr_tokens` (raw overlap count)
- **Other:** `country_match` (binary flag — whether normalized country strings match exactly between the Source 1 record and the candidate)

**Model type:** LightGBM (gradient-boosted decision trees), binary classifier — chosen over a deep-learning approach because the feature set is low-dimensional, structured/tabular similarity data, which tree ensembles handle very efficiently, train in minutes rather than hours, and satisfy the challenge's MIT/Apache 2.0, ≤8B-parameter model constraint comfortably. Trained with `class_weight="balanced"` to account for the natural class imbalance (true matches are a small minority — roughly 3% — of all blocked candidate pairs).

**Train/validation methodology:** we split at the **entity level**, not the row level — all candidate pairs belonging to a given Source 1 entity were kept together in either the training or validation split, never divided across both, to avoid leakage that a naive row-level split would introduce.

**Threshold selection method:** we swept decision thresholds from 0.30 to 0.95 in increments of 0.05, scoring each threshold's predictions against the held-out validation set using the actual **macro-averaged F₀.₅ formula specified in the problem statement** (not accuracy or standard F1), and selected the threshold that maximized it. The sweep was still improving at the top of the tested range, so we further tested threshold values up to 0.98 directly against the live leaderboard, which confirmed the higher threshold improved our real score — consistent with F₀.₅'s heavy precision weighting rewarding a conservative decision boundary.

---

## 5. Results & Error Analysis

- **F₀.₅ Score (macro), validation set:** ~0.688 (best threshold: 0.95, on the held-out entity-level validation split)
- **Leaderboard F₀.₅ Score:** 0.548 (public leaderboard, after iterative threshold tuning from an initial 0.526 baseline)
- **Blocking recall (train, full scale):** 0.6125
- **Common false positives (wrong merges):** most attributable to token overlap from generic address components (city/state names, common street types) contributing similarity signal even between genuinely different businesses at the same general location — our feature set does not currently distinguish "same building, different business" from "same business."
- **Common false negatives (missed matches):** two dominant causes — (1) matches lost entirely at the blocking stage due to our recall ceiling of ~61%, most often where the true match shared no rare/distinctive token with the Source 1 record (e.g., heavy abbreviation or transliteration divergence); (2) matches that did survive blocking but scored below our conservative 0.98 decision threshold, typically cases with only partial name/address token overlap that a purely whole-word-Jaccard feature set could not confidently distinguish from a near-miss.

---

## 6. Conclusion

Given the dataset's scale and the competition's time constraint, our priority was building a pipeline that was actually computationally tractable end-to-end on available hardware, and iterating on each stage's speed and memory footprint before layering on additional modeling sophistication. The main lesson learned was how badly naive per-row Python operations (`.apply()`, per-pair loops) scale past a few hundred thousand rows, and how consistently the fix — vectorizing with pandas/numpy/scipy — turned infeasible multi-hour steps into minutes at every stage we hit this problem (normalization, feature computation, and threshold evaluation all independently ran into this same issue). With more time, our clearest next steps would be adding character-level (n-gram) similarity features to catch typos and cross-script name variants that whole-word Jaccard misses entirely, and adding a deterministic exact-match rule for identical normalized name+address pairs to capture "easy" high-confidence matches without relying solely on the classifier's threshold.

---

## Appendix

### A. Code Artefacts

Our complete, runnable code ships under `code/business_entity_resolution/src/`, with a `README.md` documenting exact run commands and a `requirements.txt` pinning all dependencies. The pipeline consists of:

- `normalize.py` — text normalization utilities (legal suffix collapsing, address abbreviation expansion, punctuation/apostrophe handling), implemented as vectorized pandas string operations with a single combined regex pass per replacement dictionary rather than one pass per rule.
- `blocking.py` — token-based inverted-index blocking: builds country-scoped token indexes over Source 2/Source 3, selects rare/distinctive tokens per Source 1 entity via document-frequency ranking, scores and caps candidates per entity, and streams output directly to `candidate_pairs.tsv` to stay within memory bounds at full scale. Supports `--full` (entire training set, scored against ground truth) and `--test` (test set, no scoring) modes.
- `features.py` — computes pairwise similarity features for every (Source 1, candidate) pair using `HashingVectorizer`-based sparse-vector overlap (vectorized Jaccard, avoiding per-pair Python loops), processed in bounded chunks and written incrementally to disk.
- `train_model.py` — labels candidate pairs against ground truth, performs an entity-level train/validation split, trains the LightGBM classifier, and sweeps decision thresholds using a fully vectorized (pandas `groupby`-based) implementation of the macro F₀.₅ formula to select the final operating threshold.
- `predict.py` — applies the trained model and selected threshold to the test set's candidate pairs and writes `matching_results.tsv` in the required submission format, guaranteeing every test Source 1 entity receives exactly one row (empty match list where no candidate clears the threshold).

**Entry point to reproduce outputs end-to-end:** see `code/business_entity_resolution/README.md` for the full ordered command sequence, from raw data through `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results

- Dataset scale processed: ~2,206,821 Source 1, ~5,034,616 Source 2, ~5,285,603 Source 3 records (training); ~1,732,544 Source 1, ~4,887,273 Source 2, ~5,082,316 Source 3 records (test)
- Training candidate pairs generated (model training subset): 21,326,535 pairs (636,766 positive / 20,689,769 negative) from a 300,000-entity training sample
- Leaderboard score progression across iterations: 0.526 → 0.544 (threshold 0.95 → 0.98) → 0.548 (0.98 threshold + light LightGBM hyperparameter retuning: `n_estimators` 200→350, `max_depth` 6→8)

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.