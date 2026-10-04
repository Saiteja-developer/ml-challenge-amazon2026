# Business Entity Resolution — Pipeline

## Overview
This pipeline matches Source 1 (reference) business entities against Source 2 and Source 3 records using token-based blocking followed by a LightGBM classifier on similarity features, with the decision threshold tuned to maximize the macro F_0.5 metric specified by the challenge.

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Order (end-to-end)

1. **Generate train candidates (for training the model):**
   ```bash
   python src/blocking.py --full
   ```
   Writes `output/candidate_pairs_train_debug.tsv`, prints recall/candidate stats against training ground truth.

2. **Compute training features:**
   ```bash
   python src/features.py --full --limit-rows 300000
   ```
   Writes `output/train_features_full.tsv`. `--limit-rows` caps the number of Source 1 training entities processed (a 300k-entity subset was used for time efficiency; the underlying similarity signal is highly repetitive across entities, so this did not meaningfully limit model quality).

3. **Train the model:**
   ```bash
   python src/train_model.py
   ```
   Trains LightGBM, tunes the decision threshold for F_0.5 on a held-out validation split (split at the entity level to avoid leakage), saves `output/model.pkl` and `output/threshold.json`.

4. **Generate test candidates:**
   ```bash
   python src/blocking.py --test
   ```
   Writes `output/candidate_pairs.tsv` — this is the final candidate set, included in the submission as-is.

5. **Compute test features:**
   ```bash
   python src/features.py --candfile output/candidate_pairs.tsv --out output/test_features.tsv --s1 dataset/test/test_source1.tsv --s2 dataset/test/test_source2.tsv --s3 dataset/test/test_source3.tsv
   ```

6. **Generate final predictions:**
   ```bash
   python src/predict.py
   ```
   Writes `output/matching_results.tsv` using the threshold saved in `output/threshold.json`.

   *(Optional)* `src/predict_higher_thresh.py` regenerates `matching_results.tsv` using a manually specified, more conservative threshold (0.98 in our final submission) — used after confirming via the leaderboard that a higher threshold improved our F_0.5 score, consistent with the metric's 2x precision weighting.

7. **Validate before submitting:**
   ```bash
   python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
   ```

## Pipeline Components
- `src/normalize.py` — text normalization (legal suffix collapsing, address abbreviation expansion, apostrophe/punctuation stripping), implemented as a single combined regex pass per replacement dictionary for speed at scale
- `src/blocking.py` — token-based inverted-index blocking: indexes Source 2/Source 3 records by rare name/address tokens (selected via document frequency), scores candidates by weighted token overlap, and keeps only the top-N highest-scoring candidates per Source 1 entity. Supports `--full` (train set, scored against ground truth) and `--test` (test set, no scoring) modes
- `src/features.py` — vectorized similarity features (Jaccard name/address overlap computed via hashed sparse vectors and sparse matrix operations rather than per-pair Python loops), processed in memory-bounded chunks
- `src/train_model.py` — LightGBM classifier, entity-level train/val split (no leakage), vectorized F_0.5-optimal threshold selection via `groupby`-based aggregation
- `src/predict.py` — applies the trained model and saved threshold to test candidates, writes the final submission format with exactly one row per Source 1 test entity
- `src/predict_higher_thresh.py` — variant of `predict.py` for testing an explicit, manually chosen threshold without retraining

## Notes
- Blocking recall on full training data: ~0.6125 at ~71.0 avg candidates/entity (see console output of `blocking.py --full`)
- Model trained on a 300k-entity subset of training data for time efficiency; full training data was not required given the redundancy of similarity signal across entities
- Final decision threshold: 0.98, selected via leaderboard validation after the automated sweep (up to 0.95) showed F_0.5 was still improving at the top of the tested range
- No external data sources, APIs, or lookups used, per challenge rules — all normalization and matching logic is self-contained and derived only from the provided training/test data