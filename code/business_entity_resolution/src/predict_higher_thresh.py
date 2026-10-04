# src/predict_higher_thresh.py
import pandas as pd
import joblib
import csv

FEATURE_COLS = [
    "name_jaccard", "addr_jaccard", "country_match",
    "name_len_diff", "shared_name_tokens", "shared_addr_tokens",
]

THRESHOLD = 0.98

print(f"Loading model (using threshold={THRESHOLD}, overriding saved threshold.json)...")
model = joblib.load("output/model.pkl")

print("Loading test features...")
feats = pd.read_csv("output/test_features.tsv", sep="\t")
print(f"  {len(feats)} candidate pairs loaded")

print("Scoring...")
feats["score"] = model.predict_proba(feats[FEATURE_COLS])[:, 1]
matched = feats[feats["score"] >= THRESHOLD]
print(f"  {len(matched)} pairs above threshold {THRESHOLD}")

matches_by_entity = matched.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()

all_s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t")["entity_id"]

with open("output/matching_results.tsv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["source1_entity_id", "matched_entity_ids"])
    n_matched, n_empty = 0, 0
    for eid in all_s1:
        ids = matches_by_entity.get(eid, [])
        if ids:
            n_matched += 1
        else:
            n_empty += 1
        w.writerow([eid, ",".join(sorted(set(ids)))])

print(f"Done. threshold={THRESHOLD}: {n_matched} entities with matches, {n_empty} predicted as singletons.")
print("Saved to output/matching_results.tsv (this OVERWRITES your current threshold-0.95 file)")