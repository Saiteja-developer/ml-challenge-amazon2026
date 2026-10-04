# src/train_model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json

FEATURE_COLS = [
    "name_jaccard", "addr_jaccard", "country_match",
    "name_len_diff", "shared_name_tokens", "shared_addr_tokens",
]

print("Loading features + ground truth...")
feats = pd.read_csv("output/train_features_full.tsv", sep="\t")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

gt_map = {}
for row in gt.itertuples():
    val = row.matched_entity_ids
    ids = set(str(val).split(",")) if pd.notna(val) and val else set()
    gt_map[row.source1_entity_id] = ids

print("Labeling pairs...")
feats["true_set"] = feats["source1_entity_id"].map(gt_map)
feats["label"] = feats.apply(lambda r: int(r["candidate_entity_id"] in (r["true_set"] or set())), axis=1)
feats = feats.drop(columns=["true_set"])
print(f"  {feats['label'].sum()} positive pairs out of {len(feats)}")

print("Splitting by entity (80/20)...")
entities = np.array(feats["source1_entity_id"].unique(), dtype=object)
rng = np.random.default_rng(42)
perm = rng.permutation(len(entities))
entities = entities[perm]
split = int(len(entities) * 0.8)
train_ents, val_ents = set(entities[:split]), set(entities[split:])

train_df = feats[feats["source1_entity_id"].isin(train_ents)]
val_df = feats[feats["source1_entity_id"].isin(val_ents)]

print("Training LightGBM...")
model = lgb.LGBMClassifier(n_estimators=350, max_depth=8, learning_rate=0.05, class_weight="balanced")
model.fit(train_df[FEATURE_COLS], train_df["label"])

print("Scoring validation set, tuning threshold for F_0.5 (vectorized)...")
val_df = val_df.copy()
val_df["score"] = model.predict_proba(val_df[FEATURE_COLS])[:, 1]

val_ents_list = sorted(val_ents)
true_count = pd.Series({e: len(gt_map.get(e, set())) for e in val_ents_list}).reindex(val_ents_list).fillna(0)

best_thresh, best_f = 0.5, -1
for thresh in np.arange(0.3, 0.96, 0.05):
    pred_mask = val_df["score"] >= thresh
    grouped = val_df[pred_mask].groupby("source1_entity_id")["label"].agg(tp="sum", pred_count="count")
    grouped = grouped.reindex(val_ents_list, fill_value=0)

    tp = grouped["tp"].to_numpy(dtype=float)
    pred_count = grouped["pred_count"].to_numpy(dtype=float)
    tc = true_count.to_numpy(dtype=float)

    prec = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    rec = np.where(tc > 0, np.divide(tp, tc, out=np.zeros_like(tp), where=tc > 0), 1.0)

    both_empty = (tc == 0) & (pred_count == 0)
    denom = 0.25 * prec + rec
    f = np.divide(1.25 * prec * rec, denom, out=np.zeros_like(prec), where=denom > 0)
    f = np.where(both_empty, 1.0, f)

    score = f.mean()
    print(f"  thresh={thresh:.2f}  F_0.5={score:.4f}")
    if score > best_f:
        best_f, best_thresh = score, thresh

print(f"\nBest threshold: {best_thresh:.2f}  F_0.5: {best_f:.4f}")

joblib.dump(model, "output/model.pkl")
with open("output/threshold.json", "w") as f:
    json.dump({"threshold": float(best_thresh)}, f)
print("Saved model.pkl and threshold.json")