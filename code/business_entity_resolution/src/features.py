# src/features.py
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from blocking import load_and_normalize

CHUNK_SIZE = 50000

NAME_VEC = HashingVectorizer(n_features=2**16, binary=True, alternate_sign=False, token_pattern=r"(?u)\b\w+\b")
ADDR_VEC = HashingVectorizer(n_features=2**16, binary=True, alternate_sign=False, token_pattern=r"(?u)\b\w+\b")


def sparse_overlap_jaccard(vec, texts1, texts2):
    X1 = vec.transform(texts1)
    X2 = vec.transform(texts2)
    overlap = np.asarray(X1.multiply(X2).sum(axis=1)).ravel()
    size1 = np.asarray(X1.sum(axis=1)).ravel()
    size2 = np.asarray(X2.sum(axis=1)).ravel()
    union = size1 + size2 - overlap
    union[union == 0] = 1
    return overlap, overlap / union


def build_features(candidate_pairs_path, s1_path, s2_path, s3_path, out_path, limit_rows=None):
    t0 = time.time()
    print("Loading S1...")
    s1_lut = load_and_normalize(s1_path).set_index("entity_id")
    print(f"  S1 done in {time.time()-t0:.1f}s ({len(s1_lut)} rows)")

    t1 = time.time()
    print("Loading S2...")
    s2_lut = load_and_normalize(s2_path).set_index("entity_id")
    print(f"  S2 done in {time.time()-t1:.1f}s ({len(s2_lut)} rows)")

    t2 = time.time()
    print("Loading S3...")
    s3_lut = load_and_normalize(s3_path).set_index("entity_id")
    print(f"  S3 done in {time.time()-t2:.1f}s ({len(s3_lut)} rows)")

    print("Counting candidate rows...")
    total_lines = sum(1 for _ in open(candidate_pairs_path, encoding="utf-8")) - 1
    effective_total = min(total_lines, limit_rows) if limit_rows else total_lines
    print(f"  {effective_total} S1 rows to process in chunks of {CHUNK_SIZE}")

    first_write = True
    processed = 0
    run_start = time.time()

    for chunk in pd.read_csv(candidate_pairs_path, sep="\t", chunksize=CHUNK_SIZE, nrows=limit_rows):
        chunk = chunk[chunk["candidate_entity_ids"].notna() & (chunk["candidate_entity_ids"] != "")].copy()
        if chunk.empty:
            processed += CHUNK_SIZE
            continue

        chunk["candidate_entity_id"] = chunk["candidate_entity_ids"].str.split(",")
        pairs = chunk.explode("candidate_entity_id")[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)

        s1_r = s1_lut.reindex(pairs["source1_entity_id"]).reset_index(drop=True)
        s1_r.columns = ["name1", "addr1", "country1"]

        is_s2 = pairs["candidate_entity_id"].str.startswith("S2")
        cand_ids = pairs["candidate_entity_id"]

        name2 = pd.Series(index=pairs.index, dtype=object)
        addr2 = pd.Series(index=pairs.index, dtype=object)
        country2 = pd.Series(index=pairs.index, dtype=object)

        s2_hits = s2_lut.reindex(cand_ids[is_s2])
        name2.loc[is_s2] = s2_hits["name_norm"].values
        addr2.loc[is_s2] = s2_hits["address_norm"].values
        country2.loc[is_s2] = s2_hits["country"].values

        s3_hits = s3_lut.reindex(cand_ids[~is_s2])
        name2.loc[~is_s2] = s3_hits["name_norm"].values
        addr2.loc[~is_s2] = s3_hits["address_norm"].values
        country2.loc[~is_s2] = s3_hits["country"].values

        merged = pd.DataFrame({
            "source1_entity_id": pairs["source1_entity_id"].values,
            "candidate_entity_id": pairs["candidate_entity_id"].values,
            "name1": s1_r["name1"].values, "addr1": s1_r["addr1"].values, "country1": s1_r["country1"].values,
            "name2": name2.values, "addr2": addr2.values, "country2": country2.values,
        })
        merged = merged.dropna(subset=["name1", "name2"])
        for c in ["name1", "name2", "addr1", "addr2"]:
            merged[c] = merged[c].fillna("")

        merged["country_match"] = (merged["country1"] == merged["country2"]).astype(int)
        merged["name_len_diff"] = (merged["name1"].str.len() - merged["name2"].str.len()).abs()

        shared_n, name_jaccard = sparse_overlap_jaccard(NAME_VEC, merged["name1"], merged["name2"])
        shared_a, addr_jaccard = sparse_overlap_jaccard(ADDR_VEC, merged["addr1"], merged["addr2"])

        merged["shared_name_tokens"] = shared_n
        merged["shared_addr_tokens"] = shared_a
        merged["name_jaccard"] = name_jaccard
        merged["addr_jaccard"] = addr_jaccard

        out_cols = ["source1_entity_id", "candidate_entity_id", "name_jaccard", "addr_jaccard",
                    "country_match", "name_len_diff", "shared_name_tokens", "shared_addr_tokens"]

        merged[out_cols].to_csv(out_path, sep="\t", index=False, mode="w" if first_write else "a", header=first_write)
        first_write = False

        processed += len(chunk)
        print(f"  processed ~{processed} S1 rows, wrote {len(merged)} feature rows ({time.time()-run_start:.1f}s elapsed)")

    print(f"Done. Features saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--candfile", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--s1", default="dataset/train/train_source1.tsv")
    parser.add_argument("--s2", default="dataset/train/train_source2.tsv")
    parser.add_argument("--s3", default="dataset/train/train_source3.tsv")
    parser.add_argument("--limit-rows", type=int, default=None)
    args = parser.parse_args()

    cand_file = args.candfile or ("output/candidate_pairs_train_debug.tsv" if args.full else "output/candidate_pairs_train_sample.tsv")
    out_file = args.out or ("output/train_features_full.tsv" if args.full else "output/train_features_sample.tsv")

    build_features(cand_file, args.s1, args.s2, args.s3, out_file, limit_rows=args.limit_rows)