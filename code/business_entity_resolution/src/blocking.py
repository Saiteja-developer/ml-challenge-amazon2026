# src/blocking.py
import re
import csv
import argparse
from collections import Counter, defaultdict
import pandas as pd

MAX_BLOCK_SIZE = 500
NAME_TOP_N = 3
ADDR_TOP_N = 1
TOP_CANDIDATES = 30
NAME_WEIGHT = 2
ADDR_WEIGHT = 1

LEGAL_SUFFIXES = {
    "corporation": "corp", "corp.": "corp", "incorporated": "inc", "inc.": "inc",
    "limited": "ltd", "ltd.": "ltd", "private": "pvt", "pvt.": "pvt",
    "company": "co", "co.": "co", "llc": "llc", "llp": "llp",
    "sasu": "sasu", "sarl": "sarl", "sas": "sas",
}

ADDRESS_ABBREVS = {
    "road": "rd", "rd.": "rd", "street": "st", "st.": "st",
    "avenue": "ave", "ave.": "ave", "boulevard": "blvd",
    "apartment": "apt", "apt.": "apt", "floor": "fl", "fl.": "fl",
    "drive": "dr", "dr.": "dr", "lane": "ln", "ln.": "ln",
    "suite": "ste", "ste.": "ste", "circle": "cir",
    "highway": "hwy", "parkway": "pkwy",
}

NAME_STOPWORDS = {
    "inc", "corp", "ltd", "llc", "llp", "pvt", "co", "sasu", "sarl", "sas",
    "the", "and", "of", "for", "in", "at", "on", "by",
    "group", "services", "solutions", "company", "international",
    "national", "global", "enterprises", "trading", "consulting",
}

ADDRESS_STOPWORDS = {
    "rd", "st", "ave", "blvd", "apt", "fl", "dr", "ln", "ste", "cir",
    "hwy", "pkwy", "unit", "near", "opposite", "behind", "next", "to",
    "no", "plot", "floor", "road", "street",
}


def _vectorized_clean(series):
    s = series.fillna("").astype(str).str.lower()
    s = s.str.replace("'", "", regex=False)
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def _apply_token_map(series, mapping):
    if not mapping:
        return series
    clean_keys = {}
    for word, repl in mapping.items():
        w = re.sub(r"[^\w]", "", word)
        if w:
            clean_keys[w] = repl
    if not clean_keys:
        return series
    pattern = r"\b(" + "|".join(re.escape(k) for k in clean_keys.keys()) + r")\b"
    return series.str.replace(pattern, lambda m: clean_keys.get(m.group(1), m.group(1)), regex=True)


def load_and_normalize(path):
    # only read the 4 columns we actually need — cuts memory vs loading the whole file
    df = pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
    df["country"] = df["country"].fillna("").astype(str).str.strip().str.lower()
    name_clean = _vectorized_clean(df["business_name"])
    addr_clean = _vectorized_clean(df["business_address"])
    df["name_norm"] = _apply_token_map(name_clean, LEGAL_SUFFIXES)
    df["address_norm"] = _apply_token_map(addr_clean, ADDRESS_ABBREVS)
    # drop raw text columns once normalized — halves memory footprint going forward
    return df[["entity_id", "name_norm", "address_norm", "country"]]


def _tokens(text, stopwords, min_len=3):
    if not text:
        return []
    return [t for t in text.split() if len(t) >= min_len and t not in stopwords]


def build_token_index(df, text_col, stopwords, label):
    tmp = df[["entity_id", "country", text_col]].copy()
    tmp["tokens"] = tmp[text_col].apply(lambda t: _tokens(t, stopwords))
    tmp = tmp.explode("tokens").dropna(subset=["tokens"])
    tmp["key"] = tmp["country"] + "|" + tmp["tokens"]

    grouped = tmp.groupby("key")["entity_id"].apply(set)
    index = {}
    dropped = 0
    for key, ids in grouped.items():
        if len(ids) > MAX_BLOCK_SIZE:
            dropped += 1
            continue
        index[key] = ids
    print(f"  {label}: dropped {dropped} oversized blocks (> {MAX_BLOCK_SIZE})")
    return index


def compute_doc_freq(s2, s3, col, stopwords):
    freq = Counter()
    for df in (s2, s3):
        for text in df[col]:
            for tok in set(_tokens(text, stopwords)):
                freq[tok] += 1
    return freq


def top_rare_tokens(text, stopwords, freq, n):
    toks = _tokens(text, stopwords)
    toks = sorted(set(toks), key=lambda t: freq.get(t, 0))
    return toks[:n]


def block_and_score(source1_path, source2_path, source3_path, gt_path, out_path, sample_size=None):
    print("Loading and normalizing...")
    s1 = load_and_normalize(source1_path)
    s2 = load_and_normalize(source2_path)
    s3 = load_and_normalize(source3_path)

    if sample_size:
        s1 = s1.sample(n=min(sample_size, len(s1)), random_state=42).reset_index(drop=True)
        print(f"  SAMPLE MODE: testing on {len(s1)} S1 rows (full S2/S3 indexes still used)")

    print("Building name-token indexes...")
    name_idx_s2 = build_token_index(s2, "name_norm", NAME_STOPWORDS, "S2 name")
    name_idx_s3 = build_token_index(s3, "name_norm", NAME_STOPWORDS, "S3 name")

    print("Building address-token indexes...")
    addr_idx_s2 = build_token_index(s2, "address_norm", ADDRESS_STOPWORDS, "S2 addr")
    addr_idx_s3 = build_token_index(s3, "address_norm", ADDRESS_STOPWORDS, "S3 addr")

    print("Computing token document-frequencies...")
    name_freq = compute_doc_freq(s2, s3, "name_norm", NAME_STOPWORDS)
    addr_freq = compute_doc_freq(s2, s3, "address_norm", ADDRESS_STOPWORDS)

    print("Loading ground truth...")
    try:
        gt = pd.read_csv(gt_path, sep="\t")
        gt_map = {}
        for row in gt.itertuples():
            val = row.matched_entity_ids
            ids = str(val).split(",") if pd.notna(val) and val else []
            gt_map[row.source1_entity_id] = set(ids)
    except FileNotFoundError:
        gt_map = {}

    print(f"Generating candidates for {len(s1)} S1 entities (streaming to {out_path})...")
    s1 = s1.copy()
    s1["name_tokens"] = s1["name_norm"].apply(lambda t: top_rare_tokens(t, NAME_STOPWORDS, name_freq, NAME_TOP_N))
    s1["addr_tokens"] = s1["address_norm"].apply(lambda t: top_rare_tokens(t, ADDRESS_STOPWORDS, addr_freq, ADDR_TOP_N))

    hits, total_true, total_cand_size, n = 0, 0, 0, 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for row in s1.itertuples():
            scores = defaultdict(int)
            country = row.country
            for tok in row.name_tokens:
                key = country + "|" + tok
                for cid in name_idx_s2.get(key, ()):
                    scores[cid] += NAME_WEIGHT
                for cid in name_idx_s3.get(key, ()):
                    scores[cid] += NAME_WEIGHT
            for tok in row.addr_tokens:
                key = country + "|" + tok
                for cid in addr_idx_s2.get(key, ()):
                    scores[cid] += ADDR_WEIGHT
                for cid in addr_idx_s3.get(key, ()):
                    scores[cid] += ADDR_WEIGHT

            top = [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])[:TOP_CANDIDATES]]
            cand_ids = set(top)

            writer.writerow([row.entity_id, ",".join(sorted(cand_ids))])

            true_ids = gt_map.get(row.entity_id, set())
            hits += len(true_ids & cand_ids)
            total_true += len(true_ids)
            total_cand_size += len(cand_ids)
            n += 1

            if n % 200000 == 0:
                print(f"  processed {n}/{len(s1)}...")

    if total_true:
        print(f"\nRecall @ blocking: {hits}/{total_true} = {hits/total_true:.4f}")
    else:
        print("\nNo ground truth match (expected for test set) — candidates written, recall not computed.")
    print(f"Avg candidates per S1 entity: {total_cand_size/n:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sample-size", type=int, default=30000)
    args = parser.parse_args()

    if args.test:
        block_and_score(
            "dataset/test/test_source1.tsv",
            "dataset/test/test_source2.tsv",
            "dataset/test/test_source3.tsv",
            "dataset/train/train_ground_truth.tsv",
            "output/candidate_pairs.tsv",
            sample_size=None,
        )
    else:
        out_file = "output/candidate_pairs_train_debug.tsv" if args.full else "output/candidate_pairs_train_sample.tsv"
        block_and_score(
            "dataset/train/train_source1.tsv",
            "dataset/train/train_source2.tsv",
            "dataset/train/train_source3.tsv",
            "dataset/train/train_ground_truth.tsv",
            out_file,
            sample_size=None if args.full else args.sample_size,
        )