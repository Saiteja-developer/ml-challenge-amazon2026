# src/normalize.py
import re

LEGAL_SUFFIXES = {
    "corporation": "corp", "corp.": "corp", "incorporated": "inc", "inc.": "inc",
    "limited": "ltd", "ltd.": "ltd", "private": "pvt", "pvt.": "pvt",
    "company": "co", "co.": "co", "llc": "llc", "llp": "llp",
    "sasu": "sasu", "sarl": "sarl", "sas": "sas",  # French legal forms
}

ADDRESS_ABBREVS = {
    "road": "rd", "rd.": "rd", "street": "st", "st.": "st",
    "avenue": "ave", "ave.": "ave", "boulevard": "blvd",
    "apartment": "apt", "apt.": "apt", "floor": "fl", "fl.": "fl",
    "drive": "dr", "dr.": "dr", "lane": "ln", "ln.": "ln",
    "suite": "ste", "ste.": "ste", "circle": "cir",
    "highway": "hwy", "parkway": "pkwy",
}


def _strip_punct(text: str) -> str:
    text = text.replace("'", "")           # remove apostrophes entirely (don't leave a gap)
    text = re.sub(r"[^\w\s]", " ", text)   # everything else -> space
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    n = name.lower().strip()
    n = n.replace("&", " and ")
    n = _strip_punct(n)
    tokens = n.split()
    tokens = [LEGAL_SUFFIXES.get(t, t) for t in tokens]
    return " ".join(tokens)


def normalize_address(address: str) -> str:
    if not isinstance(address, str) or not address.strip():
        return ""
    a = address.lower().strip()
    a = _strip_punct(a)
    tokens = a.split()
    tokens = [ADDRESS_ABBREVS.get(t, t) for t in tokens]
    return " ".join(tokens)


def has_landmark_reference(address: str) -> bool:
    if not isinstance(address, str):
        return False
    return bool(re.search(r"\bnear\b|\bopposite\b|\bbehind\b|\bnext to\b", address.lower()))


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
    df["name_norm"] = df["business_name"].apply(normalize_name)
    df["address_norm"] = df["business_address"].apply(normalize_address)
    df["has_landmark"] = df["business_address"].apply(has_landmark_reference)

    for i in range(10):
        row = df.iloc[i]
        print(f"\n--- row {i} ---")
        print(f"name:    {row['business_name']!r}")
        print(f"  -> {row['name_norm']!r}")
        print(f"address: {row['business_address']!r}")
        print(f"  -> {row['address_norm']!r}")
        print(f"landmark: {row['has_landmark']}")