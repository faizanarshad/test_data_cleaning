"""Load and filter the retail CSV for BiLSTM training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .preprocessing import clean_text

# Classes with fewer than this many rows are dropped so stratified split works.
MIN_PER_CLASS = 2


def load_and_clean_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load CSV; keep rows with labels; dedupe; drop rare classes; build text_input column."""
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin1")
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError(f"Could not read {csv_path}")

    expected = {"ADG_CODE", "GOOD_NAME", "BRAND", "CATEGORY"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Expected columns {expected}, got {list(df.columns)}")

    before = len(df)
    df = df.copy()
    df["GOOD_NAME"] = df["GOOD_NAME"].apply(clean_text)
    df["BRAND"] = df["BRAND"].fillna("").apply(clean_text)
    df["CATEGORY"] = df["CATEGORY"].fillna("").apply(clean_text)

    df["ADG_CODE"] = pd.to_numeric(df["ADG_CODE"], errors="coerce")
    df = df.dropna(subset=["ADG_CODE"])
    df["ADG_CODE"] = df["ADG_CODE"].astype(int)

    df = df[df["GOOD_NAME"].str.len() > 0]

    df = df.drop_duplicates(subset=["GOOD_NAME", "BRAND", "CATEGORY", "ADG_CODE"])

    counts = df.groupby("ADG_CODE").size()
    rare = counts[counts < MIN_PER_CLASS].index
    if len(rare):
        df = df[~df["ADG_CODE"].isin(rare)]
        print(f"Dropped {len(rare)} ADG codes with <{MIN_PER_CLASS} samples (held out from training).")

    df["text_input"] = (
        df["GOOD_NAME"] + " [BRAND] " + df["BRAND"] + " [CAT] " + df["CATEGORY"]
    )

    after = len(df)
    print(f"Cleaning: {before} rows → {after} rows (valid ADG, non-empty name, deduped)")
    return df
