"""Text normalization for product names, brands, and categories."""

from __future__ import annotations

import re

import pandas as pd


def clean_text(s: str) -> str:
    """Normalize whitespace and punctuation in product/brand/category strings."""
    if pd.isna(s) or s is None:
        return ""
    t = str(s)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[,;]+", " ", t)
    t = t.replace("«", '"').replace("»", '"').strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t
