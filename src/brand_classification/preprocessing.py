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


def build_text_for_brand_model(good_name: str, category: str = "") -> str:
    """Input for brand classifier: name + category only (no brand label)."""
    gn = clean_text(good_name)
    cat = clean_text(category) if category else ""
    return f"{gn} [CAT] {cat}"


def training_text_to_brand_model_input(text: str) -> str:
    """Strip ``[BRAND] ... [CAT]`` from training-format string for brand prediction."""
    if "[BRAND]" in text and "[CAT]" in text:
        before, rest = text.split("[BRAND]", 1)
        _mid, after_cat = rest.split("[CAT]", 1)
        return build_text_for_brand_model(before.strip(), after_cat.strip())
    return build_text_for_brand_model(clean_text(text), "")
