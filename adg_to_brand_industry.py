"""
Given an ADG_CODE, infer the most likely BRAND and industry (CATEGORY).

Uses the training CSV: for each code we take the majority / frequency-weighted
brand and category. This is a data-driven lookup, not a neural model (a code ID
alone has no text semantics).

Cache: bilstm_artifacts/adg_brand_category_stats.json (--rebuild-cache after CSV edits)
Run: python adg_to_brand_industry.py <CODE>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "brand_task.csv"
CACHE_PATH = BASE / "bilstm_artifacts" / "adg_brand_category_stats.json"


def load_dataframe() -> pd.DataFrame:
    """Read brand_task.csv with tolerant encoding; coerce ADG_CODE; fill brand/category."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(DATA_PATH, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise FileNotFoundError(DATA_PATH)

    df["ADG_CODE"] = pd.to_numeric(df["ADG_CODE"], errors="coerce")
    df = df.dropna(subset=["ADG_CODE"])
    df["ADG_CODE"] = df["ADG_CODE"].astype(int)
    df["BRAND"] = df["BRAND"].fillna("Unknown").astype(str).str.strip()
    df["CATEGORY"] = df["CATEGORY"].fillna("Other").astype(str).str.strip()
    return df


def build_stats(df: pd.DataFrame) -> dict:
    """Build nested dict: ADG_CODE string key -> row counts per brand and per category."""
    out: dict[str, dict] = {}
    for code, g in df.groupby("ADG_CODE"):
        key = str(int(code))
        bc = g["BRAND"].value_counts()
        cc = g["CATEGORY"].value_counts()
        n = len(g)
        out[key] = {
            "n_rows": int(n),
            "brands": {str(k): int(v) for k, v in bc.items()},
            "categories": {str(k): int(v) for k, v in cc.items()},
        }
    return out


def predict_brand_industry(stats: dict, code: int, top_k: int = 3) -> dict:
    """Return majority brand/category with confidence = count/total for that code."""
    key = str(int(code))
    if key not in stats:
        return {
            "adg_code": int(code),
            "found": False,
            "message": "This ADG_CODE does not appear in the dataset.",
        }

    s = stats[key]
    n = s["n_rows"]

    def top_entries(counts: dict, k: int) -> list[dict]:
        """Sort label counts descending; attach confidence = count / n_rows."""
        items = sorted(counts.items(), key=lambda x: -x[1])[:k]
        return [
            {
                "value": name,
                "count": c,
                "confidence": round(c / n, 4),
            }
            for name, c in items
        ]

    brands = top_entries(s["brands"], top_k)
    cats = top_entries(s["categories"], top_k)

    return {
        "adg_code": int(code),
        "found": True,
        "rows_in_dataset": n,
        "brand": {
            "predicted": brands[0]["value"],
            "confidence": brands[0]["confidence"],
            "top": brands,
        },
        "industry": {
            "predicted": cats[0]["value"],
            "confidence": cats[0]["confidence"],
            "top": cats,
        },
    }


def parse_code(s: str) -> int:
    """Extract first integer from CLI string (allows spaces or extra text)."""
    s = str(s).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"Not a valid code: {s}")
    return int(m.group(1))


def main() -> None:
    """Load or rebuild JSON cache; print predictions or usage."""
    parser = argparse.ArgumentParser(
        description="Given ADG_CODE, predict brand + industry (category) from data."
    )
    parser.add_argument(
        "code",
        nargs="?",
        help="ADG code (e.g. 2101)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help=f"Rebuild {CACHE_PATH} from {DATA_PATH}",
    )
    args = parser.parse_args()

    # Build JSON histogram cache from CSV if missing or user requested rebuild.
    if args.rebuild_cache or not CACHE_PATH.is_file():
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = load_dataframe()
        stats = build_stats(df)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=0)
        print(f"Wrote cache: {CACHE_PATH} ({len(stats)} codes)")

    with open(CACHE_PATH, encoding="utf-8") as f:
        stats: dict = json.load(f)

    if args.code is None:
        if not args.rebuild_cache:
            print("Usage: python adg_to_brand_industry.py <ADG_CODE>")
            print("Example: python adg_to_brand_industry.py 2101")
        print(f"Cache: {CACHE_PATH}")
        return

    code = parse_code(args.code)
    result = predict_brand_industry(stats, code)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not result["found"]:
        print(result["message"])
        return

    print("=" * 60)
    print(f"ADG_CODE: {result['adg_code']}")
    print(f"Rows in dataset for this code: {result['rows_in_dataset']}")
    print()
    print("Predicted BRAND (majority in data):")
    print(f"  → {result['brand']['predicted']}")
    print(f"  Confidence: {result['brand']['confidence']:.1%} (share of rows with this brand)")
    print("  Other brands:")
    for b in result["brand"]["top"][1:]:
        print(f"    - {b['value']}: {b['confidence']:.1%}")
    print()
    print("Predicted INDUSTRY (category in data):")
    print(f"  → {result['industry']['predicted']}")
    print(f"  Confidence: {result['industry']['confidence']:.1%}")
    print("  Other categories:")
    for c in result["industry"]["top"][1:]:
        print(f"    - {c['value']}: {c['confidence']:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
