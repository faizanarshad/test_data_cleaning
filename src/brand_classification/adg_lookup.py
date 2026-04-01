"""ADG_CODE → majority brand and industry (category) from empirical CSV counts."""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from .config import ARTIFACT_DIR, DATA_CSV

CACHE_PATH = ARTIFACT_DIR / "adg_brand_category_stats.json"


def load_dataframe() -> pd.DataFrame:
    """Read dataset with tolerant encoding."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(DATA_CSV, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise FileNotFoundError(DATA_CSV)

    df["ADG_CODE"] = pd.to_numeric(df["ADG_CODE"], errors="coerce")
    df = df.dropna(subset=["ADG_CODE"])
    df["ADG_CODE"] = df["ADG_CODE"].astype(int)
    df["BRAND"] = df["BRAND"].fillna("Unknown").astype(str).str.strip()
    df["CATEGORY"] = df["CATEGORY"].fillna("Other").astype(str).str.strip()
    return df


def build_stats(df: pd.DataFrame) -> dict:
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
    s = str(s).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"Not a valid code: {s}")
    return int(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Given ADG_CODE, show majority brand + category from data."
    )
    parser.add_argument("code", nargs="?", help="ADG code (e.g. 2101)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild cache from data CSV",
    )
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild_cache or not CACHE_PATH.is_file():
        df = load_dataframe()
        stats = build_stats(df)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=0)
        print(f"Wrote cache: {CACHE_PATH} ({len(stats)} codes)")

    with open(CACHE_PATH, encoding="utf-8") as f:
        stats: dict = json.load(f)

    if args.code is None:
        if not args.rebuild_cache:
            print("Usage: python -m brand_classification.adg_lookup <ADG_CODE>")
            print("Example: python -m brand_classification.adg_lookup 2101")
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
    print(f"  Confidence: {result['brand']['confidence']:.1%}")
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
