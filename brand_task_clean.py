"""
Brand task data cleaning pipeline (from specification document).
Loads brand_task.csv, cleans text, extracts brands/measurements/categories, exports CSVs.
"""

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# PART 1: LOAD AND EXPLORE THE DATA
# ============================================================================


def load_and_explore_data(filepath: str = "brand_task.csv") -> pd.DataFrame:
    """Load the CSV file and perform initial exploration."""
    print("=" * 60)
    print("LOADING AND EXPLORING DATA")
    print("=" * 60)
    encodings = ["utf-8", "utf-8-sig", "cp1252", "iso-8859-1"]
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"✓ Successfully loaded with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError("Could not load file with any encoding")

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    return df


# ============================================================================
# PART 2: COMPREHENSIVE DATA CLEANING CLASS
# ============================================================================


class BrandTaskCleaner:
    """Comprehensive cleaner for brand_task.csv."""

    def __init__(self) -> None:
        self.brands = {
            "artfood": "Artfood",
            "art food": "Artfood",
            "ատրֆուդ": "Artfood",
            "nescafe": "Nescafe",
            "նեսկաֆե": "Nescafe",
            "coca-cola": "Coca-Cola",
            "coca cola": "Coca-Cola",
            "կոկա-կոլա": "Coca-Cola",
            "կոկա կոլա": "Coca-Cola",
            "royal": "Royal",
            "ռոյալ": "Royal",
            "milka": "Milka",
            "միլկա": "Milka",
            "apache": "Apache",
            "ապաչի": "Apache",
            "marianna": "Marianna",
            "մարիաննա": "Marianna",
            "grand candy": "Grand Candy",
            "գրանդ քենդի": "Grand Candy",
            "yashkino": "Yashkino",
            "յաշկինո": "Yashkino",
            "daroink": "Daroink",
            "դարոինք": "Daroink",
            "դարոյնք": "Daroink",
            "roshen": "Roshen",
            "ռոշեն": "Roshen",
            "ritter sport": "Ritter Sport",
            "ganjasar": "Ganjasar",
            "գանձասար": "Ganjasar",
            "athenk": "Athenk",
            "athens": "Athens",
            "աթենք": "Athenk",
            "art armenia": "Art Armenia",
            "արտ արմենիա": "Art Armenia",
            "armenia wine": "Armenia Wine",
            "armenia": "Armenia",
            "nivea": "Nivea",
            "նիվեա": "Nivea",
            "colgate": "Colgate",
            "քոլգեյթ": "Colgate",
            "կոլգեյթ": "Colgate",
            "garnier": "Garnier",
            "գարնիեր": "Garnier",
            "gillette": "Gillette",
            "ժիլետ": "Gillette",
            "silky soft": "Silky Soft",
            "silk soft": "Silk Soft",
            "սիլկ սոֆթ": "Silk Soft",
            "salve": "Salve",
            "սալվե": "Salve",
            "savex": "Savax",
            "սավեքս": "Savax",
            "pampers": "Pampers",
            "պամպերս": "Pampers",
            "alex": "Alex",
            "ալեքս": "Alex",
            "finder": "Finder",
            "samsung": "Samsung",
            "սամսունգ": "Samsung",
            "mercedes-benz": "Mercedes-Benz",
            "mercedes benz": "Mercedes-Benz",
            "mercedes": "Mercedes-Benz",
            "zara": "Zara",
            "զառա": "Zara",
        }

        self.categories = {
            "Beverages": [
                "գազավորված",
                "ըմպելիք",
                "coca-cola",
                "կոկա-կոլա",
                "հյութ",
                "ջուր",
                "թեյ",
                "սուրճ",
                "գինի",
                "կոմպոտ",
                "նեսկաֆե",
                "կոկա",
            ],
            "Food Products": [
                "պանիր",
                "երշիկ",
                "նրբերշիկ",
                "կոնֆետ",
                "շոկոլադ",
                "վաֆլի",
                "թխվածք",
                "կետչուպ",
                "կաթնաշոռ",
                "պաղպաղակ",
                "կեֆիր",
                "մածուն",
                "կարագ",
                "յոգուրտ",
                "կաթ",
                "պահածո",
                "մուրաբա",
                "խավիար",
                "կոտլետ",
                "խինկալի",
                "պելմենի",
                "բաստուրմա",
            ],
            "Clothing": [
                "շապիկ",
                "գուլպա",
                "տաբատ",
                "բոդի",
                "լողազգեստ",
                "տրիկոտաժե",
                "սպորտ",
                "ժիլետ",
                "վարտիք",
                "զուգագուլպա",
                "կոշիկ",
                "շորտ",
                "բաճկոն",
                "ջեմպեր",
                "սվիտեր",
                "գլխարկ",
                "պայուսակ",
                "տղ",
                "կանացի",
                "մանկական",
            ],
            "Cosmetics": [
                "ատամի մածուկ",
                "շամպուն",
                "ներկ",
                "գել",
                "կրեմ",
                "լոգանքի",
                "հոտազերծիչ",
                "դեզոդորանտ",
                "սափրվելու",
                "լոսյոն",
                "բալզամ",
                "դիմակ",
                "միցելային",
                "օճառ",
                "ածելի",
                "սափրիչ",
            ],
            "Household": [
                "տակդիր",
                "անձեռոցիկ",
                "լվացքի",
                "սպասք",
                "փոշի",
                "մաքրող",
                "սրբիչ",
                "զուգարանի թուղթ",
                "սկոչ",
                "կախիչ",
                "գորգ",
            ],
            "Electronics": [
                "հեռախոս",
                "samsung",
                "հեռուստացույց",
                "մոնիտոր",
                "պլանշետ",
                "նոութբուք",
                "սմարթֆոն",
                "համակարգիչ",
                "փոշեկուլ",
                "սառնարան",
                "լվացքի մեքենա",
                "օդակարգավորիչ",
            ],
            "Auto Parts": [
                "mercedes-benz",
                "mercedes",
                "վազ",
                "прокладка",
                "ավտոմեքենա",
                "շարժիչ",
                "արգելակ",
                "կոճղակ",
                "ռետինե",
                "ֆիլտր",
                "պոմպ",
                "մարդատար",
                "ավտո",
                "գեյզեր",
            ],
        }

    def clean_text(self, text: Any) -> str:
        """Advanced text cleaning for Armenian text."""
        if pd.isna(text) or text is None:
            return ""
        text = str(text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n+", " ", text)
        text = text.replace("«", '"').replace("»", '"')
        text = text.replace("՛՛", '"').replace("՛", "'")
        text = text.replace("`", "'").replace("´", "'")
        text = text.replace("…", "...")
        text = re.sub(r"[,;:]+\s*", " ", text)
        text = re.sub(r"[!?]+", "", text)
        text = re.sub(r"[^\w\s\u0530-\u058F\uFB00-\uFB06/-]", " ", text)
        text = text.replace("՞", "?").replace("՜", "!")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_measurements(self, text: str) -> Dict[str, Any]:
        """Extract product measurements."""
        text = str(text).lower()
        measurements: Dict[str, Any] = {
            "weight_g": None,
            "weight_kg": None,
            "volume_ml": None,
            "volume_l": None,
            "count": None,
            "size": None,
        }
        weight_patterns = [
            (r"(\d+(?:[.,]\d+)?)\s*(գ|գր|գրամ|g|gr)\b", "g"),
            (r"(\d+(?:[.,]\d+)?)\s*(կգ|kg)\b", "kg"),
            (r"(\d+(?:[.,]\d+)?)\s*(գր\.|գ\.)", "g"),
        ]
        for pattern, unit in weight_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1).replace(",", "."))
                if unit == "kg":
                    measurements["weight_kg"] = value
                    measurements["weight_g"] = value * 1000
                else:
                    measurements["weight_g"] = value
                break

        volume_patterns = [
            (r"(\d+(?:[.,]\d+)?)\s*(մլ|ml)\b", "ml"),
            (r"(\d+(?:[.,]\d+)?)\s*(լ|l)\b", "l"),
            (r"(\d+(?:[.,]\d+)?)\s*(լ\.|l\.)", "l"),
        ]
        for pattern, unit in volume_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1).replace(",", "."))
                if unit == "l":
                    measurements["volume_l"] = value
                    measurements["volume_ml"] = value * 1000
                else:
                    measurements["volume_ml"] = value
                break

        count_patterns = [
            r"(\d+)\s*(հատ|հ|шт|pcs|pc|հատ\.)",
            r"(\d+)\s*հատ\b",
        ]
        for pattern in count_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                measurements["count"] = int(match.group(1))
                break

        size_match = re.search(r"\b(S|M|L|XL|XXL|XXXL|XS|XXS)\b", text, re.IGNORECASE)
        if size_match:
            measurements["size"] = size_match.group(1).upper()
        return measurements

    def extract_brand(self, text: str, confidence_threshold: float = 0.6) -> Tuple[str, float]:
        """Extract brand with confidence score."""
        del confidence_threshold  # reserved for future use
        text_lower = str(text).lower()
        best_match = "Unknown"
        best_score = 0.0
        for brand_key, brand_name in self.brands.items():
            if brand_key in text_lower:
                pos = text_lower.find(brand_key)
                score = 1.0
                if pos > len(text_lower) * 0.7:
                    score *= 0.8
                if pos == 0:
                    score *= 1.2
                if score > best_score:
                    best_score = score
                    best_match = brand_name
        return best_match, min(best_score, 1.0)

    def categorize_product(self, text: str) -> str:
        """Categorize product with improved accuracy."""
        text_lower = str(text).lower()
        category_scores = {cat: 0 for cat in self.categories}
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    score = len(keyword) / 10
                    if text_lower.find(keyword) == 0:
                        score *= 1.5
                    category_scores[category] += score
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        if any(size in text_lower for size in ["s", "m", "l", "xl", "xxl"]):
            return "Clothing"
        if re.search(r"\d+\s*(գ|կգ|մլ|լ)", text_lower):
            return "Food Products" if "food" in text_lower else "Other"
        return "Other"

    def clean_adg_code(self, code: Any) -> int:
        """Clean ADG_CODE values."""
        if pd.isna(code) or code is None or code == "":
            return -1
        try:
            if isinstance(code, (int, float)):
                return int(code)
            code_str = str(code).strip()
            if code_str == "0":
                return 0
            numeric_match = re.search(r"(\d+)", code_str)
            if numeric_match:
                return int(numeric_match.group(1))
            return -1
        except (ValueError, TypeError):
            return -1

    def extract_price_or_quantity(self, text: str) -> Dict[str, Any]:
        """Extract price or quantity information."""
        text = str(text).lower()
        result: Dict[str, Any] = {"price_amd": None, "quantity_kg": None}
        price_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(դր|amd|դրամ)", text)
        if price_match:
            result["price_amd"] = float(price_match.group(1).replace(",", "."))
        qty_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(կգ|kg|տոննա)", text)
        if qty_match:
            result["quantity_kg"] = float(qty_match.group(1).replace(",", "."))
        return result

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete data processing pipeline."""
        print("\n" + "=" * 60)
        print("DATA CLEANING PIPELINE")
        print("=" * 60)

        df_clean = df.copy()

        print("\n1. Cleaning ADG_CODE...")
        df_clean["ADG_CODE_CLEANED"] = df_clean["ADG_CODE"].apply(self.clean_adg_code)
        missing_count = (df_clean["ADG_CODE_CLEANED"] == -1).sum()
        print(
            f"   - Missing/Invalid ADG codes: {missing_count} ({missing_count / len(df_clean) * 100:.1f}%)"
        )

        print("\n2. Cleaning product names...")
        df_clean["GOOD_NAME_CLEANED"] = df_clean["GOOD_NAME"].apply(self.clean_text)

        print("\n3. Extracting measurements...")
        measurements = df_clean["GOOD_NAME_CLEANED"].apply(self.extract_measurements)
        measurements_df = pd.DataFrame(measurements.tolist())
        df_clean = pd.concat([df_clean, measurements_df], axis=1)

        print("\n4. Extracting brands...")
        brand_info = df_clean["GOOD_NAME_CLEANED"].apply(lambda x: self.extract_brand(x))
        df_clean["BRAND"] = [b[0] for b in brand_info]
        df_clean["BRAND_CONFIDENCE"] = [b[1] for b in brand_info]

        print("\n5. Categorizing products...")
        df_clean["CATEGORY"] = df_clean["GOOD_NAME_CLEANED"].apply(self.categorize_product)

        print("\n6. Creating normalized names...")

        def normalize_name(text: str) -> str:
            t = str(text)
            t = re.sub(
                r"\d+(?:[.,]\d+)?\s*(գ|կգ|գր|գրամ|մլ|լ|ml|l|g|kg|gr|հատ|հ|шт|pcs)\b",
                "",
                t,
                flags=re.IGNORECASE,
            )
            brand_alt = "|".join(re.escape(k) for k in self.brands.keys())
            if brand_alt:
                t = re.sub(rf"\b({brand_alt})\b", "", t, flags=re.IGNORECASE)
            t = re.sub(r"\s+", " ", t)
            return t.strip()

        df_clean["GOOD_NAME_NORMALIZED"] = df_clean["GOOD_NAME_CLEANED"].apply(normalize_name)

        print("\n7. Extracting additional info...")
        additional_info = df_clean["GOOD_NAME_CLEANED"].apply(self.extract_price_or_quantity)
        additional_df = pd.DataFrame(additional_info.tolist())
        df_clean = pd.concat([df_clean, additional_df], axis=1)

        print("\n8. Adding metadata...")
        df_clean["IS_MISSING_BRAND"] = (df_clean["BRAND"] == "Unknown").astype(int)
        df_clean["HAS_MEASUREMENT"] = (
            ~df_clean["weight_g"].isna()
            | ~df_clean["volume_ml"].isna()
            | ~df_clean["count"].isna()
        ).astype(int)
        df_clean["NAME_LENGTH"] = df_clean["GOOD_NAME_CLEANED"].str.len()
        df_clean["WORD_COUNT"] = df_clean["GOOD_NAME_CLEANED"].str.split().str.len()

        measurement_cols = ["weight_g", "weight_kg", "volume_ml", "volume_l", "count", "size"]
        for col in measurement_cols:
            df_clean[col] = df_clean[col].fillna(0)
        df_clean["price_amd"] = df_clean["price_amd"].fillna(0)
        df_clean["quantity_kg"] = df_clean["quantity_kg"].fillna(0)

        df_clean["PRODUCT_ID"] = range(1, len(df_clean) + 1)
        df_clean = df_clean.sort_values("ADG_CODE_CLEANED").reset_index(drop=True)

        print("\n" + "=" * 60)
        print("CLEANING SUMMARY")
        print("=" * 60)
        print(f"✓ Total records processed: {len(df_clean):,}")
        print(f"✓ Valid ADG codes: {(df_clean['ADG_CODE_CLEANED'] != -1).sum():,}")
        print(f"✓ Unique brands identified: {df_clean['BRAND'].nunique()}")
        print(f"✓ High confidence brands (>80%): {(df_clean['BRAND_CONFIDENCE'] > 0.8).sum():,}")
        print(f"✓ Products with measurements: {df_clean['HAS_MEASUREMENT'].sum():,}")
        print("\nCategory Distribution:")
        print(df_clean["CATEGORY"].value_counts())
        print("\nTop 10 Brands:")
        print(df_clean["BRAND"].value_counts().head(10))
        return df_clean


# ============================================================================
# PART 3: EXPORT CLEANED DATA
# ============================================================================


def export_cleaned_data(df_clean: pd.DataFrame, output_prefix: str = "brand_task_cleaned") -> Dict[str, str]:
    """Export cleaned data in multiple formats."""
    print("\n" + "=" * 60)
    print("EXPORTING CLEANED DATA")
    print("=" * 60)

    full_file = f"{output_prefix}_full.csv"
    df_clean.to_csv(full_file, index=False, encoding="utf-8-sig")
    print(f"✓ Full dataset saved: {full_file}")

    ml_columns = [
        "GOOD_NAME_CLEANED",
        "GOOD_NAME_NORMALIZED",
        "ADG_CODE_CLEANED",
        "BRAND",
        "CATEGORY",
        "weight_g",
        "volume_ml",
        "count",
    ]
    df_ml = df_clean[ml_columns]
    ml_file = f"{output_prefix}_ml_ready.csv"
    df_ml.to_csv(ml_file, index=False, encoding="utf-8-sig")
    print(f"✓ ML-ready dataset saved: {ml_file}")

    brand_summary = (
        df_clean.groupby("BRAND")
        .agg(
            {
                "PRODUCT_ID": "count",
                "ADG_CODE_CLEANED": "nunique",
                "BRAND_CONFIDENCE": "mean",
            }
        )
        .round(2)
    )
    brand_summary.columns = ["Product_Count", "Unique_ADG_Codes", "Avg_Confidence"]
    brand_summary = brand_summary.sort_values("Product_Count", ascending=False)
    brand_file = f"{output_prefix}_brand_summary.csv"
    brand_summary.to_csv(brand_file, encoding="utf-8-sig")
    print(f"✓ Brand summary saved: {brand_file}")

    category_summary = (
        df_clean.groupby("CATEGORY")
        .agg(
            {
                "PRODUCT_ID": "count",
                "BRAND": "nunique",
            }
        )
        .round(2)
    )
    category_summary.columns = ["Product_Count", "Unique_Brands"]
    category_summary = category_summary.sort_values("Product_Count", ascending=False)
    cat_file = f"{output_prefix}_category_summary.csv"
    category_summary.to_csv(cat_file, encoding="utf-8-sig")
    print(f"✓ Category summary saved: {cat_file}")

    return {
        "full": full_file,
        "ml_ready": ml_file,
        "brand_summary": brand_file,
        "category_summary": cat_file,
    }


# ============================================================================
# PART 4: MAIN EXECUTION
# ============================================================================


def main() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Main execution function."""
    print("\n" + "=" * 60)
    print("BRAND TASK DATA CLEANING PIPELINE")
    print("=" * 60)

    base = Path(__file__).resolve().parent
    csv_path = base / "brand_task.csv"
    df = load_and_explore_data(str(csv_path))

    cleaner = BrandTaskCleaner()
    df_cleaned = cleaner.process_dataframe(df)
    output_files = export_cleaned_data(df_cleaned, output_prefix=str(base / "brand_task_cleaned"))

    print("\n" + "=" * 60)
    print("SAMPLE CLEANED DATA (First 10 rows)")
    print("=" * 60)
    display_cols = [
        "GOOD_NAME_CLEANED",
        "BRAND",
        "CATEGORY",
        "ADG_CODE_CLEANED",
        "weight_g",
        "volume_ml",
        "BRAND_CONFIDENCE",
    ]
    print(df_cleaned[display_cols].head(10).to_string())

    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Total records: {len(df_cleaned):,}")
    print(
        f"Missing brands: {df_cleaned['IS_MISSING_BRAND'].sum():,} ({df_cleaned['IS_MISSING_BRAND'].mean() * 100:.1f}%)"
    )
    print(
        f"Products with measurements: {df_cleaned['HAS_MEASUREMENT'].sum():,} ({df_cleaned['HAS_MEASUREMENT'].mean() * 100:.1f}%)"
    )
    print(f"Average name length: {df_cleaned['NAME_LENGTH'].mean():.1f} characters")
    print(f"Unique categories: {df_cleaned['CATEGORY'].nunique()}")
    print(f"Unique brands: {df_cleaned['BRAND'].nunique()}")
    print("\n✅ Data cleaning complete! Ready for ML model training.")
    return df_cleaned, output_files


if __name__ == "__main__":
    try:
        df_cleaned, output_files = main()
    except FileNotFoundError:
        print("\n❌ Error: brand_task.csv not found!")
        print("Please make sure the file is in the same directory as this script.")
