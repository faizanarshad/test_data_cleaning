"""CLI: predict ADG_CODE from product text using the saved BiLSTM."""

from __future__ import annotations

import argparse
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import ARTIFACT_DIR
from .preprocessing import clean_text

MODEL_PATH = ARTIFACT_DIR / "bilstm_best.keras"
CLASSES_PATH = ARTIFACT_DIR / "label_encoder_classes.json"


def build_text_input(good_name: str, brand: str = "", category: str = "") -> str:
    """Training-aligned string with [BRAND] / [CAT] markers."""
    gn = clean_text(good_name)
    br = clean_text(brand) if brand else ""
    cat = clean_text(category) if category else ""
    return f"{gn} [BRAND] {br} [CAT] {cat}"


def string_matrix(rows: list[str]) -> tf.Tensor:
    flat = tf.constant(rows, dtype=tf.string)
    return tf.reshape(flat, (-1, 1))


def load_model_and_classes():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Train first: python -m brand_classification.train"
        )
    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = [int(x) for x in json.load(f)]
    model = keras.models.load_model(MODEL_PATH)
    return model, classes


def predict_adg(
    model: keras.Model,
    classes: list[int],
    text_inputs: list[str],
    top_k: int = 5,
) -> list[dict]:
    X = string_matrix([str(t) for t in text_inputs])
    probs = model.predict(X, batch_size=32, verbose=0)
    results = []
    for row in probs:
        idx = np.argsort(-row)[:top_k]
        results.append(
            {
                "top_predictions": [
                    {
                        "adg_code": int(classes[i]),
                        "probability": float(row[i]),
                    }
                    for i in idx
                ]
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ADG_CODE from product text.")
    parser.add_argument("-n", "--name", type=str, help="Product name (GOOD_NAME)")
    parser.add_argument("-b", "--brand", type=str, default="", help="Brand")
    parser.add_argument("-c", "--category", type=str, default="", help="Category")
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        help="Full text_input as used in training",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top ADG codes to show")
    args = parser.parse_args()

    model, classes = load_model_and_classes()

    if args.text:
        texts = [args.text]
    elif args.name:
        texts = [build_text_input(args.name, args.brand, args.category)]
    else:
        demos = [
            build_text_input("Նեսկաֆե գոլդ 75գ", "Nescafe", "Beverages"),
            build_text_input("Կոմպոտ Արտֆուդ սերկևիլ ա/տ 1լ", "Artfood", "Beverages"),
            build_text_input(
                "LED Հեռուստացույց SAMSUNG UE65CU8000UXRU",
                "Samsung",
                "Electronics",
            ),
        ]
        print("No --name or --text given; running demo predictions.\n")
        texts = demos

    out = predict_adg(model, classes, texts, top_k=args.top_k)

    for i, (txt, pred) in enumerate(zip(texts, out)):
        print("=" * 60)
        print(f"Input {i + 1}:")
        print(txt[:500] + ("..." if len(txt) > 500 else ""))
        print("\nTop predictions:")
        for j, p in enumerate(pred["top_predictions"], 1):
            print(f"  {j}. ADG_CODE {p['adg_code']}: {p['probability']:.4f}")
        print()


if __name__ == "__main__":
    main()
