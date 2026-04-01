"""
Clean brand_task.csv and train a BiLSTM classifier to predict ADG_CODE from text.

Input columns: ADG_CODE, GOOD_NAME, BRAND, CATEGORY
Text fed to the model: GOOD_NAME + BRAND + CATEGORY (supervised rows only).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(42)

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "brand_task.csv"
ARTIFACT_DIR = BASE / "bilstm_artifacts"

# Minimum rows per class so stratified split works
MIN_PER_CLASS = 2


def clean_text(s: str) -> str:
    if pd.isna(s) or s is None:
        return ""
    t = str(s)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[,;]+", " ", t)
    t = t.replace("«", '"').replace("»", '"').strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_and_clean_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load CSV and return cleaned rows with valid ADG_CODE."""
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

    # Drop rows without label
    df["ADG_CODE"] = pd.to_numeric(df["ADG_CODE"], errors="coerce")
    df = df.dropna(subset=["ADG_CODE"])
    df["ADG_CODE"] = df["ADG_CODE"].astype(int)

    # Drop empty product names
    df = df[df["GOOD_NAME"].str.len() > 0]

    # Dedupe
    df = df.drop_duplicates(subset=["GOOD_NAME", "BRAND", "CATEGORY", "ADG_CODE"])

    # Drop classes with too few samples for stratified split
    counts = df.groupby("ADG_CODE").size()
    rare = counts[counts < MIN_PER_CLASS].index
    if len(rare):
        df = df[~df["ADG_CODE"].isin(rare)]
        print(f"Dropped {len(rare)} ADG codes with <{MIN_PER_CLASS} samples (held out from training).")

    # Combined text for the model (brand/category help disambiguation)
    df["text_input"] = (
        df["GOOD_NAME"] + " [BRAND] " + df["BRAND"] + " [CAT] " + df["CATEGORY"]
    )

    after = len(df)
    print(f"Cleaning: {before} rows → {after} rows (valid ADG, non-empty name, deduped)")
    return df


def main() -> None:
    print("=" * 60)
    print("BiLSTM — data cleaning + training")
    print("=" * 60)

    df = load_and_clean_dataframe(DATA_PATH)
    texts = df["text_input"].tolist()
    y_raw = df["ADG_CODE"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int32)
    num_classes = len(le.classes_)
    print(f"Classes (unique ADG_CODE): {num_classes}")

    tr_texts, va_texts, y_train, y_val = train_test_split(
        texts,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )
    y_train = np.asarray(y_train, dtype=np.int32)
    y_val = np.asarray(y_val, dtype=np.int32)

    # tf.string tensors (Keras 3 rejects numpy unicode / object for string inputs)
    def string_matrix(rows: list[str]) -> tf.Tensor:
        flat = tf.constant(rows, dtype=tf.string)
        return tf.reshape(flat, (-1, 1))

    X_train = string_matrix([str(s) for s in tr_texts])
    X_val = string_matrix([str(s) for s in va_texts])

    vocab_size = min(20_000, max(5000, int(len(df) * 3)))
    max_len = 120

    vec = layers.TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=max_len,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )
    vec.adapt(tf.constant(tr_texts, dtype=tf.string))
    embed_vocab = int(vec.vocabulary_size())

    model = keras.Sequential(
        [
            layers.Input(shape=(1,), dtype=tf.string),
            vec,
            layers.Embedding(embed_vocab, 128, mask_zero=True),
            layers.Bidirectional(layers.LSTM(64, dropout=0.3)),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Class weights for imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(np.int32(c)): float(w) for c, w in zip(classes, weights)}

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ARTIFACT_DIR / "bilstm_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation accuracy: {val_acc:.4f}")

    # Save artifacts
    model.save(ARTIFACT_DIR / "bilstm_final.keras")
    with open(ARTIFACT_DIR / "label_encoder_classes.json", "w", encoding="utf-8") as f:
        json.dump([int(x) for x in le.classes_.tolist()], f)

    df_out = df[["ADG_CODE", "GOOD_NAME", "BRAND", "CATEGORY", "text_input"]].copy()
    df_out.to_csv(ARTIFACT_DIR / "cleaned_training_data.csv", index=False, encoding="utf-8-sig")
    print(f"Saved cleaned data and model under {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
