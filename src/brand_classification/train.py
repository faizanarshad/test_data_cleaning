"""
Train BiLSTM: text → ADG_CODE. Writes models and cleaned table under artifacts/.
"""

from __future__ import annotations

import json

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from .config import ARTIFACT_DIR, DATA_CSV
from .data_loader import load_and_clean_dataframe

tf.keras.utils.set_random_seed(42)


def main() -> None:
    """Train TextVectorization + BiLSTM; save model and frozen cleaned table."""
    print("=" * 60)
    print("BiLSTM — data cleaning + training")
    print("=" * 60)

    df = load_and_clean_dataframe(DATA_CSV)
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

    model.fit(
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

    model.save(ARTIFACT_DIR / "bilstm_final.keras")
    with open(ARTIFACT_DIR / "label_encoder_classes.json", "w", encoding="utf-8") as f:
        json.dump([int(x) for x in le.classes_.tolist()], f)

    df_out = df[["ADG_CODE", "GOOD_NAME", "BRAND", "CATEGORY", "text_input"]].copy()
    df_out.to_csv(ARTIFACT_DIR / "cleaned_training_data.csv", index=False, encoding="utf-8-sig")
    print(f"Saved cleaned data and model under {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
