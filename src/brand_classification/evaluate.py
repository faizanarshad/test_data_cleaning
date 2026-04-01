"""Evaluate BiLSTM on the same validation split as training (seed=42)."""

from __future__ import annotations

import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras

import pandas as pd

from .config import ARTIFACT_DIR

MODEL_PATH = ARTIFACT_DIR / "bilstm_best.keras"
CLASSES_PATH = ARTIFACT_DIR / "label_encoder_classes.json"


def string_matrix_tf(rows: list[str]) -> tf.Tensor:
    """Batch (N, 1) tf.string for the Keras model."""
    flat = tf.constant(rows, dtype=tf.string)
    return tf.reshape(flat, (-1, 1))


def load_cleaned_frame() -> pd.DataFrame:
    """Load frozen training export from artifacts/."""
    frozen = ARTIFACT_DIR / "cleaned_training_data.csv"
    if frozen.is_file():
        return pd.read_csv(frozen, encoding="utf-8-sig")
    print("cleaned_training_data.csv missing — run training first.")
    raise FileNotFoundError(frozen)


def main() -> None:
    """Re-split data like training; print metrics; write evaluation_report.txt."""
    print("=" * 60)
    print("BiLSTM evaluation")
    print("=" * 60)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    with open(CLASSES_PATH, encoding="utf-8") as f:
        class_list = [int(x) for x in json.load(f)]

    code_to_idx = {c: i for i, c in enumerate(class_list)}

    df = load_cleaned_frame()
    texts = df["text_input"].tolist()
    y = np.array([code_to_idx[int(a)] for a in df["ADG_CODE"].values], dtype=np.int32)

    tr_texts, va_texts, y_train, y_val = train_test_split(
        texts,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    model = keras.models.load_model(MODEL_PATH)

    X_val = string_matrix_tf([str(s) for s in va_texts])
    y_val_np = np.asarray(y_val, dtype=np.int32)

    probs = model.predict(X_val, batch_size=32, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_val_np, y_pred)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_val_np, y_pred, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_val_np, y_pred, average="weighted", zero_division=0
    )

    try:
        top3 = top_k_accuracy_score(y_val_np, probs, k=min(3, probs.shape[1]))
    except (ValueError, TypeError):
        top3 = float("nan")
    if np.isnan(top3):
        top3 = float(
            np.mean(
                [
                    y_val_np[i] in np.argsort(-probs[i])[:3]
                    for i in range(len(y_val_np))
                ]
            )
        )

    print("\n--- Dataset (same split as training) ---")
    print(f"Validation samples: {len(y_val_np)}")
    print(f"Number of classes: {len(class_list)}")

    print("\n--- Performance scores (validation set) ---")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Precision (macro):  {prec_m:.4f}")
    print(f"Recall (macro):     {rec_m:.4f}")
    print(f"F1 score (macro):   {f1_m:.4f}")
    print(f"Precision (weighted): {prec_w:.4f}")
    print(f"Recall (weighted):    {rec_w:.4f}")
    print(f"F1 score (weighted):  {f1_w:.4f}")
    print(f"Top-3 accuracy:       {top3:.4f}")

    X_train = string_matrix_tf([str(s) for s in tr_texts])
    probs_tr = model.predict(X_train, batch_size=32, verbose=0)
    y_pred_tr = np.argmax(probs_tr, axis=1)
    y_train_np = np.asarray(y_train, dtype=np.int32)
    acc_tr = accuracy_score(y_train_np, y_pred_tr)
    print("\n--- Training set (same model, for reference) ---")
    print(f"Accuracy:           {acc_tr:.4f}")
    print("(Large gap vs val may indicate overfitting.)")

    print("\n--- Per-class report (validation, truncated console) ---")
    target_names = [str(c) for c in class_list]
    report = classification_report(
        y_val_np,
        y_pred,
        labels=np.arange(len(class_list)),
        target_names=target_names,
        zero_division=0,
    )
    lines = report.split("\n")
    print("\n".join(lines[:28]))
    if len(lines) > 28:
        print("  ... (truncated)")

    report_path = ARTIFACT_DIR / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BiLSTM validation metrics\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"F1 macro: {f1_m:.6f}\n")
        f.write(f"F1 weighted: {f1_w:.6f}\n")
        f.write(f"Top-3 accuracy: {top3}\n\n")
        f.write(report)
    print(f"\nFull classification report saved to: {report_path}")


if __name__ == "__main__":
    main()
