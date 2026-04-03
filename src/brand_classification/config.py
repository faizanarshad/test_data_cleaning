"""
Central paths for the repository.

``brand_classification/config.py`` → ``src`` → project root.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "brand_task.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_BRAND_PATH = ARTIFACT_DIR / "bilstm_brand_best.keras"
BRAND_CLASSES_PATH = ARTIFACT_DIR / "brand_label_encoder_classes.json"
