"""Smoke tests: imports and path resolution."""

from pathlib import Path

from brand_classification import __version__
from brand_classification.config import ARTIFACT_DIR, DATA_CSV, PROJECT_ROOT
from brand_classification.preprocessing import (
    build_text_for_brand_model,
    clean_text,
    training_text_to_brand_model_input,
)


def test_version():
    assert __version__ == "1.0.0"


def test_project_root_is_repo():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_data_csv_path():
    assert DATA_CSV.name == "brand_task.csv"
    assert DATA_CSV.parent.name == "data"


def test_artifacts_dir():
    assert ARTIFACT_DIR.name == "artifacts"


def test_clean_text():
    assert clean_text("  a  b  ") == "a b"
    assert clean_text("") == ""


def test_build_text_for_brand_model():
    assert build_text_for_brand_model("x", "Y") == "x [CAT] Y"
    assert build_text_for_brand_model("x", "") == "x [CAT] "


def test_training_text_to_brand_model_input():
    t = "foo [BRAND] Bar [CAT] Bev"
    assert training_text_to_brand_model_input(t) == "foo [CAT] Bev"


def test_data_file_exists():
    assert Path(DATA_CSV).is_file(), "Place brand_task.csv under data/"
