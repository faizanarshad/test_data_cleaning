"""
Backward-compatible launcher. Preferred: ``python -m brand_classification.train``
(from repo root, after ``pip install -e .``).
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from brand_classification.train import main  # noqa: E402

if __name__ == "__main__":
    main()
