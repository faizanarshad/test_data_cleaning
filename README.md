# Brand classification (ADG codes)

BiLSTM-based **text → ADG_CODE** classification for retail product data (Armenian / Latin), plus **evaluation**, **inference**, and an **ADG → brand / industry** lookup from the dataset.

## Project layout

```text
├── data/
│   └── brand_task.csv          # Source dataset (do not rename columns without updating loaders)
├── artifacts/                  # Generated: models, reports, caches (*.keras gitignored)
├── src/brand_classification/   # Application package
│   ├── config.py                 # Paths (project root, data, artifacts)
│   ├── preprocessing.py          # Text cleaning
│   ├── data_loader.py          # CSV → training table
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── adg_lookup.py           # Code → brand / category (empirical)
├── tests/                       # pytest smoke tests
├── pyproject.toml               # Package metadata + dependencies
├── requirements.txt             # Runtime pins (same as pyproject)
├── LICENSE                      # MIT
└── bilstm_train.py, …          # Thin wrappers (call package; work without pip install -e)
```

## Dataset (`data/brand_task.csv`)

| Column      | Description                                  |
|------------|-----------------------------------------------|
| `ADG_CODE` | Numeric label                                 |
| `GOOD_NAME`| Product name                                  |
| `BRAND`    | Brand                                         |
| `CATEGORY` | Industry / category (e.g. Beverages)         |

## Setup (client / production)

```bash
cd /path/to/test_data_cleaning
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

Optional: `pip install -r requirements-dev.txt` to run tests.

## Commands (recommended)

Run from the **repository root** after `pip install -e .`:

| Task | Command |
|------|---------|
| Train BiLSTM | `python -m brand_classification.train` |
| Evaluate | `python -m brand_classification.evaluate` |
| Predict ADG from text | `python -m brand_classification.predict -n "…" -b "…" -c "…"` |
| ADG → brand / industry | `python -m brand_classification.adg_lookup 2101` |

**Legacy scripts** (same behavior, no editable install required — they add `src/` to `PYTHONPATH`):

- `python bilstm_train.py`
- `python evaluate_bilstm.py`
- `python predict_bilstm.py`
- `python adg_to_brand_industry.py 2101`

## Artifacts (`artifacts/`)

| File | Description |
|------|-------------|
| `bilstm_best.keras`, `bilstm_final.keras` | Trained models (large; not committed) |
| `label_encoder_classes.json` | Class index → `ADG_CODE` |
| `cleaned_training_data.csv` | Frozen table for evaluation |
| `evaluation_report.txt` | Written by `evaluate` |
| `adg_brand_category_stats.json` | Cache for ADG lookup |

After clone, run **training** once to create `.keras` files locally (or copy them into `artifacts/`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## License

See [LICENSE](LICENSE) (MIT).
