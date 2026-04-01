# Brand classification (ADG codes)

BiLSTM-based **text → ADG_CODE** classification for retail product data (Armenian / Latin), plus **evaluation**, **inference**, and an **ADG → brand / industry** lookup from the dataset.

**New to the project?** Start with the beginner-friendly **[Client guide](docs/CLIENT_GUIDE.md)** (purpose, folders, workflow, commands, troubleshooting).

## Project layout

```text
├── data/
│   └── brand_task.csv          # Source dataset
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
├── docs/
│   └── CLIENT_GUIDE.md          # Step-by-step guide for non-experts
├── pyproject.toml               # Dependencies and optional dev extras
└── LICENSE                      # MIT
```

## Dataset (`data/brand_task.csv`)

| Column      | Description                                  |
|------------|-----------------------------------------------|
| `ADG_CODE` | Numeric label                                 |
| `GOOD_NAME`| Product name                                  |
| `BRAND`    | Brand                                         |
| `CATEGORY` | Industry / category (e.g. Beverages)         |

## Setup

```bash
cd /path/to/test_data_cleaning
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

Install test dependencies (pytest):

```bash
pip install -e ".[dev]"
```

## Commands

Run from the **repository root** after `pip install -e .`:

| Task | Command |
|------|---------|
| Train BiLSTM | `python -m brand_classification.train` |
| Evaluate | `python -m brand_classification.evaluate` |
| Predict ADG from text | `python -m brand_classification.predict -n "…" -b "…" -c "…"` |
| ADG → brand / industry | `python -m brand_classification.adg_lookup 2101` |

## Evaluation results (BiLSTM)

Metrics are computed on the **validation split** (15% holdout, `random_state=42`, stratified), matching the split used during training. **145** ADG classes after filtering; **282** validation rows.

| Metric | Value |
|--------|------:|
| Accuracy | 0.4965 |
| Precision (macro) | 0.2610 |
| Recall (macro) | 0.3223 |
| F1 score (macro) | 0.2684 |
| Precision (weighted) | 0.5019 |
| Recall (weighted) | 0.4965 |
| F1 score (weighted) | 0.4770 |
| Top-3 accuracy | 0.6631 |
| Training-set accuracy (same model, reference) | 0.7439 |

Macro averages are low on rare classes; weighted F1 (~0.48) reflects class frequency better. The train/validation gap suggests **overfitting** or a difficult multi-class problem with limited samples per code.

The full **per-class** precision/recall/F1 table is written to [`artifacts/evaluation_report.txt`](artifacts/evaluation_report.txt) when you run `python -m brand_classification.evaluate` (regenerates the file).

## Prediction examples (text → ADG_CODE)

Top-5 softmax probabilities from `bilstm_best.keras` (same model as evaluation). Input format:  
`GOOD_NAME [BRAND] … [CAT] …`.

### Example A — Nescafe / Beverages

Input: `Նեսկաֆե գոլդ 75գ [BRAND] Nescafe [CAT] Beverages`

| Rank | ADG_CODE | Probability |
|------|----------|------------:|
| 1 | 2101 | 0.3242 |
| 2 | 901 | 0.2894 |
| 3 | 101 | 0.0415 |
| 4 | 2009 | 0.0328 |
| 5 | 1901 | 0.0273 |

### Example B — Artfood / Beverages

Input: `Կոմպոտ Արտֆուդ սերկևիլ ա/տ 1լ [BRAND] Artfood [CAT] Beverages`

| Rank | ADG_CODE | Probability |
|------|----------|------------:|
| 1 | 2103 | 0.1318 |
| 2 | 2009 | 0.1289 |
| 3 | 2005 | 0.0867 |
| 4 | 711 | 0.0577 |
| 5 | 2001 | 0.0454 |

### Example C — Samsung / Electronics

Input: `LED Հեռուստացույց SAMSUNG UE65CU8000UXRU [BRAND] Samsung [CAT] Electronics`

| Rank | ADG_CODE | Probability |
|------|----------|------------:|
| 1 | 8517 | 0.0732 |
| 2 | 8508 | 0.0492 |
| 3 | 8450 | 0.0488 |
| 4 | 8418 | 0.0407 |
| 5 | 8471 | 0.0385 |

Reproduce:

```bash
python -m brand_classification.predict -n "Նեսկաֆե գոլդ 75գ" -b "Nescafe" -c "Beverages" --top-k 5
```

## ADG lookups (code → brand & industry)

These use **majority counts** in `data/brand_task.csv`, not the neural network.

### ADG_CODE 2101

| Field | Value | Confidence |
|-------|--------|------------:|
| Brand | Nescafe | 59.3% |
| Industry | Beverages | 70.4% |

### ADG_CODE 2009

| Field | Value | Confidence |
|-------|--------|------------:|
| Brand | Artfood (tied with Coca-Cola, 4 rows each) | 33.3% each |
| Industry | Beverages | 83.3% |

Reproduce:

```bash
python -m brand_classification.adg_lookup 2101
python -m brand_classification.adg_lookup 2009 --json
```

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
pip install -e ".[dev]"
pytest
```

## License

See [LICENSE](LICENSE) (MIT).
