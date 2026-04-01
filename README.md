# test_data_cleaning

Tools for working with Armenian retail product data: train a **BiLSTM** classifier to predict **ADG codes** from product text, **evaluate** the model, **run predictions**, and **map an ADG code** to the most common **brand** and **industry** (category) in the dataset.

## Dataset (`brand_task.csv`)

| Column      | Description                                      |
|------------|---------------------------------------------------|
| `ADG_CODE` | Numeric product classification code (label)     |
| `GOOD_NAME`| Product name (Armenian / Latin)                  |
| `BRAND`    | Brand label                                      |
| `CATEGORY` | Industry / product category (e.g. Beverages)     |

Rows with a missing `ADG_CODE` are excluded from BiLSTM training. The training pipeline also drops duplicate rows and, for stratified splitting, ADG codes that appear fewer than two times.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requirements: `pandas`, `numpy`, `scikit-learn`, `tensorflow` (see `requirements.txt`).

## Train the BiLSTM (`bilstm_train.py`)

Trains a text classifier: **text → `ADG_CODE`**. Input text is built as  
`GOOD_NAME + " [BRAND] " + BRAND + " [CAT] " + CATEGORY`.

```bash
python3 bilstm_train.py
```

Writes under `bilstm_artifacts/`:

- `bilstm_best.keras`, `bilstm_final.keras` (not committed; large files — see `.gitignore`)
- `label_encoder_classes.json`
- `cleaned_training_data.csv`

After cloning the repo, run training once to produce the `.keras` files locally.

## Evaluate (`evaluate_bilstm.py`)

Reports accuracy, precision, recall, F1 (macro and weighted), and top-3 accuracy on the **same validation split** as training (`random_state=42`).

```bash
python3 evaluate_bilstm.py
```

Saves `bilstm_artifacts/evaluation_report.txt`.

## Predict ADG from text (`predict_bilstm.py`)

Requires trained weights in `bilstm_artifacts/bilstm_best.keras`.

```bash
python3 predict_bilstm.py -n "Նեսկաֆե գոլդ 75գ" -b "Nescafe" -c "Beverages" --top-k 5
python3 predict_bilstm.py -t "full text_input string matching training format"
python3 predict_bilstm.py   # demo examples
```

## ADG code → brand and industry (`adg_to_brand_industry.py`)

Uses **empirical counts** in `brand_task.csv`: for a given `ADG_CODE`, returns the most frequent **brand** and **category** (industry), with confidence shares and runners-up.

```bash
python3 adg_to_brand_industry.py 2101
python3 adg_to_brand_industry.py --json 2009
python3 adg_to_brand_industry.py --rebuild-cache
```

Caches statistics in `bilstm_artifacts/adg_brand_category_stats.json`. Rebuild after you change `brand_task.csv`.

## Repository layout

| Path | Role |
|------|------|
| `brand_task.csv` | Source data |
| `bilstm_train.py` | Training |
| `evaluate_bilstm.py` | Metrics |
| `predict_bilstm.py` | Inference from text |
| `adg_to_brand_industry.py` | Code → brand / category lookup |
| `bilstm_artifacts/` | Outputs (small files tracked; `*.keras` ignored) |

## License

Add a license file if you distribute this project publicly.
