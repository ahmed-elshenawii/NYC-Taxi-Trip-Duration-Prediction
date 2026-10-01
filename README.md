# NYC Taxi Trip Duration Prediction

Regression pipeline to predict **trip duration** (seconds) for NYC taxi trips using **Ridge regression** (α=1) with extensive feature engineering and data cleaning.

---

## Project Structure

```
taxi_trip_duration/
├── data/                    # Input data (CSV)
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── notebooks/                # Jupyter notebooks
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   └── 02_feature_engineering_optimization.ipynb  # Feature engineering & ablation
├── src/                      # Source code
│   ├── __init__.py
│   ├── features.py          # Feature engineering & preprocessing
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Evaluation on CSV
│   └── utils.py             # Paths & utilities
├── models/                   # Saved model (output)
│   └── ridge_model.pkl
├── reports/                  # Reports (optional)
├── requirements.txt
└── README.md
```

---

## Setup

From the project root:

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, pandas, numpy, scikit-learn, joblib, matplotlib, seaborn, jupyter, scipy.

---

## Quick Start

### 1. Train the model

Trains Ridge(α=1) on `log1p(trip_duration)` and saves the full pipeline (including fitted KMeans) to `models/ridge_model.pkl`.

```bash
python src/train.py
```

Output: Training and validation **R²** and **RMSE** (on log scale), and the path to the saved model.

### 2. Evaluate on a CSV

Loads the saved pipeline and evaluates on a given CSV (e.g. validation or test). Uses the same preprocessing as training.

```bash
python src/evaluate.py                    # default: data/test.csv
python src/evaluate.py data/val.csv       # evaluate on validation set
```

Output: **R²** and **RMSE** when the CSV contains a `trip_duration` column.

---

## Data & Preprocessing

- **Input columns:** `id`, `vendor_id`, `pickup_datetime`, `passenger_count`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, `store_and_fwd_flag`, `trip_duration`.
- **Preprocessing** (in `preprocess_features` in `src/features.py`):
  - **Target:** 60 s ≤ `trip_duration` ≤ 10,800 s (3 hours).
  - **Geospatial:** NYC bounds — Latitude [40.5, 40.9], Longitude [-74.05, -73.7].
  - **Distance:** haversine distance > 0.05 km.
  - **Speed:** average speed ≤ 80 km/h.
- **Target for model:** `y = log1p(trip_duration)`.

---

## Features (aligned with `src/features.py` and `src/train.py`)

- **Numeric (StandardScaler):** `distance_km`, `passenger_count`, `distance_sq`, `distance_log`, `dist_hour_interaction`.
- **Categorical (OneHotEncoder):** `hour`, `dayofweek`, `rush_hour` (14:00–18:00), `cluster_interaction`.
- **Spatial:** KMeans(100) on pickup and dropoff → `p_cluster`, `d_cluster`, combined as `cluster_interaction` (route pattern).

**Achieved performance (as reported by `python src/train.py`):**
- **Training:** R² = 0.7656, RMSE = 0.3505 (on log1p(trip_duration)).
- **Validation:** R² = 0.7638, RMSE = 0.3524.

---

## Notebooks

- **`01_eda.ipynb`** — EDA: data overview, target analysis (raw vs log), distance/speed/temporal patterns, clustering, correlation. Paths work when run from project root or from `notebooks/`.
- **`02_feature_engineering_optimization.ipynb`** — Feature groups, ablation (baseline → time → distance → clustering), and a final section to **run training and evaluation** from the notebook (calls `src/train.py` and `src/evaluate.py`).

Run Jupyter from the project root so paths resolve correctly:

```bash
jupyter notebook notebooks/
```

---

## Model

- **Pipeline steps:** `('engineer', NYCFeatureEngineer())` → `('preprocessor', ColumnTransformer)` → `('regressor', Ridge(alpha=1))`.
- **Saved artifact:** Full pipeline (including fitted KMeans inside `NYCFeatureEngineer`) saved to `models/ridge_model.pkl` via `joblib`.

---


