import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error

from features import preprocess_features
from utils import get_data_path, get_model_path


def evaluate(csv_path=None):
    """
    Executes the final evaluation pipeline:
    1. Loads CSV (default: data/test.csv, or path from command line).
    2. Cleans data with preprocess_features (same as training).
    3. Loads saved pipeline and computes R2 and RMSE on log1p(trip_duration).
    """
    if csv_path is None and len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    if csv_path is None:
        csv_path = Path(get_data_path('test.csv'))
    else:
        csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    if not csv_path.exists():
        print(f" Error: File not found at {csv_path}. Check the path.")
        return

    model_path = get_model_path('ridge_model.pkl')

    print(f" Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Apply identical preprocessing as used during training (Outliers, GPS bounds, etc.)
    print(" Cleaning test data and applying filters...")
    df_cleaned = preprocess_features(df)

    # Ensure target variable is log-transformed for accurate metric calculation
    if 'trip_duration' in df_cleaned.columns:
        y_true = np.log1p(df_cleaned['trip_duration'])
    else:
        print(" Warning: 'trip_duration' column missing. Accuracy metrics cannot be calculated.")
        return

    # Load the comprehensive Pipeline object
    print(f" Loading trained pipeline from: {model_path}")
    try:
        model_pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print(f" Error: Model not found at {model_path}. Please run train.py first.")
        return

    # Perform inference
    # The pipeline automatically handles clustering and scaling for the test data
    print(" Making predictions using the production pipeline...")
    preds = model_pipeline.predict(df_cleaned)

    # Calculate final performance metrics
    r2 = r2_score(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))

    # Display results in a professional format
    print("\n" + "=" * 40)
    print("       FINAL EVALUATION RESULTS")
    print("=" * 40)
    print(f" Final Test R2 Score: {r2:.4f}")
    print(f" Final Test RMSE:     {rmse:.4f}")
    print("=" * 40)
    print("Status: Evaluation complete. Results reflect high-granularity feature engineering.")


if __name__ == "__main__":
    evaluate(csv_path=sys.argv[1] if len(sys.argv) > 1 else None)