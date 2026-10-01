import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
# Import custom modules
from features import preprocess_features, NYCFeatureEngineer, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from utils import get_data_path, get_model_path


def train():
    """
    Trains the Ridge regression model and evaluates performance on both
    Training and Validation sets to monitor for overfitting.
    """

    # Resolve dynamic paths
    train_path = get_data_path('train.csv')
    val_path = get_data_path('val.csv')

    print(f" Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Preprocessing and cleaning based on EDA
    print(" Cleaning and filtering data...")
    train_df = preprocess_features(train_df)
    val_df = preprocess_features(val_df)

    # Target log-transformation
    y_train = np.log1p(train_df['trip_duration'])
    y_val = np.log1p(val_df['trip_duration'])

    # Preprocessing pipeline for features
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ])

    # Complete end-to-end Pipeline
    model_pipeline = Pipeline([
        ('engineer', NYCFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1))
    ])

    print(" Training Ridge Model (alpha=1)...")
    model_pipeline.fit(train_df, y_train)

    # --- EVALUATION ON TRAINING SET ---
    train_preds = model_pipeline.predict(train_df)
    train_r2 = r2_score(y_train, train_preds)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))

    # --- EVALUATION ON VALIDATION SET ---
    val_preds = model_pipeline.predict(val_df)
    val_r2 = r2_score(y_val, val_preds)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    # Professional performance report
    print("\n" + "=" * 45)
    print("      MODEL PERFORMANCE REPORT")
    print("=" * 45)
    print(f" TRAINING METRICS:")
    print(f"   - R2 Score: {train_r2:.4f}")
    print(f"   - RMSE:     {train_rmse:.4f}")
    print("-" * 45)
    print(f" VALIDATION METRICS:")
    print(f"   - R2 Score: {val_r2:.4f}")
    print(f"   - RMSE:     {val_rmse:.4f}")
    print("=" * 45)

    # Save the pipeline
    save_path = get_model_path('ridge_model.pkl')
    joblib.dump(model_pipeline, save_path)
    print(f" Pipeline successfully saved at: {save_path}")


if __name__ == "__main__":
    train()