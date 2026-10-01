import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

# --- Project Constants ---
# Increased clusters to 100 for better geospatial granularity
N_CLUSTERS = 100

# NYC geographic boundaries for filtering invalid GPS points
NYC_LAT_MIN, NYC_LAT_MAX = 40.5, 40.9
NYC_LON_MIN, NYC_LON_MAX = -74.05, -73.7

# Target constraints based on EDA observations (1 min to 3 hours)
DURATION_MIN, DURATION_MAX = 60, 10800

# Speed limit to remove unrealistic data points (Max 80 km/h in NYC)
SPEED_MAX_KMH = 80.0


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points in kilometers."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_degrees(lat1, lon1, lat2, lon2):
    """Calculates the initial bearing (direction) between two coordinates."""
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def preprocess_features(df):
    """
    Cleans the raw dataframe by removing outliers and invalid GPS points.
    Based on EDA analysis: trip duration, distance, and speed filtering.
    """
    out = df.copy()
    out['pickup_datetime'] = pd.to_datetime(out['pickup_datetime'])

    # Calculate initial distance for filtering purposes
    dist = haversine_distance_km(out['pickup_latitude'], out['pickup_longitude'],
                                 out['dropoff_latitude'], out['dropoff_longitude'])
    out['distance_km'] = dist

    # Apply target filters if column exists (Training/Validation phase)
    if 'trip_duration' in out.columns:
        mask = (out['trip_duration'].between(DURATION_MIN, DURATION_MAX)) & \
               (out['distance_km'] > 0.05) & \
               ((out['distance_km'] / (out['trip_duration'] / 3600)) <= SPEED_MAX_KMH)
        out = out[mask]

    # Apply geospatial NYC boundary filter
    geo_mask = (out['pickup_latitude'].between(NYC_LAT_MIN, NYC_LAT_MAX)) & \
               (out['dropoff_latitude'].between(NYC_LAT_MIN, NYC_LAT_MAX)) & \
               (out['pickup_longitude'].between(NYC_LON_MIN, NYC_LON_MAX)) & \
               (out['dropoff_longitude'].between(NYC_LON_MIN, NYC_LON_MAX))

    return out[geo_mask].reset_index(drop=True)


class NYCFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for Sklearn Pipeline.
    Fits KMeans on coordinates and generates advanced interaction features.
    """

    def __init__(self, n_clusters=N_CLUSTERS):
        self.n_clusters = n_clusters
        # Initialize separate KMeans for Pickup and Dropoff locations
        self.p_km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        self.d_km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)

    def fit(self, X, y=None):
        """Fits KMeans models to identify high-traffic zones (Clusters)."""
        self.p_km.fit(X[['pickup_latitude', 'pickup_longitude']])
        self.d_km.fit(X[['dropoff_latitude', 'dropoff_longitude']])
        return self

    def transform(self, X):
        """Generates distance, time, clustering, and interaction features."""
        X = X.copy()
        X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'])

        # --- Time-based Features ---
        X['hour'] = X['pickup_datetime'].dt.hour
        X['dayofweek'] = X['pickup_datetime'].dt.dayofweek
        X['rush_hour'] = X['hour'].between(14, 18).astype(int)  # Peak hours from EDA

        # --- Distance-based Features ---
        X['distance_km'] = haversine_distance_km(X['pickup_latitude'], X['pickup_longitude'],
                                                 X['dropoff_latitude'], X['dropoff_longitude'])

        # --- Non-linear & Interaction Features (Vital for Ridge Regression) ---
        # Helps the linear model capture non-linear traffic/time behaviors
        X['dist_hour_interaction'] = X['distance_km'] * X['hour']
        X['distance_sq'] = X['distance_km'] ** 2
        X['distance_log'] = np.log1p(X['distance_km'])

        # --- Geospatial Clustering ---
        X['p_cluster'] = self.p_km.predict(X[['pickup_latitude', 'pickup_longitude']])
        X['d_cluster'] = self.d_km.predict(X[['dropoff_latitude', 'dropoff_longitude']])

        # Captured Route Pattern: Represents specific 'Start-to-End' zone behavior
        X['cluster_interaction'] = X['p_cluster'].astype(str) + "_" + X['d_cluster'].astype(str)

        return X


# Lists of features to be handled by the ColumnTransformer in train.py
NUMERIC_FEATURES = ['distance_km', 'passenger_count', 'distance_sq', 'distance_log', 'dist_hour_interaction']
CATEGORICAL_FEATURES = ['hour', 'dayofweek', 'rush_hour', 'cluster_interaction']