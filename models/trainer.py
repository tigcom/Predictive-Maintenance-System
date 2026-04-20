"""
Training module for the SMART predictive maintenance model.

Usage:
    python -m models.trainer                         # train with default data file
    python -m models.trainer path/to/custom.csv      # train with custom file
"""
import os
import sys
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SMART_FEATURES, SMART_TARGET, MODEL_FILE, DATASET_FOLDER


def _find_col(df, name_lower):
    for col in df.columns:
        if col.lower() == name_lower:
            return col
    return name_lower


def _build_column_map(df):
    mapping = {}
    lookup = {col.lower(): col for col in df.columns}
    aliases = {
        'machine_id': 'Machine_id', 'temperature': 'Temperature',
        'vibration': 'Vibration', 'humidity': 'Humidity',
        'pressure': 'Pressure', 'energy_consumption': 'Energy_consumption',
        'machine_status': 'Machine_status', 'anomaly_flag': 'Anomaly_flag',
        'downtime_risk': 'Downtime_risk', 'maintenance_required': 'Maintenance_required',
    }
    for lower_name, standard_name in aliases.items():
        if lower_name in lookup:
            mapping[standard_name] = lookup[lower_name]
    return mapping


def _engineer_features(df):
    """Advanced feature engineering for better prediction accuracy."""
    temp_col = _find_col(df, 'temperature')
    vib_col = _find_col(df, 'vibration')
    hum_col = _find_col(df, 'humidity')
    pres_col = _find_col(df, 'pressure')
    energy_col = _find_col(df, 'energy_consumption')
    anomaly_col = _find_col(df, 'anomaly_flag')
    downtime_col = _find_col(df, 'downtime_risk')
    maint_col = _find_col(df, 'maintenance_required')
    mid_col = _find_col(df, 'machine_id')

    # --- Interaction features ---
    df['temp_vibration_ratio'] = df[temp_col] / (df[vib_col] + 1)
    df['temp_x_vibration'] = df[temp_col] * df[vib_col]
    df['temp_x_anomaly'] = df[temp_col] * df[anomaly_col]
    df['vib_x_anomaly'] = df[vib_col] * df[anomaly_col]
    df['downtime_x_anomaly'] = df[downtime_col] * df[anomaly_col]
    df['energy_x_temp'] = df[energy_col] * df[temp_col]

    # --- Threshold flags ---
    df['temp_high'] = (df[temp_col] > 85).astype(int)
    df['vib_high'] = (df[vib_col] > 50).astype(int)
    df['health_score'] = df['temp_high'] + df['vib_high'] + df[anomaly_col] + df[maint_col]

    # --- Polynomial features for top correlated ---
    df['anomaly_sq'] = df[anomaly_col] ** 2
    df['downtime_sq'] = df[downtime_col] ** 2
    df['temp_sq'] = df[temp_col] ** 2

    # --- Per-machine rolling stats (sorted by timestamp) ---
    ts_col = None
    for col in df.columns:
        if col.lower() == 'timestamp':
            ts_col = col
            break
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.sort_values([mid_col, ts_col])
        df['hour'] = df[ts_col].dt.hour
        df['day_of_week'] = df[ts_col].dt.dayofweek
        df['month'] = df[ts_col].dt.month

        for col_name, src_col in [('temp', temp_col), ('vib', vib_col)]:
            df[f'{col_name}_roll3_mean'] = (
                df.groupby(mid_col)[src_col]
                .rolling(3, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )
            df[f'{col_name}_roll3_std'] = (
                df.groupby(mid_col)[src_col]
                .rolling(3, min_periods=1).std()
                .reset_index(level=0, drop=True)
            )
            df[f'{col_name}_diff'] = df.groupby(mid_col)[src_col].diff().fillna(0)

        # Per-machine cumulative count (proxy for operating time)
        df['machine_reading_count'] = df.groupby(mid_col).cumcount()
    else:
        df['hour'] = 12
        df['day_of_week'] = 0
        df['month'] = 1
        for col_name in ['temp', 'vib']:
            df[f'{col_name}_roll3_mean'] = 0
            df[f'{col_name}_roll3_std'] = 0
            df[f'{col_name}_diff'] = 0
        df['machine_reading_count'] = 0

    df = df.fillna(0)

    # Encode failure_type
    ft_col = None
    for col in df.columns:
        if col.lower() == 'failure_type':
            ft_col = col
            break
    if ft_col:
        le = LabelEncoder()
        df['Failure_type_encoded'] = le.fit_transform(df[ft_col])
    elif 'Failure_type_encoded' not in df.columns:
        df['Failure_type_encoded'] = 0

    return df


# Extended feature list (original 16 + new engineered)
EXTENDED_FEATURES = SMART_FEATURES + [
    'temp_x_vibration', 'temp_x_anomaly', 'vib_x_anomaly',
    'downtime_x_anomaly', 'energy_x_temp',
    'temp_high', 'vib_high',
    'anomaly_sq', 'downtime_sq', 'temp_sq',
    'temp_roll3_mean', 'temp_roll3_std', 'temp_diff',
    'vib_roll3_mean', 'vib_roll3_std', 'vib_diff',
    'machine_reading_count',
]


def train_smart_model(filepath: str = None):
    if filepath is None:
        filepath = os.path.join(DATASET_FOLDER, 'smart_data.csv')

    print(f'Loading data from {filepath} ...')
    df = pd.read_csv(filepath)
    print(f'  Rows: {len(df)}, Columns: {list(df.columns)}')

    df.columns = df.columns.str.strip()

    # Find target
    target_col = None
    for col in df.columns:
        if col.lower() == SMART_TARGET.lower():
            target_col = col
            break
    if target_col is None:
        raise ValueError(f"Target '{SMART_TARGET}' not found. Available: {list(df.columns)}")

    # Feature engineering
    print('  Engineering features ...')
    df = _engineer_features(df)

    # Build feature matrix
    col_map = _build_column_map(df)
    feature_df = pd.DataFrame()
    for feat in EXTENDED_FEATURES:
        src = col_map.get(feat)
        if src and src in df.columns:
            feature_df[feat] = df[src].values
        elif feat in df.columns:
            feature_df[feat] = df[feat].values
        else:
            feature_df[feat] = 0

    X = feature_df
    y = df[target_col].values
    print(f'  Features: {len(EXTENDED_FEATURES)}, Samples: {len(X)}')

    # Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train models (tuned hyperparameters)
    candidates = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(
            max_depth=20, min_samples_leaf=5, random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=25, min_samples_leaf=3,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=10, random_state=42
        ),
    }

    results = {}
    for name, model in candidates.items():
        print(f'  Training {name} ...')
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {'model': model, 'r2': r2, 'rmse': rmse, 'mae': mae}
        print(f'    R²={r2:.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}')

    # Pick best
    best_name = max(results, key=lambda k: results[k]['r2'])
    best = results[best_name]
    print(f'\n  Best model: {best_name} (R²={best["r2"]:.4f})')

    # Save — use EXTENDED_FEATURES so predictor knows the full list
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    model_info = {
        'model': best['model'],
        'model_name': best_name,
        'features': EXTENDED_FEATURES,
        'scaler': scaler,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {'r2': best['r2'], 'rmse': best['rmse'], 'mae': best['mae']},
    }
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model_info, f)
    print(f'  Model saved to {MODEL_FILE}')

    return results, model_info


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else None
    train_smart_model(path)
