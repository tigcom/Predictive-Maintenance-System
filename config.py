"""
Application configuration for Predictive Maintenance System.
"""
import os

# --- Database ---
DB_HOST = os.environ.get('DB_HOST', '127.0.0.1')
DB_USER = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'root123')
DB_NAME = os.environ.get('DB_NAME', 'qlbaotri')
DB_PORT = int(os.environ.get('DB_PORT', '3307'))

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATASET_FOLDER = os.path.join(BASE_DIR, 'data')
MODEL_FOLDER = os.path.join(BASE_DIR, 'trained_models')

# --- Model ---
MODEL_FILE = os.path.join(MODEL_FOLDER, 'smart_maintenance_model.pkl')

# --- Upload ---
ALLOWED_EXTENSIONS = {'csv'}

# --- RUL Thresholds ---
RUL_GOOD_THRESHOLD = 200
RUL_WARNING_THRESHOLD = 80
RUL_DEFAULT = 250

# --- Training defaults ---
SMART_FEATURES = [
    'Machine_id', 'Temperature', 'Vibration', 'Humidity', 'Pressure',
    'Energy_consumption', 'Machine_status', 'Anomaly_flag', 'Failure_type_encoded',
    'Downtime_risk', 'Maintenance_required', 'hour', 'day_of_week', 'month',
    'temp_vibration_ratio', 'health_score'
]

SMART_TARGET = 'predicted_remaining_life'
