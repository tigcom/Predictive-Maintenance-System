"""
Prediction module — loads the trained SMART model and provides a single
predict_rul() function used everywhere in the app.
"""
import pickle
import numpy as np
from datetime import datetime
from config import MODEL_FILE, RUL_GOOD_THRESHOLD, RUL_WARNING_THRESHOLD, RUL_DEFAULT

# --- Module-level model state ---
_model = None
_scaler = None
_features = None


def load_model():
    """Load the SMART model from disk. Call once at app startup."""
    global _model, _scaler, _features
    try:
        with open(MODEL_FILE, 'rb') as f:
            info = pickle.load(f)
        _model = info['model']
        _scaler = info.get('scaler')
        _features = info.get('features')
        print(f'Loaded SMART model ({len(_features) if _features else 0} features)')
    except Exception as e:
        print(f'Model not loaded: {e}')
        _model = _scaler = _features = None


def is_ready():
    """Return True if a model is loaded and feature list is known."""
    return _model is not None and _features is not None


def _get(machine: dict, *keys, default=0):
    """Get first non-None value from machine dict by trying multiple keys."""
    for k in keys:
        v = machine.get(k)
        if v is not None:
            return v
    return default


def _build_features(machine: dict) -> dict:
    """
    Build the feature dictionary from a machine record (DB row or CSV row).
    Handles both DB column names (PascalCase) and normalized names.
    Uses _get() to avoid the `or` falsy-zero bug (0 is a valid value).
    """
    now = datetime.now()

    temp = float(_get(machine, 'Temperature', 'temperature', default=0))
    vib = float(_get(machine, 'Vibration', 'vibration', default=0))

    # Encode Failure_type string → int if needed
    ft_encoded = _get(machine, 'Failure_type_encoded', 'failure_type_encoded', default=None)
    if ft_encoded is None:
        ft_str = _get(machine, 'Failure_type', 'failure_type', default='Normal')
        ft_encoded = 0 if str(ft_str).strip() == 'Normal' else 1
    else:
        ft_encoded = int(ft_encoded)

    temp = float(_get(machine, 'Temperature', 'temperature', default=0))
    vib = float(_get(machine, 'Vibration', 'vibration', default=0))
    anomaly = int(_get(machine, 'Anomaly_flag', 'anomaly_flag', default=0))
    downtime = float(_get(machine, 'Downtime_risk', 'downtime_risk', default=0.0))
    energy = float(_get(machine, 'Energy_consumption', 'energy_consumption', default=2.0))
    maint_req = int(_get(machine, 'Maintenance_required', 'maintenance_required', default=0))
    temp_high = 1 if temp > 85 else 0
    vib_high = 1 if vib > 50 else 0

    return {
        # Original 16 features
        'Machine_id': _get(machine, 'Machine_ID', 'Machine_id', default=0),
        'Temperature': temp,
        'Vibration': vib,
        'Humidity': float(_get(machine, 'Humidity', 'humidity', default=50.0)),
        'Pressure': float(_get(machine, 'Pressure', 'pressure', default=0)),
        'Energy_consumption': energy,
        'Machine_status': int(_get(machine, 'Machine_status', 'machine_status', default=1)),
        'Anomaly_flag': anomaly,
        'Failure_type_encoded': ft_encoded,
        'Downtime_risk': downtime,
        'Maintenance_required': maint_req,
        'hour': now.hour,
        'day_of_week': now.weekday(),
        'month': now.month,
        'temp_vibration_ratio': temp / (vib + 1),
        'health_score': temp_high + vib_high + anomaly + maint_req,
        # Extended engineered features
        'temp_x_vibration': temp * vib,
        'temp_x_anomaly': temp * anomaly,
        'vib_x_anomaly': vib * anomaly,
        'downtime_x_anomaly': downtime * anomaly,
        'energy_x_temp': energy * temp,
        'temp_high': temp_high,
        'vib_high': vib_high,
        'anomaly_sq': anomaly ** 2,
        'downtime_sq': downtime ** 2,
        'temp_sq': temp ** 2,
        'temp_roll3_mean': temp,   # single prediction: no history, use current
        'temp_roll3_std': 0.0,
        'temp_diff': 0.0,
        'vib_roll3_mean': vib,
        'vib_roll3_std': 0.0,
        'vib_diff': 0.0,
        'machine_reading_count': 0,
    }


def predict_rul(machine: dict) -> int:
    """
    Predict Remaining Useful Life for a single machine record.
    Returns an integer RUL value (hours).
    Falls back to RUL_DEFAULT when model is unavailable.
    """
    if not is_ready():
        return RUL_DEFAULT

    try:
        fd = _build_features(machine)
        arr = np.array([[fd[f] for f in _features]])
        if _scaler:
            arr = _scaler.transform(arr)
        return int(_model.predict(arr)[0])
    except Exception as e:
        print(f'Prediction error: {e}')
        return RUL_DEFAULT


def rul_status(rul: int) -> tuple:
    """
    Return (status_label, bootstrap_color) for a given RUL value.
    """
    if rul > RUL_GOOD_THRESHOLD:
        return 'Good', 'success'
    elif rul > RUL_WARNING_THRESHOLD:
        return 'Warning', 'warning'
    else:
        return 'Critical', 'danger'


def failure_risk(rul: int) -> float:
    """
    Calculate failure risk percentage based on RUL.
    RUL <= 0 → 100%, RUL >= max_rul → 0%.
    Uses inverse mapping against the data's max RUL (~499).
    """
    max_rul = 499  # max RUL in training data
    if rul <= 0:
        return 100.0
    if rul >= max_rul:
        return 0.0
    return round((1 - rul / max_rul) * 100, 1)
