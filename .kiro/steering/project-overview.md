---
inclusion: always
---

# Project Overview — Predictive Maintenance System (qlbaotri)

## Description
Flask web app for predictive maintenance of industrial machines.
Uses a SMART ML model (scikit-learn) to predict Remaining Useful Life (RUL)
from sensor data (temperature, vibration, pressure, humidity, energy, etc.).

## Tech Stack
- Backend: Python 3, Flask 2.3
- Database: MySQL via PyMySQL (database: `qlbaotri`)
- ML: scikit-learn, numpy, pandas, pickle
- Frontend: Jinja2 templates, Bootstrap 5.3, Chart.js

## Project Structure
```
├── app.py                      # Flask routes & web logic
├── config.py                   # All configuration (DB, paths, thresholds)
├── requirements.txt
├── database/
│   ├── connection.py           # get_db() helper
│   ├── schema.sql              # MySQL CREATE TABLE statements
│   └── migrate.py              # Add SMART columns to existing DB
├── models/
│   ├── predictor.py            # load_model(), predict_rul(), rul_status()
│   └── trainer.py              # train_smart_model() — runnable standalone
├── data/                       # Training datasets (CSV)
│   └── smart_data.csv          # Main SMART dataset (100k rows)
├── trained_models/             # Saved .pkl model files
├── templates/                  # Jinja2 HTML templates
└── uploads/                    # User-uploaded CSV files
```

## Key Modules
- `config.py` — DB credentials, folder paths, RUL thresholds, feature list
- `models/predictor.py` — Single source of truth for prediction logic
- `models/trainer.py` — Standalone training: `python -m models.trainer [csv_path]`
- `database/connection.py` — MySQL connection factory

## SMART Data Format (13 columns, lowercase)
```
timestamp, machine_id, temperature, vibration, humidity, pressure,
energy_consumption, machine_status, anomaly_flag, predicted_remaining_life,
failure_type, downtime_risk, maintenance_required
```
Target: `predicted_remaining_life` (RUL in hours)

## RUL Thresholds (defined in config.py)
- RUL > 1200 → Good (green)
- RUL 400–1200 → Warning (yellow)
- RUL < 400 → Critical (red)

## Running
```bash
python app.py                           # Web app on 0.0.0.0:8000
python -m models.trainer                # Train with data/smart_data.csv
python -m models.trainer path/to/file   # Train with custom CSV
```
