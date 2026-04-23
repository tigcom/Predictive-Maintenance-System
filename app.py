"""
Predictive Maintenance Web Application
"""
from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
from datetime import datetime

from config import (
    UPLOAD_FOLDER, DATASET_FOLDER, ALLOWED_EXTENSIONS,
    RUL_GOOD_THRESHOLD, RUL_WARNING_THRESHOLD,
)
from database.connection import get_db
from models.predictor import load_model, predict_rul, rul_status, is_ready, failure_risk

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model at startup
load_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _enrich_machines(machines):
    """Add RUL, Status, and FailureRisk fields to a list of machine dicts (in-place)."""
    for m in machines:
        rul = predict_rul(m)
        status, color = rul_status(rul)
        m['RUL'] = rul
        m['Status'] = status
        m['StatusColor'] = color
        m['FailureRisk'] = failure_risk(rul)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.route('/')
def dashboard():
    db = get_db()
    cursor = db.cursor()

    cursor.execute('SELECT * FROM machines')
    machines_data = cursor.fetchall()
    _enrich_machines(machines_data)

    cursor.execute('SELECT COUNT(*) as c FROM maintenance')
    maintenance_count = cursor.fetchone()['c']

    # Real aggregated metrics from DB
    cursor.execute("""
        SELECT
            COALESCE(AVG(Temperature), 0)        AS avg_temp,
            COALESCE(AVG(Humidity), 50)           AS avg_humidity,
            COALESCE(AVG(Energy_consumption), 2)  AS avg_energy,
            SUM(CASE WHEN Anomaly_flag = 1 THEN 1 ELSE 0 END) AS anomalies
        FROM machines
    """)
    stats = cursor.fetchone()

    # Real monthly maintenance costs
    cursor.execute("""
        SELECT DATE_FORMAT(Maintenance_Date, '%%Y-%%m') AS month,
               SUM(Cost) AS total_cost,
               COUNT(*) AS count
        FROM maintenance
        GROUP BY DATE_FORMAT(Maintenance_Date, '%%Y-%%m')
        ORDER BY month
    """)
    monthly = cursor.fetchall()
    months = [r['month'] for r in monthly] if monthly else ['N/A']
    costs = [float(r['total_cost'] or 0) for r in monthly] if monthly else [0]
    maintenance_trend = [r['count'] for r in monthly] if monthly else [0]

    db.close()

    good = sum(1 for m in machines_data if m['Status'] == 'Good')
    warning = sum(1 for m in machines_data if m['Status'] == 'Warning')
    critical = sum(1 for m in machines_data if m['Status'] == 'Critical')

    return render_template(
        'dashboard.html',
        machines=len(machines_data),
        maintenance=maintenance_count,
        good=good, warning=warning, critical=critical,
        avg_temp=stats['avg_temp'],
        avg_humidity=stats['avg_humidity'],
        avg_energy=stats['avg_energy'],
        anomalies=stats['anomalies'] or 0,
        months=months, costs=costs,
        maintenance_trend=maintenance_trend,
    )


# ---------------------------------------------------------------------------
# Machines
# ---------------------------------------------------------------------------

@app.route('/machines')
def machines():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM machines')
    machine_list = cursor.fetchall()
    db.close()
    _enrich_machines(machine_list)
    return render_template('machines.html', machines=machine_list)


@app.route('/add_machine', methods=['POST'])
def add_machine():
    d = request.form
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO machines
        (Date, Temperature, Vibration, Pressure, Humidity, Energy_consumption,
         Machine_status, Anomaly_flag, Failure_type, Downtime_risk,
         Maintenance_required, Maintenance_History, Failure)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        d['Date'],
        float(d['Temperature']), float(d['Vibration']), float(d['Pressure']),
        float(d.get('Humidity', 50.0)), float(d.get('Energy_consumption', 2.0)),
        int(d.get('Machine_status', 1)), int(d.get('Anomaly_flag', 0)),
        d.get('Failure_type', 'Normal'), float(d.get('Downtime_risk', 0.0)),
        int(d.get('Maintenance_required', 0)),
        d.get('Maintenance_History', ''), 0,
    ))
    db.commit()
    db.close()
    return redirect('/machines')


@app.route('/delete_machine/<int:machine_id>')
def delete_machine(machine_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute('DELETE FROM machines WHERE Machine_ID=%s', (machine_id,))
    db.commit()
    db.close()
    return redirect('/machines')


@app.route('/predict/<int:machine_id>')
def predict(machine_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM machines WHERE Machine_ID=%s', (machine_id,))
    machine = cursor.fetchone()
    db.close()

    if not machine:
        return 'Machine not found', 404

    rul = predict_rul(machine)
    status, color = rul_status(rul)
    risk = failure_risk(rul)

    messages = {
        'Good': 'Machine operating normally',
        'Warning': 'Machine may require maintenance soon',
        'Critical': 'Machine at high failure risk',
    }

    return render_template(
        'predict_result.html',
        machine=machine, rul=rul, risk=risk,
        status=status, color=color,
        message=messages[status],
    )


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

@app.route('/maintenance')
def maintenance():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM maintenance')
    records = cursor.fetchall()
    cursor.execute('SELECT Machine_ID FROM machines')
    machine_ids = cursor.fetchall()
    db.close()
    return render_template('maintenance.html', records=records, machines=machine_ids)


@app.route('/add_maintenance', methods=['POST'])
def add_maintenance():
    d = request.form
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO maintenance
        (Machine_ID, Maintenance_Date, Deadline, Status, Technician, Cost, Description)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
    """, (d['Machine_ID'], d['Maintenance_Date'],
          d.get('Deadline') or None, d.get('Status', 'Scheduled'),
          d.get('Technician') or None, d.get('Cost') or None,
          d.get('Description') or None))
    db.commit()
    db.close()
    return redirect('/maintenance')


@app.route('/delete_maintenance/<int:mid>')
def delete_maintenance(mid):
    db = get_db()
    cursor = db.cursor()
    cursor.execute('DELETE FROM maintenance WHERE Maintenance_ID=%s', (mid,))
    db.commit()
    db.close()
    return redirect('/maintenance')


@app.route('/api/maintenance/<int:mid>', methods=['POST'])
def update_maintenance(mid):
    """Inline update Technician, Cost, Status for a maintenance record."""
    from flask import jsonify
    data = request.get_json()
    if not data:
        return jsonify({'ok': False}), 400
    db = get_db()
    cursor = db.cursor()
    fields, values = [], []
    for col in ('Technician', 'Cost', 'Status'):
        if col in data:
            fields.append(f'{col}=%s')
            values.append(data[col] if data[col] != '' else None)
    if fields:
        values.append(mid)
        cursor.execute(f'UPDATE maintenance SET {",".join(fields)} WHERE Maintenance_ID=%s', values)
        db.commit()
    db.close()
    return jsonify({'ok': True})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@app.route('/train')
def train_page():
    return render_template('train.html')


@app.route('/train_model', methods=['POST'])
def train_model_route():
    file = request.files.get('dataset')
    if not file or file.filename == '':
        return 'No file uploaded', 400

    selected_model = request.form.get('model', 'All')

    os.makedirs(DATASET_FOLDER, exist_ok=True)
    filepath = os.path.join(DATASET_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    # Train synchronously (dataset is small enough)
    try:
        # Get current active model before training
        import pickle
        from config import MODEL_FILE
        current_model_info = None
        try:
            with open(MODEL_FILE, 'rb') as f:
                current_model_info = pickle.load(f)
        except:
            current_model_info = None

        from models.trainer import train_smart_model
        results, info = train_smart_model(filepath, selected_model=selected_model)
        # Reload the model in the running app
        load_model()

        # Re-predict all machines in DB with the new model
        _repredict_all_machines()

        # Prepare results for template
        model_results = []
        for name, r in results.items():
            model_results.append({
                'name': name,
                'r2': round(r['r2'], 4),
                'rmse': round(r['rmse'], 1),
                'mae': round(r['mae'], 1),
                'training_time': round(r.get('training_time', 0), 2),
                'is_best': name == info['model_name'],
            })
        model_results.sort(key=lambda x: x['r2'], reverse=True)

        # Feature importance from best model
        best_model = info['model']
        feature_importance = []
        if hasattr(best_model, 'feature_importances_'):
            from config import SMART_FEATURES
            for feat, imp in sorted(
                zip(SMART_FEATURES, best_model.feature_importances_),
                key=lambda x: x[1], reverse=True
            ):
                feature_importance.append({'name': feat, 'importance': round(imp * 100, 2)})

        # Dataset stats
        df_info = pd.read_csv(filepath)
        dataset_info = {
            'rows': len(df_info),
            'cols': len(df_info.columns),
            'train_size': int(len(df_info) * 0.8),
            'test_size': int(len(df_info) * 0.2),
        }

        # Prepare current model info
        previous_model_name = current_model_info.get('model_name', 'None') if current_model_info else 'None'
        previous_model_metrics = current_model_info.get('metrics', {}) if current_model_info else {}

        return render_template('train_results.html',
                               models=model_results,
                               best_name=info['model_name'],
                               training_date=info['training_date'],
                               filename=secure_filename(file.filename),
                               feature_importance=feature_importance,
                               dataset_info=dataset_info,
                               training_time=info.get('training_time', 0),
                               total_training_time=info.get('total_training_time', 0),
                               previous_model_name=previous_model_name,
                               previous_model_metrics=previous_model_metrics)
    except Exception as e:
        return f'Training error: {e}', 500


def _repredict_all_machines():
    """Re-predict RUL for all machines in DB using the newly trained model."""
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT * FROM machines')
        machines_list = cursor.fetchall()

        for m in machines_list:
            rul = predict_rul(m)
            status, _ = rul_status(rul)
            failure = 1 if rul < RUL_WARNING_THRESHOLD else 0
            cursor.execute("""
                UPDATE machines
                SET Failure = %s,
                    Maintenance_History = CONCAT(
                        COALESCE(Maintenance_History, ''),
                        %s
                    )
                WHERE Machine_ID = %s
            """, (
                failure,
                f'\n[{datetime.now().strftime("%Y-%m-%d %H:%M")}] Re-predicted RUL: {rul}h ({status})',
                m['Machine_ID'],
            ))

        db.commit()
        db.close()
        print(f'Re-predicted {len(machines_list)} machines with new model.')
    except Exception as e:
        print(f'Re-prediction error: {e}')


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@app.route('/datasets')
def datasets():
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    files = []
    for f in os.listdir(DATASET_FOLDER):
        if f.endswith('.csv'):
            path = os.path.join(DATASET_FOLDER, f)
            files.append({'name': f, 'size': round(os.path.getsize(path) / 1024, 2)})
    return render_template('datasets.html', files=files)


@app.route('/download_dataset/<filename>')
def download_dataset(filename):
    return send_from_directory(DATASET_FOLDER, filename)


@app.route('/delete_dataset/<filename>')
def delete_dataset(filename):
    path = os.path.join(DATASET_FOLDER, secure_filename(filename))
    if os.path.exists(path):
        os.remove(path)
    return redirect('/datasets')


@app.route('/view_dataset/<filename>')
def view_dataset(filename):
    path = os.path.join(DATASET_FOLDER, filename)
    df = pd.read_csv(path)
    data = df.head(100).to_dict(orient='records')
    return render_template('view_dataset.html', filename=filename,
                           columns=df.columns, data=data)


# ---------------------------------------------------------------------------
# Upload & Batch Predict
# ---------------------------------------------------------------------------

@app.route('/upload_predict')
def upload_predict():
    return render_template('upload_predict.html')


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if not (file and allowed_file(file.filename)):
        return 'Invalid file type. Only CSV files are allowed.', 400

    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower()

        # Column name normalization
        col_map = {
            'machine_id': 'Machine_id', 'machineid': 'Machine_id', 'id': 'Machine_id',
            'temperature': 'Temperature', 'temp': 'Temperature',
            'vibration': 'Vibration', 'vib': 'Vibration',
            'pressure': 'Pressure', 'press': 'Pressure',
            'humidity': 'Humidity', 'humid': 'Humidity',
            'energy_consumption': 'Energy_consumption', 'energy': 'Energy_consumption',
            'machine_status': 'Machine_status', 'status': 'Machine_status',
            'anomaly_flag': 'Anomaly_flag', 'anomaly': 'Anomaly_flag',
            'failure_type': 'Failure_type_encoded',
            'failure_type_encoded': 'Failure_type_encoded',
            'downtime_risk': 'Downtime_risk', 'downtime': 'Downtime_risk',
            'maintenance_required': 'Maintenance_required',
        }
        df = df.rename(columns=col_map)

        # Encode failure_type strings
        if 'Failure_type_encoded' in df.columns:
            ft_map = {'Normal': 0, 'Wear': 1, 'Overheat': 2,
                      'Electrical': 3, 'Mechanical': 4, 'Sensor': 5}
            df['Failure_type_encoded'] = df['Failure_type_encoded'].map(
                lambda x: ft_map.get(str(x).strip(), 0) if isinstance(x, str) else x
            ).fillna(0).astype(int)

        # Validate essential columns
        essential = ['Machine_id', 'Temperature', 'Vibration', 'Pressure']
        missing = [c for c in essential if c not in df.columns]
        if missing:
            return f"Missing columns: {', '.join(missing)}", 400

        # Fill optional columns with defaults
        defaults = {
            'Humidity': 50.0, 'Energy_consumption': 2.0, 'Machine_status': 1,
            'Anomaly_flag': 0, 'Failure_type_encoded': 0,
            'Downtime_risk': 0.0, 'Maintenance_required': 0,
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val

        # Predict each row
        predictions = []
        for _, row in df.iterrows():
            rul = predict_rul(row.to_dict())
            status, color = rul_status(rul)
            risk = failure_risk(rul)
            predictions.append({
                'machine_id': row['Machine_id'], 'rul': rul,
                'status': status, 'color': color, 'risk': risk,
                'temperature': row['Temperature'],
                'vibration': row['Vibration'],
                'humidity': row.get('Humidity', 50.0),
                'pressure': row['Pressure'],
                'energy_consumption': row.get('Energy_consumption', 2.0),
                'anomaly_flag': row.get('Anomaly_flag', 0),
            })

        # Insert into DB
        _insert_csv_to_db(df, [p['rul'] for p in predictions])

        return render_template('prediction_results.html',
                               predictions=predictions, filename=filename)
    except Exception as e:
        return f'Error processing file: {e}', 500


def _insert_csv_to_db(df, rul_values):
    """Clear old machines then insert new CSV data + auto-schedule maintenance."""
    try:
        db = get_db()
        cursor = db.cursor()

        # Clear old data — disable FK check to avoid constraint issues
        cursor.execute('SET FOREIGN_KEY_CHECKS = 0')
        cursor.execute('TRUNCATE TABLE maintenance')
        cursor.execute('TRUNCATE TABLE machines')
        cursor.execute('SET FOREIGN_KEY_CHECKS = 1')

        from config import RUL_WARNING_THRESHOLD, RUL_GOOD_THRESHOLD
        from datetime import timedelta

        for idx, (_, row) in enumerate(df.iterrows()):
            rul = rul_values[idx] if idx < len(rul_values) else 0
            cursor.execute("""
                INSERT INTO machines
                (Date, Operating_Hours, Temperature, Vibration, Pressure,
                 Production_Output, Humidity, Energy_consumption, Machine_status,
                 Anomaly_flag, Failure_type, Downtime_risk, Maintenance_required,
                 Maintenance_History, Failure)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                datetime.now().date(), 0.0,
                row['Temperature'], row['Vibration'], row['Pressure'], 0.0,
                row.get('Humidity', 50.0), row.get('Energy_consumption', 2.0),
                int(row.get('Machine_status', 1)), int(row.get('Anomaly_flag', 0)),
                'Normal' if row.get('Failure_type_encoded', 0) == 0 else 'Wear',
                float(row.get('Downtime_risk', 0.0)),
                int(row.get('Maintenance_required', 0)),
                f'Uploaded from CSV - RUL: {rul}',
                1 if rul < RUL_WARNING_THRESHOLD else 0,
            ))
            machine_id = cursor.lastrowid

            # Auto-schedule maintenance for Warning and Critical machines
            if rul <= RUL_GOOD_THRESHOLD:
                hours_to_critical = max(rul - RUL_WARNING_THRESHOLD, 0)
                days_to_critical = max(int(hours_to_critical / 24), 0)
                deadline = datetime.now().date() + timedelta(days=days_to_critical)
                today = datetime.now().date()

                if rul <= RUL_WARNING_THRESHOLD:
                    status = 'Urgent'
                    desc = f'URGENT: RUL={rul}h, đã vào vùng Critical. Bảo dưỡng ngay!'
                else:
                    status = 'Scheduled'
                    desc = f'RUL={rul}h, còn {days_to_critical} ngày trước khi Critical.'

                cursor.execute("""
                    INSERT INTO maintenance
                    (Machine_ID, Maintenance_Date, Deadline, Status, Description)
                    VALUES (%s, %s, %s, %s, %s)
                """, (machine_id, today, deadline, status, desc))

        db.commit()
        db.close()
    except Exception as e:
        print(f'DB insert error: {e}')


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8000)
