"""
Database migration script — adds SMART columns to existing machines table.
Run this once if upgrading from the old schema.
"""
from database.connection import get_db


def migrate():
    """Add SMART sensor columns to machines table if they don't exist."""
    alter_queries = [
        'ALTER TABLE machines ADD COLUMN Humidity FLOAT DEFAULT 50.0',
        'ALTER TABLE machines ADD COLUMN Energy_consumption FLOAT DEFAULT 2.0',
        'ALTER TABLE machines ADD COLUMN Machine_status INT DEFAULT 1',
        'ALTER TABLE machines ADD COLUMN Anomaly_flag INT DEFAULT 0',
        'ALTER TABLE machines ADD COLUMN Failure_type VARCHAR(50) DEFAULT "Normal"',
        'ALTER TABLE machines ADD COLUMN Downtime_risk FLOAT DEFAULT 0.0',
        'ALTER TABLE machines ADD COLUMN Maintenance_required INT DEFAULT 0',
    ]

    try:
        conn = get_db()
        cursor = conn.cursor()
        for query in alter_queries:
            try:
                cursor.execute(query)
                print(f'  + {query.split("ADD COLUMN")[1].strip().split()[0]}')
            except Exception:
                pass  # column already exists
        conn.commit()
        conn.close()
        print('Migration complete.')
    except Exception as e:
        print(f'Migration error: {e}')


if __name__ == '__main__':
    migrate()
