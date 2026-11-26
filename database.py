"""
database.py
===========
Stores prediction history in a small SQLite database (predictions.db).
SQLite is built into Python, so nothing extra needs to be installed.

The app saves every prediction here and shows them on the "Prediction History" page.
"""

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "predictions.db"

COLUMNS = [
    "created_at", "state", "district", "crop", "season", "year",
    "area", "predicted_yield", "estimated_production", "model_name",
]


def init_db(db_path=DB_PATH):
    """Create the history table if it does not exist yet."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                state TEXT, district TEXT, crop TEXT, season TEXT,
                year INTEGER, area REAL,
                predicted_yield REAL, estimated_production REAL,
                model_name TEXT
            )
            """
        )
        conn.commit()


def save_prediction(record, db_path=DB_PATH):
    """Save one prediction. `record` is a dict with the keys in COLUMNS (created_at is optional)."""
    init_db(db_path)
    record = dict(record)
    record.setdefault("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    values = [record.get(col) for col in COLUMNS]
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            f"INSERT INTO predictions ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
            values,
        )
        conn.commit()


def fetch_history(limit=200, db_path=DB_PATH):
    """Return the latest predictions (newest first) as a DataFrame."""
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        return pd.read_sql_query(
            f"SELECT {', '.join(COLUMNS)} FROM predictions ORDER BY id DESC LIMIT ?",
            conn,
            params=(int(limit),),
        )


def clear_history(db_path=DB_PATH):
    """Delete all saved predictions."""
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()
