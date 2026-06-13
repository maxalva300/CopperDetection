from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "calibration_data.db"

DATABASE_URL = os.getenv("DATABASE_URL")


def get_database_url() -> str:
    """
    Local development:
        SQLite database stored as calibration_data.db

    Render deployment:
        PostgreSQL database using DATABASE_URL environment variable
    """
    if DATABASE_URL:
        # Render may provide postgres:// but SQLAlchemy expects postgresql://
        if DATABASE_URL.startswith("postgres://"):
            return DATABASE_URL.replace("postgres://", "postgresql://", 1)
        return DATABASE_URL

    return f"sqlite:///{DB_PATH}"


ENGINE = create_engine(get_database_url(), future=True)


def init_db():
    db_url = get_database_url()

    if db_url.startswith("sqlite"):
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS calibration_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,

                filename TEXT,
                detected_group TEXT,
                model_file TEXT,

                uploaded_path TEXT,
                overlay_path TEXT,

                predicted_cu_area_pct REAL,
                predicted_cu_mass_pct REAL,

                total_mass_g REAL,
                copper_mass_g REAL,
                reject_mass_g REAL,

                real_cu_pct REAL,
                error_pct_points REAL,
                absolute_error_pct_points REAL,
                mass_balance_delta_g REAL,

                copper_particle_count INTEGER,
                copper_pixels INTEGER,
                material_pixels INTEGER,

                notes TEXT,
                validated INTEGER DEFAULT 0
            )
        """
    else:
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS calibration_records (
                id SERIAL PRIMARY KEY,
                created_at TEXT NOT NULL,

                filename TEXT,
                detected_group TEXT,
                model_file TEXT,

                uploaded_path TEXT,
                overlay_path TEXT,

                predicted_cu_area_pct DOUBLE PRECISION,
                predicted_cu_mass_pct DOUBLE PRECISION,

                total_mass_g DOUBLE PRECISION,
                copper_mass_g DOUBLE PRECISION,
                reject_mass_g DOUBLE PRECISION,

                real_cu_pct DOUBLE PRECISION,
                error_pct_points DOUBLE PRECISION,
                absolute_error_pct_points DOUBLE PRECISION,
                mass_balance_delta_g DOUBLE PRECISION,

                copper_particle_count INTEGER,
                copper_pixels INTEGER,
                material_pixels INTEGER,

                notes TEXT,
                validated INTEGER DEFAULT 0
            )
        """

    with ENGINE.begin() as conn:
        conn.execute(text(create_table_sql))

def insert_calibration_record(record: dict[str, Any]) -> int:
    init_db()

    record = dict(record)
    record.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    record.setdefault("validated", 0)

    columns = list(record.keys())
    placeholders = ", ".join([f":{col}" for col in columns])
    column_names = ", ".join(columns)

    query = text(
        f"""
        INSERT INTO calibration_records ({column_names})
        VALUES ({placeholders})
        """
    )

    with ENGINE.begin() as conn:
        result = conn.execute(query, record)

        try:
            return int(result.lastrowid)
        except Exception:
            row = conn.execute(text("SELECT MAX(id) FROM calibration_records")).fetchone()
            return int(row[0]) if row and row[0] is not None else -1


def fetch_all_records() -> pd.DataFrame:
    init_db()

    with ENGINE.connect() as conn:
        df = pd.read_sql_query(
            text("SELECT * FROM calibration_records ORDER BY id DESC"),
            conn,
        )

    return df


def export_records_to_excel(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    df = fetch_all_records()
    df.to_excel(output_path, index=False)
    return output_path