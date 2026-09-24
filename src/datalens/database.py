from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from datalens.analysis import DatasetOverview


class AnalysisHistory:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_key TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    column_count INTEGER NOT NULL,
                    missing_cells INTEGER NOT NULL,
                    duplicate_rows INTEGER NOT NULL,
                    quality_score REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save(self, source_name: str, overview: DatasetOverview, dataset_key: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO analysis_history (
                    dataset_key,
                    source_name,
                    row_count,
                    column_count,
                    missing_cells,
                    duplicate_rows,
                    quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_key,
                    source_name,
                    overview.rows,
                    overview.columns,
                    overview.missing_cells,
                    overview.duplicate_rows,
                    overview.quality_score,
                ),
            )
            return int(cursor.lastrowid)

    def list_recent(self, limit: int = 50) -> pd.DataFrame:
        safe_limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id AS "ID",
                    source_name AS "Dataset",
                    row_count AS "Рядки",
                    column_count AS "Колонки",
                    missing_cells AS "Пропуски",
                    duplicate_rows AS "Дублікати",
                    ROUND(quality_score, 1) AS "Якість, %",
                    created_at AS "Збережено"
                FROM analysis_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])
