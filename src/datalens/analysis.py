from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Aggregation = Literal["mean", "median", "sum", "min", "max", "count"]


@dataclass(frozen=True, slots=True)
class DatasetOverview:
    rows: int
    columns: int
    numeric_columns: int
    categorical_columns: int
    missing_cells: int
    duplicate_rows: int
    completeness: float
    quality_score: float


def dataset_overview(dataframe: pd.DataFrame) -> DatasetOverview:
    rows, columns = dataframe.shape
    total_cells = rows * columns
    missing_cells = int(dataframe.isna().sum().sum())
    duplicate_rows = int(dataframe.duplicated().sum())
    numeric_columns = len(dataframe.select_dtypes(include="number").columns)
    categorical_columns = columns - numeric_columns

    completeness = 100.0 if total_cells == 0 else 100 * (1 - missing_cells / total_cells)
    duplicate_penalty = 0.0 if rows == 0 else 100 * duplicate_rows / rows
    quality_score = max(0.0, min(100.0, completeness * 0.8 + (100 - duplicate_penalty) * 0.2))

    return DatasetOverview(
        rows=rows,
        columns=columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        missing_cells=missing_cells,
        duplicate_rows=duplicate_rows,
        completeness=completeness,
        quality_score=quality_score,
    )


def missing_values_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    row_count = len(dataframe)
    missing = dataframe.isna().sum()
    percentages = missing.mul(100).div(row_count).fillna(0) if row_count else missing.astype(float)
    return pd.DataFrame(
        {
            "Колонка": dataframe.columns,
            "Тип": dataframe.dtypes.astype(str).values,
            "Пропущено": missing.values,
            "Пропущено, %": percentages.round(2).values,
            "Унікальних": [dataframe[column].nunique(dropna=True) for column in dataframe],
        }
    ).sort_values(["Пропущено", "Колонка"], ascending=[False, True], ignore_index=True)


def descriptive_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty and len(dataframe.columns) == 0:
        return pd.DataFrame()
    return dataframe.describe(include="all").transpose().round(3)


def correlation_matrix(dataframe: pd.DataFrame) -> pd.DataFrame:
    numeric = dataframe.select_dtypes(include="number")
    if len(numeric.columns) < 2:
        return pd.DataFrame()
    return numeric.corr(min_periods=2).round(3)


def group_comparison(
    dataframe: pd.DataFrame,
    group_column: str,
    value_column: str,
    aggregation: Aggregation,
) -> pd.DataFrame:
    if group_column not in dataframe.columns:
        raise KeyError(f"Unknown group column: {group_column}")
    if value_column not in dataframe.columns:
        raise KeyError(f"Unknown value column: {value_column}")
    if not pd.api.types.is_numeric_dtype(dataframe[value_column]):
        raise TypeError("The compared value column must be numeric")
    if aggregation not in {"mean", "median", "sum", "min", "max", "count"}:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    result_name = f"{value_column}_{aggregation}"
    result = (
        dataframe.groupby(group_column, dropna=False)[value_column]
        .agg(aggregation)
        .rename(result_name)
        .reset_index()
        .sort_values(result_name, ascending=False, ignore_index=True)
    )
    return result
