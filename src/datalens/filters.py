from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd


def apply_column_filter(
    dataframe: pd.DataFrame,
    column: str,
    value: tuple[float, float] | Sequence[str] | tuple[date, date] | str,
) -> pd.DataFrame:
    if column not in dataframe.columns:
        raise KeyError(f"Unknown filter column: {column}")

    series = dataframe[column]
    if isinstance(value, str):
        mask = series.astype(str).str.contains(value, case=False, na=False, regex=False)
    elif pd.api.types.is_numeric_dtype(series) and len(value) == 2:
        lower, upper = value
        mask = series.between(lower, upper, inclusive="both")
    elif len(value) == 2 and all(isinstance(item, date) for item in value):
        converted = pd.to_datetime(series, errors="coerce")
        lower = pd.Timestamp(value[0])
        upper = pd.Timestamp(value[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        mask = converted.between(lower, upper, inclusive="both")
    else:
        selected = {str(item) for item in value}
        mask = series.astype(str).isin(selected)
    return dataframe.loc[mask].copy()
