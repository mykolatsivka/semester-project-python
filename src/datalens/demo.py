from __future__ import annotations

import numpy as np
import pandas as pd


def create_demo_dataset(rows: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    regions = np.array(["Північ", "Південь", "Схід", "Захід", "Центр"])
    categories = np.array(["Електроніка", "Одяг", "Дім", "Спорт"])
    segments = np.array(["Новий", "Постійний", "VIP"])

    units = rng.integers(1, 20, size=rows)
    unit_price = rng.lognormal(mean=4.25, sigma=0.55, size=rows).round(2)
    discount = rng.choice([0.0, 0.05, 0.1, 0.15, 0.2], size=rows, p=[0.35, 0.2, 0.25, 0.15, 0.05])
    revenue = (units * unit_price * (1 - discount)).round(2)

    dataframe = pd.DataFrame(
        {
            "date": rng.choice(dates, size=rows),
            "region": rng.choice(regions, size=rows, p=[0.18, 0.17, 0.2, 0.2, 0.25]),
            "category": rng.choice(categories, size=rows),
            "customer_segment": rng.choice(segments, size=rows, p=[0.45, 0.45, 0.1]),
            "units": units,
            "unit_price": unit_price,
            "discount": discount,
            "revenue": revenue,
            "rating": rng.normal(4.15, 0.55, size=rows).clip(1, 5).round(1),
        }
    ).sort_values("date", ignore_index=True)

    missing_indexes = rng.choice(dataframe.index, size=max(1, rows // 30), replace=False)
    dataframe.loc[missing_indexes, "rating"] = np.nan
    if rows >= 20:
        dataframe = pd.concat([dataframe, dataframe.iloc[:3]], ignore_index=True)
    return dataframe
