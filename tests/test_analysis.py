import pandas as pd
import pytest

from datalens.analysis import (
    correlation_matrix,
    dataset_overview,
    group_comparison,
    missing_values_summary,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "B"],
            "value": [10.0, 20.0, 4.0, 8.0, None],
            "other": [1, 2, 3, 4, 5],
        }
    )


def test_dataset_overview_counts_quality_issues(sample_dataframe: pd.DataFrame) -> None:
    dataframe = pd.concat([sample_dataframe, sample_dataframe.iloc[[0]]], ignore_index=True)
    overview = dataset_overview(dataframe)

    assert overview.rows == 6
    assert overview.columns == 3
    assert overview.numeric_columns == 2
    assert overview.missing_cells == 1
    assert overview.duplicate_rows == 1
    assert 0 < overview.quality_score < 100


def test_missing_values_summary_orders_missing_columns_first(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = missing_values_summary(sample_dataframe)

    assert result.iloc[0]["Колонка"] == "value"
    assert result.iloc[0]["Пропущено"] == 1


def test_group_comparison_calculates_mean(sample_dataframe: pd.DataFrame) -> None:
    result = group_comparison(sample_dataframe, "group", "value", "mean")

    assert result.to_dict("records") == [
        {"group": "A", "value_mean": 15.0},
        {"group": "B", "value_mean": 6.0},
    ]


def test_correlation_matrix_contains_numeric_columns(sample_dataframe: pd.DataFrame) -> None:
    result = correlation_matrix(sample_dataframe)

    assert result.columns.tolist() == ["value", "other"]
    assert result.index.tolist() == ["value", "other"]
