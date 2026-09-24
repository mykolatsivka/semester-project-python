import pandas as pd
import pytest

from datalens.data_loader import CSVLoadError, load_csv_bytes
from datalens.filters import apply_column_filter


def test_load_csv_detects_semicolon_delimiter_and_utf8() -> None:
    dataframe = load_csv_bytes("name;value\nКиїв;10\nЛьвів;20\n".encode())

    assert dataframe.to_dict("records") == [
        {"name": "Київ", "value": 10},
        {"name": "Львів", "value": 20},
    ]


def test_load_csv_rejects_empty_file() -> None:
    with pytest.raises(CSVLoadError, match="порожній"):
        load_csv_bytes(b"")


def test_apply_numeric_and_category_filters() -> None:
    dataframe = pd.DataFrame({"city": ["Kyiv", "Lviv", "Kyiv"], "value": [2, 5, 8]})

    numeric_result = apply_column_filter(dataframe, "value", (3.0, 8.0))
    category_result = apply_column_filter(dataframe, "city", ["Kyiv"])

    assert numeric_result["value"].tolist() == [5, 8]
    assert category_result["value"].tolist() == [2, 8]
