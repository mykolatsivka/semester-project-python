from pathlib import Path

import pandas as pd

from datalens.analysis import dataset_overview
from datalens.database import AnalysisHistory


def test_analysis_history_saves_and_lists_snapshots(tmp_path: Path) -> None:
    history = AnalysisHistory(tmp_path / "history.db")
    overview = dataset_overview(pd.DataFrame({"value": [1, 2, None]}))

    record_id = history.save("sample.csv", overview, "abc123")
    recent = history.list_recent()

    assert record_id == 1
    assert recent.iloc[0]["Dataset"] == "sample.csv"
    assert recent.iloc[0]["Рядки"] == 3
    assert recent.iloc[0]["Пропуски"] == 1
