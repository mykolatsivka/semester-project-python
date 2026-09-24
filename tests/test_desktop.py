import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from datalens.desktop import DataLensWindow


def test_desktop_window_loads_demo_and_filters_category() -> None:
    application = QApplication.instance() or QApplication([])
    window = DataLensWindow()

    assert window.tabs.count() == 6
    assert len(window.filtered_dataframe) == 603
    assert window.metric_cards["quality"].value.text() == "99.6%"

    window.filter_column.setCurrentText("region")
    window.category_value.setCurrentText("Захід")
    window.apply_filter()

    assert 0 < len(window.filtered_dataframe) < 603
    assert set(window.filtered_dataframe["region"]) == {"Захід"}

    window.close()
    application.processEvents()
