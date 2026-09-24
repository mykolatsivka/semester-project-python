# ruff: noqa: E501
from __future__ import annotations

import hashlib
import sys
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import (
    QAbstractAnimation,
    QAbstractTableModel,
    QDate,
    QEasingCurve,
    QModelIndex,
    QPropertyAnimation,
    Qt,
)
from PySide6.QtGui import QAction, QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from datalens.analysis import (
    correlation_matrix,
    dataset_overview,
    descriptive_statistics,
    group_comparison,
    missing_values_summary,
)
from datalens.data_loader import CSVLoadError, load_csv_bytes
from datalens.database import AnalysisHistory
from datalens.demo import create_demo_dataset
from datalens.filters import apply_column_filter

matplotlib.use("QtAgg")

PROJECT_ROOT = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parents[2]
)
NO_GROUP = "Без групування"
SURFACE = "#0E1727"
TEXT = "#DCE7F7"
MUTED = "#8EA0BA"
GRID = "#26344A"


class DataFrameModel(QAbstractTableModel):
    def __init__(self, dataframe: pd.DataFrame | None = None) -> None:
        super().__init__()
        self._dataframe = dataframe if dataframe is not None else pd.DataFrame()

    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008, N802
        return 0 if parent.isValid() else len(self._dataframe)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008, N802
        return 0 if parent.isValid() else len(self._dataframe.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._dataframe.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            if pd.isna(value):
                return "—"
            if isinstance(value, float):
                return f"{value:,.3f}".rstrip("0").rstrip(".")
            if isinstance(value, pd.Timestamp):
                return value.strftime("%Y-%m-%d")
            return str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role == Qt.ItemDataRole.BackgroundRole and index.row() % 2:
            return QColor("#111C2F")
        return None

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._dataframe.columns[section])
        return str(section + 1)


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(7, 4), dpi=100, facecolor=SURFACE)
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def reset(self, message: str | None = None):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor(SURFACE)
        if message:
            axis.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                color=MUTED,
                transform=axis.transAxes,
            )
            axis.set_axis_off()
        return axis

    def finish(self) -> None:
        self.figure.set_facecolor(SURFACE)
        for axis in self.figure.axes:
            axis.set_facecolor(SURFACE)
            axis.tick_params(colors=MUTED)
            axis.xaxis.label.set_color(TEXT)
            axis.yaxis.label.set_color(TEXT)
            axis.title.set_color(TEXT)
            for spine in axis.spines.values():
                spine.set_color(GRID)
        self.figure.tight_layout()
        self.draw_idle()


class MetricCard(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(3)
        label = QLabel(title)
        label.setObjectName("metricTitle")
        self.value = QLabel("—")
        self.value.setObjectName("metricValue")
        layout.addWidget(label)
        layout.addWidget(self.value)


class SmoothTableView(QTableView):
    def __init__(self) -> None:
        super().__init__()
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._vertical_target = 0
        self._vertical_animation = QPropertyAnimation(self.verticalScrollBar(), b"value", self)
        self._vertical_animation.setDuration(170)
        self._vertical_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if not delta:
            super().wheelEvent(event)
            return

        scrollbar = self.verticalScrollBar()
        if self._vertical_animation.state() != QAbstractAnimation.State.Running:
            self._vertical_target = scrollbar.value()
        distance = int(delta * 0.9)
        self._vertical_target = max(
            scrollbar.minimum(),
            min(scrollbar.maximum(), self._vertical_target - distance),
        )
        self._vertical_animation.stop()
        self._vertical_animation.setStartValue(scrollbar.value())
        self._vertical_animation.setEndValue(self._vertical_target)
        self._vertical_animation.start()
        event.accept()


def make_table() -> QTableView:
    table = SmoothTableView()
    table.setAlternatingRowColors(False)
    table.setSortingEnabled(False)
    table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
    table.verticalHeader().setDefaultSectionSize(30)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def titled(label: str, widget: QWidget) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    caption = QLabel(label)
    caption.setObjectName("fieldLabel")
    layout.addWidget(caption)
    layout.addWidget(widget)
    return container


class DataLensWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DataLens — аналіз даних")
        self.resize(1380, 860)
        self.setMinimumSize(1050, 680)

        self.dataframe = pd.DataFrame()
        self.filtered_dataframe = pd.DataFrame()
        self.source_name = ""
        self.history = AnalysisHistory(PROJECT_ROOT / "data" / "datalens.db")

        self._build_ui()
        self._build_menu()
        self._apply_style()
        self.load_demo()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setObjectName("sidebar")
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(318)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar = QWidget()
        sidebar.setObjectName("sidebarContent")
        self.sidebar_layout = QVBoxLayout(sidebar)
        self.sidebar_layout.setContentsMargins(18, 20, 18, 18)
        self.sidebar_layout.setSpacing(10)
        sidebar_scroll.setWidget(sidebar)
        root.addWidget(sidebar_scroll)

        source_title = QLabel("ДЖЕРЕЛО ДАНИХ")
        source_title.setObjectName("sectionTitle")
        self.sidebar_layout.addWidget(source_title)
        open_button = QPushButton("Відкрити CSV")
        open_button.setObjectName("primaryButton")
        open_button.clicked.connect(self.open_csv)
        demo_button = QPushButton("Демонстраційні дані")
        demo_button.setObjectName("secondaryButton")
        demo_button.clicked.connect(self.load_demo)
        self.sidebar_layout.addWidget(open_button)
        self.sidebar_layout.addWidget(demo_button)

        dataset_card = QFrame()
        dataset_card.setObjectName("sideCard")
        dataset_layout = QVBoxLayout(dataset_card)
        dataset_layout.setContentsMargins(13, 11, 13, 11)
        dataset_layout.setSpacing(3)
        dataset_caption = QLabel("АКТИВНИЙ НАБІР")
        dataset_caption.setObjectName("miniLabel")
        self.sidebar_dataset = QLabel("—")
        self.sidebar_dataset.setObjectName("datasetName")
        dataset_layout.addWidget(dataset_caption)
        dataset_layout.addWidget(self.sidebar_dataset)
        self.sidebar_layout.addWidget(dataset_card)

        filter_card = QFrame()
        filter_card.setObjectName("sideCard")
        filter_layout = QVBoxLayout(filter_card)
        filter_layout.setContentsMargins(13, 12, 13, 12)
        filter_layout.setSpacing(9)
        filter_header = QHBoxLayout()
        filter_title = QLabel("Фільтрація")
        filter_title.setObjectName("cardTitle")
        self.sidebar_filter_state = QLabel("Усі дані")
        self.sidebar_filter_state.setObjectName("statusPill")
        filter_header.addWidget(filter_title)
        filter_header.addStretch()
        filter_header.addWidget(self.sidebar_filter_state)
        filter_layout.addLayout(filter_header)
        self.filter_column = QComboBox()
        self.filter_column.currentTextChanged.connect(self._configure_filter)
        filter_layout.addWidget(titled("Колонка", self.filter_column))

        self.filter_stack = QStackedWidget()
        self.filter_stack.addWidget(self._empty_filter_page())
        self.filter_stack.addWidget(self._numeric_filter_page())
        self.filter_stack.addWidget(self._category_filter_page())
        self.filter_stack.addWidget(self._date_filter_page())
        self.filter_stack.addWidget(self._text_filter_page())
        self.filter_stack.setMinimumHeight(62)
        filter_layout.addWidget(self.filter_stack)

        apply_button = QPushButton("Застосувати")
        apply_button.setObjectName("primaryButton")
        apply_button.clicked.connect(self.apply_filter)
        reset_button = QPushButton("Скинути")
        reset_button.setObjectName("compactButton")
        reset_button.clicked.connect(self.reset_filter)
        filter_actions = QHBoxLayout()
        filter_actions.setSpacing(7)
        filter_actions.addWidget(apply_button, 2)
        filter_actions.addWidget(reset_button, 1)
        filter_layout.addLayout(filter_actions)
        self.sidebar_layout.addWidget(filter_card)

        result_card = QFrame()
        result_card.setObjectName("sideCard")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(13, 11, 13, 11)
        result_layout.setSpacing(2)
        result_caption = QLabel("РЕЗУЛЬТАТ")
        result_caption.setObjectName("miniLabel")
        self.filtered_count = QLabel("Показано рядків: —")
        self.filtered_count.setObjectName("sidebarMetric")
        result_layout.addWidget(result_caption)
        result_layout.addWidget(self.filtered_count)
        self.sidebar_layout.addWidget(result_card)

        actions_title = QLabel("ШВИДКІ ДІЇ")
        actions_title.setObjectName("sectionTitle")
        self.sidebar_layout.addWidget(actions_title)
        export_button = QPushButton("Експортувати CSV")
        export_button.setObjectName("secondaryButton")
        export_button.clicked.connect(self.export_csv)
        snapshot_button = QPushButton("Зберегти знімок")
        snapshot_button.setObjectName("secondaryButton")
        snapshot_button.clicked.connect(self.save_snapshot)
        self.sidebar_layout.addWidget(export_button)
        self.sidebar_layout.addWidget(snapshot_button)
        self.sidebar_layout.addStretch(1)

        content = QWidget()
        content.setObjectName("contentArea")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(30, 24, 30, 24)
        content_layout.setSpacing(18)
        root.addWidget(content, 1)

        heading_row = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Аналітична панель")
        title.setObjectName("pageTitle")
        self.dataset_label = QLabel("Dataset не завантажено")
        self.dataset_label.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(self.dataset_label)
        heading_row.addLayout(title_box)
        heading_row.addStretch()
        content_layout.addLayout(heading_row)

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.metric_cards = {
            key: MetricCard(title)
            for key, title in [
                ("rows", "Рядки"),
                ("columns", "Колонки"),
                ("numeric", "Числові"),
                ("missing", "Пропуски"),
                ("quality", "Якість"),
            ]
        }
        for card in self.metric_cards.values():
            metrics.addWidget(card)
        content_layout.addLayout(metrics)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._overview_tab(), "Огляд")
        self.tabs.addTab(self._quality_tab(), "Якість даних")
        self.tabs.addTab(self._statistics_tab(), "Статистика")
        self.tabs.addTab(self._visualization_tab(), "Візуалізація")
        self.tabs.addTab(self._comparison_tab(), "Порівняння")
        self.tabs.addTab(self._history_tab(), "Історія")
        self.tabs.currentChanged.connect(self._tab_changed)
        content_layout.addWidget(self.tabs, 1)

        self.setStatusBar(QStatusBar())

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("Файл")
        open_action = QAction("Відкрити CSV…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_csv)
        export_action = QAction("Експортувати…", self)
        export_action.setShortcut("Ctrl+S")
        export_action.triggered.connect(self.export_csv)
        exit_action = QAction("Вийти", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(open_action)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        help_menu = self.menuBar().addMenu("Довідка")
        about_action = QAction("Про DataLens", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _empty_filter_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 4)
        label = QLabel("Оберіть колонку для фільтрації")
        label.setWordWrap(True)
        label.setObjectName("hint")
        layout.addWidget(label)
        return page

    def _numeric_filter_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(7)
        self.numeric_min = QDoubleSpinBox()
        self.numeric_max = QDoubleSpinBox()
        for control in (self.numeric_min, self.numeric_max):
            control.setDecimals(4)
            control.setRange(-1e15, 1e15)
        layout.addWidget(titled("Від", self.numeric_min))
        layout.addWidget(titled("До", self.numeric_max))
        return page

    def _category_filter_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 4)
        self.category_value = QComboBox()
        layout.addWidget(titled("Значення", self.category_value))
        return page

    def _date_filter_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(7)
        self.date_from = QDateEdit(calendarPopup=True)
        self.date_to = QDateEdit(calendarPopup=True)
        self.date_from.setDisplayFormat("dd.MM.yyyy")
        self.date_to.setDisplayFormat("dd.MM.yyyy")
        layout.addWidget(titled("Від", self.date_from))
        layout.addWidget(titled("До", self.date_to))
        return page

    def _text_filter_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 4)
        self.text_query = QLineEdit()
        self.text_query.setPlaceholderText("Введіть фрагмент тексту")
        self.text_query.returnPressed.connect(self.apply_filter)
        layout.addWidget(titled("Текст містить", self.text_query))
        return page

    def _overview_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 16, 12, 12)
        self.overview_table = make_table()
        self.overview_model = DataFrameModel()
        self.overview_table.setModel(self.overview_model)
        self.overview_note = QLabel()
        self.overview_note.setObjectName("hint")
        layout.addWidget(self.overview_table, 1)
        layout.addWidget(self.overview_note)
        return tab

    def _quality_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 16, 12, 12)
        self.quality_summary = QLabel()
        self.quality_summary.setObjectName("infoBanner")
        self.quality_table = make_table()
        self.quality_model = DataFrameModel()
        self.quality_table.setModel(self.quality_model)
        layout.addWidget(self.quality_summary)
        layout.addWidget(self.quality_table, 1)
        return tab

    def _statistics_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 16, 12, 12)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.statistics_table = make_table()
        self.statistics_model = DataFrameModel()
        self.statistics_table.setModel(self.statistics_model)
        self.correlation_canvas = PlotCanvas()
        splitter.addWidget(self.statistics_table)
        splitter.addWidget(self.correlation_canvas)
        splitter.setSizes([300, 400])
        layout.addWidget(splitter)
        return tab

    def _visualization_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 16, 12, 12)
        controls = QHBoxLayout()
        self.chart_type = QComboBox()
        self.chart_type.addItems(["Гістограма", "Box plot", "Scatter plot", "Стовпчикова"])
        self.chart_x = QComboBox()
        self.chart_y = QComboBox()
        self.chart_group = QComboBox()
        draw_button = QPushButton("Побудувати")
        draw_button.setObjectName("primaryButton")
        draw_button.clicked.connect(self.draw_chart)
        controls.addWidget(titled("Тип", self.chart_type))
        controls.addWidget(titled("Колонка X", self.chart_x))
        controls.addWidget(titled("Колонка Y", self.chart_y))
        controls.addWidget(titled("Групування", self.chart_group))
        controls.addWidget(draw_button)
        self.chart_canvas = PlotCanvas()
        layout.addLayout(controls)
        layout.addWidget(self.chart_canvas, 1)
        return tab

    def _comparison_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 16, 12, 12)
        controls = QHBoxLayout()
        self.compare_group = QComboBox()
        self.compare_value = QComboBox()
        self.compare_aggregation = QComboBox()
        self.compare_aggregation.addItems(["mean", "median", "sum", "min", "max", "count"])
        compare_button = QPushButton("Порівняти")
        compare_button.setObjectName("primaryButton")
        compare_button.clicked.connect(self.update_comparison)
        controls.addWidget(titled("Групувати за", self.compare_group))
        controls.addWidget(titled("Показник", self.compare_value))
        controls.addWidget(titled("Агрегація", self.compare_aggregation))
        controls.addWidget(compare_button)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.comparison_table = make_table()
        self.comparison_model = DataFrameModel()
        self.comparison_table.setModel(self.comparison_model)
        self.comparison_canvas = PlotCanvas()
        splitter.addWidget(self.comparison_table)
        splitter.addWidget(self.comparison_canvas)
        splitter.setSizes([430, 650])
        layout.addLayout(controls)
        layout.addWidget(splitter, 1)
        return tab

    def _history_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 16, 12, 12)
        row = QHBoxLayout()
        label = QLabel("Збережені аналітичні знімки")
        label.setObjectName("tabTitle")
        refresh = QPushButton("Оновити")
        refresh.clicked.connect(self.update_history)
        row.addWidget(label)
        row.addStretch()
        row.addWidget(refresh)
        self.history_table = make_table()
        self.history_model = DataFrameModel()
        self.history_table.setModel(self.history_model)
        layout.addLayout(row)
        layout.addWidget(self.history_table, 1)
        return tab

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                color: #DCE7F7;
                font-family: "Segoe UI";
                font-size: 13px;
            }
            QMainWindow { background: #080E1A; }
            QLabel { background: transparent; }
            QWidget#contentArea { background: #0B1220; }
            QMenuBar {
                background: #090F1C;
                color: #DCE7F7;
                border-bottom: 1px solid #202C40;
                padding: 4px;
            }
            QMenuBar::item {
                padding: 7px 18px;
                margin: 1px 4px;
                border-radius: 5px;
            }
            QMenuBar::item:selected, QMenu::item:selected { background: #1A2942; }
            QMenu {
                background: #101A2B;
                color: #DCE7F7;
                border: 1px solid #2A3850;
                padding: 6px;
            }
            QMenu::item {
                min-width: 210px;
                padding: 8px 34px 8px 14px;
                margin: 2px 0;
                border-radius: 5px;
            }
            QMenu::separator {
                height: 1px;
                background: #26354B;
                margin: 5px 10px;
            }
            QScrollArea#sidebar, QWidget#sidebarContent {
                background: #070C16;
                border: none;
            }
            QLabel#sectionTitle, QLabel#miniLabel {
                color: #667995;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QFrame#sideCard {
                background: #0D1727;
                border: 1px solid #1F2D43;
                border-radius: 10px;
            }
            QLabel#cardTitle { color: #E6EDF8; font-size: 14px; font-weight: 650; }
            QLabel#statusPill {
                background: #17315B;
                color: #8CB4FF;
                border: 1px solid #294A7A;
                border-radius: 8px;
                padding: 2px 7px;
                font-size: 10px;
            }
            QLabel#datasetName { color: #C8D5E8; font-size: 13px; font-weight: 600; }
            QLabel#fieldLabel { color: #8293AC; font-size: 10px; font-weight: 600; }
            QLabel#hint { color: #72839D; }
            QLabel#sidebarMetric { color: #F5F8FF; font-size: 19px; font-weight: 700; }
            QLabel#pageTitle { font-size: 27px; font-weight: 700; color: #EEF4FF; }
            QLabel#subtitle { color: #8293AC; }
            QLabel#tabTitle { color: #E6EDF8; font-size: 16px; font-weight: 600; }
            QLabel#infoBanner {
                background: #101F36;
                border: 1px solid #294363;
                border-radius: 8px;
                padding: 10px;
                color: #AAC7F0;
            }
            QFrame#metricCard {
                background: #101A2B;
                border: 1px solid #202E43;
                border-radius: 10px;
            }
            QLabel#metricTitle { color: #7587A2; font-size: 11px; }
            QLabel#metricValue { color: #F1F5FD; font-size: 23px; font-weight: 700; }
            QPushButton {
                background: #131E30;
                color: #DCE7F7;
                border: 1px solid #2A3A52;
                border-radius: 7px;
                padding: 9px 13px;
                font-weight: 600;
            }
            QPushButton:hover { background: #1A2940; border-color: #3B5272; }
            QPushButton:pressed { background: #101827; }
            QPushButton#primaryButton {
                background: #5B7CFA;
                color: #FFFFFF;
                border-color: #5B7CFA;
            }
            QPushButton#primaryButton:hover { background: #6A89FF; }
            QPushButton#secondaryButton { text-align: left; }
            QPushButton#compactButton { padding-left: 8px; padding-right: 8px; }
            QComboBox, QLineEdit, QDateEdit, QDoubleSpinBox {
                background: #111C2E;
                color: #E1EAF7;
                border: 1px solid #2A3A52;
                border-radius: 6px;
                padding: 7px 9px;
                min-height: 20px;
                selection-background-color: #3C5CCC;
            }
            QComboBox:hover, QLineEdit:hover, QDateEdit:hover, QDoubleSpinBox:hover {
                border-color: #466083;
            }
            QComboBox QAbstractItemView {
                background: #111C2E;
                color: #E1EAF7;
                border: 1px solid #2A3A52;
                selection-background-color: #294A7A;
            }
            QTabWidget::pane {
                background: #0E1727;
                border: 1px solid #202E43;
                border-radius: 9px;
                top: -1px;
            }
            QTabBar::tab {
                background: transparent;
                color: #7F91AA;
                padding: 11px 16px;
                border-bottom: 2px solid transparent;
            }
            QTabBar::tab:hover { color: #B8C8DE; }
            QTabBar::tab:selected {
                color: #8EAAFF;
                border-bottom-color: #6D8BFF;
                font-weight: 600;
            }
            QTableView {
                background: #0E1727;
                alternate-background-color: #111C2F;
                color: #D6E0EE;
                border: 1px solid #202E43;
                border-radius: 6px;
                gridline-color: #202E43;
                selection-background-color: #223E6A;
                selection-color: #FFFFFF;
            }
            QHeaderView::section {
                background: #121E30;
                color: #91A2BA;
                border: none;
                border-right: 1px solid #26354B;
                border-bottom: 1px solid #26354B;
                padding: 8px;
                font-weight: 600;
            }
            QTableCornerButton::section { background: #121E30; border: 1px solid #26354B; }
            QScrollBar:vertical { background: #0C1422; width: 10px; margin: 0; }
            QScrollBar::handle:vertical { background: #31415A; min-height: 28px; border-radius: 5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollArea#sidebar QScrollBar:vertical { width: 0; background: transparent; }
            QStatusBar {
                background: #090F1C;
                color: #74869F;
                border-top: 1px solid #202C40;
            }
            """
        )

    def load_demo(self) -> None:
        self.dataframe = create_demo_dataset()
        self.source_name = "demo_sales.csv"
        self.reset_filter(show_message=False)
        self.statusBar().showMessage("Завантажено демонстраційний набір даних", 4000)

    def open_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Відкрити CSV",
            str(PROJECT_ROOT),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        try:
            self.dataframe = load_csv_bytes(Path(path).read_bytes())
        except (OSError, CSVLoadError) as error:
            QMessageBox.critical(self, "Не вдалося відкрити файл", str(error))
            return
        self.source_name = Path(path).name
        self.reset_filter(show_message=False)
        self.statusBar().showMessage(f"Відкрито {self.source_name}", 5000)

    def reset_filter(self, show_message: bool = True) -> None:
        self.filtered_dataframe = self.dataframe.copy()
        self._set_combo_items(self.filter_column, ["Без фільтра", *self.dataframe.columns])
        self.filter_stack.setCurrentIndex(0)
        self.sidebar_filter_state.setText("Усі дані")
        self.refresh_all()
        if show_message:
            self.statusBar().showMessage("Фільтр скинуто", 3000)

    def _configure_filter(self, column: str) -> None:
        if not column or column == "Без фільтра" or column not in self.dataframe.columns:
            self.filter_stack.setCurrentIndex(0)
            return
        series = self.dataframe[column]
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if clean.empty:
                self.filter_stack.setCurrentIndex(0)
                return
            low, high = float(clean.min()), float(clean.max())
            step = max((high - low) / 100, 0.01)
            self.numeric_min.setSingleStep(step)
            self.numeric_max.setSingleStep(step)
            self.numeric_min.setValue(low)
            self.numeric_max.setValue(high)
            self.filter_stack.setCurrentIndex(1)
            return
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        if len(series) and parsed.notna().mean() >= 0.8:
            clean = parsed.dropna()
            self.date_from.setDate(QDate(clean.min().year, clean.min().month, clean.min().day))
            self.date_to.setDate(QDate(clean.max().year, clean.max().month, clean.max().day))
            self.filter_stack.setCurrentIndex(3)
            return
        values = sorted(series.dropna().astype(str).unique().tolist())
        if len(values) <= 100:
            self._set_combo_items(self.category_value, values)
            self.filter_stack.setCurrentIndex(2)
        else:
            self.text_query.clear()
            self.filter_stack.setCurrentIndex(4)

    def apply_filter(self) -> None:
        column = self.filter_column.currentText()
        if column == "Без фільтра" or column not in self.dataframe.columns:
            self.filtered_dataframe = self.dataframe.copy()
        else:
            page = self.filter_stack.currentIndex()
            if page == 1:
                value = (self.numeric_min.value(), self.numeric_max.value())
            elif page == 2:
                value = [self.category_value.currentText()]
            elif page == 3:
                value = (
                    date(
                        self.date_from.date().year(),
                        self.date_from.date().month(),
                        self.date_from.date().day(),
                    ),
                    date(
                        self.date_to.date().year(),
                        self.date_to.date().month(),
                        self.date_to.date().day(),
                    ),
                )
            elif page == 4:
                value = self.text_query.text()
            else:
                value = ""
            self.filtered_dataframe = apply_column_filter(self.dataframe, column, value)
        self.sidebar_filter_state.setText("Усі дані" if column == "Без фільтра" else column)
        self.refresh_all(update_filters=False)
        self.statusBar().showMessage(
            f"Фільтр застосовано: {len(self.filtered_dataframe)} рядків", 4000
        )

    def refresh_all(self, update_filters: bool = True) -> None:
        data = self.filtered_dataframe
        overview = dataset_overview(data)
        self.dataset_label.setText(f"{self.source_name}  •  локальна обробка")
        self.sidebar_dataset.setText(self.source_name or "—")
        self.metric_cards["rows"].value.setText(f"{overview.rows:,}".replace(",", " "))
        self.metric_cards["columns"].value.setText(str(overview.columns))
        self.metric_cards["numeric"].value.setText(str(overview.numeric_columns))
        self.metric_cards["missing"].value.setText(str(overview.missing_cells))
        self.metric_cards["quality"].value.setText(f"{overview.quality_score:.1f}%")
        self.filtered_count.setText(f"Показано рядків: {overview.rows:,}".replace(",", " "))

        self.overview_model.set_dataframe(data.head(500))
        self.overview_note.setText(
            f"Відображено до 500 із {len(data):,} відфільтрованих рядків".replace(",", " ")
        )
        self.quality_model.set_dataframe(missing_values_summary(data))
        self.quality_summary.setText(
            f"Заповненість: {overview.completeness:.1f}%    •    "
            f"Дублікати: {overview.duplicate_rows}    •    "
            f"Пропущені клітинки: {overview.missing_cells}"
        )
        statistics = descriptive_statistics(data).reset_index()
        if not statistics.empty:
            statistics = statistics.rename(columns={statistics.columns[0]: "Колонка"})
        self.statistics_model.set_dataframe(statistics)
        self._draw_correlation()
        if update_filters:
            self._update_analysis_controls()
        self.draw_chart()
        self.update_comparison()
        self.update_history()
        for table in (
            self.overview_table,
            self.quality_table,
            self.statistics_table,
            self.comparison_table,
            self.history_table,
        ):
            table.resizeColumnsToContents()

    def _update_analysis_controls(self) -> None:
        columns = [str(column) for column in self.filtered_dataframe.columns]
        numeric = [
            str(column)
            for column in self.filtered_dataframe.select_dtypes(include="number").columns
        ]
        categorical = [column for column in columns if column not in numeric]
        self._set_combo_items(self.chart_x, columns)
        if numeric:
            self.chart_x.setCurrentText(numeric[0])
        self._set_combo_items(self.chart_y, numeric)
        if len(numeric) > 1:
            self.chart_y.setCurrentIndex(1)
        self._set_combo_items(self.chart_group, [NO_GROUP, *columns])
        self._set_combo_items(self.compare_group, columns)
        if categorical:
            preferred_group = next(
                (
                    column
                    for column in categorical
                    if not pd.api.types.is_datetime64_any_dtype(self.filtered_dataframe[column])
                ),
                categorical[0],
            )
            self.compare_group.setCurrentText(preferred_group)
        self._set_combo_items(self.compare_value, numeric)

    @staticmethod
    def _set_combo_items(combo: QComboBox, items) -> None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems([str(item) for item in items])
        if current in items:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _draw_correlation(self) -> None:
        correlations = correlation_matrix(self.filtered_dataframe)
        axis = self.correlation_canvas.reset(
            "Для кореляції потрібні щонайменше дві числові колонки" if correlations.empty else None
        )
        if correlations.empty:
            self.correlation_canvas.finish()
            return
        image = axis.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1)
        axis.set_xticks(
            range(len(correlations.columns)), correlations.columns, rotation=35, ha="right"
        )
        axis.set_yticks(range(len(correlations.index)), correlations.index)
        for row in range(len(correlations.index)):
            for column in range(len(correlations.columns)):
                axis.text(
                    column,
                    row,
                    f"{correlations.iat[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=("#F3F7FF" if abs(correlations.iat[row, column]) >= 0.45 else "#101827"),
                )
        axis.set_title("Матриця кореляцій")
        self.correlation_canvas.figure.colorbar(image, ax=axis, fraction=0.03, pad=0.03)
        self.correlation_canvas.finish()

    def draw_chart(self) -> None:
        data = self.filtered_dataframe
        if data.empty or not len(data.columns):
            self.chart_canvas.reset("Немає даних для візуалізації")
            self.chart_canvas.finish()
            return
        chart_type = self.chart_type.currentText()
        x_column = self.chart_x.currentText()
        y_column = self.chart_y.currentText()
        group_column = self.chart_group.currentText()
        axis = self.chart_canvas.reset()
        try:
            if chart_type == "Гістограма":
                values = pd.to_numeric(data[x_column], errors="coerce").dropna()
                axis.hist(values, bins=24, color="#6D8BFF", alpha=0.9, edgecolor=SURFACE)
                axis.set_title(f"Розподіл {x_column}")
                axis.set_xlabel(x_column)
                axis.set_ylabel("Кількість")
            elif chart_type == "Box plot":
                if group_column != NO_GROUP and group_column in data.columns:
                    groups = []
                    labels = []
                    for label, frame in list(data.groupby(group_column, dropna=False))[:12]:
                        values = pd.to_numeric(frame[x_column], errors="coerce").dropna()
                        if len(values):
                            groups.append(values)
                            labels.append(str(label))
                    boxes = axis.boxplot(groups, labels=labels, patch_artist=True)
                    axis.tick_params(axis="x", rotation=30)
                else:
                    boxes = axis.boxplot(
                        pd.to_numeric(data[x_column], errors="coerce").dropna(),
                        patch_artist=True,
                    )
                    axis.set_xticks([1], [x_column])
                for box in boxes["boxes"]:
                    box.set_facecolor("#5273E8")
                    box.set_edgecolor("#9CB2FF")
                for median in boxes["medians"]:
                    median.set_color("#F6C760")
                axis.set_title(f"Box plot: {x_column}")
            elif chart_type == "Scatter plot":
                if not y_column:
                    raise ValueError("Оберіть числову колонку Y")
                x = pd.to_numeric(data[x_column], errors="coerce")
                y = pd.to_numeric(data[y_column], errors="coerce")
                mask = x.notna() & y.notna()
                axis.scatter(x[mask], y[mask], s=28, alpha=0.65, color="#6D8BFF")
                axis.set_xlabel(x_column)
                axis.set_ylabel(y_column)
                axis.set_title(f"{y_column} залежно від {x_column}")
            else:
                counts = data[x_column].astype(str).value_counts().head(20)
                axis.bar(counts.index, counts.values, color="#6D8BFF")
                axis.tick_params(axis="x", rotation=35)
                axis.set_title(f"Кількість записів за {x_column}")
                axis.set_ylabel("Кількість")
            axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.85)
        except (KeyError, TypeError, ValueError) as error:
            axis = self.chart_canvas.reset(str(error))
        self.chart_canvas.finish()

    def update_comparison(self) -> None:
        group = self.compare_group.currentText()
        value = self.compare_value.currentText()
        aggregation = self.compare_aggregation.currentText()
        if not group or not value:
            self.comparison_model.set_dataframe(pd.DataFrame())
            self.comparison_canvas.reset("Оберіть колонки для порівняння")
            self.comparison_canvas.finish()
            return
        try:
            comparison = group_comparison(
                self.filtered_dataframe,
                group,
                value,
                aggregation,
            )
        except (KeyError, TypeError, ValueError) as error:
            self.comparison_model.set_dataframe(pd.DataFrame())
            self.comparison_canvas.reset(str(error))
            self.comparison_canvas.finish()
            return
        self.comparison_model.set_dataframe(comparison)
        chart = comparison.head(20)
        result_column = f"{value}_{aggregation}"
        axis = self.comparison_canvas.reset()
        axis.bar(chart[group].astype(str), chart[result_column], color="#6D8BFF")
        axis.set_title(f"{aggregation}: {value} за {group}")
        axis.set_ylabel(result_column)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        self.comparison_canvas.finish()

    def update_history(self) -> None:
        self.history_model.set_dataframe(self.history.list_recent())

    def save_snapshot(self) -> None:
        overview = dataset_overview(self.filtered_dataframe)
        payload = f"{self.source_name}:{self.filtered_dataframe.shape}:{list(self.filtered_dataframe.columns)}"
        dataset_key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        self.history.save(self.source_name, overview, dataset_key)
        self.update_history()
        self.statusBar().showMessage("Аналітичний знімок збережено в SQLite", 5000)

    def export_csv(self) -> None:
        default_name = f"filtered_{self.source_name or 'dataset.csv'}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Експортувати CSV",
            str(PROJECT_ROOT / default_name),
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            self.filtered_dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        except OSError as error:
            QMessageBox.critical(self, "Не вдалося зберегти файл", str(error))
            return
        self.statusBar().showMessage(f"Експортовано: {path}", 6000)

    def _tab_changed(self, index: int) -> None:
        if index == 3:
            self.draw_chart()
        elif index == 4:
            self.update_comparison()
        elif index == 5:
            self.update_history()

    def show_about(self) -> None:
        QMessageBox.information(
            self,
            "Про DataLens",
            "DataLens Desktop\n\nЛокальний інструмент для аналізу, статистики, "
            "порівняння та візуалізації CSV-даних.",
        )


def run() -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName("DataLens")
    application.setOrganizationName("Semester Project")
    palette = application.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#080E1A"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor("#0E1727"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#111C2F"))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor("#131E30"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#5B7CFA"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    application.setPalette(palette)
    window = DataLensWindow()
    window.show()
    return application.exec()
