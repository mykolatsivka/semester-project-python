from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datalens.analysis import (  # noqa: E402
    correlation_matrix,
    dataset_overview,
    descriptive_statistics,
    group_comparison,
    missing_values_summary,
)
from datalens.data_loader import CSVLoadError, load_csv_bytes  # noqa: E402
from datalens.database import AnalysisHistory  # noqa: E402
from datalens.demo import create_demo_dataset  # noqa: E402
from datalens.filters import apply_column_filter  # noqa: E402

st.set_page_config(
    page_title="DataLens",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_demo_dataset() -> pd.DataFrame:
    return create_demo_dataset()


@st.cache_data(show_spinner=False)
def cached_csv_dataset(file_bytes: bytes) -> pd.DataFrame:
    return load_csv_bytes(file_bytes)


def dataframe_fingerprint(dataframe: pd.DataFrame, source_name: str) -> str:
    payload = f"{source_name}:{dataframe.shape}:{list(dataframe.columns)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def render_filter(dataframe: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.subheader("Фільтр")
    selected_column = st.sidebar.selectbox(
        "Колонка",
        options=["Без фільтра", *dataframe.columns.tolist()],
    )
    if selected_column == "Без фільтра":
        return dataframe

    series = dataframe[selected_column]
    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if clean.empty:
            st.sidebar.info("У колонці немає числових значень.")
            return dataframe
        minimum = float(clean.min())
        maximum = float(clean.max())
        if minimum == maximum:
            st.sidebar.info("Усі числові значення однакові.")
            return dataframe
        bounds = st.sidebar.slider(
            "Діапазон значень",
            min_value=minimum,
            max_value=maximum,
            value=(minimum, maximum),
        )
        return apply_column_filter(dataframe, selected_column, bounds)

    datetime_values = pd.to_datetime(series, errors="coerce", format="mixed")
    if datetime_values.notna().mean() >= 0.8:
        clean = datetime_values.dropna()
        date_bounds = st.sidebar.date_input(
            "Діапазон дат",
            value=(clean.min().date(), clean.max().date()),
        )
        if len(date_bounds) == 2:
            return apply_column_filter(dataframe, selected_column, date_bounds)
        return dataframe

    values = sorted(series.dropna().astype(str).unique().tolist())
    if len(values) > 100:
        query = st.sidebar.text_input("Текст містить")
        return apply_column_filter(dataframe, selected_column, query) if query else dataframe
    selected_values = st.sidebar.multiselect("Значення", values, default=values)
    return apply_column_filter(dataframe, selected_column, selected_values)


def plot_builder(dataframe: pd.DataFrame) -> None:
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    all_columns = dataframe.columns.tolist()
    chart_type = st.selectbox(
        "Тип графіка",
        ["Гістограма", "Box plot", "Scatter plot", "Стовпчикова діаграма"],
    )

    if chart_type == "Гістограма":
        if not numeric_columns:
            st.warning("Для гістограми потрібна числова колонка.")
            return
        column = st.selectbox("Числова колонка", numeric_columns)
        color = st.selectbox("Групування кольором", ["Без групування", *all_columns])
        figure = px.histogram(
            dataframe,
            x=column,
            color=None if color == "Без групування" else color,
            marginal="box",
            title=f"Розподіл {column}",
        )
    elif chart_type == "Box plot":
        if not numeric_columns:
            st.warning("Для box plot потрібна числова колонка.")
            return
        value = st.selectbox("Значення", numeric_columns)
        group = st.selectbox("Група", ["Без групування", *all_columns])
        figure = px.box(
            dataframe,
            x=None if group == "Без групування" else group,
            y=value,
            color=None if group == "Без групування" else group,
            points="outliers",
            title=f"Розподіл {value} за групами",
        )
    elif chart_type == "Scatter plot":
        if len(numeric_columns) < 2:
            st.warning("Для scatter plot потрібні щонайменше дві числові колонки.")
            return
        x_column = st.selectbox("Вісь X", numeric_columns, index=0)
        y_column = st.selectbox("Вісь Y", numeric_columns, index=1)
        color = st.selectbox("Колір", ["Без групування", *all_columns])
        figure = px.scatter(
            dataframe,
            x=x_column,
            y=y_column,
            color=None if color == "Без групування" else color,
            opacity=0.75,
            title=f"{y_column} залежно від {x_column}",
        )
    else:
        category = st.selectbox("Категорія", all_columns)
        value_options = ["Кількість записів", *numeric_columns]
        value = st.selectbox("Показник", value_options)
        if value == "Кількість записів":
            chart_data = dataframe[category].astype(str).value_counts().head(30).reset_index()
            chart_data.columns = [category, "count"]
            figure = px.bar(chart_data, x=category, y="count", title=f"Кількість за {category}")
        else:
            chart_data = (
                dataframe.groupby(category, dropna=False)[value].mean().reset_index().head(30)
            )
            figure = px.bar(
                chart_data,
                x=category,
                y=value,
                title=f"Середнє {value} за {category}",
            )
    figure.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(figure, width="stretch")


st.title("DataLens")
st.caption("Інтерактивний аналіз, статистика, порівняння та візуалізація CSV-даних")

st.sidebar.header("Джерело даних")
source_mode = st.sidebar.radio(
    "Оберіть джерело",
    ["Демонстраційний dataset", "Завантажити CSV"],
)

source_name: str
try:
    if source_mode == "Демонстраційний dataset":
        dataframe = cached_demo_dataset()
        source_name = "demo_sales.csv"
        st.sidebar.caption("Вбудовані демонстраційні дані про продажі")
    else:
        uploaded_file = st.sidebar.file_uploader("CSV до 25 MB", type=["csv"])
        if uploaded_file is None:
            st.info("Завантажте CSV-файл у боковій панелі або відкрийте демонстраційний dataset.")
            st.stop()
        file_bytes = uploaded_file.getvalue()
        dataframe = cached_csv_dataset(file_bytes)
        source_name = uploaded_file.name
except CSVLoadError as error:
    st.error(str(error))
    st.stop()

filtered_dataframe = render_filter(dataframe)
overview = dataset_overview(filtered_dataframe)
history = AnalysisHistory(PROJECT_ROOT / "data" / "datalens.db")

st.sidebar.divider()
st.sidebar.metric("Показано рядків", f"{len(filtered_dataframe):,}".replace(",", " "))
csv_download = filtered_dataframe.to_csv(index=False).encode("utf-8-sig")
st.sidebar.download_button(
    "Завантажити відфільтровані дані",
    data=csv_download,
    file_name=f"filtered_{source_name}",
    mime="text/csv",
    width="stretch",
)

save_key = dataframe_fingerprint(filtered_dataframe, source_name)
if st.sidebar.button("Зберегти знімок аналізу", width="stretch"):
    history.save(source_name, overview, save_key)
    st.sidebar.success("Знімок збережено в SQLite")

metric_columns = st.columns(5)
metric_columns[0].metric("Рядки", f"{overview.rows:,}".replace(",", " "))
metric_columns[1].metric("Колонки", overview.columns)
metric_columns[2].metric("Числові", overview.numeric_columns)
metric_columns[3].metric("Пропуски", overview.missing_cells)
metric_columns[4].metric("Якість", f"{overview.quality_score:.1f}%")

tabs = st.tabs(["Огляд", "Якість даних", "Статистика", "Візуалізація", "Порівняння", "Історія"])

with tabs[0]:
    st.subheader("Попередній перегляд")
    st.dataframe(filtered_dataframe.head(200), width="stretch", hide_index=True)
    st.caption(f"Відображено до 200 із {len(filtered_dataframe)} відфільтрованих рядків")
    schema = pd.DataFrame(
        {
            "Колонка": filtered_dataframe.columns,
            "Тип": filtered_dataframe.dtypes.astype(str).values,
            "Унікальних": [
                filtered_dataframe[column].nunique(dropna=True) for column in filtered_dataframe
            ],
        }
    )
    st.subheader("Схема dataset")
    st.dataframe(schema, width="stretch", hide_index=True)

with tabs[1]:
    quality_columns = st.columns(3)
    quality_columns[0].metric("Пропущені клітинки", overview.missing_cells)
    quality_columns[1].metric("Дублікати рядків", overview.duplicate_rows)
    quality_columns[2].metric("Заповненість", f"{overview.completeness:.1f}%")
    missing_summary = missing_values_summary(filtered_dataframe)
    st.subheader("Пропуски за колонками")
    st.dataframe(missing_summary, width="stretch", hide_index=True)
    if overview.duplicate_rows:
        st.subheader("Приклади дублікатів")
        st.dataframe(
            filtered_dataframe[filtered_dataframe.duplicated(keep=False)].head(100),
            width="stretch",
            hide_index=True,
        )

with tabs[2]:
    st.subheader("Описова статистика")
    st.dataframe(descriptive_statistics(filtered_dataframe), width="stretch")
    correlations = correlation_matrix(filtered_dataframe)
    st.subheader("Кореляції")
    if correlations.empty:
        st.info("Для кореляційного аналізу потрібні щонайменше дві числові колонки.")
    else:
        heatmap = px.imshow(
            correlations,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Матриця кореляцій",
        )
        st.plotly_chart(heatmap, width="stretch")

with tabs[3]:
    plot_builder(filtered_dataframe)

with tabs[4]:
    numeric_columns = filtered_dataframe.select_dtypes(include="number").columns.tolist()
    group_columns = filtered_dataframe.columns.tolist()
    if not numeric_columns or not group_columns:
        st.info("Для порівняння потрібні категоріальна та числова колонки.")
    else:
        left, middle, right = st.columns(3)
        group_column = left.selectbox("Групувати за", group_columns)
        value_column = middle.selectbox("Числовий показник", numeric_columns)
        aggregation = right.selectbox(
            "Агрегація",
            ["mean", "median", "sum", "min", "max", "count"],
        )
        comparison = group_comparison(
            filtered_dataframe,
            group_column,
            value_column,
            aggregation,
        )
        st.dataframe(comparison, width="stretch", hide_index=True)
        comparison_chart = px.bar(
            comparison.head(30),
            x=group_column,
            y=f"{value_column}_{aggregation}",
            title=f"{aggregation}: {value_column} за {group_column}",
        )
        st.plotly_chart(comparison_chart, width="stretch")

with tabs[5]:
    st.subheader("Збережені знімки")
    recent = history.list_recent(limit=50)
    if recent.empty:
        st.info("Історія порожня. Збережіть перший знімок у боковій панелі.")
    else:
        st.dataframe(recent, width="stretch", hide_index=True)
