# DataLens

DataLens is an interactive web application for exploratory analysis of CSV datasets. It helps a user inspect data quality, calculate descriptive statistics, compare groups, filter records, visualize distributions and relationships, and keep a local history of analyses.

The project is being developed as a semester project for the Python programming course. The current `mvp` branch contains the first functional version.

## MVP functionality

- Upload a CSV file or open the built-in demonstration dataset.
- Inspect dataset dimensions, column roles, missing values, duplicates, and a data-quality score.
- Filter rows by numeric, categorical, or date columns.
- Calculate descriptive statistics and a correlation matrix.
- Build interactive histograms, box plots, scatter plots, and bar charts.
- Compare groups using count, mean, median, sum, minimum, or maximum.
- Download the filtered dataset as CSV.
- Save and review analysis snapshots in a local SQLite database.

## Technology areas

1. Data Science: pandas and NumPy.
2. Web user interface: Streamlit and Plotly.
3. Databases: SQLite through Python's standard `sqlite3` module.

## Project structure

```text
.
├── app.py
├── src/datalens/
│   ├── analysis.py
│   ├── data_loader.py
│   ├── database.py
│   ├── demo.py
│   └── filters.py
├── tests/
├── data/
├── docs/
└── pyproject.toml
```

## Installation

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Running

```powershell
streamlit run app.py
```

The application opens at `http://localhost:8501`. Start with the demonstration dataset or upload a CSV file up to 25 MB.

## Tests

```powershell
pytest
```

## Data and privacy

Uploaded CSV content is processed in the running application and is not sent to an external API. The SQLite history stores dataset metadata only: file name, dimensions, missing-cell count, duplicate-row count, and the calculated quality score.

