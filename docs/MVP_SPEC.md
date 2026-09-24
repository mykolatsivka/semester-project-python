# DataLens MVP specification

## Problem

Exploratory data analysis usually requires programming knowledge and repeated manual work. DataLens gives a student or non-technical user one interface for inspecting a CSV dataset, finding quality problems, calculating statistics, comparing groups, and creating interactive visualizations.

## MVP goal

Deliver a locally runnable web application that demonstrates the main project idea and meaningfully uses Data Science, a web user interface, and a relational database.

## User scenarios

1. Open the built-in sales dataset and immediately see its size and quality indicators.
2. Upload a CSV file and inspect its schema, missing values, duplicates, and descriptive statistics.
3. Filter the dataset and download the filtered records.
4. Build a histogram, box plot, scatter plot, or bar chart without writing code.
5. Compare a numeric indicator across groups with different aggregation methods.
6. View a correlation matrix for numeric columns.
7. Save metadata about an analysis to SQLite and view recent snapshots.

## MVP acceptance criteria

- The application starts with `streamlit run app.py`.
- The demonstration dataset works without network access.
- A valid CSV can be uploaded and analyzed.
- Invalid or empty files produce a readable error instead of a crash.
- Data-quality metrics, descriptive statistics, comparison, and at least four visualization types are available.
- Analysis history persists in SQLite.
- Core analysis, loading, filtering, and database behavior is covered by automated tests.

## Planned after MVP

- Multi-step cleaning with an undoable transformation log.
- Saved dashboard configurations.
- Statistical hypothesis tests and outlier detection.
- PDF or HTML analytical report generation.
- Optional external dataset API.
- Authentication and multiple local workspaces if the project scope permits.

