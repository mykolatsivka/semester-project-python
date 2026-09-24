from __future__ import annotations

import csv
import io

import pandas as pd

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "latin-1")


class CSVLoadError(ValueError):
    """Raised when uploaded bytes cannot be loaded as a usable CSV dataset."""


def _decode(file_bytes: bytes) -> str:
    if not file_bytes:
        raise CSVLoadError("CSV-файл порожній.")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise CSVLoadError("CSV-файл перевищує ліміт 25 MB.")
    for encoding in SUPPORTED_ENCODINGS:
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise CSVLoadError("Не вдалося визначити кодування CSV-файлу.")


def _detect_delimiter(text: str) -> str:
    sample = text[:10_000]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    text = _decode(file_bytes)
    delimiter = _detect_delimiter(text)
    try:
        dataframe = pd.read_csv(io.StringIO(text), sep=delimiter)
    except (pd.errors.ParserError, ValueError) as error:
        raise CSVLoadError(f"Не вдалося прочитати CSV: {error}") from error

    if dataframe.empty and len(dataframe.columns) == 0:
        raise CSVLoadError("CSV не містить колонок.")
    if any(str(column).startswith("Unnamed:") for column in dataframe.columns):
        dataframe = dataframe.drop(
            columns=[column for column in dataframe.columns if str(column).startswith("Unnamed:")]
        )
    dataframe.columns = [
        str(column).strip() or f"column_{index + 1}"
        for index, column in enumerate(dataframe.columns)
    ]
    return dataframe
