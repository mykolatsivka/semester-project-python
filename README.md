# DataLens Desktop

DataLens — нативна Windows-програма для дослідницького аналізу CSV-даних. Вона
працює локально, відкривається в окремому вікні та не потребує браузера.

## Можливості

- відкриття CSV або вбудованого демонстраційного набору даних;
- огляд таблиці, структури, пропусків, дублікатів і показника якості;
- фільтрація числових, категоріальних, текстових і датованих колонок;
- описова статистика та матриця кореляцій;
- гістограми, box plot, scatter plot і стовпчикові діаграми;
- порівняння груп за середнім, медіаною, сумою, мінімумом, максимумом або кількістю;
- експорт відфільтрованої вибірки в CSV;
- локальна історія аналітичних знімків у SQLite.

## Встановлення

```powershell
cd "C:\IT\Study\Semester project Python"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Запуск

```powershell
cd "C:\IT\Study\Semester project Python"
.\.venv\Scripts\python.exe app.py
```

## Створення Windows EXE

```powershell
.\build_desktop.ps1
```

Готовий застосунок з'явиться у `dist\DataLens\DataLens.exe`.

## Тести

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Вміст CSV обробляється лише локально. В SQLite зберігаються тільки метадані
аналізу, а не сам набір даних.
