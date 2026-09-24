$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Не знайдено .venv. Спочатку створіть середовище та встановіть залежності."
}

Push-Location $ProjectRoot
try {
    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --windowed `
        --name DataLens `
        --paths src `
        --collect-all matplotlib `
        app.py
}
finally {
    Pop-Location
}

Write-Host "Готово: dist\DataLens\DataLens.exe"
