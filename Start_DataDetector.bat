@echo off
:: Startet das DataDetector-Startfenster (DataMatrixReader, Benchmark, Log Analyzer) ohne Konsolenfenster.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [FEHLER] Die Python-Umgebung .venv wurde nicht gefunden.
    echo Bitte zuerst setup.bat ausfuehren.
    echo.
    pause
    exit /b 1
)

:: Ohne Konsolenfenster starten (pythonw.exe einer uv-.venv ist selbst ein Konsolenprogramm)
".venv\Scripts\python.exe" -c "import subprocess, sys; subprocess.Popen([sys.executable, 'launcher.py'], creationflags=subprocess.CREATE_NO_WINDOW)"
