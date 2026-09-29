@echo off
echo ============================================
echo  DataDetector - Benchmark ^& Annotation Tool
echo ============================================
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo Starte Benchmark GUI...
    .venv\Scripts\python.exe benchmark_gui.py %*
) else (
    echo FEHLER: Python venv nicht gefunden!
    echo Bitte zuerst setup.bat ausfuehren.
    pause
)
