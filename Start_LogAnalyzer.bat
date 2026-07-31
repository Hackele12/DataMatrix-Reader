@echo off
title Log Analyzer - DataDetector v4
cd /d "%~dp0"

:: Prüfen ob virtuelle Umgebung existiert
if not exist ".venv" (
    echo [FEHLER] Virtuelle Umgebung '.venv' wurde nicht gefunden!
    echo Bitte fuehre zuerst 'setup.bat' aus.
    pause
    exit /b
)

echo Starte Log Analyzer App...
call ".venv\Scripts\activate.bat"
start "" pythonw log_analyzer_app.py
exit
