@echo off
title DataDetector - Watchdog [24/7]
cd /d "%~dp0"

:: ============================================================
::  WATCHDOG: Startet die App automatisch neu nach jedem Crash
:: ============================================================

:WATCHDOG_LOOP
echo.
echo ============================================================
echo  DataDetector Watchdog - %date% %time%
echo  Starte App... (Dieses Fenster schliessen = App beenden)
echo ============================================================
echo.

:: Virtuelle Umgebung aktivieren
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

:: App starten
python vision_app.py

:: Wenn wir hier ankommen, ist die App gestorben
echo.
echo [WATCHDOG] App beendet (Exit Code: %errorlevel%). Neustart in 2 Sekunden...
timeout /t 2 /nobreak >nul

:: Zurueck zum Start
goto WATCHDOG_LOOP
