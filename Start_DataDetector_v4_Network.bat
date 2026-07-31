@echo off
title DataDetector Headless Multi-Camera TCP Server - Watchdog [24/7]
cd /d "%~dp0"

:: ============================================================
::  WATCHDOG: Startet die Multi-Kamera App automatisch neu
:: ============================================================

:WATCHDOG_LOOP
echo.
echo ============================================================
echo  DataDetector Headless Multi-Camera Watchdog - %date% %time%
echo  Starte TCP-Server fuer Kamera 1 (Port 9500) und Kamera 2 (Port 9501)...
echo  (Dieses Fenster schliessen = Server beenden)
echo ============================================================
echo.

:: Virtuelle Umgebung aktivieren
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

:: Multi-Kamera App mit zentraler config.json starten
python vision_app_v4_network.py config.json

:: Wenn wir hier ankommen, ist die App gestorben
echo.
echo [WATCHDOG] Server beendet (Exit Code: %errorlevel%). Neustart in 3 Sekunden...
timeout /t 3 /nobreak >nul

:: Zurueck zum Start
goto WATCHDOG_LOOP
