@echo off
title DataDetector Headless - CAMERA 1 [Port 9500]
cd /d "%~dp0"

:: ============================================================
::  WATCHDOG: Startet die App automatisch neu nach jedem Crash
:: ============================================================

:WATCHDOG_LOOP
echo.
echo ============================================================
echo  DataDetector Headless - CAMERA 1 Watchdog - %date% %time%
echo  Konfiguration: config_cam1.json (Port 9500)
echo  (Dieses Fenster schliessen = Server beenden)
echo ============================================================
echo.

:: Virtuelle Umgebung aktivieren
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

:: Headless App mit config_cam1.json starten
python vision_app_v4_network.py config_cam1.json

:: Wenn wir hier ankommen, ist die App gestorben
echo.
echo [WATCHDOG] Camera 1 beendet (Exit Code: %errorlevel%). Neustart in 3 Sekunden...
timeout /t 3 /nobreak >nul

:: Zurueck zum Start
goto WATCHDOG_LOOP
