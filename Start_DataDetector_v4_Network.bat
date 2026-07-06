@echo off
title DataDetector Headless TCP Server - Watchdog [24/7]
cd /d "%~dp0"

:: ============================================================
::  WATCHDOG: Startet die App automatisch neu nach jedem Crash
:: ============================================================

:WATCHDOG_LOOP
echo.
echo ============================================================
echo  DataDetector Headless Network Watchdog - %date% %time%
echo  Starte TCP-Server an Port 9500...
echo  (Dieses Fenster schliessen = Server beenden)
echo ============================================================
echo.

:: Virtuelle Umgebung aktivieren
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

:: Headless App starten
python vision_app_v4_network.py

:: Wenn wir hier ankommen, ist die App gestorben
echo.
echo [WATCHDOG] Server beendet (Exit Code: %errorlevel%). Neustart in 3 Sekunden...
timeout /t 3 /nobreak >nul

:: Zurueck zum Start
goto WATCHDOG_LOOP
