@echo off
title DataDetector - KI Training (YOLOv10)
cd /d "%~dp0"

echo ============================================================
echo  Starte Training der YOLOv10-KI...
echo ============================================================
echo.

:: Virtuelle Umgebung aktivieren
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

python train.py

echo.
echo Training abgeschlossen!
pause
