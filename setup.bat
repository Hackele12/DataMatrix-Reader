@echo off
setlocal EnableExtensions
title DataDetector v24 - Setup
cd /d "%~dp0"
set "PYTHONUTF8=1"

echo ============================================================
echo  DataDetector v24 - Einrichtung
echo ============================================================
echo.

if not exist "scanner\__init__.py" (
    echo [FEHLER] Programmdateien nicht gefunden.
    echo Bitte die ZIP-Datei zuerst vollstaendig entpacken ^(Rechtsklick - Alle extrahieren^)
    echo und setup.bat im entpackten Ordner starten.
    echo.
    pause
    exit /b 1
)

set "CUR_DIR=%CD%\"
if /i not "%CUR_DIR:\AppData\Local\Temp\=%"=="%CUR_DIR%" (
    echo [HINWEIS] Der Ordner liegt in einem temporaeren Verzeichnis.
    echo           Empfehlung: nach z.B. C:\DataDetector verschieben und setup.bat dort starten.
    echo.
)

:: Python-Laufzeit: mitgelieferte Version bevorzugen (die Offline-Pakete sind fuer Python 3.14 gebaut)
set "BASE_PY="
if exist "python_runtime\python.exe" set "BASE_PY=python_runtime\python.exe"
if not defined BASE_PY py -3.14 -c "import sys" >nul 2>&1 && set "BASE_PY=py -3.14"
if not defined BASE_PY python -c "import sys; sys.exit(sys.version_info[:2] != (3, 14))" >nul 2>&1 && set "BASE_PY=python"
if not defined BASE_PY (
    echo [FEHLER] Keine Python-Laufzeit gefunden.
    echo Der Ordner "python_runtime" fehlt - bitte die ZIP-Datei vollstaendig entpacken.
    echo.
    pause
    exit /b 1
)
echo [INFO] Python-Laufzeit: %BASE_PY%

:: 1. Virtuelle Umgebung (defekte oder falsche Version, z.B. nach Verschieben des Ordners, wird neu erstellt)
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info[:2] != (3, 14))" >nul 2>&1
    if errorlevel 1 (
        echo [INFO] Vorhandene .venv ist nicht lauffaehig - wird neu erstellt.
        rmdir /s /q ".venv"
    )
)
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Erstelle virtuelle Umgebung .venv ...
    %BASE_PY% -m venv .venv
    if errorlevel 1 (
        echo [FEHLER] Virtuelle Umgebung konnte nicht erstellt werden.
        echo.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtuelle Umgebung .venv ist vorhanden.
)
set "VENV_PY=.venv\Scripts\python.exe"

:: 2. Bibliotheken installieren
set "OFFLINE=0"
if exist "requirements_lock.txt" if exist "packages\" set "OFFLINE=1"
if "%OFFLINE%"=="1" (
    echo [2/4] Installiere Bibliotheken offline aus "packages" - das dauert einige Minuten ...
    "%VENV_PY%" -m pip install --no-index --find-links packages -r requirements_lock.txt --disable-pip-version-check --no-warn-script-location
) else (
    echo [2/4] Kein Offline-Paketordner gefunden - installiere online aus dem Internet ...
    "%VENV_PY%" -m pip install -r requirements.txt --disable-pip-version-check --no-warn-script-location
)
if errorlevel 1 (
    echo.
    echo [FEHLER] Installation der Bibliotheken fehlgeschlagen.
    echo TIPP bei Pfadlaengen-Fehler - WinError 206: Ordner in ein kurzes Verzeichnis
    echo wie C:\DataDetector verschieben und setup.bat erneut starten.
    echo.
    pause
    exit /b 1
)

:: opencv-python und opencv-python-headless liefern beide "cv2" - die Vollversion zuletzt installieren
set "OPENCV_PIN="
if "%OFFLINE%"=="1" for /f "usebackq delims=" %%L in (`findstr /b /i /c:"opencv-python==" requirements_lock.txt`) do set "OPENCV_PIN=%%L"
if defined OPENCV_PIN (
    "%VENV_PY%" -m pip install --no-index --find-links packages --force-reinstall --no-deps "%OPENCV_PIN%" --disable-pip-version-check -q
    if errorlevel 1 (
        echo [FEHLER] OpenCV konnte nicht installiert werden.
        echo.
        pause
        exit /b 1
    )
)

:: 3. Pruefung
echo [3/4] Pruefe Installation ...
"%VENV_PY%" -c "import cv2, numpy, torch, ultralytics, easyocr, zxingcpp, onnxruntime, customtkinter, tkinter, setuptools; from pylibdmtx import pylibdmtx; import scanner; print('      OpenCV', cv2.__version__, '/ Torch', torch.__version__, '/ Ultralytics', ultralytics.__version__)"
if errorlevel 1 (
    echo [FEHLER] Pruefung der Bibliotheken fehlgeschlagen - siehe Meldung oben.
    echo.
    pause
    exit /b 1
)
"%VENV_PY%" -c "from ids_peak import ids_peak; from ids_peak_ipl import ids_peak_ipl" >nul 2>&1
if errorlevel 1 (
    echo [HINWEIS] IDS-Kamerabibliothek nicht ladbar. Fuer den Kamerabetrieb muss die
    echo           Software "IDS peak" auf diesem PC installiert sein.
)

echo.
echo [4/4] Setup erfolgreich abgeschlossen.
echo.
echo Programm starten mit Start_DataDetector.bat
echo   Startfenster fuer DataMatrixReader, Benchmark und Log Analyzer
echo.
pause

