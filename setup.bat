@echo off
title DataDetector v17 - Automatisches Setup
cd /d "%~dp0"

echo ============================================================
echo  DataDetector v17 - Installation der Abhängigkeiten
echo ============================================================
echo.

:: Prüfen, ob Python oder py im Pfad ist
set "PYTHON_CMD="
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python"
) else (
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_CMD=py"
    )
)

if "%PYTHON_CMD%"=="" (
    echo [FEHLER] Python wurde auf diesem PC nicht gefunden!
    echo Bitte stelle sicher, dass Python installiert ist und bei der
    echo Installation das Haeckchen bei "Add Python to PATH" gesetzt wurde.
    echo.
    pause
    exit /b
)

:: 1. Virtuelle Umgebung erstellen
if not exist ".venv" (
    echo [1/3] Erstelle virtuelle Umgebung venv...
    %PYTHON_CMD% -m venv .venv
    if %errorlevel% neq 0 (
        echo [FEHLER] Konnte virtuelle Umgebung nicht erstellen.
        pause
        exit /b
    )
) else (
    echo [1/3] Virtuelle Umgebung venv existiert bereits.
)

:: 2. Pip und Abhängigkeiten installieren
echo [2/3] Installiere Bibliotheken...
call ".venv\Scripts\activate.bat"

if exist "packages" (
    echo [INFO] Offline-Modus: Installiere Bibliotheken lokal aus dem Ordner 'packages'...
    python -m pip install --no-index --find-links=packages -r requirements.txt
) else (
    echo [INFO] Online-Modus: Versuche Online-Download aus dem Internet...
    echo (Dies kann je nach Internetverbindung 1-2 Minuten dauern)
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
)

if %errorlevel% neq 0 (
    echo.
    echo [FEHLER] Installation der Bibliotheken fehlgeschlagen!
    pause
    exit /b
)

echo.
echo [3/3] Setup erfolgreich abgeschlossen!
echo Sie koennen das Programm jetzt ueber "Start_DataDetector.bat" oder "Start_DataDetector_v4_Network.bat" starten.
echo.
pause

