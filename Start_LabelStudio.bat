@echo off
title DataDetector - Label Studio
cd /d "%~dp0"

echo ============================================================
echo  Starte Label Studio fuer Etiketten-Beschriftung...
echo ============================================================
echo.

:: Virtuelle Umgebung aktivieren
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

:: Pruefen ob label-studio installiert ist, andernfalls installieren
pip show label-studio >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Label Studio wird in der virtuellen Umgebung installiert...
    pip install "django-environ>=0.14.0" "django-csp==3.7" "Django>=5.1.8,<5.2.0" label-studio
)

echo Starten von Label Studio...
label-studio start --port 8080
