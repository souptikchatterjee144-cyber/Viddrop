@echo off
title VidDrop Companion

echo Checking for Python...
python --version 2>nul
if errorlevel 1 (
    echo.
    echo  Python not found!
    echo  Install Python 3.8+ from https://python.org
    echo.
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt --quiet

cls
echo ================================
echo   VidDrop Companion Starting...
echo ================================
echo.

python companion.py

pause
