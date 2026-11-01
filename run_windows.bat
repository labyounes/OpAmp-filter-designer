@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python 3 was not found. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in the installer, then run this file again.
  pause
  exit /b 1
)
py -3 -c "import numpy, scipy, matplotlib" 2>nul || py -3 -m pip install -r requirements.txt
py -3 main.py
if errorlevel 1 pause
