@echo off
rem Builds a single OpAmpLab.exe (no Python needed on the target PC). Result: dist\OpAmpLab.exe
cd /d "%~dp0"
py -3 -m pip install -r requirements.txt pyinstaller
py -3 -m PyInstaller --noconsole --onefile --name OpAmpLab --hidden-import matplotlib.backends.backend_tkagg main.py
echo.
echo Done. Your program is in the dist folder.
pause
