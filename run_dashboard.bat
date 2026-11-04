@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python environment missing. Set up .venv in this project folder first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run dashboard.py
pause