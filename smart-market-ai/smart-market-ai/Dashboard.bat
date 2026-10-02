@echo off
title Smart Market AI - Dashboard
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo         Run install.bat first, then try again.
  pause
  exit /b 1
)

echo ============================================================
echo  SMART MARKET AI - DASHBOARD
echo  A browser window will open at http://localhost:8501
echo  Press Ctrl+C here to stop the dashboard.
echo ============================================================
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
pause
