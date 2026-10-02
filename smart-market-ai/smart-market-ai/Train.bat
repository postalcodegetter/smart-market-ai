@echo off
title Smart Market AI - Train
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo         Run install.bat first, then try again.
  pause
  exit /b 1
)

echo ============================================================
echo  SMART MARKET AI - TRAINING
echo  Uses default settings. First run downloads model weights
echo  and live market data, so it can take a while.
echo ============================================================
".venv\Scripts\python.exe" quick_train.py %*
echo.
echo Training finished. Close this window or run Dashboard.bat.
pause
