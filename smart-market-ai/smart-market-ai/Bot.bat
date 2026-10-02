@echo off
title Smart Market AI - Bot
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo         Run install.bat first, then try again.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo  SMART MARKET AI - AUTO TRADING BOT
echo  Long/short portfolio book, fractional shares via Alpaca.
echo  Paper-safe by default (needs ALPACA_LIVE=1 for real orders).
echo  To really trade, edit bot_config.json ^(live: true^) and set
echo  ALPACA_LIVE=1 in .env. Real shorts need a margin account.
echo  Press Ctrl+C to stop. Place a STOP_BOT file to flatten.
echo ============================================================
echo.

if "%1"=="" (
  ".venv\Scripts\python.exe" trade_bot.py
) else (
  ".venv\Scripts\python.exe" trade_bot.py %*
)
pause
