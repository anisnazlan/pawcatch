@echo off
REM Double-click to start WildWatch, then the dashboard opens in your browser.
REM Close this window (or press Ctrl+C) to stop the app.
cd /d "%~dp0"
title WildWatch Elmina server
echo Starting WildWatch Elmina...
python -m backend.generate_data --if-empty
start "" cmd /c "timeout /t 4 >nul & start http://localhost:8000/dashboard"
python -m uvicorn backend.app:app --reload
pause
