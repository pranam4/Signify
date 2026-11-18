@echo off
title SignBridge Backend (Keep This Window Open)
cd /d "%~dp0backend"

if not exist ".venv\Scripts\python.exe" (
    echo First-time setup: creating Python virtual environment and installing dependencies...
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 (
        echo Python 3.10+ was not found. Install Python from https://www.python.org/downloads/ and check "Add Python to PATH".
        pause
        exit /b 1
    )
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install Python dependencies.
        pause
        exit /b 1
    )
)

echo Starting SignBridge MediaPipe + Whisper backend on http://localhost:8000 ...
echo Please wait ~15-20 seconds until you see "Uvicorn running on http://127.0.0.1:8000".
echo KEEP THIS WINDOW OPEN while using the React frontend.
echo.
.venv\Scripts\python.exe -m uvicorn signbridge.server:app --host 127.0.0.1 --port 8000
if errorlevel 1 (
    echo.
    echo Backend exited with an error. Check the output above.
    pause
)
