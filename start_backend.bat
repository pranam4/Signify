@echo off
title Signify Backend (Keep This Window Open)
cd /d "%~dp0backend"

if not exist ".venv\Scripts\python.exe" goto :recreate_venv
".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
if errorlevel 1 goto :recreate_venv
goto :check_deps

:recreate_venv
echo First-time or cross-device setup: recreating Python virtual environment...
if exist ".venv" rmdir /s /q ".venv"
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

:check_deps
".venv\Scripts\python.exe" -c "import importlib.util as u, sys; mods=('uvicorn','fastapi','mediapipe','transformers','torch','av','soundfile','python_multipart'); sys.exit(0 if all(u.find_spec(m) for m in mods) else 1)" >nul 2>&1
if not errorlevel 1 goto :start_server

echo Installing / updating Python backend dependencies (this may take a few minutes)...
where uv >nul 2>&1
if not errorlevel 1 (
    uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
    if not errorlevel 1 goto :start_server
)

".venv\Scripts\python.exe" -m pip --version >nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m ensurepip --upgrade
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install Python dependencies.
    pause
    exit /b 1
)

:start_server
echo Starting Signify MediaPipe + Whisper backend on http://localhost:8000 ...
echo Please wait ~15-20 seconds until you see "Uvicorn running on http://127.0.0.1:8000".
echo KEEP THIS WINDOW OPEN while using the React frontend.
echo.
".venv\Scripts\python.exe" -m uvicorn signbridge.server:app --host 127.0.0.1 --port 8000
if errorlevel 1 (
    echo.
    echo Backend exited with an error. Check the output above.
    pause
)
