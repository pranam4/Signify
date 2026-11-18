@echo off
title SignBridge Backend (Keep This Window Open)
cd /d "%~dp0backend"
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
