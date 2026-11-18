@echo off
title SignBridge Frontend (Keep This Window Open)
cd /d "%~dp0frontend"
if exist "%~dp0.tools\node\npm.cmd" (
    set "PATH=%~dp0.tools\node;%PATH%"
)
echo Starting SignBridge React UI on http://localhost:5173 ...
echo Make sure SignBridge Backend (port 8000) is also running in another window.
echo.
npm run dev
if errorlevel 1 (
    echo.
    echo Frontend exited with an error. Check the output above.
    pause
)
