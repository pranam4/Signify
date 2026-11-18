@echo off
title SignBridge Frontend (Keep This Window Open)
cd /d "%~dp0frontend"
if exist "%~dp0.tools\node\npm.cmd" (
    set "PATH=%~dp0.tools\node;%PATH%"
)
where npm >nul 2>&1
if errorlevel 1 (
    echo Node.js / npm was not found. Install Node.js 18+ from https://nodejs.org/ and re-run.
    pause
    exit /b 1
)
if not exist "node_modules\.bin\vite.cmd" goto :install_deps
if not exist "node_modules\three\package.json" goto :install_deps
goto :start_ui

:install_deps
echo First-time or cross-device setup: installing frontend npm packages...
call npm install
if errorlevel 1 (
    echo Failed to install npm dependencies.
    pause
    exit /b 1
)

:start_ui
echo Starting SignBridge React UI on http://localhost:5173 ...
echo Make sure SignBridge Backend (port 8000) is also running in another window.
echo.
call npm run dev
if errorlevel 1 (
    echo.
    echo Frontend exited with an error. Check the output above.
    pause
)
