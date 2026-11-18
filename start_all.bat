@echo off
echo Launching Signify Backend and Frontend in two separate windows...
echo Keep BOTH windows open while using http://localhost:5173
echo.
start "Signify Backend" cmd /k ""%~dp0start_backend.bat""
start "Signify Frontend" cmd /k ""%~dp0start_frontend.bat""
