@echo off
echo Launching SignBridge Backend and Frontend in two separate windows...
echo Keep BOTH windows open while using http://localhost:5173
echo.
start "SignBridge Backend" cmd /k ""%~dp0start_backend.bat""
start "SignBridge Frontend" cmd /k ""%~dp0start_frontend.bat""
