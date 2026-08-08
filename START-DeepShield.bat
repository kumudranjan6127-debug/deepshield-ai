@echo off
REM Double-click to run DeepShield: starts the backend and opens the app.
cd /d "%~dp0"
call node scripts\ds.js start
start "" http://localhost:5000
echo.
echo App opened in your browser. Close this window whenever you like -
echo the server keeps running until you use STOP-DeepShield.bat
echo.
pause
