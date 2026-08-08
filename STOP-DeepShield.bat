@echo off
REM Double-click to stop the DeepShield backend.
cd /d "%~dp0"
call node scripts\ds.js stop
pause
