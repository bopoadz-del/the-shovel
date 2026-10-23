@echo off
REM fork-tunnel-autostart.cmd - silent autostart for the Startup folder.
REM Runs at every login. No prompts, no pause, no console window.

start "" /B powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%USERPROFILE%\.local\bin\fork-tunnel-start.ps1"
timeout /t 12 /nobreak >nul 2>&1
start "" /B powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%USERPROFILE%\.local\bin\fork-tunnel-update.ps1"
exit /b 0
