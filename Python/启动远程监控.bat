@echo off
setlocal
cd /d "%~dp0"

if not defined FBG_MQTT_USERNAME for /f "tokens=2,*" %%A in ('reg query "HKCU\Environment" /v FBG_MQTT_USERNAME 2^>nul') do set "FBG_MQTT_USERNAME=%%B"
if not defined FBG_MQTT_PASSWORD for /f "tokens=2,*" %%A in ('reg query "HKCU\Environment" /v FBG_MQTT_PASSWORD 2^>nul') do set "FBG_MQTT_PASSWORD=%%B"

if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" "launch_fbg_studio.pyw" --source mqtt %*
  exit /b 0
)

where pythonw.exe >nul 2>nul
if not errorlevel 1 (
  start "" pythonw.exe "launch_fbg_studio.pyw" --source mqtt %*
  exit /b 0
)

echo.
echo FBG OptiSense Studio failed to start: pythonw.exe was not found.
echo Install Python or create Python\.venv, then try again.
pause
exit /b 1
