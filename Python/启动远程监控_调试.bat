@echo off
setlocal
cd /d "%~dp0"

if not defined FBG_MQTT_USERNAME for /f "tokens=2,*" %%A in ('reg query "HKCU\Environment" /v FBG_MQTT_USERNAME 2^>nul') do set "FBG_MQTT_USERNAME=%%B"
if not defined FBG_MQTT_PASSWORD for /f "tokens=2,*" %%A in ('reg query "HKCU\Environment" /v FBG_MQTT_PASSWORD 2^>nul') do set "FBG_MQTT_PASSWORD=%%B"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "fbg_unified_app.py" --source mqtt %*
) else (
  py -3 "fbg_unified_app.py" --source mqtt %*
)

if not errorlevel 1 exit /b 0
echo.
echo FBG OptiSense Studio failed. Check the error message above.
pause
exit /b 1
