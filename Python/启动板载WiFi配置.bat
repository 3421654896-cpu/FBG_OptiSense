@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" "launch_fbg_studio.pyw" --view settings
) else (
  start "" pythonw.exe "launch_fbg_studio.pyw" --view settings
)
endlocal
