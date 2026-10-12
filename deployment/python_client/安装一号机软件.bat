@echo off
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if errorlevel 1 (
  echo.
  echo 安装失败，请把安装日志发给维护人员：
  echo %LOCALAPPDATA%\FBGOptiSenseStudio\install.log
  pause
  exit /b 1
)
echo.
echo 安装完成。请从桌面启动 FBG OptiSense Studio（一号机）。
pause
