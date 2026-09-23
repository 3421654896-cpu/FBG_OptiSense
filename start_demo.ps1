[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$bundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $bundleRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "尚未创建开发环境，请先运行 .\setup_dev.ps1"
}
$env:QT_OPENGL = "software"
& $python (Join-Path $bundleRoot "Python\fbg_unified_app.py") --demo

