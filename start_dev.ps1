[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$bundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $bundleRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "尚未创建开发环境，请先运行 .\setup_dev.ps1"
}
# pyqtgraph.opengl requires a real desktop OpenGL context.  Qt's software
# backend creates a nominal context on this workstation but every GL call
# fails with GL_INVALID_OPERATION, leaving the mechanical-finger model blank.
$env:QT_OPENGL = "desktop"
& $python (Join-Path $bundleRoot "Python\fbg_unified_app.py") --source local

