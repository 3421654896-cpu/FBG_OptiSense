[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$bundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $bundleRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & $launcher.Source -3.12 -m venv (Join-Path $bundleRoot ".venv")
    } else {
        $python = Get-Command python -ErrorAction Stop
        $version = & $python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($version.Trim() -ne "3.12") {
            throw "需要 CPython 3.12 x64；当前 python 是 $version"
        }
        & $python.Source -m venv (Join-Path $bundleRoot ".venv")
    }
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install --requirement (Join-Path $bundleRoot "requirements-lock.txt")
& $venvPython -c "import PyQt5, numpy, scipy, pyqtgraph, serial, pyvisa, yaml; print('依赖导入检查通过')"

Write-Host "开发环境准备完成：$venvPython"

