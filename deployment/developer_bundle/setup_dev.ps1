[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$bundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $bundleRoot "Python\.venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & $launcher.Source -3.12 -m venv (Join-Path $bundleRoot "Python\.venv")
    } else {
        $python = Get-Command python -ErrorAction Stop
        $version = & $python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($version.Trim() -ne "3.12") {
            throw "需要 CPython 3.12 x64；当前 python 是 $version"
        }
        & $python.Source -m venv (Join-Path $bundleRoot "Python\.venv")
    }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip 准备失败" }
& $venvPython -m pip install --requirement (Join-Path $bundleRoot "requirements-lock.txt")
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败" }
& $venvPython -c "import PyQt5, numpy, scipy, pyqtgraph, serial, pyvisa, yaml, cryptography; print('依赖导入检查通过')"
if ($LASTEXITCODE -ne 0) { throw "依赖导入检查失败" }

Write-Host "开发环境准备完成：$venvPython"

