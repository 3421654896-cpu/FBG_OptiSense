[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$ZipPath)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot "Python\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "请先运行 setup_dev.ps1 创建 Python\.venv" }
$env:PYTHONIOENCODING = "utf-8"
& $python (Join-Path $PSScriptRoot "package_developer_snapshot.py") --verify $ZipPath
if ($LASTEXITCODE -ne 0) { throw "开发源码包校验失败" }
