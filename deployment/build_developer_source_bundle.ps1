[CmdletBinding()]
param([string]$Version = "", [string]$OutputDirectory = "")
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot "Python\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "请先运行 setup_dev.ps1 创建 Python\.venv" }
$arguments = @((Join-Path $PSScriptRoot "package_developer_snapshot.py"))
if ($Version) { $arguments += @("--version", $Version) }
if ($OutputDirectory) { $arguments += @("--output-directory", $OutputDirectory) }
$env:PYTHONIOENCODING = "utf-8"
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "开发源码包生成/校验失败" }
