param(
    [switch]$ReuseWheelhouse
)

$ErrorActionPreference = 'Stop'
$deployment = Split-Path -Parent $MyInvocation.MyCommand.Path
$repository = (Resolve-Path (Join-Path $deployment '..\..')).Path
$source = Join-Path $repository 'Python'
$buildPython = Join-Path $source '.venv\Scripts\python.exe'
$firmware = Join-Path $repository 'outputs\ota\JDSU_F205RE_v1.0.86.fbgfw'
$buildRoot = Join-Path $repository 'build\python-client'
$packageRoot = Join-Path $buildRoot 'FBG_OptiSense_Studio_一号机_Python客户端'
$payload = Join-Path $packageRoot 'payload'
$wheelhouse = Join-Path $packageRoot 'wheelhouse'
$releaseRoot = Join-Path $repository 'release'

if (-not (Test-Path -LiteralPath $buildPython)) { throw "构建 Python 不存在：$buildPython" }
if (-not (Test-Path -LiteralPath $firmware)) { throw "OTA 固件不存在：$firmware" }
if (Test-Path -LiteralPath $packageRoot) { Remove-Item -LiteralPath $packageRoot -Recurse -Force }
New-Item -ItemType Directory -Force -Path $payload, $wheelhouse, $releaseRoot | Out-Null

& $buildPython (Join-Path $deployment 'collect_client_payload.py') $source $payload `
    --installed-launcher (Join-Path $deployment 'installed_launcher.pyw') `
    --requirements (Join-Path $deployment 'client_requirements.txt') `
    --firmware $firmware
if ($LASTEXITCODE -ne 0) { throw '收集客户程序失败。' }

Copy-Item -LiteralPath (Join-Path $deployment 'install.ps1') -Destination $packageRoot
Copy-Item -LiteralPath (Join-Path $deployment '安装一号机软件.bat') -Destination $packageRoot
Copy-Item -LiteralPath (Join-Path $deployment '客户安装说明.txt') -Destination $packageRoot

if (-not $ReuseWheelhouse -or -not (Get-ChildItem -LiteralPath $wheelhouse -File -ErrorAction SilentlyContinue)) {
    & $buildPython -m pip download --index-url https://pypi.org/simple --only-binary=:all: --dest $wheelhouse -r (Join-Path $deployment 'client_requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw '下载离线依赖失败。' }
}

$release = Get-Content -Raw -LiteralPath (Join-Path $payload 'release.json') | ConvertFrom-Json
$zipPath = Join-Path $releaseRoot ("FBG_OptiSense_Studio_一号机_Python客户端_v" + $release.app_version + ".zip")
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Compress-Archive -Path $packageRoot -DestinationPath $zipPath -CompressionLevel Optimal
$hash = Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath
$checksumPath = $zipPath + '.sha256.txt'
Set-Content -LiteralPath $checksumPath -Value ($hash.Hash.ToLowerInvariant() + '  ' + (Split-Path -Leaf $zipPath)) -Encoding ASCII
Write-Host $zipPath
Write-Host ("SHA256=" + $hash.Hash.ToLowerInvariant())
