param(
    [string]$Version = "1.0.3"
)

$ErrorActionPreference = 'Stop'
$deployment = Split-Path -Parent $MyInvocation.MyCommand.Path
$repository = (Resolve-Path (Join-Path $deployment '..')).Path
$pythonDir = Join-Path $repository 'Python'
$buildPython = Join-Path $pythonDir '.venv\Scripts\python.exe'
$sourceVenv = (Resolve-Path (Join-Path $pythonDir '.venv')).Path
$spec = Join-Path $deployment 'pyinstaller\fbg_optisense_studio_machine1.spec'
$installerScript = Join-Path $deployment 'installer\fbg_optisense_studio_machine1.iss'
$iscc = Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe'
$output = Join-Path $pythonDir ("installer\FBG_OptiSense_Studio_Machine1_v$Version`_Setup.exe")

foreach ($required in ($buildPython, $spec, $installerScript, $iscc)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "缺少一号机客户更新构建依赖：$required"
    }
}

# Qt 的 Windows 收集钩子不能稳定处理含中文的虚拟环境路径，因此通过
# 纯 ASCII 目录联接运行同一个虚拟环境；源码仍保留在原目录。
$asciiBuildRoot = Join-Path $env:LOCALAPPDATA 'FBGOptiSenseBuild'
$asciiVenv = Join-Path $asciiBuildRoot 'venv'
if (-not (Test-Path -LiteralPath $asciiBuildRoot)) {
    New-Item -ItemType Directory -Path $asciiBuildRoot | Out-Null
}
if (Test-Path -LiteralPath $asciiVenv) {
    $junction = Get-Item -LiteralPath $asciiVenv
    if ($junction.LinkType -ne 'Junction' -or [string]$junction.Target -ne $sourceVenv) {
        throw "ASCII 构建路径已存在但不是当前虚拟环境的目录联接：$asciiVenv"
    }
} else {
    New-Item -ItemType Junction -Path $asciiVenv -Target $sourceVenv | Out-Null
}
$asciiPython = Join-Path $asciiVenv 'Scripts\python.exe'

$appVersionText = Get-Content -Raw -LiteralPath (Join-Path $pythonDir 'app_version.py')
$installerText = Get-Content -Raw -LiteralPath $installerScript
if ($appVersionText -notmatch ('APP_VERSION\s*=\s*"' + [regex]::Escape($Version) + '"')) {
    throw "app_version.py 与请求版本 $Version 不一致"
}
if ($installerText -notmatch ('MyAppVersion\s+"' + [regex]::Escape($Version) + '"')) {
    throw "Inno Setup 脚本与请求版本 $Version 不一致"
}

Push-Location $repository
try {
    & $asciiPython -m PyInstaller --noconfirm --clean `
        --workpath (Join-Path $repository 'build\pyinstaller_machine1') `
        --distpath $pythonDir `
        $spec
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller 一号机客户程序构建失败' }
    & $iscc $installerScript
    if ($LASTEXITCODE -ne 0) { throw 'Inno Setup 一号机客户更新安装包构建失败' }
} finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $output)) {
    throw "没有生成预期安装包：$output"
}
$hash = Get-FileHash -Algorithm SHA256 -LiteralPath $output
$checksum = "$output.sha256.txt"
Set-Content -LiteralPath $checksum -Encoding ASCII -Value (
    $hash.Hash.ToLowerInvariant() + '  ' + (Split-Path -Leaf $output)
)
Write-Host "一号机客户更新安装包：$output"
Write-Host "SHA256=$($hash.Hash.ToLowerInvariant())"
