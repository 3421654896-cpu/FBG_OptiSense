param(
    [switch]$NoDesktopShortcut,
    [string]$PythonExecutable = ''
)

$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$payloadRoot = Join-Path $packageRoot 'payload'
$wheelRoot = Join-Path $packageRoot 'wheelhouse'
$productRoot = Join-Path $env:LOCALAPPDATA 'Programs\FBG OptiSense Studio'
$dataRoot = Join-Path $env:LOCALAPPDATA 'FBGOptiSenseStudio'
$version = (Get-Content -Raw -LiteralPath (Join-Path $payloadRoot 'release.json') | ConvertFrom-Json).app_version
$target = Join-Path $productRoot ("app-" + $version)
$staging = Join-Path $productRoot (".installing-" + $version + "-" + [Guid]::NewGuid().ToString('N'))
$logPath = Join-Path $dataRoot 'install.log'

New-Item -ItemType Directory -Force -Path $productRoot, $dataRoot | Out-Null
Start-Transcript -Path $logPath -Append | Out-Null

try {
    $python = $null
    $pythonArgs = @()
    if ($PythonExecutable) {
        $candidate = (Resolve-Path -LiteralPath $PythonExecutable -ErrorAction Stop).Path
        & $candidate -c "import struct,sys; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $python = $candidate
        } else {
            throw '指定的 Python 不是 64 位 Python 3.12。'
        }
    }
    $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if (-not $python -and $pyLauncher) {
        & $pyLauncher.Source -3.12 -c "import struct,sys; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $python = $pyLauncher.Source
            $pythonArgs = @('-3.12')
        }
    }
    if (-not $python) {
        $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($pythonCommand) {
            & $pythonCommand.Source -c "import struct,sys; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $python = $pythonCommand.Source
            }
        }
    }
    if (-not $python) {
        throw '未找到 64 位 Python 3.12。请先安装 Python 3.12 x64，并勾选 Add Python to PATH。'
    }

    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    Copy-Item -Path (Join-Path $payloadRoot '*') -Destination $staging -Recurse -Force
    & $python @pythonArgs -m venv (Join-Path $staging '.venv')
    if ($LASTEXITCODE -ne 0) { throw '创建隔离 Python 环境失败。' }

    $venvPython = Join-Path $staging '.venv\Scripts\python.exe'
    & $venvPython -m pip install --no-index --find-links $wheelRoot -r (Join-Path $staging 'client_requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw '安装离线依赖失败。' }

    $env:QT_QPA_PLATFORM = 'offscreen'
    $env:FBG_STUDIO_DATA_DIR = $dataRoot
    Push-Location $staging
    try {
        & $venvPython -c "import fbg_unified_app, machine_profile; assert machine_profile.MACHINE_ID == 'machine_1'; print('client import smoke: OK')"
        if ($LASTEXITCODE -ne 0) { throw '客户版启动前检查失败。' }
    } finally {
        Pop-Location
    }
    Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue

    if (Test-Path -LiteralPath $target) {
        $backup = Join-Path $productRoot ("app-" + $version + ".previous-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
        Move-Item -LiteralPath $target -Destination $backup
    }
    Move-Item -LiteralPath $staging -Destination $target

    $pythonw = Join-Path $target '.venv\Scripts\pythonw.exe'
    $entry = Join-Path $target 'installed_launcher.pyw'
    $shell = New-Object -ComObject WScript.Shell
    $startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
    New-Item -ItemType Directory -Force -Path $startMenu | Out-Null
    $startLink = $shell.CreateShortcut((Join-Path $startMenu 'FBG OptiSense Studio（一号机）.lnk'))
    $startLink.TargetPath = $pythonw
    $startLink.Arguments = '"' + $entry + '"'
    $startLink.WorkingDirectory = $target
    $startLink.Description = 'FBG OptiSense Studio 一号机客户版'
    $startLink.Save()
    if (-not $NoDesktopShortcut) {
        $desktop = [Environment]::GetFolderPath('Desktop')
        $desktopLink = $shell.CreateShortcut((Join-Path $desktop 'FBG OptiSense Studio（一号机）.lnk'))
        $desktopLink.TargetPath = $pythonw
        $desktopLink.Arguments = '"' + $entry + '"'
        $desktopLink.WorkingDirectory = $target
        $desktopLink.Description = 'FBG OptiSense Studio 一号机客户版'
        $desktopLink.Save()
    }

    $current = @{
        schema = 'fbg_python_client_install_v1'
        app_version = $version
        install_path = $target
        data_path = $dataRoot
        installed_at = (Get-Date).ToString('o')
    } | ConvertTo-Json
    Set-Content -LiteralPath (Join-Path $productRoot 'current.json') -Value $current -Encoding UTF8
    Write-Host "安装完成：FBG OptiSense Studio $version（一号机）" -ForegroundColor Green
    Write-Host "客户数据：$dataRoot"
} catch {
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
    Write-Host $_.Exception.Message -ForegroundColor Red
    throw
} finally {
    Stop-Transcript | Out-Null
}
