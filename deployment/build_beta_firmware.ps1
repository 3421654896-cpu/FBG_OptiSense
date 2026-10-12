param([string]$OutputName = 'fw118_beta_single_adc_20261011', [switch]$FastDacPolling)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$mdk = Join-Path $repo 'JDSU\MDK-ARM'
if ($OutputName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'OutputName must be a simple directory name' }
$betaOutput = Join-Path $repo ('build\' + $OutputName)
New-Item -ItemType Directory -Path $betaOutput -Force | Out-Null
$project = Join-Path $mdk 'JDSU_beta_generated.uvprojx'
[xml]$projectXml = Get-Content -LiteralPath (Join-Path $mdk 'JDSU.uvprojx') -Raw
$target = $projectXml.Project.Targets.Target
$target.TargetName = 'JDSU Beta'
$target.TargetOption.TargetCommonOption.OutputDirectory = '..\..\build\' + $OutputName + '\'
$target.TargetOption.TargetCommonOption.OutputName = 'JDSU_Beta'
$target.TargetOption.TargetArmAds.Cads.VariousControls.Define += ',FBG_FEATURE_BETA_SINGLE_ADC=1'
if ($FastDacPolling) {
    $target.TargetOption.TargetArmAds.Cads.VariousControls.Define += ',FBG_FEATURE_BETA_DAC_POLLING=1'
}
$xmlSettings = New-Object System.Xml.XmlWriterSettings
$xmlSettings.Encoding = New-Object System.Text.UTF8Encoding($false)
$xmlSettings.Indent = $true
$xmlWriter = [System.Xml.XmlWriter]::Create($project, $xmlSettings)
try { $projectXml.Save($xmlWriter) } finally { $xmlWriter.Dispose() }
$log = Join-Path $betaOutput 'keil_build.log'
$build = Start-Process -FilePath 'D:\Keil_v5\UV4\UV4.exe' -WorkingDirectory $mdk -WindowStyle Hidden -PassThru -Wait -ArgumentList @('-b', 'JDSU_beta_generated.uvprojx', '-j0', '-o', ('..\..\build\' + $OutputName + '\keil_build.log'))
if ($build.ExitCode -ne 0) { throw "Beta build failed ($($build.ExitCode)); see $log" }
$buildText = Get-Content -LiteralPath $log -Raw
if ($buildText -notmatch '0 Error\(s\), 0 Warning\(s\)') { throw "Check Beta build diagnostics: $log" }
$versionHeader = Get-Content -LiteralPath (Join-Path $repo 'JDSU\Core\Inc\firmware_version.h') -Raw
$versionBranch = if ($FastDacPolling) { '#if FBG_FEATURE_BETA_DAC_POLLING' } else { '#elif FBG_FEATURE_BETA_SINGLE_ADC' }
$betaMatch = [regex]::Match($versionHeader, '(?s)' + [regex]::Escape($versionBranch) + '.*?APP_FIRMWARE_VERSION_PATCH (\d+)U')
if (-not $betaMatch.Success) { throw 'Cannot resolve Beta firmware version from source header' }
$betaVersion = '1.0.' + $betaMatch.Groups[1].Value
$bin = Join-Path $betaOutput ('JDSU_F205RE_v' + $betaVersion + '_beta.bin')
& 'D:\Keil_v5\ARM\ARMCLANG\Bin\fromelf.exe' --bin --output $bin (Join-Path $betaOutput 'JDSU_Beta.axf')
if ($LASTEXITCODE -ne 0) { throw 'fromelf failed' }
$python = Join-Path $repo 'Python\.venv\Scripts\python.exe'
& $python (Join-Path $repo 'Python\package_firmware.py') $bin --version $betaVersion --output (Join-Path $betaOutput ('JDSU_F205RE_v' + $betaVersion + '_beta.fbgfw'))
if ($LASTEXITCODE -ne 0) { throw 'Beta firmware packaging failed' }
Write-Output "Beta candidate only; NOT installed or added to any customer update catalog: $betaOutput"
