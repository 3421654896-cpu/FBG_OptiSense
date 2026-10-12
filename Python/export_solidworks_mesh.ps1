param(
    [Parameter(Mandatory = $true)]
    [string]$PartPath,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"
$part = [IO.Path]::GetFullPath($PartPath)
$output = [IO.Path]::GetFullPath($OutputPath)
if (-not [IO.File]::Exists($part)) {
    throw "SolidWorks part not found: $part"
}

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$source = Join-Path $scriptDirectory "solidworks_export.cs"
$solidWorksDirectory = "D:\Program Files\SOLIDWORKS Corp\SOLIDWORKS"
$interopSldWorks = Join-Path $solidWorksDirectory "SolidWorks.Interop.sldworks.dll"
$interopConstants = Join-Path $solidWorksDirectory "SolidWorks.Interop.swconst.dll"
$compiler = "C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe"
foreach ($requiredFile in @($source, $interopSldWorks, $interopConstants, $compiler)) {
    if (-not [IO.File]::Exists($requiredFile)) {
        throw "SolidWorks export dependency not found: $requiredFile"
    }
}

$buildDirectory = Join-Path ([IO.Path]::GetTempPath()) "fbg-solidworks-export"
$null = [IO.Directory]::CreateDirectory($buildDirectory)
$executable = Join-Path $buildDirectory "solidworks_export.exe"
Copy-Item -LiteralPath $interopSldWorks -Destination $buildDirectory -Force
Copy-Item -LiteralPath $interopConstants -Destination $buildDirectory -Force

& $compiler /nologo /platform:x86 /optimize+ "/out:$executable" `
    "/reference:$interopSldWorks" "/reference:$interopConstants" $source
if ($LASTEXITCODE -ne 0) {
    throw "Failed to compile the SolidWorks mesh exporter (exit=$LASTEXITCODE)"
}

& $executable $part $output
if ($LASTEXITCODE -ne 0) {
    throw "SolidWorks mesh export failed (exit=$LASTEXITCODE)"
}
