param(
    [Parameter(Mandatory = $true)]
    [string]$DllPath
)

$ErrorActionPreference = "Stop"
$resolved = (Resolve-Path -LiteralPath $DllPath).Path.Replace("\", "\\")
$source = @"
using System;
using System.Runtime.InteropServices;
public static class WchProbe {
    [DllImport("$resolved", CallingConvention=CallingConvention.Cdecl)]
    public static extern int McuCompiler_EnumLink_DAP();
    [DllImport("$resolved", CallingConvention=CallingConvention.Cdecl)]
    public static extern int McuCompiler_SetLinkIndex(int index);
    [DllImport("$resolved", CallingConvention=CallingConvention.Cdecl)]
    public static extern int McuCompiler_OpenDevice_DAP();
    [DllImport("$resolved", CallingConvention=CallingConvention.Cdecl)]
    public static extern int McuCompiler_GetDeviceMode_DAP();
    [DllImport("$resolved", CallingConvention=CallingConvention.Cdecl)]
    public static extern int McuCompiler_GetDeviceVersion_DAP();
    [DllImport("$resolved", CallingConvention=CallingConvention.Cdecl)]
    public static extern int McuCompiler_CloseDevice_DAP();
}
"@

Add-Type -TypeDefinition $source
$enumerated = [WchProbe]::McuCompiler_EnumLink_DAP()
$selected = [WchProbe]::McuCompiler_SetLinkIndex(0)
$opened = [WchProbe]::McuCompiler_OpenDevice_DAP()
try {
    $mode = [WchProbe]::McuCompiler_GetDeviceMode_DAP()
    $version = [WchProbe]::McuCompiler_GetDeviceVersion_DAP()
} finally {
    $closed = [WchProbe]::McuCompiler_CloseDevice_DAP()
}

"enum=$enumerated set=$selected open=$opened mode=$mode version=$version close=$closed"
