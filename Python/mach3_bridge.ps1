param()

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$mach = $null
$machScript = $null

# Some Mach3 installations expose the active object correctly but intermittently
# fail ProgID-to-CLSID resolution with CO_E_CLASSSTRING.  Keep a direct ROT
# lookup by the locally registered Mach3 CLSID as a non-launching fallback.
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class Mach3RotLookup
{
    [DllImport("oleaut32.dll", PreserveSig = false)]
    [return: MarshalAs(UnmanagedType.Interface)]
    private static extern object GetActiveObject(ref Guid clsid, IntPtr reserved);

    public static object Attach(string clsidText)
    {
        Guid clsid = new Guid(clsidText);
        return GetActiveObject(ref clsid, IntPtr.Zero);
    }
}
"@

function Connect-Mach3 {
    if ($null -ne $mach -and $null -ne $machScript) {
        return
    }
    # Mach3 exposes an already-running automation object.  Creating a new COM
    # server can start an empty/default profile, so attach only to the active
    # Mach3Mill instance.
    try {
        $script:mach = [Runtime.InteropServices.Marshal]::GetActiveObject(
            "Mach4.Document"
        )
    }
    catch {
        $script:mach = [Mach3RotLookup]::Attach(
            "{CA7992B2-2653-4342-8061-D7D385C07809}"
        )
    }
    $script:machScript = $script:mach.GetScriptDispatch()
    if ($null -eq $script:machScript) {
        throw "Mach3 script dispatch is unavailable"
    }
}

function Get-Mach3Status {
    Connect-Mach3
    [ordered]@{
        connected = $true
        work_x = [double]$machScript.GetOEMDRO(800)
        work_y = [double]$machScript.GetOEMDRO(801)
        work_z = [double]$machScript.GetOEMDRO(802)
        machine_x = [double]$machScript.GetOEMDRO(83)
        machine_y = [double]$machScript.GetOEMDRO(84)
        machine_z = [double]$machScript.GetOEMDRO(85)
        moving = [bool]$machScript.IsMoving()
        stopped = [bool]$machScript.IsStopped()
        estop = [bool]$machScript.IsEstop()
        estop_led = [bool]$machScript.GetOEMLED(800)
        x_homed = [bool]$machScript.GetOEMLED(807)
        y_homed = [bool]$machScript.GetOEMLED(808)
        z_homed = [bool]$machScript.GetOEMLED(809)
        soft_limits = [bool]$machScript.GetOEMLED(815)
        x_pos_limit = [bool]$machScript.GetOEMLED(828)
        x_neg_limit = [bool]$machScript.GetOEMLED(829)
        y_pos_limit = [bool]$machScript.GetOEMLED(831)
        y_neg_limit = [bool]$machScript.GetOEMLED(832)
        z_pos_limit = [bool]$machScript.GetOEMLED(834)
        z_neg_limit = [bool]$machScript.GetOEMLED(835)
        units_code = [int]$machScript.GetSetupUnits()
    }
}

function Get-Mach3PositionSample {
    # Read-only timing diagnostic. Never use this reduced snapshot in place of
    # the full homing/limit/estop checks required by every motion operation.
    Connect-Mach3
    $sampleStart = [Diagnostics.Stopwatch]::GetTimestamp()
    $sampleX = [double]$machScript.GetOEMDRO(83)
    $sampleY = [double]$machScript.GetOEMDRO(84)
    $sampleZ = [double]$machScript.GetOEMDRO(85)
    $sampleEnd = [Diagnostics.Stopwatch]::GetTimestamp()
    [ordered]@{
        machine_x = $sampleX; machine_y = $sampleY; machine_z = $sampleZ
        read_start_ticks = $sampleStart; read_end_ticks = $sampleEnd
        clock_frequency_hz = [Diagnostics.Stopwatch]::Frequency
        coordinates_simultaneous = $false
        encoder_feedback_verified = $false
        safety_status_included = $false
    }
}

function Invoke-EmergencyStop {
    Connect-Mach3
    # Feed-hold and Stop are intentionally issued before latching Reset/E-stop.
    # DoOEMButton(1021) is a toggle, so it must never be called when E-stop is
    # already active.
    try { $null = $mach.FeedHold() } catch {}
    try { $null = $machScript.DoOEMButton(1003) } catch {}
    Start-Sleep -Milliseconds 30
    $isEstop = [bool]$machScript.IsEstop()
    $estopLed = [bool]$machScript.GetOEMLED(800)
    if (-not ($isEstop -or $estopLed)) {
        $null = $machScript.DoOEMButton(1021)
    }
    $deadline = [DateTime]::UtcNow.AddSeconds(2)
    do {
        Start-Sleep -Milliseconds 20
        $status = Get-Mach3Status
        # Some Mach3 3.x builds leave IsMoving() asserted while Reset/E-stop
        # is active.  IsStopped() is the dedicated planner-stop indication and
        # the coordinates are checked separately during commissioning.
        if (($status.estop -or $status.estop_led) -and $status.stopped) {
            return $status
        }
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Mach3 did not confirm the emergency-stop state"
}

function Release-EmergencyStop($request) {
    Connect-Mach3
    if (-not [bool]$request.operator_confirmed) {
        throw "The operator has not confirmed that the machine is safe"
    }

    $before = Get-Mach3Status
    if (-not $before.stopped) {
        throw "Mach3 must be stopped before Reset/E-stop can be released"
    }
    $limitActive = (
        $before.x_pos_limit -or $before.x_neg_limit -or
        $before.y_pos_limit -or $before.y_neg_limit -or
        $before.z_pos_limit -or $before.z_neg_limit
    )
    if ($limitActive) {
        throw "A hardware limit is active; Reset/E-stop remains latched"
    }

    # OEM button 1021 is a toggle.  Use it only when both Mach3 indicators say
    # Reset/E-stop is latched; this prevents an already-ready controller from
    # being stopped accidentally by a repeated release request.
    if ($before.estop -or $before.estop_led) {
        $null = $machScript.DoOEMButton(1021)
    }

    $deadline = [DateTime]::UtcNow.AddSeconds(2)
    do {
        Start-Sleep -Milliseconds 20
        $released = Get-Mach3Status
        if (-not ($released.estop -or $released.estop_led)) {
            break
        }
    } while ([DateTime]::UtcNow -lt $deadline)

    if ($released.estop -or $released.estop_led) {
        throw "Mach3 did not confirm that Reset/E-stop was released"
    }

    # The spindle is forbidden for this fixture.  M5 is deliberately sent
    # after Reset is clear so Mach3 cannot discard it while Reset is active.
    $null = $machScript.Code("M5")
    Start-Sleep -Milliseconds 200
    $after = Get-Mach3Status
    $unsafe = (
        $after.estop -or $after.estop_led -or -not $after.stopped -or
        $after.x_pos_limit -or $after.x_neg_limit -or
        $after.y_pos_limit -or $after.y_neg_limit -or
        $after.z_pos_limit -or $after.z_neg_limit
    )
    if ($unsafe) {
        Invoke-EmergencyStop | Out-Null
        throw "Mach3 was not safely stopped after release; E-stop was re-applied"
    }

    return [ordered]@{
        spindle_command = "M5"
        before = $before
        after = $after
    }
}

function Invoke-MicroMove($request) {
    Connect-Mach3
    $axis = ([string]$request.axis).Trim().ToUpperInvariant()
    if ($axis -notin @("X", "Y", "Z")) {
        throw "Only X, Y and Z axes are permitted"
    }
    $delta = [double]$request.delta_mm
    $feed = [double]$request.feed_mm_min
    if ([double]::IsNaN($delta) -or [double]::IsInfinity($delta) -or
        [Math]::Abs($delta) -le 0.0 -or [Math]::Abs($delta) -gt 1.0000001) {
        throw "Each verified staged move must be greater than 0 and no more than 1 mm"
    }
    if ([double]::IsNaN($feed) -or [double]::IsInfinity($feed) -or
        $feed -le 0.0 -or $feed -gt 30.0) {
        throw "Commissioning feed must be greater than 0 and no more than 30 mm/min"
    }
    if (-not [bool]$request.safe_zone_confirmed) {
        throw "The physical safe zone has not been confirmed"
    }
    $before = Get-Mach3Status
    if ($before.estop -or $before.estop_led) {
        throw "Mach3 is in E-stop/Reset; release it manually before a test move"
    }
    if (-not $before.stopped) {
        throw "Mach3 is already moving"
    }

    $current = switch ($axis) {
        "X" { [double]$before.work_x }
        "Y" { [double]$before.work_y }
        "Z" { [double]$before.work_z }
    }
    $target = $current + $delta
    $targetText = $target.ToString("0.000000", [Globalization.CultureInfo]::InvariantCulture)
    $feedText = $feed.ToString("0.000", [Globalization.CultureInfo]::InvariantCulture)

    # Absolute positioning is deliberate: an interrupted command cannot leave
    # Mach3 in incremental mode and affect a later operation.
    # The spindle is forbidden for the finger fixture, including commissioning
    # micro moves.
    $null = $machScript.Code("M5")
    Start-Sleep -Milliseconds 50
    $command = "G21 G90 G1 $axis$targetText F$feedText"
    $null = $machScript.Code($command)
    # Low-speed contact moves can legitimately take longer than five seconds
    # (for example 0.5 mm at 3 mm/min takes ten seconds).  Size the watchdog
    # from commanded travel time and retain a three-second controller margin.
    $expectedSeconds = [Math]::Abs($delta) / $feed * 60.0
    $deadline = [DateTime]::UtcNow.AddSeconds([Math]::Max(5.0, $expectedSeconds + 3.0))
    do {
        Start-Sleep -Milliseconds 20
        $after = Get-Mach3Status
        $actual = switch ($axis) {
            "X" { [double]$after.work_x }
            "Y" { [double]$after.work_y }
            "Z" { [double]$after.work_z }
        }
        # Some Mach3 builds assert IsStopped slightly before IsMoving clears.
        # Do not publish an arrived status until both indicators agree; callers
        # use the returned snapshot as a safety gate for the next move.
        if ($after.stopped -and -not $after.moving -and
            [Math]::Abs($actual - $target) -le 0.002) {
            return [ordered]@{
                command = $command
                before = $before
                after = $after
            }
        }
    } while ([DateTime]::UtcNow -lt $deadline)

    Invoke-EmergencyStop | Out-Null
        throw "The staged move timed out and E-stop was applied"
}

function Invoke-ConfirmedLongAxisMove($request) {
    Connect-Mach3
    $axis = ([string]$request.axis).Trim().ToUpperInvariant()
    if ($axis -notin @("X", "Y", "Z")) {
        throw "Only X, Y and Z axes are permitted"
    }
    $delta = [double]$request.delta_mm
    $feed = [double]$request.feed_mm_min
    if ([double]::IsNaN($delta) -or [double]::IsInfinity($delta) -or
        [Math]::Abs($delta) -le 0.003 -or [Math]::Abs($delta) -gt 10.0000001) {
        throw "Confirmed continuous move must be greater than 0.003 and no more than 10 mm"
    }
    if ([double]::IsNaN($feed) -or [double]::IsInfinity($feed) -or
        $feed -le 0.0 -or $feed -gt 60.0) {
        throw "Continuous feed must be greater than 0 and no more than 60 mm/min"
    }
    if (-not [bool]$request.safe_zone_confirmed) {
        throw "The full continuous axis path has not been confirmed"
    }

    $before = Get-Mach3Status
    if ($before.estop -or $before.estop_led) {
        throw "Mach3 is in E-stop/Reset"
    }
    if (-not $before.stopped) {
        throw "Mach3 is already moving"
    }
    $isHomed = switch ($axis) {
        "X" { [bool]$before.x_homed }
        "Y" { [bool]$before.y_homed }
        "Z" { [bool]$before.z_homed }
    }
    $positiveLimit = switch ($axis) {
        "X" { [bool]$before.x_pos_limit }
        "Y" { [bool]$before.y_pos_limit }
        "Z" { [bool]$before.z_pos_limit }
    }
    $negativeLimit = switch ($axis) {
        "X" { [bool]$before.x_neg_limit }
        "Y" { [bool]$before.y_neg_limit }
        "Z" { [bool]$before.z_neg_limit }
    }
    if (-not $isHomed) {
        throw "$axis axis is not homed"
    }
    if ($positiveLimit -or $negativeLimit) {
        throw "A $axis-axis hardware limit is already active"
    }

    $currentWork = switch ($axis) {
        "X" { [double]$before.work_x }
        "Y" { [double]$before.work_y }
        "Z" { [double]$before.work_z }
    }
    $currentMachine = switch ($axis) {
        "X" { [double]$before.machine_x }
        "Y" { [double]$before.machine_y }
        "Z" { [double]$before.machine_z }
    }
    $targetWork = $currentWork + $delta
    $targetMachine = $currentMachine + $delta
    # Mach3Mill.xml configures X/Y/Z to -100..+100 mm but soft limits are off.
    # Retain a one-millimetre software guard at either end.
    if ($targetMachine -lt -99.0 -or $targetMachine -gt 99.0) {
        throw "Target $axis machine coordinate violates the one-millimetre boundary guard"
    }
    $targetText = $targetWork.ToString(
        "0.000000", [Globalization.CultureInfo]::InvariantCulture
    )
    $feedText = $feed.ToString(
        "0.000", [Globalization.CultureInfo]::InvariantCulture
    )
    $null = $machScript.Code("M5")
    Start-Sleep -Milliseconds 50
    $command = "G21 G90 G1 $axis$targetText F$feedText"
    $null = $machScript.Code($command)
    $lastPosition = $currentWork
    $travelSeconds = ([Math]::Abs($delta) / $feed) * 60.0
    $deadline = [DateTime]::UtcNow.AddSeconds($travelSeconds + 20.0)
    do {
        Start-Sleep -Milliseconds 20
        $after = Get-Mach3Status
        if ($after.estop -or $after.estop_led) {
            throw "Continuous $axis move was interrupted by E-stop"
        }
        $positiveLimit = switch ($axis) {
            "X" { [bool]$after.x_pos_limit }
            "Y" { [bool]$after.y_pos_limit }
            "Z" { [bool]$after.z_pos_limit }
        }
        $negativeLimit = switch ($axis) {
            "X" { [bool]$after.x_neg_limit }
            "Y" { [bool]$after.y_neg_limit }
            "Z" { [bool]$after.z_neg_limit }
        }
        if ($positiveLimit -or $negativeLimit) {
            Invoke-EmergencyStop | Out-Null
            throw "$axis hardware limit became active; E-stop was applied"
        }
        $actualPosition = switch ($axis) {
            "X" { [double]$after.work_x }
            "Y" { [double]$after.work_y }
            "Z" { [double]$after.work_z }
        }
        if (($delta -lt 0.0 -and $actualPosition -gt $lastPosition + 0.01) -or
            ($delta -gt 0.0 -and $actualPosition -lt $lastPosition - 0.01)) {
            Invoke-EmergencyStop | Out-Null
            throw "$axis moved opposite to the commanded direction; E-stop was applied"
        }
        $lastPosition = $actualPosition
        if ($after.stopped -and -not $after.moving -and
            [Math]::Abs($actualPosition - $targetWork) -le 0.002) {
            foreach ($otherAxis in @("X", "Y", "Z")) {
                if ($otherAxis -eq $axis) { continue }
                $beforeOther = switch ($otherAxis) {
                    "X" { [double]$before.work_x }
                    "Y" { [double]$before.work_y }
                    "Z" { [double]$before.work_z }
                }
                $afterOther = switch ($otherAxis) {
                    "X" { [double]$after.work_x }
                    "Y" { [double]$after.work_y }
                    "Z" { [double]$after.work_z }
                }
                if ([Math]::Abs($afterOther - $beforeOther) -gt 0.002) {
                    Invoke-EmergencyStop | Out-Null
                    throw "$otherAxis moved unexpectedly; E-stop was applied"
                }
            }
            return [ordered]@{
                command = $command
                before = $before
                after = $after
            }
        }
    } while ([DateTime]::UtcNow -lt $deadline)

    Invoke-EmergencyStop | Out-Null
    throw "Continuous $axis move timed out and E-stop was applied"
}

function Invoke-ConfirmedXYMove($request) {
    Connect-Mach3
    $targetX = [double]$request.target_x
    $targetY = [double]$request.target_y
    $feed = [double]$request.feed_mm_min
    if ([double]::IsNaN($targetX) -or [double]::IsInfinity($targetX) -or
        [double]::IsNaN($targetY) -or [double]::IsInfinity($targetY)) {
        throw "XY target must contain finite coordinates"
    }
    if ([double]::IsNaN($feed) -or [double]::IsInfinity($feed) -or
        $feed -le 0.0 -or $feed -gt 120.0) {
        throw "XY continuous feed must be greater than 0 and no more than 120 mm/min"
    }
    if (-not [bool]$request.safe_zone_confirmed) {
        throw "The complete XY path at safe Z has not been confirmed"
    }

    $before = Get-Mach3Status
    if ($before.estop -or $before.estop_led) {
        throw "Mach3 is in E-stop/Reset"
    }
    if (-not $before.stopped) {
        throw "Mach3 is already moving"
    }
    if (-not $before.x_homed -or -not $before.y_homed) {
        throw "X and Y must both be homed"
    }
    if ($before.x_pos_limit -or $before.x_neg_limit -or
        $before.y_pos_limit -or $before.y_neg_limit) {
        throw "An X/Y hardware limit is already active"
    }

    $deltaX = $targetX - [double]$before.work_x
    $deltaY = $targetY - [double]$before.work_y
    if ([Math]::Abs($deltaX) -gt 30.0000001 -or
        [Math]::Abs($deltaY) -gt 30.0000001) {
        throw "Each XY axis delta must not exceed 30 mm"
    }
    if ([Math]::Abs($deltaX) -le 0.003 -and [Math]::Abs($deltaY) -le 0.003) {
        return [ordered]@{ command = ""; before = $before; after = $before }
    }

    $targetMachineX = [double]$before.machine_x + $deltaX
    $targetMachineY = [double]$before.machine_y + $deltaY
    if ($targetMachineX -lt -99.0 -or $targetMachineX -gt 99.0 -or
        $targetMachineY -lt -99.0 -or $targetMachineY -gt 99.0) {
        throw "XY target violates the one-millimetre machine boundary guard"
    }

    $xText = $targetX.ToString(
        "0.000000", [Globalization.CultureInfo]::InvariantCulture
    )
    $yText = $targetY.ToString(
        "0.000000", [Globalization.CultureInfo]::InvariantCulture
    )
    $feedText = $feed.ToString(
        "0.000", [Globalization.CultureInfo]::InvariantCulture
    )
    # The spindle is forbidden for this fixture.  Reinforce M5 immediately
    # before every coordinated positioning command.
    $null = $machScript.Code("M5")
    Start-Sleep -Milliseconds 50
    $command = "G21 G90 G1 X$xText Y$yText F$feedText"
    $null = $machScript.Code($command)

    $lastX = [double]$before.work_x
    $lastY = [double]$before.work_y
    $beforeZ = [double]$before.work_z
    $distance = [Math]::Sqrt($deltaX * $deltaX + $deltaY * $deltaY)
    $deadline = [DateTime]::UtcNow.AddSeconds(($distance / $feed) * 60.0 + 12.0)
    do {
        Start-Sleep -Milliseconds 20
        $after = Get-Mach3Status
        if ($after.estop -or $after.estop_led) {
            throw "Coordinated XY move was interrupted by E-stop"
        }
        if ($after.x_pos_limit -or $after.x_neg_limit -or
            $after.y_pos_limit -or $after.y_neg_limit) {
            Invoke-EmergencyStop | Out-Null
            throw "An X/Y hardware limit became active; E-stop was applied"
        }
        $actualX = [double]$after.work_x
        $actualY = [double]$after.work_y
        $actualZ = [double]$after.work_z
        if (($deltaX -lt -0.003 -and $actualX -gt $lastX + 0.01) -or
            ($deltaX -gt 0.003 -and $actualX -lt $lastX - 0.01) -or
            ($deltaY -lt -0.003 -and $actualY -gt $lastY + 0.01) -or
            ($deltaY -gt 0.003 -and $actualY -lt $lastY - 0.01)) {
            Invoke-EmergencyStop | Out-Null
            throw "XY moved opposite to the commanded direction; E-stop was applied"
        }
        if ([Math]::Abs($actualZ - $beforeZ) -gt 0.002) {
            Invoke-EmergencyStop | Out-Null
            throw "Z moved unexpectedly during XY positioning; E-stop was applied"
        }
        $lastX = $actualX
        $lastY = $actualY
        if ($after.stopped -and -not $after.moving -and
            [Math]::Abs($actualX - $targetX) -le 0.003 -and
            [Math]::Abs($actualY - $targetY) -le 0.003) {
            return [ordered]@{
                command = $command
                before = $before
                after = $after
            }
        }
    } while ([DateTime]::UtcNow -lt $deadline)

    Invoke-EmergencyStop | Out-Null
    throw "Coordinated XY move timed out and E-stop was applied"
}

while ($true) {
    $line = [Console]::In.ReadLine()
    if ($null -eq $line) { break }
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    try {
        $request = $line | ConvertFrom-Json
        switch ([string]$request.op) {
            "status" { $result = Get-Mach3Status }
            "position_sample" { $result = Get-Mach3PositionSample }
            "estop" { $result = Invoke-EmergencyStop }
            "release_estop" { $result = Release-EmergencyStop $request }
            "move" { $result = Invoke-MicroMove $request }
            "long_axis_move" { $result = Invoke-ConfirmedLongAxisMove $request }
            "xy_move" { $result = Invoke-ConfirmedXYMove $request }
            "quit" {
                [Console]::Out.WriteLine('{"ok":true,"result":{"closed":true}}')
                [Console]::Out.Flush()
                break
            }
            default { throw "Unsupported operation" }
        }
        $response = [ordered]@{ ok = $true; result = $result }
    } catch {
        $response = [ordered]@{ ok = $false; error = $_.Exception.Message }
        $mach = $null
        $machScript = $null
    }
    [Console]::Out.WriteLine(($response | ConvertTo-Json -Compress -Depth 8))
    [Console]::Out.Flush()
}
