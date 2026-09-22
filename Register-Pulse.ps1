# Register-Pulse.ps1 - schedule the operator's pulse (docs/QPSI_SELF_ORGANISING.md).
#
#   powershell -ExecutionPolicy Bypass -File "C:\Users\agard\Documents\VS Code\QUIPU\Register-Pulse.ps1"
#   powershell -ExecutionPolicy Bypass -File "C:\Users\agard\Documents\VS Code\QUIPU\Register-Pulse.ps1" -Minutes 30
#   powershell -ExecutionPolicy Bypass -File "C:\Users\agard\Documents\VS Code\QUIPU\Register-Pulse.ps1" -Unregister
#
# Registers "QuipuPulse", pointed at Start-Pulse.ps1 -Route, every -Minutes
# (default 10) and at logon.  Each fire applies one pulse of the field along
# the allocation the self-organising loop recorded, inside the operator's
# grant (self_organising.constrained_gate: sources within corpus_ingest.SOURCES
# as narrowed by QUIPU_PULSE_SOURCES, documents within QUIPU_SOMN_BUDGET_DOCS),
# then takes one expansion step so the flux is read.
#
# Ruling (operator, 2026-09-22): the operator gave the access, so r-ADMIN has
# enabled a constrained gate -- allocation within the grant is not a widening
# under Invariance #7.  That ruling is what permits this task to exist.
# Registering it is still the operator's act: this script only runs when a
# person runs it.
#
# Ten minutes is the default because the emergence detector wants an onset
# and an offset of the field inside its 16-row window (~21 min at the live
# cadence) and the flux window (QUIPU_FLUX_WINDOW_S) is 5 min.

param(
    [int]$Minutes = 10,
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$Repo = "C:\Users\agard\Documents\VS Code\QUIPU"
$TaskName = "QuipuPulse"
$Runner = "$Repo\Start-Pulse.ps1"

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'." -ForegroundColor Yellow
    } else {
        Write-Host "Task '$TaskName' not found." -ForegroundColor Yellow
    }
    return
}

if (-not (Test-Path $Runner)) {
    Write-Host "Runner not found: $Runner" -ForegroundColor Red
    exit 1
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Runner`" -Route"

$atLogon = New-ScheduledTaskTrigger -AtLogOn
$repeating = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes $Minutes)

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes ([Math]::Max(5, $Minutes - 1)))

Register-ScheduledTask -TaskName $TaskName -Action $action `
    -Trigger @($atLogon, $repeating) -Settings $settings `
    -Description "QUIPU System Entirety - operator's pulse every $Minutes min: corpus_ingest along the self-organising allocation inside the granted budget/sources (constrained gate, 2026-09-22 ruling), then one expansion step. Edges still held at the qpsi gates." `
    -Force | Out-Null

Write-Host "Registered '$TaskName' (every $Minutes min + at logon), pointing at $Runner -Route." -ForegroundColor Green
