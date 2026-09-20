# Register-Expansion.ps1 - a working successor to the QuipuExpansion
# scheduled task.
#
#   powershell -ExecutionPolicy Bypass -File "C:\Users\agard\Documents\VS Code\QUIPU\Register-Expansion.ps1"
#
# The original QuipuExpansion task has been firing every 10 minutes against
# SCB-Cleanup\Expansion-Watchdog.ps1, which no longer exists on disk --
# every fire has been failing (Last Result -196608) since some point after
# QUIPU was extracted from the parent Supply Chain Architect project.
#
# That task could not be repaired in place: both Register-ScheduledTask
# -Force (overwrite) and Unregister-ScheduledTask (delete) on the name
# "QuipuExpansion" returned Access is denied, and schtasks /change prompted
# for agard's account password interactively, which is not something this
# session has or should collect. Creating a NEW task from scratch is not
# blocked, so this registers "QuipuExpansion2" instead, pointed at
# Start-Expansion.ps1, same 10-minute cadence the original had. The old
# QuipuExpansion task is still present and still failing every 10 minutes --
# harmlessly, since it does nothing but log an error -- and needs a human
# with full rights on this machine (Task Scheduler GUI as the account that
# created it, or an elevated prompt) to actually remove it.
#
# Re-run this script to update the task definition. Use -Unregister to
# remove QuipuExpansion2, -Minutes N to change the cadence.

param(
    [int]$Minutes = 10,
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$Repo = "C:\Users\agard\Documents\VS Code\QUIPU"
$TaskName = "QuipuExpansion2"
$Runner = "$Repo\Start-Expansion.ps1"

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
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Runner`" -Once"

$atLogon = New-ScheduledTaskTrigger -AtLogOn
$repeating = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $Minutes)

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action `
    -Trigger @($atLogon, $repeating) -Settings $settings `
    -Description "QUIPU System Entirety - oscillating_expansion_step every $Minutes min (research-only scope; all mesh writes still held at the qpsi gates)." `
    -Force | Out-Null

Write-Host "Re-registered '$TaskName' (every $Minutes min + at logon), now pointing at $Runner." -ForegroundColor Green
Write-Host "Running one step now..." -ForegroundColor Cyan
Start-ScheduledTask -TaskName $TaskName
