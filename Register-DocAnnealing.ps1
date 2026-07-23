# Register-DocAnnealing.ps1 - install the QUIPU doc-annealing worker as a
# recurring Windows Scheduled Task so the living system map stays current.
#
#   powershell -ExecutionPolicy Bypass -File "C:\Users\agard\Documents\VS Code\QUIPU\Register-DocAnnealing.ps1"
#
# Creates task "QUIPUDocAnnealing": runs one anneal cycle at logon and every
# 30 minutes thereafter. Re-run this script to update the task definition.
# Use -Unregister to remove it, -Minutes N to change the cadence.

param(
    [int]$Minutes = 30,
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$Repo = "C:\Users\agard\Documents\VS Code\QUIPU"
$TaskName = "QUIPUDocAnnealing"
$Runner = "$Repo\Start-DocAnnealing.ps1"

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

# One anneal cycle per fire; the schedule provides the cadence.
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Runner`" -Once"

$atLogon = New-ScheduledTaskTrigger -AtLogOn
$repeating = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $Minutes)

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action `
    -Trigger @($atLogon, $repeating) -Settings $settings `
    -Description "QUIPU Entirety doc annealing - regenerates docs/system_entirety_map.md every $Minutes min." `
    -Force | Out-Null

Write-Host "Registered '$TaskName' (every $Minutes min + at logon)." -ForegroundColor Green
Write-Host "Running one anneal cycle now to seed the map..." -ForegroundColor Cyan
Start-ScheduledTask -TaskName $TaskName
