# Show-TodaysLearning.ps1 - "what has the System Entirety learned today?"
#
#   powershell -ExecutionPolicy Bypass -File "C:\Users\agard\Documents\VS Code\QUIPU\Show-TodaysLearning.ps1"
#
# Optional: -Date "2026-07-19" to view a past day; -Json for raw JSON output.
# Read-only - queries QUIPU's own local_brain.sqlite, never writes anything.

param(
    [string]$Date = "",
    [switch]$Json
)

$ErrorActionPreference = "Continue"
$Repo = "C:\Users\agard\Documents\VS Code\QUIPU"

function Get-Python() {
    $venv = "$Repo\.venv\Scripts\python.exe"
    if (Test-Path $venv) { return $venv }
    foreach ($c in @("python", "python3")) {
        $f = Get-Command $c -ErrorAction SilentlyContinue
        if ($f -and $f.Source) { return $f.Source }
    }
    return $null
}

$py = Get-Python
if (-not $py) { Write-Host "No Python interpreter found." -ForegroundColor Red; exit 1 }

Set-Location $Repo
$cliArgs = @("-m", "src.quipu.daily_digest")
if ($Date) { $cliArgs += @("--date", $Date) }
if ($Json) { $cliArgs += @("--json") }

& $py @cliArgs
