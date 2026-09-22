# Start-Pulse.ps1 - apply one pulse of the field to the System Entirety along the
# allocation the self-organising loop recorded (docs/QPSI_SELF_ORGANISING.md).
#
#   powershell -ExecutionPolicy Bypass -File "C:\Users\agard\Documents\VS Code\QUIPU\Start-Pulse.ps1"          # plan only
#   powershell -ExecutionPolicy Bypass -File "C:\Users\agard\Documents\VS Code\QUIPU\Start-Pulse.ps1" -Route   # apply it
#
# Without -Route this prints the plan (documents per corpus_ingest source from
# the Kirchhoff partition of the budget) and changes nothing.  With -Route it
# runs corpus_ingest over exactly those sources with exactly those counts,
# refinement off unless -Refine, then takes one expansion step so the flux is
# read.  Routing is an explicit switch at the call site, never an environment
# variable.  The sources are corpus_ingest.SOURCES and nothing else.
#
# This script does not schedule itself.  A recurring pulse is a Windows
# Scheduled Task the operator registers (compare Register-Expansion.ps1), and
# whether a scheduled -Route is admissible under Invariance #7 is the
# operator's ruling, not this script's.
#
# The loop itself is wired only when QUIPU_SELF_ORGANISING=1 is set for the
# process that steps the Entirety (Start-Expansion.ps1 / the scheduled task).
# This script enables it for its own process so the step it takes is a
# self-organising one.
#
#   -Route        apply the plan (operator's act)
#   -Refine       also run Ring-5 refinement per Weyl cycle
#   -MaxSeconds   time budget for the ingest (0 = none)

param(
    [switch]$Route,
    [switch]$Refine,
    [double]$MaxSeconds = 0
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
$env:QUIPU_SELF_ORGANISING = "1"

$args = @("-m", "src.quipu.qpsi.self_organising", "pulse")
if ($Route)  { $args += "--route" }
if ($Refine) { $args += "--refine" }
if ($MaxSeconds -gt 0) { $args += @("--max-seconds", "$MaxSeconds") }

& $py @args
