# Start-DocAnnealing.ps1 - run the QUIPU Entirety doc-annealing worker.
#
#   powershell -ExecutionPolicy Bypass -File "C:\Users\agard\Documents\VS Code\QUIPU\Start-DocAnnealing.ps1"
#
# Regenerates docs/system_entirety_map.md whenever the System Entirety's
# structural fingerprint changes.
#
#   -Once      run a single anneal cycle and exit (default is a 30-min loop)
#   -Force     rewrite the map even if unchanged
#   -Interval  seconds between cycles in loop mode (default 1800)

param(
    [switch]$Once,
    [switch]$Force,
    [int]$Interval = 1800
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
$cliArgs = @("-m", "src.quipu.doc_annealing")
if ($Once) {
    if ($Force) { $cliArgs += @("--force") }
} else {
    $cliArgs += @("--loop", "--interval", "$Interval")
}

& $py @cliArgs
