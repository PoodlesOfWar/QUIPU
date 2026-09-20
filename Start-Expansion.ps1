# Start-Expansion.ps1 - run the QUIPU System Entirety's 7th-axis expansion step.
#
#   powershell -ExecutionPolicy Bypass -File "C:\Users\agard\Documents\VS Code\QUIPU\Start-Expansion.ps1"
#
# Calls system_entirety.oscillating_expansion_step() once per fire. The
# function has its own 90-second rate limit, so calling it more often than
# that just no-ops the extra calls rather than double-stepping.
#
# Every write this step can make to the mesh runs through DIVINE_BLESSING_SQRT(-1)
# (src/quipu/divine_blessing.py) -- held at the six gates unless
# QUIPU_REALISE_GRANT_REF is set. Research-only scope by default.
#
# This replaces the QuipuExpansion scheduled task's prior target,
# SCB-Cleanup\Expansion-Watchdog.ps1, which no longer exists on disk (it
# looks like it was left behind when QUIPU was extracted from the parent
# Supply Chain Architect project) -- every 10-minute fire had been failing
# since. This restores the Entirety-stepping half of what that task did.
# There is no surviving copy of Expansion-Watchdog.ps1, so if it also drove
# research-corpus crawling, that half is not reconstructed here.
#
#   -Once      run a single step and exit (default)
#   -Loop      keep running in this process, one attempt every -Interval seconds
#   -Interval  seconds between attempts in loop mode (default 600; the
#              function's own limiter is 90s, so anything below that just
#              no-ops more often without stepping any faster)

param(
    [switch]$Once,
    [switch]$Loop,
    [int]$Interval = 600
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
$stepCode = "import sys; sys.path.insert(0, '.'); from src.quipu import system_entirety as se; " +
            "import json; print(json.dumps(se.oscillating_expansion_step()))"

function Invoke-Step {
    & $py -c $stepCode
}

if ($Loop) {
    while ($true) {
        Invoke-Step
        Start-Sleep -Seconds $Interval
    }
} else {
    Invoke-Step
}
