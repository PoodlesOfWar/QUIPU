# Start-Expansion.ps1 - one expansion step, inside the quipu container (v0.48.0).
#
# The container steps the Entirety on its own every QUIPU_EXPANSION_INTERVAL_S
# (src/quipu/entirety_service.py).  This runs one step by hand; -Loop repeats it.
param(
    [switch]$Once,
    [switch]$Loop,
    [int]$Interval = 600
)
$ErrorActionPreference = "Continue"
$code = "import json; from src.quipu import system_entirety as se; print(json.dumps(se.oscillating_expansion_step(), default=str))"
function Invoke-Step { & docker exec -e QUIPU_SELF_ORGANISING=1 quipu python -c $code }
if ($Loop) {
    while ($true) { Invoke-Step; Start-Sleep -Seconds $Interval }
} else {
    Invoke-Step
}
