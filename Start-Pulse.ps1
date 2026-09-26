# Start-Pulse.ps1 - one operator's pulse, inside the quipu container (v0.48.0).
#
# The brain lives in the container (src/quipu/entirety_service.py), which also
# pulses on its own when QUIPU_PULSE_ROUTE=1.  This runs one pulse by hand:
#   .\Start-Pulse.ps1            plan only
#   .\Start-Pulse.ps1 -Route     ingest along the plan inside the grant, then one step
param(
    [switch]$Route,
    [switch]$Refine,
    [double]$MaxSeconds = 0
)
$ErrorActionPreference = "Continue"
$cmd = @("exec", "quipu", "python", "-m", "src.quipu.qpsi.self_organising", "pulse")
if ($Route)  { $cmd += "--route" }
if ($Refine) { $cmd += "--refine" }
if ($MaxSeconds -gt 0) { $cmd += @("--max-seconds", "$MaxSeconds") }
& docker @cmd
