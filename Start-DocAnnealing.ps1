# Start-DocAnnealing.ps1 - one doc-annealing cycle, inside the quipu container (v0.48.0).
#
# The container anneals the docs every QUIPU_DOC_ANNEAL_MINUTES and writes the
# map into the repository's docs/ (mounted).  This runs one cycle by hand.
param(
    [switch]$Force,
    [switch]$Loop,
    [int]$Interval = 1800
)
$ErrorActionPreference = "Continue"
$cliArgs = @("exec", "quipu", "python", "-m", "src.quipu.doc_annealing")
if ($Force) { $cliArgs += @("--force") }
if ($Loop)  { $cliArgs += @("--loop", "--interval", "$Interval") }
& docker @cliArgs
