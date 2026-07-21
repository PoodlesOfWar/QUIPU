# Init-QuipuRepo.ps1 - create the fresh QUIPU git repo (single clean initial commit).
# Run AFTER Build-Quipu.ps1 reports CLEAN.
#
#   powershell -ExecutionPolicy Bypass -File "C:\Users\agard\Documents\VS Code\QUIPU\Init-QuipuRepo.ps1"

$ErrorActionPreference = "Continue"
$Repo = "C:\Users\agard\Documents\VS Code\QUIPU"

Set-Location $Repo
if (Test-Path "$Repo\.git") {
    Write-Host "Git repo already exists here - nothing done." -ForegroundColor Yellow
    exit 0
}
git init -b main
git add -A
git commit -m "QUIPU v0.24.1 - Supply Chain Brain model core, clean-room extraction (MESH-SLM-SCM-GLM-GNN, Quipu, GARD-shard, System Entirety)"
Write-Host ""
Write-Host "Done. To publish:" -ForegroundColor Cyan
Write-Host "  1. Create an empty GitHub repo (e.g. loadopoly/QUIPU) - do NOT fork Supply-Chain-Brain."
Write-Host "  2. git remote add origin https://github.com/loadopoly/QUIPU.git"
Write-Host "  3. git push -u origin main"
