<#
.SYNOPSIS
    Starts the GARD Lite self-sustainable compression/decompression application.
#>

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Starting GARD Lite Application on http://127.0.0.1:8780/..." -ForegroundColor Cyan
python "$ScriptDir\run_gard_lite.py"
