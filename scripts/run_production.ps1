<#
  Run the gateway in production (foreground). For a background Windows service,
  use install_service_windows.ps1 instead.
  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\run_production.ps1
    powershell -ExecutionPolicy Bypass -File scripts\run_production.ps1 -Port 8000 -Workers 2
#>
param(
    [string]$AppHost = "0.0.0.0",
    [int]$Port = 8000,
    [int]$Workers = 1
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv not found. Run setup_local_windows.ps1 first." }

if (-not (Test-Path ".env")) {
    Write-Host "No .env found — copying .env.production.example. EDIT IT before serving." -ForegroundColor Yellow
    Copy-Item ".env.production.example" ".env"
}

Write-Host "SLM gateway (production) → http://$AppHost`:$Port/v1  workers=$Workers" -ForegroundColor Green
# One worker is usually enough: the GPU model (Ollama) is the throughput limiter,
# and each extra worker loads its own OCR/embedding models into RAM.
& $py -m uvicorn app.main:app --host $AppHost --port $Port --workers $Workers --no-access-log
