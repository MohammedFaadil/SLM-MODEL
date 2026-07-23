<#
  Install the SLM gateway as a Windows service so it starts on boot and stays up.
  Uses NSSM (https://nssm.cc). Run in an ELEVATED PowerShell.

  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\install_service_windows.ps1
    powershell -ExecutionPolicy Bypass -File scripts\install_service_windows.ps1 -Port 8000 -ServiceName SLMGateway
#>
param(
    [string]$ServiceName = "SLMGateway",
    [string]$AppHost = "0.0.0.0",
    [int]$Port = 8000,
    [string]$NssmPath = "nssm"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "logs"

if (-not (Test-Path $py)) { throw "venv not found at $py. Run setup_local_windows.ps1 first." }
if (-not (Get-Command $NssmPath -ErrorAction SilentlyContinue)) {
    Write-Error "NSSM not found. Download from https://nssm.cc/download, unzip, and either add nssm.exe to PATH or pass -NssmPath 'C:\path\to\nssm.exe'."
    exit 1
}
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Write-Host "Installing service '$ServiceName'..." -ForegroundColor Cyan
& $NssmPath install $ServiceName $py "-m" "uvicorn" "app.main:app" "--host" $AppHost "--port" "$Port" "--workers" "1"
& $NssmPath set $ServiceName AppDirectory $root
& $NssmPath set $ServiceName AppStdout (Join-Path $logDir "gateway.out.log")
& $NssmPath set $ServiceName AppStderr (Join-Path $logDir "gateway.err.log")
& $NssmPath set $ServiceName Start SERVICE_AUTO_START
& $NssmPath set $ServiceName AppRotateFiles 1
& $NssmPath set $ServiceName AppRotateBytes 10485760

Write-Host "Starting service..." -ForegroundColor Cyan
& $NssmPath start $ServiceName

Write-Host "`nService '$ServiceName' installed and started." -ForegroundColor Green
Write-Host "  Health : http://localhost:$Port/health"
Write-Host "  Manage : nssm restart $ServiceName | nssm stop $ServiceName | nssm remove $ServiceName confirm"
Write-Host "  Logs   : $logDir"
