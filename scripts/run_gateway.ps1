<#
  Start the SLM gateway.
  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\run_gateway.ps1
    powershell -ExecutionPolicy Bypass -File scripts\run_gateway.ps1 -Port 8000 -Reload
#>
param(
    [string]$AppHost = "0.0.0.0",
    [int]$Port = 8000,
    [switch]$Reload
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$args = @("-m", "uvicorn", "app.main:app", "--host", $AppHost, "--port", "$Port")
if ($Reload) { $args += "--reload" }

Write-Host "Gateway → http://$AppHost`:$Port   (OpenAI base_url = http://localhost:$Port/v1)" -ForegroundColor Green
& $py @args
