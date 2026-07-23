<#
  Configure Ollama for efficient GPU serving on Windows and pull the model.
  Run in an ELEVATED PowerShell (sets machine-wide env vars + restarts Ollama).
  Usage: powershell -ExecutionPolicy Bypass -File scripts\setup_ollama_gpu.ps1 [-Model qwen3:8b]
#>
param(
    [string]$Model = "qwen3:8b",
    [int]$NumParallel = 2
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Error "Ollama not installed. Install the Windows build from https://ollama.com/download (it uses your NVIDIA GPU automatically)."
    exit 1
}

Write-Host "Setting Ollama efficiency env vars (machine scope)..." -ForegroundColor Cyan
# Keep the model resident in VRAM (no reload between requests).
[Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "-1", "Machine")
# Serve a few concurrent requests from the product.
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "$NumParallel", "Machine")
[Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "Machine")
# Listen on all interfaces (only needed if the gateway runs on another host).
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0", "Machine")

Write-Host "Restarting Ollama so the new settings apply..." -ForegroundColor Cyan
Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

Write-Host "Pulling $Model (first time downloads ~5 GB)..." -ForegroundColor Cyan
ollama pull $Model

Write-Host "`nGPU / loaded model:" -ForegroundColor Green
ollama ps
Write-Host "`nVerify GPU usage with:  nvidia-smi   (you should see an 'ollama' process on the GPU)."
Write-Host "Done. Now run the gateway: scripts\run_production.ps1" -ForegroundColor Green
