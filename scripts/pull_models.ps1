<#
  Pull the model into Ollama for local CPU inference.
  Usage: powershell -ExecutionPolicy Bypass -File scripts\pull_models.ps1 [-Model qwen3:8b]
#>
param([string]$Model = "qwen3:8b")

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Error "Ollama is not installed. Get it from https://ollama.com/download"
    exit 1
}

Write-Host "Pulling $Model ..." -ForegroundColor Cyan
ollama pull $Model

Write-Host "`nInstalled models:" -ForegroundColor Green
ollama list
Write-Host "`nTip: for a lower-memory test use 'qwen3:4b'; for more accuracy on a"
Write-Host "bigger machine use 'qwen3:14b'. Set MODEL_NAME in .env to match."
