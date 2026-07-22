<#
  One-shot local setup for Windows (CPU box).
  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\setup_local_windows.ps1
    powershell -ExecutionPolicy Bypass -File scripts\setup_local_windows.ps1 -WithOCR -WithEmbeddings
#>
param(
    [switch]$WithOCR,
    [switch]$WithEmbeddings,
    [string]$Model = "qwen3:8b"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host "== SLM Gateway local setup ==" -ForegroundColor Cyan

# 1) venv
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual env (.venv)..."
    python -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"

# 2) core deps
Write-Host "Installing core requirements..." -ForegroundColor Cyan
& $py -m pip install --disable-pip-version-check -q -r requirements.txt

if ($WithEmbeddings) {
    Write-Host "Installing embeddings (sentence-transformers + torch CPU)..." -ForegroundColor Cyan
    & $py -m pip install -q -r requirements-embeddings.txt
}
if ($WithOCR) {
    Write-Host "Installing PaddleOCR..." -ForegroundColor Cyan
    & $py -m pip install -q -r requirements-ocr.txt
}

# 3) .env
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from template." -ForegroundColor Green
}

# 4) model backend (Ollama)
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollama) {
    Write-Host "Pulling model '$Model' via Ollama (this downloads ~5 GB the first time)..." -ForegroundColor Cyan
    ollama pull $Model
} else {
    Write-Host "Ollama not found." -ForegroundColor Yellow
    Write-Host "  Install it from https://ollama.com/download, then run:" -ForegroundColor Yellow
    Write-Host "      ollama pull $Model" -ForegroundColor Yellow
    Write-Host "  (Or test without a model now: set LLM_BACKEND=mock in .env)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done. Start the gateway with:" -ForegroundColor Green
Write-Host "    powershell -ExecutionPolicy Bypass -File scripts\run_gateway.ps1" -ForegroundColor Green
