<#
  Open the gateway port on the Windows firewall so the product can reach it.
  Run ELEVATED. Usage: powershell -ExecutionPolicy Bypass -File scripts\open_firewall.ps1 -Port 8000
#>
param([int]$Port = 8000, [string]$RuleName = "SLM Gateway")

$ErrorActionPreference = "Stop"
if (Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue) {
    Write-Host "Firewall rule '$RuleName' already exists." -ForegroundColor Yellow
} else {
    New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Action Allow `
        -Protocol TCP -LocalPort $Port -Profile Any | Out-Null
    Write-Host "Opened inbound TCP $Port ('$RuleName')." -ForegroundColor Green
}
Write-Host "The product can now reach:  http://<this-server-ip>:$Port/v1"
