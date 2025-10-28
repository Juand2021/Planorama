# ========================================
# Configurar API Key de Gemini (PowerShell)
# ========================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CONFIGURAR GEMINI API KEY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Pedir la API key
$apiKey = Read-Host "Ingresa tu GOOGLE_API_KEY"

if ([string]::IsNullOrWhiteSpace($apiKey)) {
    Write-Host ""
    Write-Host "[ERROR] No ingresaste ninguna API key." -ForegroundColor Red
    Write-Host ""
    Read-Host "Presiona Enter para salir"
    exit 1
}

# Guardar permanentemente (nivel usuario)
[System.Environment]::SetEnvironmentVariable("GOOGLE_API_KEY", $apiKey, [System.EnvironmentVariableTarget]::User)

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  API KEY CONFIGURADA EXITOSAMENTE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "La API key se guardo permanentemente." -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANTE:" -ForegroundColor Yellow
Write-Host "1. Cierra esta ventana de PowerShell" -ForegroundColor White
Write-Host "2. Abre una NUEVA ventana de PowerShell" -ForegroundColor White
Write-Host "3. Ve a la carpeta del proyecto" -ForegroundColor White
Write-Host "4. Ejecuta: .\dev.bat" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Presiona Enter para cerrar"

