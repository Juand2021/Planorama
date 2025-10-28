@echo off
REM ========================================
REM Configurar API Key de Gemini (Permanente)
REM ========================================

echo.
echo ========================================
echo   CONFIGURAR GEMINI API KEY
echo ========================================
echo.

REM Pedir la API key al usuario
set /p API_KEY="Ingresa tu GOOGLE_API_KEY: "

if "%API_KEY%"=="" (
    echo [ERROR] No ingresaste ninguna API key.
    pause
    exit /b 1
)

REM Guardar en variable de entorno del USUARIO (permanente)
setx GOOGLE_API_KEY "%API_KEY%"

echo.
echo ========================================
echo   API KEY CONFIGURADA EXITOSAMENTE
echo ========================================
echo.
echo La API key se guardo permanentemente.
echo Cierra y vuelve a abrir PowerShell/CMD.
echo Luego ejecuta: .\dev.bat
echo.
echo ========================================
pause

