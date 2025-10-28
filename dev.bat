@echo off
REM ========================================
REM Planorama - Script de Desarrollo (Windows)
REM ========================================

echo.
echo ========================================
echo   PLANORAMA - Iniciando Desarrollo
echo ========================================
echo.

REM Verificar que existe el entorno virtual
if exist ".venv\" (
    echo [1/3] Activando entorno virtual (.venv^)...
    call .venv\Scripts\activate.bat
) else if exist "venv\" (
    echo [1/3] Activando entorno virtual (venv^)...
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] No se encontro el entorno virtual.
    echo Por favor, ejecuta: python -m venv venv
    echo.
    pause
    exit /b 1
)
echo.

REM Verificar que uvicorn este instalado
python -c "import uvicorn" 2>nul
if errorlevel 1 (
    echo [ERROR] uvicorn no esta instalado.
    echo Por favor, ejecuta: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Verificar variable de entorno o archivo GOOGLE_API_KEY
if "%GOOGLE_API_KEY%"=="" (
    if exist "gemini_api_key.txt" (
        echo [INFO] API key sera cargada desde gemini_api_key.txt
        echo.
    ) else (
        echo [ADVERTENCIA] GOOGLE_API_KEY no esta configurada.
        echo El chat usara modo fallback sin Gemini.
        echo Para configurarla:
        echo   1. Crea el archivo gemini_api_key.txt con tu API key
        echo   2. O ejecuta: set GOOGLE_API_KEY=tu_api_key
        echo.
    )
) else (
    echo [INFO] API key configurada en variable de entorno
    echo.
)

REM Iniciar backend
echo [2/3] Iniciando Backend (FastAPI)...
echo Backend en: http://localhost:8000
echo Docs en: http://localhost:8000/docs
echo.
start cmd /k "title Backend - FastAPI && uvicorn api:app --reload --host 0.0.0.0 --port 8000"

REM Esperar 3 segundos
timeout /t 3 /nobreak >nul

REM Iniciar frontend
echo [3/3] Iniciando Frontend (HTTP Server)...
echo Frontend en: http://localhost:3000
echo.
start cmd /k "title Frontend - HTTP Server && python -m http.server 3000 --directory frontend"

REM Esperar 2 segundos
timeout /t 2 /nobreak >nul

REM Abrir navegador
echo.
echo ========================================
echo   PLANORAMA - Desarrollo Iniciado
echo ========================================
echo.
echo Backend API:  http://localhost:8000
echo API Docs:     http://localhost:8000/docs
echo Frontend:     http://localhost:3000
echo.
echo Para detener: Cierra las ventanas de cmd
echo ========================================
echo.

REM Abrir navegador con frontend
start http://localhost:3000

REM Mantener esta ventana abierta
echo Presiona cualquier tecla para cerrar esta ventana...
pause >nul

