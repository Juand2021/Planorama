#!/bin/bash
# ========================================
# Planorama - Script de Desarrollo (Linux/Mac)
# ========================================

echo ""
echo "========================================"
echo "  PLANORAMA - Iniciando Desarrollo"
echo "========================================"
echo ""

# Verificar que existe el entorno virtual
if [ ! -d "venv" ]; then
    echo "[ERROR] No se encontró el entorno virtual."
    echo "Por favor, ejecuta: python -m venv venv"
    echo ""
    exit 1
fi

# Activar entorno virtual
echo "[1/3] Activando entorno virtual..."
source venv/bin/activate
echo ""

# Verificar que uvicorn esté instalado
python -c "import uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] uvicorn no está instalado."
    echo "Por favor, ejecuta: pip install -r requirements.txt"
    echo ""
    exit 1
fi

# Verificar variable de entorno GOOGLE_API_KEY
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "[ADVERTENCIA] GOOGLE_API_KEY no está configurada."
    echo "El chat usará modo fallback sin Gemini."
    echo "Para configurarla: export GOOGLE_API_KEY=tu_api_key"
    echo ""
fi

# Iniciar backend en background
echo "[2/3] Iniciando Backend (FastAPI)..."
echo "Backend en: http://localhost:8000"
echo "Docs en: http://localhost:8000/docs"
echo ""
uvicorn api:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Esperar 3 segundos
sleep 3

# Iniciar frontend en background
echo "[3/3] Iniciando Frontend (HTTP Server)..."
echo "Frontend en: http://localhost:3000"
echo ""
python -m http.server 3000 --directory frontend &
FRONTEND_PID=$!

# Esperar 2 segundos
sleep 2

echo ""
echo "========================================"
echo "  PLANORAMA - Desarrollo Iniciado"
echo "========================================"
echo ""
echo "Backend API:  http://localhost:8000"
echo "API Docs:     http://localhost:8000/docs"
echo "Frontend:     http://localhost:3000"
echo ""
echo "Para detener: Presiona Ctrl+C"
echo "========================================"
echo ""

# Abrir navegador (intenta detectar el sistema)
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:3000 &
elif command -v open > /dev/null; then
    open http://localhost:3000 &
fi

# Función para limpiar al salir
cleanup() {
    echo ""
    echo "Deteniendo servicios..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Servicios detenidos."
    exit 0
}

# Capturar Ctrl+C
trap cleanup SIGINT SIGTERM

# Mantener el script corriendo
wait

