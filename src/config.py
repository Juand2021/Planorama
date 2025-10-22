# config.py (opcional)
DATA_PATH = "data/Planorama_BD.csv"
GEMINI_MODEL_NAME = "gemini-1.5-flash"
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"

# Pesos del ranking (usados en app.py si decides importarlos)
W_CONT  = 0.50
W_PREC  = 0.20
W_DIST  = 0.25
W_FECHA = 0.05

# Bogotá centro aproximado (para el mapa)
DEFAULT_CENTER = (4.6486, -74.0649)
DEFAULT_ZOOM = 12
