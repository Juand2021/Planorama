# Planorama 🎟️ — FastAPI + HTML Version

**Recomendador de eventos en Bogotá con interfaz web profesional**

## 🎯 Stack Tecnológico

- **Backend:** FastAPI + Python
- **Frontend:** HTML5 + CSS3 + JavaScript Vanilla + Leaflet.js
- **IA:** Google Gemini (normalización de lenguaje natural)
- **Recomendaciones:** TF-IDF + Cosine Similarity + Scoring contextual

---

## 📁 Estructura del Proyecto

```
planorama/
├── api.py                      # ⭐ Backend FastAPI
├── frontend/                   # ⭐ Frontend web
│   ├── index.html             # Interfaz principal
│   ├── css/
│   │   └── style.css          # Estilos
│   └── js/
│       ├── app.js             # Lógica principal
│       ├── chat.js            # Manejo del chat
│       └── map.js             # Mapa interactivo
├── src/                        # Lógica compartida
│   ├── geo_utils.py
│   ├── llm_interviewer.py
│   ├── recommender.py
│   └── config.py
├── data/
│   └── Planorama_BD.csv       # Dataset de eventos
├── last_versions/
│   └── app_streamlit_backup.py # Backup de Streamlit
├── dev.bat                     # Script desarrollo (Windows)
├── dev.sh                      # Script desarrollo (Linux/Mac)
├── requirements.txt            # Dependencias
└── README_FASTAPI.md          # Esta documentación
```

---

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio (si aplica)

```bash
git clone <tu-repo>
cd planorama
```

### 2. Crear entorno virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar API Key de Gemini

**Opción A: Variable de entorno (recomendado)**

**Windows:**
```bash
set GOOGLE_API_KEY=tu_api_key_aqui
```

**Linux/Mac:**
```bash
export GOOGLE_API_KEY=tu_api_key_aqui
```

**Opción B: Archivo .env (opcional)**

Copia `env.example.txt` a `.env` y edita:
```bash
GOOGLE_API_KEY=tu_api_key_aqui
```

> 🔑 Obtén tu API key en: https://makersuite.google.com/app/apikey

---

## 🎮 Desarrollo

### Método 1: Script automático (RECOMENDADO)

**Windows:**
```bash
dev.bat
```

**Linux/Mac:**
```bash
chmod +x dev.sh
./dev.sh
```

Esto abrirá:
- ✅ Backend en `http://localhost:8000`
- ✅ API Docs en `http://localhost:8000/docs`
- ✅ Frontend en `http://localhost:3000`
- ✅ Navegador automáticamente

---

### Método 2: Manual (paso a paso)

**Terminal 1 - Backend:**
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
python -m http.server 3000 --directory frontend
```

**Luego abre tu navegador en:**
```
http://localhost:3000
```

---

## 🔄 Flujo de Desarrollo

### Editar funcionalidad (backend)

1. Modifica archivos en `src/` (ej: `recommender.py`)
2. Guarda el archivo
3. **Hot-reload automático** en ~1 segundo
4. Refresca el navegador para ver cambios

### Editar interfaz (frontend)

1. Modifica archivos en `frontend/` (HTML/CSS/JS)
2. Guarda el archivo
3. Refresca el navegador (F5)
4. Cambios visibles inmediatamente

### Probar API directamente

Ve a `http://localhost:8000/docs` para:
- 📚 Ver todos los endpoints
- 🧪 Probar requests directamente
- 📊 Ver respuestas en tiempo real

---

## 📡 API Endpoints

### `GET /`
Health check de la API

### `POST /api/chat`
Procesar mensaje del usuario

**Request:**
```json
{
  "text": "Quiero un concierto mañana",
  "profile": { ... }
}
```

**Response:**
```json
{
  "reply": "¿Cuál es tu presupuesto máximo?",
  "profile": { ... },
  "done": false,
  "smalltalk": "¡Genial! Tomo nota."
}
```

### `POST /api/recommend`
Obtener recomendaciones

**Request:**
```json
{
  "profile": { ... },
  "user_lat": 4.6533,
  "user_lon": -74.0836
}
```

**Response:**
```json
{
  "events": [ ... ],
  "count": 5,
  "profile_summary": { ... }
}
```

### `GET /api/events/{event_id}`
Obtener detalles de un evento

---

## 🎨 Personalización

### Colores y estilos

Edita `frontend/css/style.css`:

```css
/* Cambiar colores principales */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### Textos y mensajes

Edita `frontend/js/chat.js` y `frontend/index.html`

### Lógica de recomendación

Edita `src/recommender.py`:
- Pesos de scoring
- Filtros adicionales
- Categorías personalizadas

---

## 🐛 Troubleshooting

### Error: "Cannot connect to API"

**Solución:**
1. Verifica que el backend esté corriendo: `http://localhost:8000`
2. Revisa la consola del navegador (F12)
3. Verifica CORS en `api.py`

### Error: "Gemini not connected"

**Solución:**
1. Verifica tu API key: `echo %GOOGLE_API_KEY%` (Windows) o `echo $GOOGLE_API_KEY` (Mac/Linux)
2. Revisa que la key sea válida en https://makersuite.google.com
3. La app funcionará en modo fallback sin Gemini (menos inteligente)

### Frontend no se actualiza

**Solución:**
1. Fuerza recarga: `Ctrl + Shift + R` (Windows/Linux) o `Cmd + Shift + R` (Mac)
2. Limpia caché del navegador
3. Revisa la consola (F12) por errores JS

### Hot-reload no funciona

**Solución:**
1. Detén uvicorn y reinicia: `Ctrl + C` → `uvicorn api:app --reload`
2. Verifica que guardaste el archivo
3. Revisa errores en la terminal

---

## 📦 Deploy a Producción

### Backend (Render.com - Gratis)

1. Sube tu código a GitHub
2. Ve a https://render.com
3. New → Web Service
4. Conecta tu repo
5. Configura:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables:** Agrega `GOOGLE_API_KEY`

### Frontend (Netlify - Gratis)

1. Ve a https://netlify.com
2. New Site → Import from Git
3. Conecta tu repo
4. Configura:
   - **Base directory:** `frontend/`
   - **Publish directory:** `.`
5. Actualiza `frontend/js/app.js`:
   ```javascript
   const API_URL = 'https://tu-backend.onrender.com';
   ```

---

## 🧪 Testing

### Probar manualmente

1. Abre `http://localhost:3000`
2. Envía mensajes en el chat
3. Verifica que el perfil se actualiza
4. Marca ubicación en el mapa (si aplica)
5. Revisa que aparecen recomendaciones

### Casos de prueba sugeridos

- "Quiero un concierto gratis mañana"
- "Algo de comedia cerca de mí este fin de semana"
- "Teatro de pago máximo 50 mil"
- "Experiencias para toda la familia"

---

## 📚 Recursos Adicionales

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Leaflet.js Docs](https://leafletjs.com/)
- [Google Gemini API](https://ai.google.dev/)
- [Render Deploy Guide](https://render.com/docs)

---

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'Agrega nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

---

## 📄 Licencia

Uso académico/educativo. Ajusta según tu contexto.

---

## ✨ Créditos

Desarrollado por [Tu Nombre] para el curso de Inteligencia Artificial, Universidad Jorge Tadeo Lozano.

---

## 🆘 Soporte

¿Problemas? Abre un Issue en GitHub o contacta a [tu email].

---

**¡Disfruta desarrollando en Planorama! 🎟️**

