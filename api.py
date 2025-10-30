# api.py - FastAPI Backend for Planorama
# -*- coding: utf-8 -*-
"""
Backend REST API that replaces Streamlit.
Uses the same logic from src/ modules.
"""

import os
import sys
import json
import warnings
from typing import Optional, Dict, List
from datetime import datetime

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Suppress pandas date parsing warnings (we handle errors with errors="coerce")
warnings.filterwarnings('ignore', category=UserWarning, message='.*Could not infer format.*')

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Local modules
from geo_utils import parse_date_pref
# Removed dependency on llm_interviewer - now 100% Gemini-driven
from recommender import compute_recommendations, _normtxt, _expand_cats

# Initialize FastAPI
app = FastAPI(
    title="Planorama API",
    description="API para recomendación de eventos en Bogotá",
    version="1.0.0"
)

# CORS configuration (permite que el frontend se conecte)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",  # VS Code Live Server
        "http://127.0.0.1:5500",
        "http://localhost:8000",  # FastAPI puede servir el frontend también
        "http://127.0.0.1:8000",
        "null",  # Para file:// protocol (desarrollo local)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ==================== GEMINI SETUP ====================
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_OK = False
GEMINI_MODEL = None

def _init_gemini():
    """Initialize Gemini from environment variable or gemini_api_key.txt file."""
    global GEMINI_OK, GEMINI_MODEL
    try:
        # Try to get from environment variable first
        _api_key = os.environ.get("GOOGLE_API_KEY", "")
        
        # If not in environment, try to read from gemini_api_key.txt
        if not _api_key:
            key_file = os.path.join(BASE_DIR, "gemini_api_key.txt")
            if os.path.exists(key_file):
                try:
                    with open(key_file, 'r') as f:
                        _api_key = f.read().strip()
                    print("✅ API key loaded from gemini_api_key.txt")
                except Exception as e:
                    print(f"⚠️  Could not read gemini_api_key.txt: {e}")
        
        if not _api_key:
            print("⚠️  WARNING: GOOGLE_API_KEY not found. Chat normalization will use fallback.")
            print("💡 TIP: Create a file 'gemini_api_key.txt' with your API key, or set GOOGLE_API_KEY environment variable.")
            GEMINI_OK, GEMINI_MODEL = False, None
            return GEMINI_OK, GEMINI_MODEL
        import google.generativeai as genai
        genai.configure(api_key=_api_key)
        GEMINI_MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)
        GEMINI_OK = True
        print("✅ Gemini initialized successfully")
        return GEMINI_OK, GEMINI_MODEL
    except Exception as e:
        print(f"⚠️  WARNING: Gemini initialization failed: {e}")
        GEMINI_OK, GEMINI_MODEL = False, None
        return GEMINI_OK, GEMINI_MODEL

_init_gemini()

# Gemini prompt - Fully AI-driven conversational flow
_GEMINI_SYSTEM = (
    "Eres Planorama, un asistente conversacional inteligente para recomendar eventos en Bogotá. "
    "Tu tarea es mantener una conversación natural con el usuario, entender sus preferencias, y determinar cuándo tienes suficiente información para hacer recomendaciones.\n\n"
    
    "FORMATO DE RESPUESTA (OBLIGATORIO - SOLO JSON):\n"
    "{\n"
    '  "smalltalk": "respuesta breve y cálida al mensaje del usuario (ej: \'¡Perfecto!\', \'Entendido\', \'Genial\')",\n'
    '  "next_question": "pregunta natural sobre lo que falta O string vacío \"\" si ya tienes suficiente información",\n'
    '  "show_categories": true/false,  // true si el usuario no sabe qué categoría quiere y debe ver la lista\n'
    '  "profile_complete": true/false,  // true solo si tienes: categoría + fecha + gratis/pago + presupuesto (si es pago) + cercanía + edad\n'
    '  "fecha": "hoy|mañana|fin_de_semana|YYYY-MM-DD|" o string vacío "",\n'
    '  "fecha_rango": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} o null,\n'
    '  "categorias": ["concierto|teatro|experiencia", ...] o [],\n'
    '  "keywords": ["palabra1", "palabra2", ...] o [],  // artistas, géneros, temas específicos\n'
    '  "es_gratis": "gratis|pago|indiferente|" o string vacío "",\n'
    '  "precio_max_cop": 120000 o null,  // entero en COP\n'
    '  "dist_importa": "si|no|" o string vacío "",\n'
    '  "parte_del_dia": ["mañana|tarde|noche", ...] o "indiferente" o string vacío "",\n'
    '  "edad_usuario": null o número entero,\n'
    '  "excluir_restriccion_edad": "si|no|indiferente|" o string vacío ""\n'
    "}\n\n"
    
    "REGLAS CRÍTICAS:\n\n"
    
    "1. CONVERSACIÓN NATURAL:\n"
    "- Analiza qué información YA TIENES en el perfil actual.\n"
    "- Identifica qué falta para poder recomendar (categoría, fecha, presupuesto, etc.).\n"
    "- Haz PREGUNTAS CONVERSACIONALES Y NATURALES (no robóticas). Ejemplos:\n"
    "   ❌ MAL: '¿Para cuándo te gustaría el plan? Responde: hoy, mañana o fecha YYYY-MM-DD'\n"
    "   ✅ BIEN: '¿Para cuándo estarías interesado en ir?' o '¿Tienes alguna fecha en mente?'\n"
    "- Si el usuario NO MENCIONA una categoría específica en su mensaje:\n"
    "   ✅ DEBES mencionar las categorías disponibles DIRECTAMENTE en 'next_question' o 'smalltalk'.\n"
    "   ✅ Dí algo como: '¡Perfecto! Las categorías disponibles son: [lista de categorías]. Puedes elegir una o más. ¿Cuál te interesa?'\n"
    "   ✅ SIEMPRE pon 'show_categories': true para que el sistema muestre los botones también.\n"
    "   ✅ NO solo preguntes '¿Qué tipo de plan te gustaría?' - MENCIONA las categorías específicas en tu respuesta.\n"
    "- Si el usuario dice explícitamente que no sabe (dice 'no sé', 'cualquiera', 'me da igual', 'muéstrame opciones', 'qué hay disponible'):\n"
    "   ✅ TAMBIÉN pon 'show_categories': true y muestra las categorías directamente.\n"
    "- Si el usuario da información incompleta, haz follow-up. Ej: si dice 'el próximo mes' pero no año, calcula la fecha.\n"
    "- Si ya tienes suficiente información, pon profile_complete=true y next_question=\"\".\n\n"
    
    "2. PERFIL COMPLETO (profile_complete=true) cuando tengas:\n"
    "   ✅ categorias: al menos una categoría\n"
    "   ✅ fecha o fecha_rango: fecha específica o rango\n"
    "   ✅ es_gratis: 'gratis', 'pago' o 'indiferente'\n"
    "   ✅ precio_max_cop: número si es_gratis='pago', sino puede ser null\n"
    "   ✅ dist_importa: 'si' o 'no'\n"
    "   ✅ (edad_usuario O excluir_restriccion_edad): al menos uno de estos\n"
    "   ✅ parte_del_dia: lista, 'indiferente' o puede ser vacío (opcional)\n\n"
    
    "3. NORMALIZACIÓN DE CAMPOS:\n"
    "- CATEGORÍAS: musica|música|concierto|festival|show|live|rock|pop|salsa|jazz|reggaeton|trap → 'concierto'\n"
    "- CATEGORÍAS: teatro|comedia|stand up|stand-up|humor|danza|ballet|circo|musical familiar → 'teatro'\n"
    "- CATEGORÍAS: feria|expo|exposición|taller|workshop|tour|experiencia inmersiva → 'experiencia'\n"
    "- FECHAS: 'mañana' o 'tomorrow' → calcula la fecha real de mañana en formato YYYY-MM-DD y usa fecha_rango.\n"
    "- FECHAS: 'pasado mañana' o 'day after tomorrow' → calcula la fecha real de pasado mañana en formato YYYY-MM-DD y usa fecha_rango.\n"
    "- FECHAS: 'hoy' o 'today' → calcula la fecha de hoy en formato YYYY-MM-DD y usa fecha_rango.\n"
    "- FECHAS: 'este sábado', 'este domingo', etc. → calcula la fecha real de ese día de la semana (si es hoy usa hoy, si es futuro usa esa fecha, si ya pasó esta semana usa la siguiente semana).\n"
    "- FECHAS: 'próximo sábado', 'proximo domingo', etc. → calcula la fecha del siguiente [día] de la semana.\n"
    "- FECHAS: 'próximo mes' → calcula fecha_rango desde HOY. 'próxima semana' → calcula fecha_rango 7 días desde HOY. NUNCA uses fechas pasadas.\n"
    "- IMPORTANTE: Cuando el usuario dice 'mañana', 'pasado mañana', 'este sábado', etc., SIEMPRE convierte a fecha_rango con la fecha real en formato YYYY-MM-DD, NO dejes strings relativos en el campo fecha.\n"
    "- PRECIO: '50 mil'→50000, '120k'→120000, '$150.000'→150000\n"
    "- KEYWORDS: Extrae términos específicos (géneros, artistas, temas) para búsqueda semántica mejorada\n\n"
    
    "4. CONTEXTO:\n"
    "- Si el perfil actual ya tiene información, NO la sobrescribas a menos que el usuario la cambie explícitamente.\n"
    "- Completa SOLO los campos que faltan.\n"
    "- Mantén un tono amigable y conversacional en smalltalk.\n\n"
    
    "EJEMPLO 1 - Usuario nuevo dice: 'hola, quiero ir a un concierto'\n"
    '{"smalltalk":"¡Hola! Me encanta ayudarte a encontrar el plan perfecto.","next_question":"¿Para cuándo te gustaría ir? ¿Esta semana, el próximo fin de semana, o tienes alguna fecha en mente?","show_categories":false,"profile_complete":false,"categorias":["concierto"],"keywords":["concierto"],"fecha":"","fecha_rango":null,"es_gratis":"","precio_max_cop":null,"dist_importa":"","parte_del_dia":"","edad_usuario":null,"excluir_restriccion_edad":""}\n\n'
    
    "EJEMPLO 1b - Usuario dice: 'hola, me gustaría hacer un plan el día de mañana' (NO menciona categoría)\n"
    '{"smalltalk":"¡Hola! Perfecto, tienes fecha. Las categorías disponibles son: Música / Clásica, Teatro / Comedia, Experiencia, etc. Puedes elegir una o más. ¿Cuál te interesa?","next_question":"","show_categories":true,"profile_complete":false,"categorias":[],"keywords":[],"fecha":"","fecha_rango":{"start":"2025-01-XX","end":"2025-01-XX"},"es_gratis":"","precio_max_cop":null,"dist_importa":"","parte_del_dia":"","edad_usuario":null,"excluir_restriccion_edad":""}\n'
    "NOTA: Como NO mencionó categoría, DEBES mencionar las categorías disponibles DIRECTAMENTE en smalltalk o next_question. Menciona al menos las primeras 5-10 categorías de la lista. show_categories=true para mostrar botones.\n\n"
    
    "EJEMPLO 2 - Usuario dice: 'mañana' (solo fecha, sin categoría)\n"
    '{"smalltalk":"¡Perfecto!","next_question":"Elige una categoría:","show_categories":true,"profile_complete":false,"categorias":[],"keywords":[],"fecha":"","fecha_rango":{"start":"2025-01-XX","end":"2025-01-XX"},"es_gratis":"","precio_max_cop":null,"dist_importa":"","parte_del_dia":"","edad_usuario":null,"excluir_restriccion_edad":""}\n'
    "NOTA: Como NO mencionó categoría, show_categories=true. NO preguntes textualmente qué tipo quiere, solo muestra las categorías.\n\n"
    
    "EJEMPLO 2b - Usuario dice: 'este sábado' (solo fecha, sin categoría - si hoy es miércoles 2025-01-15)\n"
    '{"smalltalk":"¡Perfecto!","next_question":"Elige una categoría:","show_categories":true,"profile_complete":false,"categorias":[],"keywords":[],"fecha":"","fecha_rango":{"start":"2025-01-18","end":"2025-01-19"},"es_gratis":"","precio_max_cop":null,"dist_importa":"","parte_del_dia":"","edad_usuario":null,"excluir_restriccion_edad":""}\n'
    "NOTA: Como NO mencionó categoría, show_categories=true. 'este sábado' significa el sábado más cercano. Si hoy es miércoles, el sábado de esta semana es 2025-01-18. NO preguntes qué tipo quiere, solo muestra las categorías.\n\n"
    
    "EJEMPLO 3 - Usuario dice: 'el próximo mes y tengo 50 mil pesos de presupuesto' (NO menciona categoría)\n"
    '{"smalltalk":"¡Perfecto! Ya tengo tu presupuesto.","next_question":"Elige una categoría:","show_categories":true,"profile_complete":false,"categorias":[],"keywords":[],"fecha":"","fecha_rango":{"start":"2025-02-01","end":"2025-03-01"},"precio_max_cop":50000,"es_gratis":"pago","dist_importa":"","parte_del_dia":"","edad_usuario":null,"excluir_restriccion_edad":""}\n'
    "NOTA: Como NO mencionó categoría, show_categories=true. Primero muestra las categorías, luego continuará preguntando sobre la distancia.\n\n"
    
    "EJEMPLO 4 - Perfil completo\n"
    '{"smalltalk":"¡Excelente! Ya tengo toda la información que necesito.","next_question":"","profile_complete":true,"categorias":["teatro"],"keywords":["comedia"],"fecha_rango":{"start":"2025-02-01","end":"2025-03-01"},"es_gratis":"pago","precio_max_cop":50000,"dist_importa":"no","parte_del_dia":"indiferente","edad_usuario":null,"excluir_restriccion_edad":"no"}\n\n'
    
    "IMPORTANTE: Responde SOLO con JSON válido, sin texto adicional, sin backticks, sin comentarios."
)

def _strip_to_json(text: str) -> str:
    """Strip markdown code blocks to get plain JSON."""
    if not text:
        return "{}"
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i:j+1]
    return "{}"

def _ensure_schema(d: dict) -> dict:
    """Ensure JSON contract with safe defaults."""
    pdia = d.get("parte_del_dia", "")
    if isinstance(pdia, list):
        pdia_normalized = pdia
    elif pdia == "indiferente":
        pdia_normalized = "indiferente"
    else:
        pdia_normalized = ""
    
    # Handle keywords - ensure it's a list
    keywords = d.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    
    # Handle profile_complete, next_question, and show_categories
    profile_complete = d.get("profile_complete", False)
    if not isinstance(profile_complete, bool):
        profile_complete = False
    
    next_question = d.get("next_question", "")
    if not isinstance(next_question, str):
        next_question = ""
    
    show_categories = d.get("show_categories", False)
    if not isinstance(show_categories, bool):
        show_categories = False
    
    return {
        "smalltalk": d.get("smalltalk", ""),
        "next_question": next_question,
        "show_categories": show_categories,
        "profile_complete": profile_complete,
        "fecha": d.get("fecha", ""),
        "fecha_rango": d.get("fecha_rango") if isinstance(d.get("fecha_rango"), dict) else None,
        "categorias": d.get("categorias", []) or [],
        "keywords": keywords,
        "es_gratis": d.get("es_gratis", ""),
        "precio_max_cop": d.get("precio_max_cop", None),
        "dist_importa": d.get("dist_importa", ""),
        "parte_del_dia": pdia_normalized,
        "edad_usuario": d.get("edad_usuario", None),
        "excluir_restriccion_edad": d.get("excluir_restriccion_edad", ""),
    }

def gemini_normalize(user_text: str, current_profile: Dict) -> dict:
    """Call Gemini to normalize user input into structured preferences."""
    if not GEMINI_OK or GEMINI_MODEL is None or not (user_text or "").strip():
        return _ensure_schema({"smalltalk": "¡Listo! Tomo nota. 😉"})

    try:
        # Ensure current_profile has keywords
        safe_profile = dict(current_profile or {})
        if "keywords" not in safe_profile:
            safe_profile["keywords"] = []
        
        perfil_json = json.dumps({
            k: v for k, v in safe_profile.items()
            if k in {"fecha","fecha_rango","categorias","keywords","es_gratis","precio_max_cop","dist_importa","parte_del_dia","edad_usuario","excluir_restriccion_edad"}
        }, ensure_ascii=False)

        # Build categories list text for prompt
        categories_text = ", ".join(AVAILABLE_CATEGORIES[:20])  # Show top 20
        if len(AVAILABLE_CATEGORIES) > 20:
            categories_text += f", y {len(AVAILABLE_CATEGORIES) - 20} más"

        prompt = (
            _GEMINI_SYSTEM
            + f"\n\nCATEGORÍAS DISPONIBLES EN LA BASE DE DATOS:\n{categories_text}\n"
              "IMPORTANTE: El usuario puede elegir UNA O MÁS categorías de esta lista.\n"
              "Cuando el usuario NO mencione una categoría específica, debes mencionar las categorías disponibles directamente en tu respuesta.\n\n"
            + "\n\nPERFIL_ACTUAL (información que ya tienes):\n" + perfil_json
            + "\n\nMENSAJE_DEL_USUARIO:\n" + (user_text or "").strip()
            + "\n\nINSTRUCCIONES:\n"
              "1. Analiza el mensaje del usuario y actualiza el perfil con la nueva información.\n"
              "2. Mantén toda la información previa que no sea contradicha.\n"
              "3. Determina qué información aún falta para hacer una recomendación.\n"
              "4. **CRÍTICO**: Si NO hay categoría en categorias[] (lista vacía), DEBES:\n"
              "   a) Mencionar las categorías disponibles en 'next_question' o 'smalltalk'.\n"
              "   b) Decirle al usuario que puede elegir UNA o MÁS categorías.\n"
              "   c) Pon show_categories=true para que el sistema muestre los botones.\n"
              "   Ejemplo: '¡Perfecto! Las categorías disponibles son: [lista de categorías]. Puedes elegir una o más. ¿Cuál te interesa?'\n"
              "5. Si falta información, genera una pregunta natural en 'next_question' y pon profile_complete=false.\n"
              "6. Si ya tienes suficiente información (categoría + fecha + precio + cercanía + edad), pon profile_complete=true y next_question=\"\".\n"
              "7. Responde SOLO con el JSON del contrato, sin texto adicional.\n"
        )

        generation_config = {"temperature": 0.2}
        resp = GEMINI_MODEL.generate_content(prompt, generation_config=generation_config)
        raw = getattr(resp, "text", "") or ""
        
        # Try to parse JSON
        json_str = _strip_to_json(raw)
        data = json.loads(json_str)
        
        # Ensure schema is correct
        result = _ensure_schema(data)
        return result
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error in gemini_normalize: {e}")
        print(f"Raw response: {raw[:500] if 'raw' in locals() else 'N/A'}")
        return _ensure_schema({"smalltalk": "¡Perfecto! Continuemos. 😊"})
    except Exception as e:
        import traceback
        print(f"❌ Error in gemini_normalize: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return _ensure_schema({"smalltalk": "¡Perfecto! Continuemos. 😊"})

# ==================== DATA LOADING ====================
DATA_PATH = os.path.join(BASE_DIR, "data", "Planorama_BD.csv")

def load_events_from_csv(path: str) -> pd.DataFrame:
    """Load and preprocess events data (same as Streamlit version)."""
    import unicodedata
    import re
    
    def _norm_ascii_lower(s: str) -> str:
        s = str(s or "")
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        return s.lower().strip()
    
    try:
        df = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(path, encoding="latin-1", dtype=str, keep_default_na=False)

    expected = [
        "event_id","title","Artist_name","description","category","audience","tags",
        "date_start","doors_open_time","date_end","time_start","duration_min",
        "venue_name","venue_address","barrio","localidad","city","lat","lon",
        "price_min_cop","price_max_cop","is_free","age_min","organizer_name",
        "organizer_url","source_name","source_url","image_url","status"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = ""

    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    df["uid"] = df["event_id"].where(df["event_id"].str.strip() != "", other=df.index.astype(str))
    
    # Parse dates - suppress warnings since format may vary and we handle errors gracefully
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='.*Could not infer format.*')
        df["date_start_parsed"] = pd.to_datetime(df["date_start"], errors="coerce", dayfirst=True)
        df["date_end_parsed"]   = pd.to_datetime(df["date_end"],   errors="coerce", dayfirst=True)

    def _to_hour(x) -> float:
        try:
            t = pd.to_datetime(str(x), errors="coerce").time()
            return float(t.hour) if t else np.nan
        except Exception:
            return np.nan
    df["hour_start"] = df["time_start"].apply(_to_hour)

    for c in ["price_min_cop", "price_max_cop", "lat", "lon", "age_min", "duration_min"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Fix coordinates
    def fix_coordinate(val, is_lat=True):
        if pd.isna(val):
            return val
        if is_lat:
            if val > 10:
                str_val = str(int(val))
                if len(str_val) >= 8:
                    return float(str_val[0] + "." + str_val[1:])
                elif len(str_val) >= 5:
                    return float(str_val[0] + "." + str_val[1:])
            return val
        else:
            if abs(val) > 100:
                str_val = str(int(abs(val)))
                if len(str_val) >= 10:
                    fixed = float(str_val[:2] + "." + str_val[2:])
                elif len(str_val) >= 7:
                    fixed = float(str_val[:2] + "." + str_val[2:])
                else:
                    fixed = val
                return -fixed if val < 0 else fixed
            return val
    
    df["lat"] = df["lat"].apply(lambda x: fix_coordinate(x, is_lat=True))
    df["lon"] = df["lon"].apply(lambda x: fix_coordinate(x, is_lat=False))

    def _is_free(v: str):
        s = (v or "").strip().lower()
        if s in {"true","verdadero","1","si","sí","gratis"}:  return True
        if s in {"false","falso","0","no","pago"}:            return False
        return np.nan
    df["is_free"] = df["is_free"].apply(_is_free)

    df["city_norm"] = df["city"].apply(_norm_ascii_lower)
    now = pd.Timestamp.now(tz=None).normalize()
    df["is_future"] = df["date_start_parsed"] >= now

    def _blob(r):
        parts = []
        for col in ["title","description","tags","category","Artist_name","barrio","localidad","venue_name"]:
            v = r.get(col, "")
            if isinstance(v, str) and v:
                parts.append(v)
        return " ".join(parts)
    df["text_blob"] = df.apply(_blob, axis=1)

    def _age_to_int(s: str):
        s = str(s or "").strip().lower()
        if s in {"", "todas", "toda", "all"}:
            return 0
        m = re.search(r"(\d+)", s)
        if m:
            try:
                return int(m.group(1))
            except:
                return None
        return None
    df["age_min_num"] = df["age_min"].apply(_age_to_int)

    return df

def build_tfidf(texts: pd.Series, ids: List[str]):
    """Build TF-IDF vectorizer and matrix."""
    vec = TfidfVectorizer(min_df=1, max_df=0.95)
    X = vec.fit_transform(texts.fillna(""))
    return vec, X, ids

# Load data on startup
print("📚 Loading events data...")
df_events = load_events_from_csv(DATA_PATH)
vectorizer, Xmatrix, IDS = build_tfidf(df_events["text_blob"], df_events["uid"].tolist())
print(f"✅ Loaded {len(df_events)} events")

# Extract available categories from future events
def get_available_categories(df: pd.DataFrame) -> List[str]:
    """
    Get list of unique categories from future events.
    Consolidates categories that share the same prefix before "/" (e.g., "Teatro/Danza" and "Teatro/Familia" -> "Teatro").
    """
    future_events = df[df["is_future"] == True]
    category_counts = future_events["category"].value_counts().to_dict()
    
    # Consolidate categories: group by prefix before "/"
    consolidated = {}
    for category, count in category_counts.items():
        if not category or str(category).strip() == "" or str(category).lower() in {"nan", "none", ""}:
            continue
        
        category_str = str(category).strip()
        
        # Extract base category (before "/") or use full category if no "/"
        if "/" in category_str:
            base_category = category_str.split("/")[0].strip()
        else:
            base_category = category_str
        
        # Normalize to title case to avoid duplicates like "Música" and "musica"
        base_category = base_category.capitalize()
        
        # Sum counts for consolidated categories
        if base_category in consolidated:
            consolidated[base_category] += count
        else:
            consolidated[base_category] = count
    
    # Sort by count (most popular first)
    valid_categories_sorted = sorted(
        consolidated.keys(), 
        key=lambda x: consolidated.get(x, 0), 
        reverse=True
    )
    return valid_categories_sorted

AVAILABLE_CATEGORIES = get_available_categories(df_events)
print(f"📋 Available categories ({len(AVAILABLE_CATEGORIES)}): {', '.join(AVAILABLE_CATEGORIES[:10])}...")

# ==================== PYDANTIC MODELS ====================
class ChatMessage(BaseModel):
    text: str
    profile: Optional[Dict] = None

class RecommendRequest(BaseModel):
    profile: Dict
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None
    skip_top: Optional[bool] = False  # For alternative recommendations

class LocationUpdate(BaseModel):
    lat: float
    lon: float

# ==================== ENDPOINTS ====================

@app.get("/health")
@app.get("/api/health")
async def health_check():
    """Health check endpoint - returns API status and Gemini connection state."""
    return {
        "status": "ok",
        "service": "Planorama API",
        "version": "1.0.0",
        "gemini_connected": GEMINI_OK,
        "events_loaded": len(df_events),
        "gemini_model": GEMINI_MODEL_NAME if GEMINI_OK else None
    }

@app.get("/")
async def root(accept: Optional[str] = Header(None)):
    """Serve frontend HTML or return health check as JSON based on Accept header."""
    # If Accept header requests JSON, return health check
    if accept and "application/json" in accept:
        return await health_check()
    
    # Otherwise serve the frontend HTML
    index_path = os.path.join(BASE_DIR, "frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    # Fallback to health check JSON if HTML not found
    return await health_check()

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage):
    """
    Process user message and return bot response + updated profile.
    
    Args:
        message: ChatMessage with text and current profile
    
    Returns:
        {
            "reply": str,           # Bot response
            "profile": dict,        # Updated profile
            "done": bool,          # Whether profile is complete
            "smalltalk": str       # Gemini's friendly response
        }
    """
    try:
        user_text = message.text.strip()
        current_profile = message.profile or {}
        
        # Normalize with Gemini - now includes conversational flow
        gemini_response = gemini_normalize(user_text, current_profile)
        
        # Merge profiles (Gemini already updated the profile fields)
        profile = merge_profiles(current_profile, gemini_response)
        
        # Ensure relative dates are converted to actual date ranges
        profile = _convert_relative_dates_to_ranges(profile)
        
        # Gemini tells us if profile is complete and what to ask next
        profile_complete = gemini_response.get("profile_complete", False)
        next_question = gemini_response.get("next_question", "").strip()
        smalltalk = gemini_response.get("smalltalk", "").strip()
        show_categories = gemini_response.get("show_categories", False)
        
        # FALLBACK: Si no hay categoría y el perfil no está completo, mostrar categorías automáticamente
        categorias = profile.get("categorias", [])
        # Normalize categorias - ensure it's a list and check if it's truly empty
        if categorias is None:
            categorias = []
        if not isinstance(categorias, list):
            categorias = []
        categorias = [c for c in categorias if c and str(c).strip()]  # Filter out empty strings
        
        if not profile_complete:
            if len(categorias) == 0:
                show_categories = True
                print(f"🔍 Auto-activating show_categories: no categories found in profile (len={len(profile.get('categorias', []))})")
        
        # Update profile with normalized categorias
        profile["categorias"] = categorias
        
        print(f"📊 Chat response - show_categories: {show_categories}, categorias: {categorias}, profile_complete: {profile_complete}")
        
        # Build bot reply: smalltalk + next question
        # Si vamos a mostrar categorías, no usar next_question que pregunte sobre categorías
        if show_categories:
            # Si vamos a mostrar categorías, solo usar smalltalk o un mensaje breve
            if smalltalk:
                bot_reply = smalltalk
            else:
                bot_reply = "Te muestro las opciones disponibles:"
        elif profile_complete:
            # Emit required confirmation phrase when we have all info
            bot_reply = "perfecto ya tengo la informacion requerida y el plan que recomiento es el siguiente"
        else:
            # Combine smalltalk with question naturally
            if smalltalk and next_question:
                bot_reply = f"{smalltalk} {next_question}"
            elif next_question:
                bot_reply = next_question
            else:
                bot_reply = smalltalk or "Entiendo, continuemos."
        
        # Ensure profile has all required fields with defaults
        if "keywords" not in profile:
            profile["keywords"] = []
        
        return {
            "reply": bot_reply,
            "profile": profile,
            "done": profile_complete,
            "show_categories": show_categories,
            # Frontend hint: when profile is complete, show only the top recommendation card
            "trigger_top_recommendation": True if profile_complete else False,
            "hide_results_list": True if profile_complete else False
        }
    
    except Exception as e:
        import traceback
        print(f"❌ Error in chat_endpoint: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_ai_probability(score_final: float, max_score: float, min_score: float) -> float:
    """
    Convert score_final to a probability percentage (0-100).
    Uses min-max normalization and applies a sigmoid-like curve for better interpretation.
    """
    if max_score == min_score:
        return 85.0  # Default high probability if all scores are the same
    
    # Normalize to 0-1 range
    normalized = (score_final - min_score) / (max_score - min_score)
    
    # Apply sigmoid-like transformation for more interpretable percentages
    # This gives better spread: scores near max get 85-95%, medium scores get 60-80%
    # Formula: percentage = 60 + (normalized * 35) - ensures range of roughly 60-95%
    percentage = 60.0 + (normalized * 35.0)
    
    # Clamp to reasonable range
    percentage = max(50.0, min(98.0, percentage))
    
    return round(percentage, 1)

@app.post("/api/recommend")
async def recommend_endpoint(request: RecommendRequest):
    """
    Get event recommendations based on user profile.
    When there are many options, uses AI (TF-IDF + cosine similarity) to identify the top recommendation.
    
    Args:
        request: RecommendRequest with profile and optional location
    
    Returns:
        {
            "events": [...],              # List of recommended events
            "count": int,                 # Number of results
            "profile_summary": {},        # Profile used for recommendations
            "top_recommendation": {...},  # Top AI recommendation (if count > 5)
            "ai_enabled": bool            # Whether AI recommendation was used
        }
    """
    try:
        profile = request.profile
        user_lat = request.user_lat
        user_lon = request.user_lon
        
        # Get recommendations
        df_rank = compute_recommendations(
            perfil=profile,
            df_events=df_events,
            vectorizer=vectorizer,
            Xmatrix=Xmatrix,
            IDS=IDS,
            user_lat=user_lat,
            user_lon=user_lon,
        )
        
        if df_rank.empty:
            return {
                "events": [],
                "count": 0,
                "profile_summary": profile,
                "top_recommendation": None,
                "ai_enabled": False
            }
        
        # Convert to JSON-serializable format
        events = []
        scores = []
        for _, row in df_rank.iterrows():
            score_final = float(row.get("score_final", 0))
            scores.append(score_final)
            event = {
                "uid": row.get("uid", ""),
                "title": row.get("title", ""),
                "artist_name": row.get("Artist_name", ""),
                "description": row.get("description", ""),
                "category": row.get("category", ""),
                "date_start": row.get("date_start", ""),
                "time_start": row.get("time_start", ""),
                "venue_name": row.get("venue_name", ""),
                "venue_address": row.get("venue_address", ""),
                "barrio": row.get("barrio", ""),
                "localidad": row.get("localidad", ""),
                "lat": float(row.get("lat")) if pd.notna(row.get("lat")) else None,
                "lon": float(row.get("lon")) if pd.notna(row.get("lon")) else None,
                "price_min_cop": float(row.get("price_min_cop")) if pd.notna(row.get("price_min_cop")) else None,
                "price_max_cop": float(row.get("price_max_cop")) if pd.notna(row.get("price_max_cop")) else None,
                "is_free": bool(row.get("is_free")) if pd.notna(row.get("is_free")) else False,
                "age_min": str(row.get("age_min", "")),
                "image_url": row.get("image_url", ""),
                "source_url": row.get("source_url", ""),
                "organizer_url": row.get("organizer_url", ""),
                "score_final": score_final,
                "dist_km": float(row.get("dist_km")) if pd.notna(row.get("dist_km")) else None,
            }
            events.append(event)
        
        # AI Recommendation System: Always compute a single top recommendation
        # This uses TF-IDF + cosine similarity that was already computed in compute_recommendations
        top_recommendation = None
        ai_enabled = False

        # Get the top event (highest score_final) if available
        if len(events) >= 1:
            top_event = events[0]

            # Calculate probability percentage
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            probability = calculate_ai_probability(top_event["score_final"], max_score, min_score)

            # Add probability to top event and craft explanation
            top_event["ai_probability"] = probability
            top_recommendation = {
                **top_event,
                "ai_probability": probability,
                "explanation": f"Nuestro sistema de IA analizó {len(events)} opciones y determinó que este plan tiene {probability}% de compatibilidad con tus preferencias."
            }
            ai_enabled = True
            print(f"🤖 AI Recommendation: {top_event['title'][:50]}... (Score: {top_event['score_final']:.3f}, Probability: {probability}%)")

        return {
            # To remove the results block in the UI, we signal to hide the list and focus on top card
            "events": [],
            "count": len(events),
            "profile_summary": profile,
            "top_recommendation": top_recommendation,
            "ai_enabled": ai_enabled,
            "present_top_only": True,
            "hide_results_list": True
        }
    
    except Exception as e:
        print(f"Error in recommend_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend/all")
async def recommend_all_endpoint(request: RecommendRequest):
    """
    Get all ranked event recommendations based on user profile.
    This endpoint returns the full ranked list for showing alternative options.
    
    Args:
        request: RecommendRequest with profile, optional location, and skip_top flag
    
    Returns:
        {
            "recommendations": [...],     # List of all recommended events
            "count": int,                 # Number of results
            "profile_summary": {}         # Profile used for recommendations
        }
    """
    try:
        profile = request.profile
        user_lat = request.user_lat
        user_lon = request.user_lon
        skip_top = request.skip_top
        
        # Get recommendations
        df_rank = compute_recommendations(
            perfil=profile,
            df_events=df_events,
            vectorizer=vectorizer,
            Xmatrix=Xmatrix,
            IDS=IDS,
            user_lat=user_lat,
            user_lon=user_lon,
        )
        
        if df_rank.empty:
            return {
                "recommendations": [],
                "count": 0,
                "profile_summary": profile
            }
        
        # Convert to JSON-serializable format
        events = []
        scores = []
        for idx, row in df_rank.iterrows():
            score_final = float(row.get("score_final", 0))
            scores.append(score_final)
            event = {
                "uid": row.get("uid", ""),
                "title": row.get("title", ""),
                "artist_name": row.get("Artist_name", ""),
                "description": row.get("description", ""),
                "category": row.get("category", ""),
                "date_start": row.get("date_start", ""),
                "time_start": row.get("time_start", ""),
                "venue_name": row.get("venue_name", ""),
                "venue_address": row.get("venue_address", ""),
                "barrio": row.get("barrio", ""),
                "localidad": row.get("localidad", ""),
                "lat": float(row.get("lat")) if pd.notna(row.get("lat")) else None,
                "lon": float(row.get("lon")) if pd.notna(row.get("lon")) else None,
                "price_min_cop": float(row.get("price_min_cop")) if pd.notna(row.get("price_min_cop")) else None,
                "price_max_cop": float(row.get("price_max_cop")) if pd.notna(row.get("price_max_cop")) else None,
                "is_free": bool(row.get("is_free")) if pd.notna(row.get("is_free")) else False,
                "age_min": str(row.get("age_min", "")),
                "image_url": row.get("image_url", ""),
                "source_url": row.get("source_url", ""),
                "organizer_url": row.get("organizer_url", ""),
                "score_final": score_final,
                "dist_km": float(row.get("dist_km")) if pd.notna(row.get("dist_km")) else None,
            }
            events.append(event)
        
        # Skip the top recommendation if requested (user already saw it)
        if skip_top and len(events) > 1:
            events = events[1:]  # Skip first (top) event
            scores = scores[1:]
        
        # Calculate probabilities for all events
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        
        for event in events:
            probability = calculate_ai_probability(event["score_final"], max_score, min_score)
            event["ai_probability"] = probability
            # Create a specific explanation for each
            event["explanation"] = f"Este evento tiene {probability}% de compatibilidad con tus preferencias."
        
        return {
            "recommendations": events[:10],  # Limit to top 10 alternatives
            "count": len(events),
            "profile_summary": profile
        }
    
    except Exception as e:
        print(f"Error in recommend_all_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/categories")
async def get_categories():
    """
    Get all available categories from the database, with counts of events per category.
    Returns categories sorted by count (most popular first).
    Consolidates categories that share the same prefix before "/" (e.g., "Teatro/Danza" and "Teatro/Familia" -> "Teatro").
    """
    try:
        # Get unique categories from future events only
        future_events = df_events[df_events["is_future"] == True]
        
        # Count categories
        category_counts = future_events["category"].value_counts().to_dict()
        
        # Consolidate categories: group by prefix before "/"
        consolidated = {}
        for category, count in category_counts.items():
            if not category or str(category).strip() == "" or str(category).lower() in {"nan", "none", ""}:
                continue
            
            category_str = str(category).strip()
            
            # Extract base category (before "/") or use full category if no "/"
            if "/" in category_str:
                base_category = category_str.split("/")[0].strip()
            else:
                base_category = category_str
            
            # Normalize to title case to avoid duplicates like "Música" and "musica"
            base_category = base_category.capitalize()
            
            # Sum counts for consolidated categories
            if base_category in consolidated:
                consolidated[base_category] += int(count)
            else:
                consolidated[base_category] = int(count)
        
        # Convert to list sorted by count (most popular first)
        categories = [
            {
                "name": category,
                "count": count
            }
            for category, count in sorted(consolidated.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return {
            "categories": categories,
            "total": len(categories)
        }
    except Exception as e:
        print(f"Error in get_categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/events/{event_id}")
async def get_event_detail(event_id: str):
    """Get detailed information about a specific event."""
    try:
        event = df_events[df_events["uid"] == event_id]
        if event.empty:
            raise HTTPException(status_code=404, detail="Event not found")
        
        row = event.iloc[0]
        return {
            "uid": row.get("uid", ""),
            "title": row.get("title", ""),
            "artist_name": row.get("Artist_name", ""),
            "description": row.get("description", ""),
            "category": row.get("category", ""),
            "date_start": row.get("date_start", ""),
            "date_end": row.get("date_end", ""),
            "time_start": row.get("time_start", ""),
            "venue_name": row.get("venue_name", ""),
            "venue_address": row.get("venue_address", ""),
            "barrio": row.get("barrio", ""),
            "localidad": row.get("localidad", ""),
            "city": row.get("city", ""),
            "lat": float(row.get("lat")) if pd.notna(row.get("lat")) else None,
            "lon": float(row.get("lon")) if pd.notna(row.get("lon")) else None,
            "price_min_cop": float(row.get("price_min_cop")) if pd.notna(row.get("price_min_cop")) else None,
            "price_max_cop": float(row.get("price_max_cop")) if pd.notna(row.get("price_max_cop")) else None,
            "is_free": bool(row.get("is_free")) if pd.notna(row.get("is_free")) else False,
            "age_min": str(row.get("age_min", "")),
            "organizer_name": row.get("organizer_name", ""),
            "organizer_url": row.get("organizer_url", ""),
            "source_name": row.get("source_name", ""),
            "source_url": row.get("source_url", ""),
            "image_url": row.get("image_url", ""),
            "status": row.get("status", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_event_detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== HELPER FUNCTIONS ====================

def _convert_relative_dates_to_ranges(perfil: Dict) -> Dict:
    """
    Convierte fechas relativas como "mañana", "pasado mañana", "hoy", "este sábado" a fecha_rango con fechas reales.
    Esto asegura que el perfil siempre tenga fechas concretas para filtrado.
    """
    p = dict(perfil)
    
    # Si tenemos fecha relativa, convertirla a fecha_rango
    fecha_str = p.get("fecha", "").strip().lower()
    
    if fecha_str and not p.get("fecha_rango"):
        from geo_utils import parse_date_pref
        try:
            dr = parse_date_pref(fecha_str)
            # Convertir a fecha_rango ISO format
            start_date = dr.start.date().isoformat()
            end_date = dr.end.date().isoformat()
            p["fecha_rango"] = {
                "start": start_date,
                "end": end_date
            }
            # Limpiar fecha para evitar confusión
            p["fecha"] = ""
            # Debug: log conversion for troubleshooting
            print(f"🔄 Converted relative date '{fecha_str}' → fecha_rango: {start_date} → {end_date}")
        except Exception as e:
            # Si no se puede parsear, dejar como está
            print(f"⚠️ Could not parse date '{fecha_str}': {e}")
            pass
    
    return p

def _normalize_future_like_range_if_past(dr_dict: dict) -> dict:
    """
    Si fecha_rango (start/end) viene en el pasado, la reubica al FUTURO relativo a hoy:
      - ~7 días  -> la próxima semana (lunes a lunes)
      - ~30 días -> el próximo mes (1er día del mes siguiente a 1er día del subsiguiente)
      - ~365 días-> el próximo año (1 enero a 1 enero siguiente)
      - otro     -> mismo largo empezando HOY
    Devuelve un dict {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}.
    """
    if not isinstance(dr_dict, dict):
        return dr_dict

    start = pd.to_datetime(dr_dict.get("start"), errors="coerce")
    end   = pd.to_datetime(dr_dict.get("end"),   errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return dr_dict

    today = pd.Timestamp.now(tz=None).normalize()
    if end >= today:
        return {"start": start.date().isoformat(), "end": end.date().isoformat()}
    dur_days = (end - start).days
    dur_days = max(1, int(dur_days))

    # Heurísticas por tipo de rango
    if 26 <= dur_days <= 32:
        # MES parecido
        y, m = today.year, today.month
        # 1er día del próximo mes
        if m == 12:
            m2, y2 = 1, y + 1
        else:
            m2, y2 = m + 1, y
        new_start = pd.Timestamp(year=y2, month=m2, day=1)
        # 1er día del mes subsiguiente
        if m2 == 12:
            new_end = pd.Timestamp(year=y2 + 1, month=1, day=1)
        else:
            new_end = pd.Timestamp(year=y2, month=m2 + 1, day=1)
        return {"start": new_start.date().isoformat(), "end": new_end.date().isoformat()}

    if 6 <= dur_days <= 8:
        dow = today.weekday()  
        days_to_next_monday = (7 - dow) % 7
        next_monday = today + pd.Timedelta(days=days_to_next_monday or 7)
        new_start = next_monday
        new_end = new_start + pd.Timedelta(days=7)
        return {"start": new_start.date().isoformat(), "end": new_end.date().isoformat()}

    if 360 <= dur_days <= 370:
        y = today.year + 1
        new_start = pd.Timestamp(year=y, month=1, day=1)
        new_end = pd.Timestamp(year=y + 1, month=1, day=1)
        return {"start": new_start.date().isoformat(), "end": new_end.date().isoformat()}

    new_start = today
    new_end = today + pd.Timedelta(days=dur_days)
    return {"start": new_start.date().isoformat(), "end": new_end.date().isoformat()}

def merge_profiles(base: Dict, delta: Dict) -> Dict:
    """Merge profile updates (same logic as Streamlit version)."""
    p = dict(base or {})
    
    # Ensure keywords exists as empty list if not present
    if "keywords" not in p:
        p["keywords"] = []

    stalk = (delta.get("smalltalk") or "").strip()
    if stalk:
        p["smalltalk"] = stalk

    # Handle fecha_rango first (takes priority)
    dr = delta.get("fecha_rango")
    if isinstance(dr, dict) and (dr.get("start") and dr.get("end")):
        # Normalize date range to future if it's in the past
        p["fecha_rango"] = _normalize_future_like_range_if_past(dr)
        p["fecha"] = ""
    else:
        # Handle fecha - update if provided
        if delta.get("fecha"):
            p["fecha"] = str(delta["fecha"])
        
        # Convert relative dates (mañana, pasado mañana, hoy) to fecha_rango
        p = _convert_relative_dates_to_ranges(p)

    if delta.get("categorias"):
        cats_delta = [str(c).strip().lower() for c in delta["categorias"] if str(c).strip()]
        p["categorias"] = sorted(set((p.get("categorias") or [])) | set(cats_delta))
    
    # keywords - merge lists, keep unique (handle None/empty cases)
    delta_keywords = delta.get("keywords")
    if delta_keywords:
        if isinstance(delta_keywords, list) and len(delta_keywords) > 0:
            kw_delta = [str(kw).strip().lower() for kw in delta_keywords if str(kw).strip()]
            existing_kw = p.get("keywords") or []
            if not isinstance(existing_kw, list):
                existing_kw = []
            p["keywords"] = list(set(existing_kw) | set(kw_delta))

    if not p.get("es_gratis") and delta.get("es_gratis"):
        p["es_gratis"] = str(delta["es_gratis"]).strip().lower()

    if p.get("es_gratis") == "pago" and delta.get("precio_max_cop") not in (None, "", []):
        try:
            p["precio_max_cop"] = int(float(delta["precio_max_cop"]))
        except Exception:
            pass

    # cercanía - Siempre actualizar si viene algo nuevo
    if delta.get("dist_importa"):
        dist_value = str(delta["dist_importa"]).strip().lower()
        # Normalizar variantes de sí/no
        if dist_value in {"si", "sí", "yes", "s"}:
            p["dist_importa"] = "si"
        elif dist_value in {"no", "n"}:
            p["dist_importa"] = "no"
        elif dist_value:
            p["dist_importa"] = dist_value

    if not p.get("parte_del_dia"):
        delta_pdia = delta.get("parte_del_dia")
        if isinstance(delta_pdia, list):
            normalized = [str(x).strip().lower() for x in delta_pdia if str(x).strip().lower() in {"mañana", "tarde", "noche"}]
            if normalized:
                p["parte_del_dia"] = normalized
        elif str(delta_pdia).strip().lower() == "indiferente":
            p["parte_del_dia"] = "indiferente"

    # edad - Actualizar si viene un valor
    if delta.get("edad_usuario") not in (None, "", []):
        try:
            p["edad_usuario"] = int(delta["edad_usuario"])
        except Exception:
            pass
    
    # restricción de edad - Siempre actualizar si viene algo nuevo
    delta_excl = str(delta.get("excluir_restriccion_edad") or "").strip().lower()
    if delta_excl in {"si", "sí", "no", "indiferente"}:
        # Normalizar sí con tilde
        if delta_excl in {"sí"}:
            p["excluir_restriccion_edad"] = "si"
        else:
            p["excluir_restriccion_edad"] = delta_excl

    return p

# ==================== STARTUP ====================

@app.on_event("startup")
async def startup_event():
    """Run on API startup."""
    print("=" * 60)
    print("🎟️  PLANORAMA API STARTED")
    print("=" * 60)
    print(f"✅ Events loaded: {len(df_events)}")
    
    if GEMINI_OK:
        print(f"✅ Gemini connected: YES (Model: {GEMINI_MODEL_NAME})")
    else:
        print(f"⚠️  Gemini connected: NO")
        print(f"   💡 TIP: Create 'gemini_api_key.txt' or set GOOGLE_API_KEY env var")
    
    print(f"✅ TF-IDF vectorizer ready")
    print("=" * 60)
    print("📡 API running on http://localhost:8000")
    print("📚 Docs available at http://localhost:8000/docs")
    print("🌐 Frontend available at http://localhost:8000/")
    print("💚 Health check at http://localhost:8000/health")
    print("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

