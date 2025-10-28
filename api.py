# api.py - FastAPI Backend for Planorama
# -*- coding: utf-8 -*-
"""
Backend REST API that replaces Streamlit.
Uses the same logic from src/ modules.
"""

import os
import sys
import json
from typing import Optional, Dict, List
from datetime import datetime

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Local modules
from geo_utils import parse_date_pref
from llm_interviewer import process_turn
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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ==================== GEMINI SETUP ====================
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"
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

# Gemini prompt (same as Streamlit version)
_GEMINI_SYSTEM = (
    "Eres un normalizador de preferencias para un recomendador de eventos en Bogotá. "
    "Debes responder SOLO con JSON válido (sin texto fuera del JSON) y con EXACTAMENTE estas claves:\n"
    "{\n"
    '  "smalltalk": "texto breve, cálido y SIN preguntas",\n'
    '  "fecha": "hoy|mañana|fin_de_semana|YYYY-MM-DD|",\n'
    '  "fecha_rango": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},\n'
    '  "categorias": ["concierto|teatro|experiencia", ...],\n'
    '  "es_gratis": "gratis|pago|indiferente|",\n'
    '  "precio_max_cop": 120000,\n'
    '  "dist_importa": "si|no|",\n'
    '  "parte_del_dia": ["mañana|tarde|noche", ...] or "indiferente",\n'
    '  "edad_usuario": null,\n'
    '  "excluir_restriccion_edad": "si|no|indiferente"\n'
    "}\n"
    "Reglas de normalización (OBLIGATORIAS):\n"
    "- Mapea TODO lo relacionado con música en vivo a 'concierto': musica|música|musica en vivo|festival|show|live|rock|pop|salsa|jazz|reggaeton|trap → concierto.\n"
    "- Mapea stand up|stand-up|comedia a 'teatro'.\n"
    "- Mapea feria|expo|exposición|taller|workshop|tour|recorrido|experiencia inmersiva a 'experiencia'.\n"
    "- Permite múltiples categorías si el usuario las menciona, pero NUNCA inventes etiquetas fuera de {concierto, teatro, experiencia}.\n"
    "- Si el usuario usa palabras fuera del set (ej. 'música'), normalízalas al set (ej. 'concierto').\n"
    "- IMPORTANTE: Si el usuario menciona un RANGO RELATIVO (ej. 'próximo mes', 'próxima semana', 'en los próximos días'), calcula las fechas relativas a HOY usando formato YYYY-MM-DD. NUNCA uses fechas del pasado.\n"
    "- Si el usuario menciona un RANGO ABSOLUTO (ej. 'entre el 10 y 12 de noviembre'), usa 'fecha_rango' con start/end en YYYY-MM-DD.\n"
    "- Si da un día específico (hoy, mañana o una fecha), usa 'fecha' y deja 'fecha_rango' vacío.\n"
    "- 'precio_max_cop' debe ser entero en COP (ej. '120k'→120000; '$150.000'→150000).\n"
    "- 'edad_usuario' solo debe tener un número si el usuario menciona una edad ESPECÍFICA. Si dice 'no importa', 'indiferente', o no menciona edad, déjalo en null.\n"
    "- 'excluir_restriccion_edad': 'si' si el usuario quiere filtrar por edad (menor de edad o va con niños), 'no' o 'indiferente' si no le importa la restricción de edad.\n"
    "- 'parte_del_dia': puede ser LISTA si el usuario menciona múltiples momentos (ej. 'tarde o noche' → ['tarde','noche']). Si menciona uno solo, usa lista con un elemento (ej. ['tarde']). Si dice 'no importa', usa string 'indiferente'.\n"
    "- 'dist_importa': debe ser EXACTAMENTE 'si' o 'no' (sin tilde). Normaliza 'sí'/'yes'/'s' → 'si'. Normaliza 'no'/'n' → 'no'.\n"
    "- 'excluir_restriccion_edad': debe ser EXACTAMENTE 'si', 'no' o 'indiferente' (sin tilde). Normaliza 'sí' → 'si'.\n"
    "- 'smalltalk' DEBE ser una frase cálida SIN preguntas (ej.: '¡Listo! Tomo nota.') y SIEMPRE debe venir.\n"
    "- NO incluyas comentarios, ni backticks, ni bloques de código; SOLO JSON estricto.\n"
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
    
    return {
        "smalltalk": d.get("smalltalk", ""),
        "fecha": d.get("fecha", ""),
        "fecha_rango": d.get("fecha_rango") if isinstance(d.get("fecha_rango"), dict) else None,
        "categorias": d.get("categorias", []) or [],
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
        perfil_json = json.dumps({
            k: v for k, v in (current_profile or {}).items()
            if k in {"fecha","fecha_rango","categorias","es_gratis","precio_max_cop","dist_importa","parte_del_dia","edad_usuario","excluir_restriccion_edad"}
        }, ensure_ascii=False)

        prompt = (
            _GEMINI_SYSTEM
            + "\n\nContexto_perfil_actual_JSON:\n" + perfil_json
            + "\n\nNuevo_mensaje_usuario:\n" + (user_text or "").strip()
            + "\n\nTarea:\n"
              "- Interpreta el 'Nuevo_mensaje_usuario' y completa SOLO los campos que estén vacíos en 'Contexto_perfil_actual_JSON'. "
              "Si el usuario cambia explícitamente una preferencia, actualízala. Responde SOLO con el JSON del contrato."
        )

        generation_config = {"temperature": 0.2}
        resp = GEMINI_MODEL.generate_content(prompt, generation_config=generation_config)
        raw = getattr(resp, "text", "") or ""
        data = json.loads(_strip_to_json(raw))
        return _ensure_schema(data)
    except Exception as e:
        print(f"Error in gemini_normalize: {e}")
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

# ==================== PYDANTIC MODELS ====================
class ChatMessage(BaseModel):
    text: str
    profile: Optional[Dict] = None

class RecommendRequest(BaseModel):
    profile: Dict
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None

class LocationUpdate(BaseModel):
    lat: float
    lon: float

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Planorama API",
        "version": "1.0.0",
        "gemini_connected": GEMINI_OK,
        "events_loaded": len(df_events)
    }

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
        
        # Normalize with Gemini
        delta = gemini_normalize(user_text, current_profile)
        
        # Merge profiles
        profile = merge_profiles(current_profile, delta)
        
        # Check if profile is complete
        reply_text, _, done = process_turn("", profile)
        
        smalltalk = profile.get("smalltalk", "").strip()
        
        if done:
            bot_reply = smalltalk or "¡Perfecto! Con esa info ya puedo recomendarte."
        else:
            ask = reply_text.strip()
            bot_reply = f"{smalltalk} {ask}" if smalltalk else ask
        
        return {
            "reply": bot_reply,
            "profile": profile,
            "done": done,
            "smalltalk": smalltalk
        }
    
    except Exception as e:
        print(f"Error in chat_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend")
async def recommend_endpoint(request: RecommendRequest):
    """
    Get event recommendations based on user profile.
    
    Args:
        request: RecommendRequest with profile and optional location
    
    Returns:
        {
            "events": [...],        # List of recommended events
            "count": int,          # Number of results
            "profile_summary": {}  # Profile used for recommendations
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
                "profile_summary": profile
            }
        
        # Convert to JSON-serializable format
        events = []
        for _, row in df_rank.iterrows():
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
                "score_final": float(row.get("score_final", 0)),
                "dist_km": float(row.get("dist_km")) if pd.notna(row.get("dist_km")) else None,
            }
            events.append(event)
        
        return {
            "events": events,
            "count": len(events),
            "profile_summary": profile
        }
    
    except Exception as e:
        print(f"Error in recommend_endpoint: {e}")
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

    stalk = (delta.get("smalltalk") or "").strip()
    if stalk:
        p["smalltalk"] = stalk

    dr = delta.get("fecha_rango")
    if isinstance(dr, dict) and (dr.get("start") and dr.get("end")):
        # Normalize date range to future if it's in the past
        p["fecha_rango"] = _normalize_future_like_range_if_past(dr)
        p["fecha"] = ""
    else:
        if not p.get("fecha") and delta.get("fecha"):
            p["fecha"] = str(delta["fecha"])

    if delta.get("categorias"):
        cats_delta = [str(c).strip().lower() for c in delta["categorias"] if str(c).strip()]
        p["categorias"] = sorted(set((p.get("categorias") or [])) | set(cats_delta))

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
    print(f"✅ Gemini connected: {GEMINI_OK}")
    print(f"✅ TF-IDF vectorizer ready")
    print("=" * 60)
    print("📡 API running on http://localhost:8000")
    print("📚 Docs available at http://localhost:8000/docs")
    print("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

