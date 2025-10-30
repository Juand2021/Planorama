
# Planorama — app.py (estructura final ordenada y limpia)

# 0) Imports y bootstrap de rutas
import os, sys, json, unicodedata, re
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Librerías externas
import streamlit as st 
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

# Módulos locales
from geo_utils import (
    haversine,
    compute_distance_score,
    require_user_point,
    normalize_distance,
    parse_date_pref,
    score_part_of_day,
    TAU_DEFAULT,
    R_DEFAULT,
)
# Removed dependency on llm_interviewer - now 100% Gemini-driven
from recommender import compute_recommendations, _normtxt, _expand_cats
from ui_utils import render_results, render_location_map


# 1) Config de pagina
st.set_page_config(page_title="Planorama", page_icon="🎟️", layout="wide")

# 2) Gemini — API key 

GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_OK = False
GEMINI_MODEL = None

def _init_gemini():
    """Configura Gemini desde st.secrets['GOOGLE_API_KEY'] u os.environ['GOOGLE_API_KEY']."""
    global GEMINI_OK, GEMINI_MODEL
    try:
        _api_key = st.secrets.get("GOOGLE_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if not _api_key:
            GEMINI_OK, GEMINI_MODEL = False, None
            return GEMINI_OK, GEMINI_MODEL
        import google.generativeai as genai
        genai.configure(api_key=_api_key)
        GEMINI_MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)
        GEMINI_OK = True
        return GEMINI_OK, GEMINI_MODEL
    except Exception:
        GEMINI_OK, GEMINI_MODEL = False, None
        return GEMINI_OK, GEMINI_MODEL

_init_gemini()

def _strip_to_json(text: str) -> str:
    """Recorta ```json ...``` a JSON plano."""
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
    """Aplica contrato JSON esperado por la app (valores por defecto seguros)."""
    # Handle parte_del_dia: can be a list or "indiferente" string
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
    
    # Handle profile_complete and next_question
    profile_complete = d.get("profile_complete", False)
    if not isinstance(profile_complete, bool):
        profile_complete = False
    
    next_question = d.get("next_question", "")
    if not isinstance(next_question, str):
        next_question = ""
    
    return {
        "smalltalk": d.get("smalltalk", ""),
        "next_question": next_question,
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

_GEMINI_SYSTEM = (
    "Eres Planorama, un asistente conversacional inteligente para recomendar eventos en Bogotá. "
    "Tu tarea es mantener una conversación natural con el usuario, entender sus preferencias, y determinar cuándo tienes suficiente información para hacer recomendaciones.\n\n"
    
    "FORMATO DE RESPUESTA (OBLIGATORIO - SOLO JSON):\n"
    "{\n"
    '  "smalltalk": "respuesta breve y cálida al mensaje del usuario (ej: \'¡Perfecto!\', \'Entendido\', \'Genial\')",\n'
    '  "next_question": "pregunta natural sobre lo que falta O string vacío \"\" si ya tienes suficiente información",\n'
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
    '{"smalltalk":"¡Hola! Me encanta ayudarte a encontrar el plan perfecto.","next_question":"¿Para cuándo te gustaría ir? ¿Esta semana, el próximo fin de semana, o tienes alguna fecha en mente?","profile_complete":false,"categorias":["concierto"],"keywords":["concierto"],"fecha":"","fecha_rango":null,"es_gratis":"","precio_max_cop":null,"dist_importa":"","parte_del_dia":"","edad_usuario":null,"excluir_restriccion_edad":""}\n\n'
    
    "EJEMPLO 2 - Usuario dice: 'mañana'\n"
    '{"smalltalk":"¡Perfecto!","next_question":"¿Qué tipo de evento te gustaría? ¿Concierto, teatro o alguna experiencia?","profile_complete":false,"fecha":"","fecha_rango":{"start":"2025-01-XX","end":"2025-01-XX"},"es_gratis":"","precio_max_cop":null,"dist_importa":"","parte_del_dia":"","edad_usuario":null,"excluir_restriccion_edad":""}\n'
    "NOTA: Calcula la fecha real de mañana (día de hoy + 1 día) en formato YYYY-MM-DD y usa fecha_rango, NO pongas 'mañana' en el campo fecha.\n\n"
    
    "EJEMPLO 2b - Usuario dice: 'este sábado' (si hoy es miércoles 2025-01-15)\n"
    '{"smalltalk":"¡Perfecto!","next_question":"¿Qué tipo de evento te gustaría?","profile_complete":false,"fecha":"","fecha_rango":{"start":"2025-01-18","end":"2025-01-19"},"es_gratis":"","precio_max_cop":null,"dist_importa":"","parte_del_dia":"","edad_usuario":null,"excluir_restriccion_edad":""}\n'
    "NOTA: 'este sábado' significa el sábado más cercano. Si hoy es miércoles, el sábado de esta semana es 2025-01-18. Calcula correctamente la fecha y úsala en fecha_rango con formato YYYY-MM-DD.\n\n"
    
    "EJEMPLO 3 - Usuario dice: 'el próximo mes y tengo 50 mil pesos de presupuesto'\n"
    '{"smalltalk":"¡Perfecto! Ya tengo tu presupuesto.","next_question":"¿Te importa que el evento esté cerca de donde estás, o no tienes problema con la distancia?","profile_complete":false,"fecha":"","fecha_rango":{"start":"2025-02-01","end":"2025-03-01"},"precio_max_cop":50000,"es_gratis":"pago","dist_importa":"","parte_del_dia":"","edad_usuario":null,"excluir_restriccion_edad":""}\n\n'
    
    "EJEMPLO 4 - Perfil completo\n"
    '{"smalltalk":"¡Excelente! Ya tengo toda la información que necesito.","next_question":"","profile_complete":true,"categorias":["teatro"],"keywords":["comedia"],"fecha_rango":{"start":"2025-02-01","end":"2025-03-01"},"es_gratis":"pago","precio_max_cop":50000,"dist_importa":"no","parte_del_dia":"indiferente","edad_usuario":null,"excluir_restriccion_edad":"no"}\n\n'
    
    "IMPORTANTE: Responde SOLO con JSON válido, sin texto adicional, sin backticks, sin comentarios."
)

def gemini_normalize(user_text: str, current_profile: Dict) -> dict:
    """
    Llama a Gemini para normalizar el turno, dándole contexto del perfil actual.
    Devuelve SIEMPRE el contrato con defaults si algo falta.
    """
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

        prompt = (
            _GEMINI_SYSTEM
            + "\n\nPERFIL_ACTUAL (información que ya tienes):\n" + perfil_json
            + "\n\nMENSAJE_DEL_USUARIO:\n" + (user_text or "").strip()
            + "\n\nINSTRUCCIONES:\n"
              "1. Analiza el mensaje del usuario y actualiza el perfil con la nueva información.\n"
              "2. Mantén toda la información previa que no sea contradicha.\n"
              "3. Determina qué información aún falta para hacer una recomendación.\n"
              "4. Si falta información, genera una pregunta natural en 'next_question' y pon profile_complete=false.\n"
              "5. Si ya tienes suficiente información (categoría + fecha + precio + cercanía + edad), pon profile_complete=true y next_question=\"\".\n"
              "6. Responde SOLO con el JSON del contrato, sin texto adicional.\n"
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
        return _ensure_schema({"smalltalk": "¡Perfecto! Continuemos. 😊", "next_question": "¿Podrías repetir eso de otra manera?", "profile_complete": False})
    except Exception as e:
        import traceback
        print(f"❌ Error in gemini_normalize: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return _ensure_schema({"smalltalk": "¡Perfecto! Continuemos. 😊", "next_question": "¿Podrías repetir eso de otra manera?", "profile_complete": False})

# 3) Carga y preparación del dataset (CSV + normalización + TF-IDF)
DATA_PATH = os.path.join(BASE_DIR, "data", "Planorama_BD.csv")

@st.cache_data(show_spinner=False)
def _norm_ascii_lower(s: str) -> str:
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()

@st.cache_data(show_spinner=True)
def load_events_from_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(path, encoding="latin-1", dtype=str, keep_default_na=False)

    # Columnas esperadas (si falta alguna, la creamos vacía)
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

    # Limpieza básica
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    # UID estable
    df["uid"] = df["event_id"].where(df["event_id"].str.strip() != "", other=df.index.astype(str))

    # Fechas y horas
    df["date_start_parsed"] = pd.to_datetime(df["date_start"], errors="coerce", dayfirst=True)
    df["date_end_parsed"]   = pd.to_datetime(df["date_end"],   errors="coerce", dayfirst=True)

    def _to_hour(x) -> float:
        try:
            t = pd.to_datetime(str(x), errors="coerce").time()
            return float(t.hour) if t else np.nan
        except Exception:
            return np.nan
    df["hour_start"] = df["time_start"].apply(_to_hour)

    # Numéricos
    for c in ["price_min_cop", "price_max_cop", "lat", "lon", "age_min", "duration_min"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Fix lat/lon: Bogotá coordinates should be ~4.6 lat, ~-74.0 lon
    # If values are way off (missing decimals), normalize them
    def fix_coordinate(val, is_lat=True):
        """Fix coordinates that are missing decimal points"""
        if pd.isna(val):
            return val
        # Expected ranges: lat ~4-5, lon ~-73 to -75
        if is_lat:
            # If lat > 10, it's likely missing decimals (e.g., 461222222 → 4.61222222)
            if val > 10:
                # Count digits and divide appropriately
                str_val = str(int(val))
                if len(str_val) >= 8:  # e.g., 461222222
                    return float(str_val[0] + "." + str_val[1:])
                elif len(str_val) >= 5:  # e.g., 468485
                    return float(str_val[0] + "." + str_val[1:])
            return val
        else:  # longitude
            # If abs(lon) > 100, it's likely missing decimals
            if abs(val) > 100:
                str_val = str(int(abs(val)))
                if len(str_val) >= 10:  # e.g., 7406888889
                    fixed = float(str_val[:2] + "." + str_val[2:])
                elif len(str_val) >= 7:  # e.g., 7407321
                    fixed = float(str_val[:2] + "." + str_val[2:])
                else:
                    fixed = val
                return -fixed if val < 0 else fixed
            return val
    
    df["lat"] = df["lat"].apply(lambda x: fix_coordinate(x, is_lat=True))
    df["lon"] = df["lon"].apply(lambda x: fix_coordinate(x, is_lat=False))

    # is_free → booleano si es claro
    def _is_free(v: str):
        s = (v or "").strip().lower()
        if s in {"true","verdadero","1","si","sí","gratis"}:  return True
        if s in {"false","falso","0","no","pago"}:            return False
        return np.nan
    df["is_free"] = df["is_free"].apply(_is_free)

    # Auxiliares
    df["city_norm"] = df["city"].apply(_norm_ascii_lower)
    now = pd.Timestamp.now(tz=None).normalize()
    df["is_future"] = df["date_start_parsed"] >= now

    # Texto para TF-IDF
    def _blob(r):
        parts = []
        for col in ["title","description","tags","category","Artist_name","barrio","localidad","venue_name"]:
            v = r.get(col, "")
            if isinstance(v, str) and v:
                parts.append(v)
        return " ".join(parts)
    df["text_blob"] = df.apply(_blob, axis=1)

    # age_min → entero robusto
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

@st.cache_resource(show_spinner=False)
def build_tfidf(texts: pd.Series, ids: List[str]):
    vec = TfidfVectorizer(min_df=1, max_df=0.95)
    X = vec.fit_transform(texts.fillna(""))
    return vec, X, ids

df = load_events_from_csv(DATA_PATH)
vectorizer, Xmatrix, IDS = build_tfidf(df["text_blob"], df["uid"].tolist())

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

# 4–5) Estado de la app
if "perfil" not in st.session_state:
    st.session_state.perfil = {
        "smalltalk": "",
        "fecha": "",
        "fecha_rango": None,
        "categorias": [],
        "keywords": [],
        "es_gratis": "",
        "precio_max_cop": None,
        "dist_importa": "",
        "parte_del_dia": "",  # Empty until user specifies (mañana/tarde/noche/indiferente)
        "edad_usuario": None,
        "excluir_restriccion_edad": "",  # Empty until user specifies
    }
if "chat" not in st.session_state:
    st.session_state.chat = [
        ("bot", "¡Hola! 👋 Soy Planorama, tu asistente para encontrar planes en Bogotá. Cuéntame qué tipo de evento buscas y te ayudo a encontrar las mejores opciones. 😊")
    ]
if "user_lat" not in st.session_state:
    st.session_state.user_lat = None
if "user_lon" not in st.session_state:
    st.session_state.user_lon = None
if "ready" not in st.session_state:
    st.session_state.ready = False
if "last_recs" not in st.session_state:
    st.session_state.last_recs = None

def _convert_relative_dates_to_ranges(perfil: Dict) -> Dict:
    """
    Convierte fechas relativas como "mañana", "pasado mañana", "hoy", "este sábado" a fecha_rango con fechas reales.
    Esto asegura que el perfil siempre tenga fechas concretas para filtrado.
    """
    p = dict(perfil)
    
    # Si tenemos fecha relativa, convertirla a fecha_rango
    fecha_str = p.get("fecha", "").strip().lower()
    
    if fecha_str and not p.get("fecha_rango"):
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

def merge_profiles(base: Dict, delta: Dict) -> Dict:
    p = dict(base or {})

    # smalltalk
    stalk = (delta.get("smalltalk") or "").strip()
    if stalk:
        p["smalltalk"] = stalk

    # fecha / rango
    dr = delta.get("fecha_rango")
    if isinstance(dr, dict) and (dr.get("start") and dr.get("end")):
        p["fecha_rango"] = _normalize_future_like_range_if_past(dr)
        p["fecha"] = ""
    else:
        # Handle fecha - update if provided
        if delta.get("fecha"):
            p["fecha"] = str(delta["fecha"])
        
        # Convert relative dates (mañana, pasado mañana, hoy) to fecha_rango
        p = _convert_relative_dates_to_ranges(p)

    # categorías
    if delta.get("categorias"):
        cats_delta = [str(c).strip().lower() for c in delta["categorias"] if str(c).strip()]
        p["categorias"] = sorted(set((p.get("categorias") or [])) | set(cats_delta))
    
    # keywords - merge lists, keep unique (handle None/empty cases)
    # Ensure keywords exists as empty list if not present
    if "keywords" not in p:
        p["keywords"] = []
    
    delta_keywords = delta.get("keywords")
    if delta_keywords:
        if isinstance(delta_keywords, list) and len(delta_keywords) > 0:
            kw_delta = [str(kw).strip().lower() for kw in delta_keywords if str(kw).strip()]
            existing_kw = p.get("keywords") or []
            if not isinstance(existing_kw, list):
                existing_kw = []
            p["keywords"] = list(set(existing_kw) | set(kw_delta))

    # gratis/pago
    if not p.get("es_gratis") and delta.get("es_gratis"):
        p["es_gratis"] = str(delta["es_gratis"]).strip().lower()

    # presupuesto
    if p.get("es_gratis") == "pago" and delta.get("precio_max_cop") not in (None, "", []):
        try:
            p["precio_max_cop"] = int(float(delta["precio_max_cop"]))
        except Exception:
            pass

    # cercanía
    if not p.get("dist_importa") and delta.get("dist_importa"):
        p["dist_importa"] = str(delta["dist_importa"]).strip().lower()

    # parte del día (opcional) - can be list or string "indiferente"
    if not p.get("parte_del_dia"):
        delta_pdia = delta.get("parte_del_dia")
        if isinstance(delta_pdia, list):
            # Normalize list elements
            normalized = [str(x).strip().lower() for x in delta_pdia if str(x).strip().lower() in {"mañana", "tarde", "noche"}]
            if normalized:
                p["parte_del_dia"] = normalized
        elif str(delta_pdia).strip().lower() == "indiferente":
            p["parte_del_dia"] = "indiferente"

    # edad / política
    # Only update edad_usuario if delta has an actual number (not None)
    if delta.get("edad_usuario") not in (None, "", []):
        try:
            p["edad_usuario"] = int(delta["edad_usuario"])
        except Exception:
            pass
    # Update excluir_restriccion_edad if we don't have it yet or if delta provides one
    delta_excl = str(delta.get("excluir_restriccion_edad") or "").strip().lower()
    if delta_excl in {"si", "no", "indiferente"} and not p.get("excluir_restriccion_edad"):
        p["excluir_restriccion_edad"] = delta_excl

    return p

def is_profile_complete(perfil: Dict) -> bool:
    has_range = isinstance(perfil.get("fecha_rango"), dict) and perfil["fecha_rango"].get("start") and perfil["fecha_rango"].get("end")
    has_fecha = bool(perfil.get("fecha"))
    if not (has_range or has_fecha):
        return False
    if not perfil.get("categorias"):
        return False
    eg = (perfil.get("es_gratis") or "").lower()
    if eg not in {"gratis", "pago", "indiferente"}:
        return False
    if eg == "pago" and perfil.get("precio_max_cop") in (None, "", []):
        return False
    di = (perfil.get("dist_importa") or "").lower()
    if di not in {"si", "no"}:
        return False
    if di == "si" and (st.session_state.user_lat is None or st.session_state.user_lon is None):
        return False
    # Age: we need either an age OR a policy. "indiferente" or "no" means don't filter by age.
    has_age = perfil.get("edad_usuario") is not None
    has_policy = (perfil.get("excluir_restriccion_edad") or "").lower() in {"si", "no", "indiferente"}
    if not (has_age or has_policy):
        return False
    return True

def handle_user_message(text: str) -> None:
    user_msg = (text or "").strip()
    if not user_msg:
        return
    st.session_state.chat.append(("user", user_msg))
    # Gemini now handles complete conversational flow
    gemini_response = gemini_normalize(user_msg, st.session_state.perfil)
    st.session_state.perfil = merge_profiles(st.session_state.perfil, gemini_response)
    
    # Ensure relative dates are converted to actual date ranges
    st.session_state.perfil = _convert_relative_dates_to_ranges(st.session_state.perfil)
    
    # Gemini tells us if profile is complete and what to ask next
    profile_complete = gemini_response.get("profile_complete", False)
    next_question = gemini_response.get("next_question", "").strip()
    smalltalk = gemini_response.get("smalltalk", "").strip()
    
    # Build bot reply: smalltalk + next question
    if profile_complete:
        bot_text = smalltalk or "¡Perfecto! Ya tengo toda la información que necesito para recomendarte."
        st.session_state.ready = True
    else:
        # Combine smalltalk with question naturally
        if smalltalk and next_question:
            bot_text = f"{smalltalk} {next_question}"
        elif next_question:
            bot_text = next_question
        else:
            bot_text = smalltalk or "Entiendo, continuemos."
        st.session_state.ready = False
    
    st.session_state.chat.append(("bot", bot_text))

# 6) UI — Chat, perfil, mapa condicional y resultados
st.title("Planorama 🎟️ — Recomendador de planes en Bogotá")
st.caption(f"Gemini conectado: {'sí' if GEMINI_OK else 'no'}")

with st.sidebar:
    st.header("⚙️ Opciones")
    if st.button("🧹 Nueva búsqueda"):
        st.session_state.perfil = {
            "smalltalk": "",
            "fecha": "",
            "fecha_rango": None,
            "categorias": [],
            "keywords": [],
            "es_gratis": "",
            "precio_max_cop": None,
            "dist_importa": "",
            "parte_del_dia": "",  # Empty until user specifies (mañana/tarde/noche/indiferente)
            "edad_usuario": None,
            "excluir_restriccion_edad": "",  # Empty until user specifies
        }
        st.session_state.chat = [
            ("bot", "¡Hola! 👋 Soy Planorama, tu asistente para encontrar planes en Bogotá. Cuéntame qué tipo de evento buscas y te ayudo a encontrar las mejores opciones. 😊")
        ]
        st.session_state.user_lat = None
        st.session_state.user_lon = None
        st.session_state.ready = False
        st.session_state.last_recs = None
        st.rerun()

left_col, right_col = st.columns([1.0, 1.2], gap="large")

with left_col:
    st.subheader("💬 Conversación")
    st.caption("Habla como gustes. Yo me encargo de entenderte y normalizar la info. 😉")

    if st.session_state.chat:
        for who, msg in st.session_state.chat:
            if who == "user":
                st.markdown(f"**Tú:** {msg}")
            else:
                st.markdown(f"**Planorama:** {msg}")

    with st.form("chat_form", clear_on_submit=True):
        user_text = st.text_input("Escribe aquí…", placeholder="Ej.: Mañana quiero algo de comedia cerca de mí")
        sent = st.form_submit_button("Enviar")
    if sent:
        handle_user_message(user_text)
        st.rerun()

with right_col:
    def _fecha_o_rango_display(perfil: dict) -> str:
        """Devuelve texto legible de la fecha/rango, aplicando normalización si es un rango pasado."""
        fr = perfil.get("fecha_rango")
        if isinstance(fr, dict) and fr.get("start") and fr.get("end"):
            fr2 = _normalize_future_like_range_if_past(fr)
            return f"{fr2['start']} → {fr2['end']}"
        f = (perfil.get("fecha") or "").strip()
        if f:
            dr = parse_date_pref(f)  # usa la lógica que ya tienes en geo_utils
            try:
                return dr.start.strftime("%Y-%m-%d")  # fecha concreta resuelta
            except Exception:
                return str(dr.start)
        return "—"

    st.subheader("🧾 Tu perfil (solo para ti)")
    perfil = st.session_state.perfil

    with st.container(border=True):
        fecha_txt = _fecha_o_rango_display(perfil)
        if (perfil.get("fecha") or "").strip():
            dr = parse_date_pref(perfil["fecha"])
            st.session_state["_fecha_resuelta"] = {
                "start": dr.start.date().isoformat(),
                "end": dr.end.date().isoformat()
            }
        elif isinstance(perfil.get("fecha_rango"), dict):
            st.session_state["_fecha_resuelta"] = {
                "start": perfil["fecha_rango"].get("start"),
                "end": perfil["fecha_rango"].get("end"),
            }

        cats_txt = ", ".join(perfil.get("categorias") or []) or "—"
        eg = (perfil.get("es_gratis") or "—").lower()
        presu = perfil.get("precio_max_cop")
        presu_txt = f"${int(presu):,}".replace(",", ".") if presu not in (None, "", []) else "—"
        cerc = (perfil.get("dist_importa") or "—").lower()
        edad = perfil.get("edad_usuario")
        edad_txt = f"{edad} años" if edad is not None else "no especificada"
        restr_raw = (perfil.get("excluir_restriccion_edad") or "—").lower()
        restr = restr_raw if restr_raw in {"si", "no", "indiferente"} else "—"
        # parte_del_dia can be list or string
        pdia_raw = perfil.get("parte_del_dia")
        if isinstance(pdia_raw, list):
            pdia = ", ".join(pdia_raw) if pdia_raw else "—"
        else:
            pdia = pdia_raw if pdia_raw else "—"

        st.markdown(
            f"- **Fecha / Rango**: {fecha_txt}\n"
            f"- **Categorías**: {cats_txt}\n"
            f"- **Gratis/Pago**: {eg} · **Presupuesto máx**: {presu_txt}\n"
            f"- **Cercanía importa**: {cerc}\n"
            f"- **Edad**: {edad_txt} · **Excluir por restricción**: {restr}\n"
            f"- **Parte del día**: {pdia}"
        )


    # Depuración: ver por dónde se caen los eventos
    with st.expander("🔧 Depuración (filtros paso a paso)"):
        try:
            # Copiamos df original
            _d0 = df.copy()

            # 0) ciudad
            _d1 = _d0[_d0["city_norm"].str.contains("bogota", na=False)] if "city_norm" in _d0.columns else _d0
            # 1) estado
            if "status" in _d1.columns:
                ok_states = {"scheduled","activo","active","programado",""}
                _d1 = _d1[_d1["status"].str.lower().isin(ok_states)]
            # 2) fecha o rango
            def _apply_range(df_in, start, end):
                return df_in[df_in["date_start_parsed"].apply(lambda ts: isinstance(ts, pd.Timestamp) and (ts>=start) and (ts<end))]
            _d2 = _d1.copy()
            if isinstance(perfil.get("fecha_rango"), dict) and perfil["fecha_rango"].get("start") and perfil["fecha_rango"].get("end"):
                start = pd.to_datetime(perfil["fecha_rango"]["start"], errors="coerce")
                end   = pd.to_datetime(perfil["fecha_rango"]["end"], errors="coerce")
                if pd.notna(start) and pd.notna(end):
                    _d2 = _apply_range(_d2, start, end)
            elif perfil.get("fecha"):
                dr = parse_date_pref(perfil["fecha"])
                _d2 = _apply_range(_d2, dr.start, dr.end)

            # 3) categoría flexible (igual que en compute_recommendations)
            _d3 = _d2.copy()
            cats = [c.lower() for c in (perfil.get("categorias") or [])]
            if cats and "category" in _d3.columns:
                if "text_blob_norm" not in _d3.columns:
                    _d3["text_blob_norm"] = _d3["text_blob"].apply(_normtxt)
                wanted = _expand_cats(cats)
                def _cat_ok(row) -> bool:
                    cat_ok  = _normtxt(row.get("category","")) in wanted
                    blob_ok = any(w in row.get("text_blob_norm","") for w in wanted)
                    return cat_ok or blob_ok
                _d3 = _d3[_d3.apply(_cat_ok, axis=1)]

            # 4) gratis/pago + presupuesto
            _d4 = _d3.copy()
            eg = (perfil.get("es_gratis") or "indiferente").lower()
            if eg == "gratis" and "is_free" in _d4.columns:
                _d4 = _d4[_d4["is_free"] == True]
            elif eg == "pago":
                if "is_free" in _d4.columns:
                    _d4 = _d4[_d4["is_free"] == False]
                if perfil.get("precio_max_cop") is not None:
                    budget = float(perfil["precio_max_cop"])
                    _d4 = _d4[pd.to_numeric(_d4["price_min_cop"], errors="coerce").fillna(np.inf) <= budget]

            # 5) edad/restricción (solo si el usuario pidió excluir)
            _d5 = _d4.copy()
            edad_usuario = perfil.get("edad_usuario")
            excluir = (perfil.get("excluir_restriccion_edad") or "").lower()
            # Only apply filter if user wants to exclude age-restricted events
            if excluir == "si" and edad_usuario is not None:
                _d5 = _d5[(_d5["age_min_num"].isna()) | (_d5["age_min_num"] <= int(edad_usuario))]
            # If "no" or "indiferente", _d5 stays as _d4

            st.write(f"Total CSV: **{len(_d0)}**")
            st.write(f"Tras CIUDAD/ESTADO: **{len(_d1)}** (quedan)")
            st.write(f"Tras FECHA: **{len(_d2)}** (quedan)")
            st.write(f"Tras CATEGORÍA: **{len(_d3)}** (quedan)")
            st.write(f"Tras PRECIO: **{len(_d4)}** (quedan)")
            st.write(f"Tras EDAD: **{len(_d5)}** (quedan)")
            
            # Show coordinate samples
            if "lat" in _d5.columns and "lon" in _d5.columns:
                st.write("\n**Muestra de coordenadas (primeras 3 filas):**")
                coord_sample = _d5[["title", "lat", "lon"]].head(3)
                st.dataframe(coord_sample)
                
                # If user has location and dist_importa, show distance info
                if (perfil.get("dist_importa") or "").lower() == "si" and st.session_state.user_lat and st.session_state.user_lon:
                    st.write(f"\n**Tu ubicación:** ({st.session_state.user_lat:.4f}, {st.session_state.user_lon:.4f})")
                    st.write("**Las distancias se calcularán con Haversine desde tu ubicación.**")
        except Exception as e:
            st.warning(f"No pude ejecutar el debug: {e}")

    if (perfil.get("dist_importa") or "").lower() == "si":
        st.subheader("📍 Marca tu zona")
        st.caption("Haz clic en el mapa para fijar tu ubicación aproximada. La usaremos solo para calcular distancias.")
        map_data = render_location_map(
            user_lat=st.session_state.user_lat,
            user_lon=st.session_state.user_lon
        )
        if map_data and map_data.get("last_clicked"):
            st.session_state.user_lat = map_data["last_clicked"]["lat"]
            st.session_state.user_lon = map_data["last_clicked"]["lng"]
        has_point = (st.session_state.user_lat is not None and st.session_state.user_lon is not None)
        if not has_point:
            st.error("Para continuar con cercanía, **haz clic en el mapa** y marca tu zona.")
        else:
            st.success("Zona marcada. Usaré tu ubicación para priorizar la cercanía.")

    st.subheader("🎯 Resultados")
    ready = st.session_state.ready

    if not ready:
        st.info("Aún estoy reuniendo tus preferencias. Sigue respondiendo en el chat.")
    else:
        need_point = (perfil.get("dist_importa") or "").lower() == "si"
        has_point = (st.session_state.user_lat is not None and st.session_state.user_lon is not None)

        if need_point and not has_point:
            st.warning("Marcaste **cercanía = sí**. Falta que **marques tu zona en el mapa** para calcular distancias.")
        else:
            # Este bloque se ejecuta si el perfil está listo 
            with st.spinner("Buscando los mejores planes para ti..."):
                df_rank = compute_recommendations(
                    perfil=perfil,
                    df_events=df,
                    vectorizer=vectorizer,
                    Xmatrix=Xmatrix,
                    IDS=IDS,
                    user_lat=st.session_state.user_lat,
                    user_lon=st.session_state.user_lon,
                )
                st.session_state.last_recs = df_rank
                render_results(df_rank, perfil)

#744
