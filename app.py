
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
from sklearn.metrics.pairwise import cosine_similarity

from streamlit_folium import st_folium
import folium

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
from llm_interviewer import process_turn


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
    return {
        "smalltalk": d.get("smalltalk", ""),
        "fecha": d.get("fecha", ""),
        "fecha_rango": d.get("fecha_rango") if isinstance(d.get("fecha_rango"), dict) else None,
        "categorias": d.get("categorias", []) or [],
        "es_gratis": d.get("es_gratis", ""),
        "precio_max_cop": d.get("precio_max_cop", None),
        "dist_importa": d.get("dist_importa", ""),
        "parte_del_dia": d.get("parte_del_dia", ""),
        "edad_usuario": d.get("edad_usuario", None),
        "excluir_restriccion_edad": d.get("excluir_restriccion_edad", ""),
    }

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
    '  "parte_del_dia": "mañana|tarde|noche|",\n'
    '  "edad_usuario": 21,\n'
    '  "excluir_restriccion_edad": "si|no"\n'
    "}\n"
    "Reglas de normalización (OBLIGATORIAS):\n"
    "- Mapea TODO lo relacionado con música en vivo a 'concierto': musica|música|musica en vivo|festival|show|live|rock|pop|salsa|jazz|reggaeton|trap → concierto.\n"
    "- Mapea stand up|stand-up|comedia a 'teatro'.\n"
    "- Mapea feria|expo|exposición|taller|workshop|tour|recorrido|experiencia inmersiva a 'experiencia'.\n"
    "- Permite múltiples categorías si el usuario las menciona, pero NUNCA inventes etiquetas fuera de {concierto, teatro, experiencia}.\n"
    "- Si el usuario usa palabras fuera del set (ej. 'música'), normalízalas al set (ej. 'concierto').\n"
    "- Si el usuario menciona un RANGO (ej. 'la próxima semana', 'entre el 10 y 12'), usa 'fecha_rango' con start/end en YYYY-MM-DD.\n"
    "- Si da un día específico (hoy, mañana o una fecha), usa 'fecha' y deja 'fecha_rango' vacío.\n"
    "- 'precio_max_cop' debe ser entero en COP (ej. '120k'→120000; '$150.000'→150000).\n"
    "- 'smalltalk' DEBE ser una frase cálida SIN preguntas (ej.: '¡Listo! Tomo nota.') y SIEMPRE debe venir.\n"
    "- NO incluyas comentarios, ni backticks, ni bloques de código; SOLO JSON estricto.\n"
    "\n"
    "Ejemplo 1 (usuario): \"quiero algo de música en vivo, tal vez festival\"\n"
    "Salida JSON: {\"smalltalk\":\"¡Genial! Lo tengo.\",\"fecha\":\"\",\"fecha_rango\":null,"
    "\"categorias\":[\"concierto\"],\"es_gratis\":\"\",\"precio_max_cop\":null,\"dist_importa\":\"\","
    "\"parte_del_dia\":\"\",\"edad_usuario\":null,\"excluir_restriccion_edad\":\"\"}\n"
    "\n"
    "Ejemplo 2 (usuario): \"stand up o comedia barata este finde\"\n"
    "Salida JSON: {\"smalltalk\":\"¡De una!\",\"fecha\":\"fin_de_semana\",\"fecha_rango\":null,"
    "\"categorias\":[\"teatro\"],\"es_gratis\":\"pago\",\"precio_max_cop\":null,\"dist_importa\":\"\","
    "\"parte_del_dia\":\"\",\"edad_usuario\":null,\"excluir_restriccion_edad\":\"\"}\n"
)

def gemini_normalize(user_text: str, current_profile: Dict) -> dict:
    """
    Llama a Gemini para normalizar el turno, dándole contexto del perfil actual.
    Devuelve SIEMPRE el contrato con defaults si algo falta.
    """
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
    except Exception:
        return _ensure_schema({"smalltalk": "¡Perfecto! Continuemos. 😊"})

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
        "es_gratis": "",
        "precio_max_cop": None,
        "dist_importa": "",
        "parte_del_dia": "",
        "edad_usuario": None,
        "excluir_restriccion_edad": "",
    }
if "chat" not in st.session_state:
    st.session_state.chat = []
if "user_lat" not in st.session_state:
    st.session_state.user_lat = None
if "user_lon" not in st.session_state:
    st.session_state.user_lon = None
if "ready" not in st.session_state:
    st.session_state.ready = False
if "last_recs" not in st.session_state:
    st.session_state.last_recs = None

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
        if not p.get("fecha") and delta.get("fecha"):
            p["fecha"] = str(delta["fecha"])

    # categorías
    if delta.get("categorias"):
        cats_delta = [str(c).strip().lower() for c in delta["categorias"] if str(c).strip()]
        p["categorias"] = sorted(set((p.get("categorias") or [])) | set(cats_delta))

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

    # parte del día (opcional)
    if not p.get("parte_del_dia") and delta.get("parte_del_dia"):
        p["parte_del_dia"] = str(delta["parte_del_dia"]).strip().lower()

    # edad / política
    if delta.get("edad_usuario") not in (None, "", []):
        try:
            p["edad_usuario"] = int(delta["edad_usuario"])
        except Exception:
            pass
    if not p.get("excluir_restriccion_edad") and delta.get("excluir_restriccion_edad"):
        p["excluir_restriccion_edad"] = str(delta["excluir_restriccion_edad"]).strip().lower()

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
    has_age = perfil.get("edad_usuario") is not None
    has_policy = (perfil.get("excluir_restriccion_edad") or "").lower() in {"si", "no"}
    if not (has_age or has_policy):
        return False
    return True

def handle_user_message(text: str) -> None:
    user_msg = (text or "").strip()
    if not user_msg:
        return
    st.session_state.chat.append(("user", user_msg))
    delta = gemini_normalize(user_msg, st.session_state.perfil)
    st.session_state.perfil = merge_profiles(st.session_state.perfil, delta)
    reply_text, _, done = process_turn("", st.session_state.perfil)
    smalltalk = st.session_state.perfil.get("smalltalk", "").strip()
    if done:
        bot_text = smalltalk or "¡Perfecto! Con esa info ya puedo recomendarte."
        st.session_state.chat.append(("bot", bot_text))
        st.session_state.ready = True
    else:
        ask = reply_text.strip()
        bot_text = f"{smalltalk} {ask}" if smalltalk else ask
        st.session_state.chat.append(("bot", bot_text))
        st.session_state.ready = False

# 7–8) Recomendaciones (filtros + scoring) y render
W_CONT  = 0.50
W_PREC  = 0.20
W_DIST  = 0.25
W_FECHA = 0.05
TOP_N_DEFAULT = 10

def _in_date_range(ts: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if not isinstance(ts, pd.Timestamp) or pd.isna(ts): return False
    t = ts.to_pydatetime().replace(tzinfo=None)
    return (t >= start.replace(tzinfo=None)) and (t < end.replace(tzinfo=None))

def _part_of_day_bonus(hour_start: float, desired: str) -> float:
    if not desired or pd.isna(hour_start): return 0.0
    h = float(hour_start)
    if desired == "mañana": return 1.0 if h < 12 else 0.0
    if desired == "tarde":  return 1.0 if (13 <= h <= 18) else 0.0
    if desired == "noche":  return 1.0 if h > 18 else 0.0
    return 0.0
# --- helpers para matching flexible de categorías ---
def _normtxt(x: str) -> str:
    import unicodedata as _ud
    s = str(x or "")
    s = _ud.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    return s.lower().strip()

_CAT_SYNONYMS = {
    "concierto": {"concierto","musica","musica en vivo","musical","show","festival","live",
                  "reggaeton","trap","rock","pop","salsa","jazz"},
    "teatro": {"teatro","obra","drama","comedia","stand up","stand-up"},
    "experiencia": {"experiencia","feria","expo","exposicion","exposición","taller","workshop","tour","recorrido"},
}
def _expand_cats(cats: list[str]) -> set[str]:
    out = set()
    for c in cats:
        key = _normtxt(c)
        out.add(key)
        out |= _CAT_SYNONYMS.get(key, set())
    return out

def compute_recommendations(
    perfil: Dict,
    df_events: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    Xmatrix,
    IDS: list,
    user_lat: Optional[float],
    user_lon: Optional[float],
    w_cont: float = W_CONT,
    w_prec: float = W_PREC,
    w_dist: float = W_DIST,
    w_fecha: float = W_FECHA,
    tau_km: float = TAU_DEFAULT,
    R_km: float = R_DEFAULT,
) -> pd.DataFrame:

    df_filt = df_events.copy()

    # Ciudad: Bogotá
    if "city_norm" in df_filt.columns:
        df_filt = df_filt[df_filt["city_norm"].str.contains("bogota", na=False)]

    # Estado activo
    if "status" in df_filt.columns:
        ok_states = {"scheduled", "activo", "active", "programado", ""}
        df_filt = df_filt[df_filt["status"].str.lower().isin(ok_states)]

    # Futuro 
    has_user_date = bool(perfil.get("fecha")) or (
        isinstance(perfil.get("fecha_rango"), dict)
        and perfil["fecha_rango"].get("start")
        and perfil["fecha_rango"].get("end")
    )
    if "is_future" in df_filt.columns and not has_user_date:
        df_filt = df_filt[df_filt["is_future"] == True]

    # Fecha / Rango
    if isinstance(perfil.get("fecha_rango"), dict) and perfil["fecha_rango"].get("start") and perfil["fecha_rango"].get("end"):
        start = pd.to_datetime(perfil["fecha_rango"]["start"], errors="coerce")
        end   = pd.to_datetime(perfil["fecha_rango"]["end"], errors="coerce")
        if pd.notna(start) and pd.notna(end):
            df_filt = df_filt[df_filt["date_start_parsed"].apply(lambda ts: _in_date_range(ts, start, end))]
    else:
        pref_fecha = perfil.get("fecha") or ""
        if pref_fecha:
            dr = parse_date_pref(pref_fecha)
            df_filt = df_filt[df_filt["date_start_parsed"].apply(lambda ts: _in_date_range(ts, dr.start, dr.end))]

    # --- Categorías (filtro flexible) ---
    cats = [c.lower() for c in (perfil.get("categorias") or [])]
    if cats and "category" in df_filt.columns:
        if "text_blob_norm" not in df_filt.columns:
            df_filt["text_blob_norm"] = df_filt["text_blob"].apply(_normtxt)
        wanted = _expand_cats(cats)
        def _cat_ok(row) -> bool:
            cat_ok  = _normtxt(row.get("category","")) in wanted
            blob_ok = any(w in row.get("text_blob_norm","") for w in wanted)
            return cat_ok or blob_ok
        df_filt = df_filt[df_filt.apply(_cat_ok, axis=1)]

    # Gratis / pago + presupuesto
    eg = (perfil.get("es_gratis") or "indiferente").lower()
    if eg == "gratis":
        if "is_free" in df_filt.columns:
            df_filt = df_filt[df_filt["is_free"] == True]
    elif eg == "pago":
        if "is_free" in df_filt.columns:
            df_filt = df_filt[df_filt["is_free"] == False]
        budget = perfil.get("precio_max_cop")
        if budget is not None:
            df_filt = df_filt[pd.to_numeric(df_filt["price_min_cop"], errors="coerce").fillna(np.inf) <= float(budget)]

    # Edad / restricción
    if "age_min_num" not in df_filt.columns:
        def _age_to_int(s: str):
            s = (s or "").strip().lower()
            if s in {"", "todas", "toda", "all"}: return 0
            import re as _re
            m = _re.search(r"(\d+)", s)
            return int(m.group(1)) if m else np.nan
        df_filt["age_min_num"] = df_filt["age_min"].apply(_age_to_int)

    edad_usuario = perfil.get("edad_usuario")
    excluir_restriccion = (perfil.get("excluir_restriccion_edad") or "").lower()
    if excluir_restriccion == "si" and edad_usuario is not None:
        df_filt = df_filt[(df_filt["age_min_num"].isna()) | (df_filt["age_min_num"] <= int(edad_usuario))]

    if df_filt.empty:
        return df_filt

    # 1) Similitud de contenido
    if cats:
        qv = vectorizer.transform([" ".join(cats)])
        sims_full = cosine_similarity(qv, Xmatrix).ravel()
        sim_map = {id_: float(s) for id_, s in zip(IDS, sims_full)}
        df_filt["sim_contenido"] = df_filt["uid"].map(sim_map).fillna(0.0)
    else:
        df_filt["sim_contenido"] = 0.0

    # 2) Precio
    def price_score(row) -> float:
        price_min = pd.to_numeric(row.get("price_min_cop"), errors="coerce")
        if pd.isna(price_min):
            return 0.6
        if eg == "gratis":
            return 1.0 if (row.get("is_free") is True) else 0.2
        if eg == "pago":
            if perfil.get("precio_max_cop") is None:
                return 0.7
            budget = float(perfil["precio_max_cop"])
            if price_min <= budget:
                return 1.0
            ratio = min(1.5, price_min / max(1.0, budget))
            return max(0.0, 1.0 - (ratio - 1.0))
        return 0.85 if (row.get("is_free") is True) else 0.7

    df_filt["score_precio"] = df_filt.apply(price_score, axis=1)

    # 3) Parte del día (bonus)
    desired_part = perfil.get("parte_del_dia") or ""
    df_filt["score_fecha"] = df_filt["hour_start"].apply(lambda h: _part_of_day_bonus(h, desired_part))

    # 4) Distancia (Haversine + exponencial)
    dist_importa = (perfil.get("dist_importa") or "").lower() == "si"
    has_user_point = (user_lat is not None and user_lon is not None)

    def dist_component(row) -> Tuple[float, Optional[float]]:
        if not dist_importa or not has_user_point:
            return 0.0, None
        ev_lat = pd.to_numeric(row.get("lat"), errors="coerce")
        ev_lon = pd.to_numeric(row.get("lon"), errors="coerce")
        if pd.isna(ev_lat) or pd.isna(ev_lon):
            return 0.0, None
        res = compute_distance_score(
            dist_importa=True,
            user_lat=float(user_lat),
            user_lon=float(user_lon),
            event_lat=float(ev_lat),
            event_lon=float(ev_lon),
            mode="exp",
            tau=float(TAU_DEFAULT),
            R=float(R_DEFAULT),
        )
        return (res.score if res.ok else 0.0), (res.dist_km if res.ok else None)

    scores_dist, dists_km = [], []
    for _, r in df_filt.iterrows():
        s, dkm = dist_component(r)
        scores_dist.append(s)
        dists_km.append(dkm)
    df_filt["score_dist"] = scores_dist
    df_filt["dist_km"] = dists_km

    w_dist_eff = W_DIST if dist_importa else 0.0

    # 5) Score final y ranking
    df_filt["score_final"] = (
        W_CONT  * df_filt["sim_contenido"].fillna(0.0) +
        W_PREC  * df_filt["score_precio"].fillna(0.0) +
        w_dist_eff * df_filt["score_dist"].fillna(0.0) +
        W_FECHA * df_filt["score_fecha"].fillna(0.0)
    )

    df_rank = (
        df_filt
        .drop_duplicates(subset=["title", "date_start_parsed"], keep="first")
        .sort_values(["score_final", "date_start_parsed"], ascending=[False, True])
    )

    # Top-N dinámico: si hay pocos, muestra 2–3; si no, hasta 10
    if len(df_rank) <= 3:
        top_n = len(df_rank)
    elif len(df_rank) <= TOP_N_DEFAULT:
        top_n = len(df_rank)
    else:
        top_n = TOP_N_DEFAULT

    return df_rank.head(top_n)

def render_results(df_rank: pd.DataFrame, perfil: Dict) -> None:
    if df_rank.empty:
        st.warning("No encontré resultados con tus preferencias. Prueba ampliar categorías, fecha o presupuesto.")
        return

    dist_importa = (perfil.get("dist_importa") or "").lower() == "si"

    for _, r in df_rank.iterrows():
        with st.container(border=True):
            title = r.get("title", "(sin título)") or "(sin título)"
            ts = r.get("date_start_parsed", "")
            date_txt = str(ts) if pd.notna(ts) else "—"
            zona = ", ".join([str(r.get("barrio", "") or ""), str(r.get("localidad", "") or "")]).strip(", ").strip() or "—"

            is_free = bool(r.get("is_free") is True)
            price = r.get("price_min_cop", None)
            if is_free:
                price_txt = "Gratis"
            elif pd.notna(price):
                try:
                    price_txt = f"{int(float(price)):,} COP".replace(",", ".")
                except Exception:
                    price_txt = f"{price} COP"
            else:
                price_txt = "N/D"

            dist_txt = ""
            if dist_importa and r.get("dist_km") is not None:
                try:
                    dist_txt = f" · 📏 ~{float(r['dist_km']):.1f} km de tu zona"
                except Exception:
                    dist_txt = ""

            st.markdown(f"### {title}")
            st.markdown(f"**Fecha:** {date_txt} · **Zona:** {zona} · **Desde:** {price_txt}{dist_txt}")

            img = r.get("image_url", "")
            if isinstance(img, str) and img.startswith("http"):
                st.image(img, use_column_width=True)

            url = r.get("organizer_url") or r.get("source_url") or ""
            if isinstance(url, str) and url.startswith("http"):
                st.link_button("Ver más / Comprar", url, use_container_width=True)

    with st.expander("Ver tabla (detalles)"):
        show_cols = [
            "title","category","date_start_parsed","barrio","localidad",
            "price_min_cop","is_free","age_min","lat","lon",
            "sim_contenido","score_precio","score_dist","score_fecha","score_final","dist_km"
        ]
        show_cols = [c for c in show_cols if c in df_rank.columns]
        st.dataframe(df_rank[show_cols])

# 6) UI — Chat, perfil, mapa condicional y resultados (después de definir 7–8)
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
            "es_gratis": "",
            "precio_max_cop": None,
            "dist_importa": "",
            "parte_del_dia": "",
            "edad_usuario": None,
            "excluir_restriccion_edad": "",
        }
        st.session_state.chat = []
        st.session_state.user_lat = None
        st.session_state.user_lon = None
        st.session_state.ready = False
        st.session_state.last_recs = None
        st.experimental_rerun()

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
        edad_txt = f"{edad} años" if edad is not None else "no indicada"
        restr = (perfil.get("excluir_restriccion_edad") or "—").lower()
        pdia = perfil.get("parte_del_dia") or "—"

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
            if excluir == "si" and edad_usuario is not None:
                _d5 = _d5[(_d5["age_min_num"].isna()) | (_d5["age_min_num"] <= int(edad_usuario))]

            st.write(f"Total CSV: **{len(_d0)}**")
            st.write(f"Tras CIUDAD/ESTADO: **{len(_d1)}** (quedan)")
            st.write(f"Tras FECHA: **{len(_d2)}** (quedan)")
            st.write(f"Tras CATEGORÍA: **{len(_d3)}** (quedan)")
            st.write(f"Tras PRECIO: **{len(_d4)}** (quedan)")
            st.write(f"Tras EDAD: **{len(_d5)}** (quedan)")
        except Exception as e:
            st.warning(f"No pude ejecutar el debug: {e}")

    if (perfil.get("dist_importa") or "").lower() == "si":
        st.subheader("📍 Marca tu zona")
        st.caption("Haz clic en el mapa para fijar tu ubicación aproximada. La usaremos solo para calcular distancias.")
        center_lat, center_lon = 4.6486, -74.0649
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)
        if st.session_state.user_lat is not None and st.session_state.user_lon is not None:
            folium.Marker([st.session_state.user_lat, st.session_state.user_lon], tooltip="Tu zona").add_to(m)
        map_data = st_folium(m, width=900, height=420)
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
