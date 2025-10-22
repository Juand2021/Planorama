# app.py — Planorama MVP en un solo archivo (simple y funcional)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os, sys, streamlit as st

# (opcional) si alguna vez Streamlit cambia el cwd, nos aseguramos de que la raíz esté en el path
# sys.path.append(os.path.dirname(__file__))

# Inyecta la API key desde .streamlit/secrets.toml (si ya lo tienes, deja tu versión)
if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"]:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# IMPORT DEL ENTREVISTADOR IA con fallback para que no rompa la app
try:
    from src.llm_interviewer import llm_next_turn
    LLM_OK = True
except Exception as e:
    LLM_OK = False
    def llm_next_turn(perfil, historial):
        # fallback: no rompe la app si falla el import
        return f"(IA no disponible: {e}). Comparte categoría, fecha, parte del día, presupuesto y zona.", perfil

st.set_page_config(page_title="Planorama", page_icon="🎟️", layout="wide")
st.title("Planorama 🎟️ — MVP")

# --------- 1) Carga de datos (CSV) con detección de encoding/sep y tolerancia ----------
@st.cache_data(show_spinner=False)
def load_events(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    seps = [None, ",", ";", "\t", "|"]  # None => autodetección (engine='python')
    last_err = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    sep=sep,
                    engine="python",            # tolerante con comillas/saltos
                    quoting=csv.QUOTE_MINIMAL,
                    on_bad_lines="skip"         # filas dañadas se omiten
                )
                # Heurística mínima: varias columnas y alguna conocida
                if df.shape[1] >= 5 and any(c in df.columns for c in ["title", "category", "date_start"]):
                    # Normalizaciones básicas
                    for c in ["title","description","tags","city","barrio","localidad","category","status","booking_url","source_url"]:
                        if c in df.columns:
                            df[c] = df[c].fillna("").astype(str).str.strip()

                    # Fechas/horas
                    if "date_start" in df.columns:
                        df["date_start_parsed"] = pd.to_datetime(df["date_start"], errors="coerce")
                    else:
                        df["date_start_parsed"] = pd.NaT

                    if "time_start" in df.columns:
                        df["hour_start"] = pd.to_datetime(df["time_start"], errors="coerce").dt.hour
                    else:
                        df["hour_start"] = np.nan

                    # Futuro
                    df["is_future"] = df["date_start_parsed"] >= pd.Timestamp.now().normalize()

                    # Texto combinado para similitud
                    def blob(r):
                        parts = []
                        for col in ["title","description","tags","category","barrio","localidad"]:
                            if col in r and pd.notna(r[col]) and r[col]:
                                parts.append(str(r[col]))
                        return " ".join(parts)
                    df["text_blob"] = df.apply(blob, axis=1)

                    # ID
                    if "event_id" in df.columns:
                        df["event_id"] = df["event_id"].astype(str)
                    else:
                        df["event_id"] = df.index.astype(str)

                    return df
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"No pude leer el CSV (encodings/separadores comunes). Último error: {last_err}")

# ⚠️ Usa slash normal para evitar '\p' en Windows
df = load_events("data/Planorama_BD.csv")

# --------- 2) Índice TF-IDF ----------
@st.cache_resource(show_spinner=False)
def build_index(texts: pd.Series, ids: list[str]):
    vec = TfidfVectorizer(min_df=2, max_df=0.9)
    X = vec.fit_transform(texts.fillna(""))
    return vec, X, ids

vectorizer, Xmatrix, IDS = build_index(df["text_blob"], df["event_id"].tolist())

# --------- 3) Perfil (sintonización simple sin LLM) ----------
DEFAULT_PROFILE = {
    "categorias": [],
    "precio_max_cop": None,
    "fecha": "",
    "parte_del_dia": "",
    "zona": [],
    "idioma": "indiferente",
    "indoor_outdoor": "indiferente",
    "grupo": ""
}
if "perfil" not in st.session_state:
    st.session_state.perfil = DEFAULT_PROFILE.copy()
if "paso" not in st.session_state:
    st.session_state.paso = 0

PREGUNTAS = [
    ("categorias", "¿Qué tipos de planes te atraen? (ej.: concierto, stand-up, teatro) — puedes escribir varios separados por comas"),
    ("fecha", "¿Para cuándo? (hoy, mañana, fin_de_semana o rango:YYYY-MM-DD..YYYY-MM-DD)"),
    ("parte_del_dia", "¿Mañana, tarde, noche o indiferente?"),
    ("precio_max_cop", "¿Presupuesto máximo (COP)?"),
    ("zona", "¿Alguna zona/barrio/localidad preferida? (varias separadas por comas)"),
    ("grupo", "¿Vas solo, en pareja, con amigos o en familia?")
]

def aplicar_respuesta(campo, texto):
    t = (texto or "").strip()
    if campo in ("categorias","zona"):
        st.session_state.perfil[campo] = [p.strip() for p in t.replace(";",",").split(",") if p.strip()]
    elif campo == "precio_max_cop":
        nums = "".join([c for c in t if c.isdigit()])
        st.session_state.perfil[campo] = int(nums) if nums else None
    else:
        st.session_state.perfil[campo] = t

# --------- 4) UI: Entrevista a la izquierda / Resultados a la derecha ----------
col_chat, col_res = st.columns([1, 1.2])

with col_chat:
    st.subheader("Preferencias (chat)")

    # Toggle para activar/desactivar IA
    use_ai = st.toggle("Usar IA (Gemini)", value=True)
    if os.environ.get("GOOGLE_API_KEY"):
        st.caption("🔑 Gemini listo")
    else:
        st.caption("⚠️ Falta GOOGLE_API_KEY en .streamlit/secrets.toml")

    # Estado de sesión para el chat y el perfil
    if "chat" not in st.session_state:
        st.session_state.chat = []  # lista de tuplas (who, text), where who ∈ {"user","bot"}
    if "turnos" not in st.session_state:
        st.session_state.turnos = 0
    if "perfil" not in st.session_state:
        st.session_state.perfil = {
            "categorias": [],
            "audiencia": "indiferente",
            "precio_max_cop": None,
            "fecha": "",
            "parte_del_dia": "",
            "zona": [],
            "idioma": "indiferente",
            "indoor_outdoor": "indiferente",
            "grupo": ""
        }

    # Render del historial
    for who, msg in st.session_state.chat:
        st.markdown(f"**{'Planorama' if who=='bot' else 'Tú'}:** {msg}")

    # Entrada del usuario
    user_msg = st.text_input("Escribe aquí y presiona Enter", key="chat_input", label_visibility="collapsed")
    if st.button("Enviar", use_container_width=True):
        txt = (user_msg or "").strip()
        if txt:
            st.session_state.chat.append(("user", txt))
            st.session_state.turnos += 1

            if use_ai:
                # Llama al entrevistador IA (Gemini)
                bot_msg, perfil_new = llm_next_turn(st.session_state.perfil, st.session_state.chat)
                # Actualiza perfil con lo que devolvió el LLM
                st.session_state.perfil = perfil_new
                st.session_state.chat.append(("bot", bot_msg if bot_msg else "Listo, voy a recomendarte"))
            else:
                # Modo sin IA (placeholder): pide campos clave
                st.session_state.chat.append(("bot", "Modo IA apagado. Activa el toggle para que el chatbot te haga preguntas."))

            st.rerun()

    # Indicador de que ya es suficiente para recomendar
    def perfil_suficiente(p):
        claves = ["categorias","precio_max_cop","fecha","parte_del_dia","zona"]
        return all( (isinstance(p[k], list) and p[k]) or (not isinstance(p[k], list) and p[k]) for k in claves )

    if perfil_suficiente(st.session_state.perfil) or st.session_state.turnos >= 6:
        st.info("Listo, ya puedo recomendarte con este perfil.")

    st.markdown("#### Perfil actual")
    st.json(st.session_state.perfil)

with col_res:
    st.subheader("Recomendaciones")
    perfil = st.session_state.perfil
    df_filt = df.copy()

    # Ciudad y estado (si existen)
    if "city" in df_filt:
        df_filt = df_filt[df_filt["city"].str.lower().str.contains("bogotá|bogota", na=False)]
    if "status" in df_filt:
        df_filt = df_filt[df_filt["status"].str.lower().isin(["scheduled","activo","active","programado"]) | df_filt["status"].eq("")]

    # Futuro
    df_filt = df_filt[df_filt["is_future"] == True]

    # Categorías
    if perfil["categorias"]:
        cats = [c.lower() for c in perfil["categorias"]]
        if "category" in df_filt:
            df_filt = df_filt[df_filt["category"].str.lower().isin(cats)]

    # Presupuesto
    if perfil["precio_max_cop"] is not None and "price_min_cop" in df_filt:
        df_filt = df_filt[(df_filt["price_min_cop"].fillna(0) <= perfil["precio_max_cop"])]

    # Fecha
    def fecha_ok(ts, pref):
        if not isinstance(ts, pd.Timestamp) or pd.isna(ts):
            return False
        today = pd.Timestamp.now().normalize()
        if pref == "hoy":
            return ts.normalize() == today
        if pref == "mañana":
            return ts.normalize() == (today + pd.Timedelta(days=1))
        if pref == "fin_de_semana":
            return ts.weekday() in (5, 6)
        if pref.startswith("rango:"):
            try:
                _, rng = pref.split(":", 1)
                a, b = rng.split("..")
                a = pd.to_datetime(a).normalize()
                b = pd.to_datetime(b).normalize()
                return (ts.normalize() >= a) and (ts.normalize() <= b)
            except:
                return True
        return True

    if perfil["fecha"]:
        df_filt = df_filt[df_filt["date_start_parsed"].apply(lambda ts: fecha_ok(ts, perfil["fecha"]))]

    # Parte del día
    def parte_ok(h, pref):
        if not pref or pd.isna(h):
            return True
        try:
            h = int(h)
        except:
            return True
        bands = {"mañana": range(6,12), "tarde": range(12,19), "noche": range(19,24)}
        return h in bands.get(pref, range(0,24))

    if perfil["parte_del_dia"] and "hour_start" in df_filt:
        df_filt = df_filt[df_filt["hour_start"].apply(lambda h: parte_ok(h, perfil["parte_del_dia"]))]

    # Zona
    if perfil["zona"]:
        zones = [z.lower() for z in perfil["zona"]]
        cond_b = "barrio" in df_filt and df_filt["barrio"].str.lower().isin(zones)
        cond_l = "localidad" in df_filt and df_filt["localidad"].str.lower().isin(zones)
        if "barrio" in df_filt and "localidad" in df_filt:
            df_filt = df_filt[ cond_b | cond_l ]
        elif "barrio" in df_filt:
            df_filt = df_filt[ cond_b ]
        elif "localidad" in df_filt:
            df_filt = df_filt[ cond_l ]

    # Similitud (usa categorías como consulta)
    qtext = " ".join(perfil["categorias"])
    if qtext:
        qv = vectorizer.transform([qtext])
        sims_full = cosine_similarity(qv, Xmatrix).ravel()
        sim_map = {id_: float(s) for id_, s in zip(IDS, sims_full)}  # <- corregido IDS
        df_filt["sim_contenido"] = df_filt["event_id"].map(sim_map).fillna(0.0)
    else:
        df_filt["sim_contenido"] = 0.0

    # Score (simple)
    if "price_min_cop" in df_filt and df_filt["price_min_cop"].notna().any():
        p = df_filt["price_min_cop"].fillna(df_filt["price_min_cop"].median())
        if (p.max() - p.min()) > 0:
            price_score = 1 - (p - p.min())/(p.max()-p.min())
        else:
            price_score = 0.5
    else:
        price_score = 0.5

    df_filt["score"] = 0.6*df_filt["sim_contenido"] + 0.25*price_score + 0.15
    df_show = df_filt.sort_values("score", ascending=False).head(10)

    if df_show.empty:
        st.warning("No encontré resultados con tu perfil actual. Prueba subir presupuesto, cambiar fecha o ampliar zona/categoría.")
    else:
        for _, r in df_show.iterrows():
            with st.container(border=True):
                title = r.get("title","(sin título)")
                date = r.get("date_start_parsed","")
                zona = ", ".join([str(r.get("barrio","")), str(r.get("localidad",""))]).strip(", ")
                price = r.get("price_min_cop", None)
                url = r.get("booking_url", r.get("source_url",""))

                st.markdown(f"### {title}")
                st.markdown(f"**Fecha:** {date}  |  **Zona:** {zona}  |  **Desde:** {('%.0f' % price) + ' COP' if pd.notna(price) else 'N/D'}")
                if isinstance(url, str) and url.startswith("http"):
                    st.link_button("Ver más / Comprar", url)
