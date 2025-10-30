# src/recommender.py
# -*- coding: utf-8 -*-
"""
Event recommendation engine using TF-IDF + cosine similarity, 
with contextual scoring (price, distance, time of day).
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from geo_utils import (
    compute_distance_score,
    parse_date_pref,
    TAU_DEFAULT,
    R_DEFAULT,
)

# Scoring weights
W_CONT  = 0.50  # Content similarity
W_PREC  = 0.20  # Price
W_DIST  = 0.25  # Distance
W_FECHA = 0.05  # Time of day
TOP_N_DEFAULT = 10


def _in_date_range(ts: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Check if timestamp falls within date range."""
    if not isinstance(ts, pd.Timestamp) or pd.isna(ts):
        return False
    t = ts.to_pydatetime().replace(tzinfo=None)
    return (t >= start.replace(tzinfo=None)) and (t < end.replace(tzinfo=None))


def _part_of_day_bonus(hour_start: float, desired: str) -> float:
    """Calculate bonus score for time of day preference."""
    if not desired or pd.isna(hour_start):
        return 0.0
    h = float(hour_start)
    if desired == "mañana":
        return 1.0 if h < 12 else 0.0
    if desired == "tarde":
        return 1.0 if (13 <= h <= 18) else 0.0
    if desired == "noche":
        return 1.0 if h > 18 else 0.0
    return 0.0


def _normtxt(x: str) -> str:
    """Normalize text to ASCII lowercase."""
    import unicodedata as _ud
    s = str(x or "")
    s = _ud.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()


# Category synonyms for flexible matching - EXPANDED based on CSV data analysis
_CAT_SYNONYMS = {
    "concierto": {
        "concierto", "musica", "musica en vivo", "musical", "show", "festival", "live",
        "reggaeton", "trap", "rock", "pop", "salsa", "jazz", "hip hop", "rap", "blues",
        "folk", "clasica", "sinfonica", "orquesta", "electronica", "house", "alternativo",
        "regional mexicano", "cumbia", "bachata", "metal", "punk", "garage rock",
        "fado", "chanson", "clasico", "clásico"
    },
    "teatro": {
        "teatro", "obra", "drama", "comedia", "stand up", "stand-up", "standup",
        "musical familiar", "danza", "ballet", "danza contemporanea", "danza clásica",
        "circo", "familiar", "infantil", "teatro familiar", "circo danza"
    },
    "experiencia": {
        "experiencia", "feria", "expo", "exposicion", "exposición", "taller", "workshop",
        "tour", "recorrido", "experiencia inmersiva", "inmersivo", "inmersiva",
        "evento interactivo", "festival"
    },
}

def _normalize_category_from_csv(cat_str: str) -> str:
    """Normalize category from CSV to canonical categories: concierto, teatro, experiencia."""
    if not cat_str:
        return ""
    cat_lower = _normtxt(cat_str)
    
    # Check for music-related
    music_indicators = ["musica", "música", "concierto", "festival", "show", "live", "clasica", "sinfonica"]
    for indicator in music_indicators:
        if indicator in cat_lower:
            return "concierto"
    
    # Check for theater-related
    theater_indicators = ["teatro", "comedia", "drama", "stand", "musical", "ballet", "danza", "circo", "familiar"]
    for indicator in theater_indicators:
        if indicator in cat_lower:
            return "teatro"
    
    # Check for experience-related
    experience_indicators = ["experiencia", "feria", "expo", "exposición", "taller", "workshop", "tour", "recorrido", "inmersivo"]
    for indicator in experience_indicators:
        if indicator in cat_lower:
            return "experiencia"
    
    # Special cases from CSV
    if "boxing" in cat_lower:
        # Boxing could be experience or just leave as-is
        return "experiencia"
    
    return cat_lower  # Return as-is if can't normalize


def _expand_cats(cats: List[str]) -> set:
    """Expand categories to include synonyms."""
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
    """
    Main recommendation function: filters + scores + ranks events.
    
    Args:
        perfil: User preference profile
        df_events: Full events DataFrame
        vectorizer: Trained TF-IDF vectorizer
        Xmatrix: TF-IDF matrix for all events
        IDS: List of event UIDs matching Xmatrix rows
        user_lat: User latitude (if distance matters)
        user_lon: User longitude (if distance matters)
        w_cont: Weight for content similarity
        w_prec: Weight for price score
        w_dist: Weight for distance score
        w_fecha: Weight for time-of-day score
        tau_km: Distance decay parameter (km)
        R_km: Distance radius parameter (km)
    
    Returns:
        DataFrame with top-N ranked events
    """
    df_filt = df_events.copy()
    
    # 1) FILTERS
    
    # City: Bogotá
    if "city_norm" in df_filt.columns:
        df_filt = df_filt[df_filt["city_norm"].str.contains("bogota", na=False)]
    
    # Status: active
    if "status" in df_filt.columns:
        ok_states = {"scheduled", "activo", "active", "programado", ""}
        df_filt = df_filt[df_filt["status"].str.lower().isin(ok_states)]
    
    # Future events
    has_user_date = bool(perfil.get("fecha")) or (
        isinstance(perfil.get("fecha_rango"), dict)
        and perfil["fecha_rango"].get("start")
        and perfil["fecha_rango"].get("end")
    )
    if "is_future" in df_filt.columns and not has_user_date:
        df_filt = df_filt[df_filt["is_future"] == True]
    
    # Date / Range
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
    
    # Categories (flexible matching) - IMPROVED with better normalization
    cats = [c.lower() for c in (perfil.get("categorias") or [])]
    if cats and "category" in df_filt.columns:
        if "text_blob_norm" not in df_filt.columns:
            df_filt["text_blob_norm"] = df_filt["text_blob"].apply(_normtxt)
        
        # Normalize categories from CSV to canonical ones
        if "category_normalized" not in df_filt.columns:
            df_filt["category_normalized"] = df_filt["category"].apply(_normalize_category_from_csv)
        
        wanted = _expand_cats(cats)
        
        def _cat_ok(row) -> bool:
            # Check normalized category first
            cat_norm = _normtxt(row.get("category_normalized", ""))
            if cat_norm in wanted or cat_norm in cats:
                return True
            # Check original category (with flexible matching for consolidated categories)
            orig_cat = _normtxt(row.get("category", ""))
            
            # Direct match
            if orig_cat in wanted:
                return True
            
            # Check if any wanted category matches the prefix (for consolidated categories)
            # e.g., if user selected "teatro" and event has "teatro/danza", it should match
            for wanted_cat in wanted:
                # If the event category starts with wanted_cat + "/", it's a match
                if orig_cat.startswith(wanted_cat + "/"):
                    return True
                # If wanted_cat starts with orig_cat + "/" (less common but possible)
                if "/" in wanted_cat and orig_cat == wanted_cat.split("/")[0]:
                    return True
            
            # Check in text blob
            blob_ok = any(w in row.get("text_blob_norm", "") for w in wanted)
            return blob_ok
        
        df_filt = df_filt[df_filt.apply(_cat_ok, axis=1)]
    
    # Free / Paid + Budget
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
    
    # Age restriction
    if "age_min_num" not in df_filt.columns:
        def _age_to_int(s: str):
            s = (s or "").strip().lower()
            if s in {"", "todas", "toda", "all"}:
                return 0
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
    
    # 2) SCORING
    
    # Content similarity (TF-IDF + cosine) - IMPROVED to use keywords
    # Build query text from categories + keywords for better semantic matching
    query_parts = []
    if cats:
        query_parts.extend(cats)
    
    # Add keywords from user preferences (artists, genres, themes, etc.)
    keywords = perfil.get("keywords", [])
    if keywords:
        # Keywords are already lowercased in merge_profiles
        query_parts.extend([str(kw).strip() for kw in keywords if str(kw).strip()])
    
    if query_parts:
        # Join all query parts for TF-IDF search
        query_text = " ".join(query_parts)
        qv = vectorizer.transform([query_text])
        sims_full = cosine_similarity(qv, Xmatrix).ravel()
        sim_map = {id_: float(s) for id_, s in zip(IDS, sims_full)}
        df_filt["sim_contenido"] = df_filt["uid"].map(sim_map).fillna(0.0)
    else:
        # If no categories or keywords, use a very basic similarity
        df_filt["sim_contenido"] = 0.3  # Small default similarity instead of 0
    
    # Price score
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
    
    # Time of day (bonus)
    desired_part = perfil.get("parte_del_dia")
    if desired_part == "indiferente" or not desired_part:
        df_filt["score_fecha"] = 0.0
    elif isinstance(desired_part, list):
        # Multiple time preferences
        def multi_part_bonus(h):
            if pd.isna(h):
                return 0.0
            for part in desired_part:
                if _part_of_day_bonus(h, part) > 0:
                    return 1.0
            return 0.0
        df_filt["score_fecha"] = df_filt["hour_start"].apply(multi_part_bonus)
    else:
        df_filt["score_fecha"] = df_filt["hour_start"].apply(lambda h: _part_of_day_bonus(h, desired_part))
    
    # Distance (Haversine + exponential decay)
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
            tau=float(tau_km),
            R=float(R_km),
        )
        return (res.score if res.ok else 0.0), (res.dist_km if res.ok else None)
    
    scores_dist, dists_km = [], []
    for _, r in df_filt.iterrows():
        s, dkm = dist_component(r)
        scores_dist.append(s)
        dists_km.append(dkm)
    df_filt["score_dist"] = scores_dist
    df_filt["dist_km"] = dists_km
    
    w_dist_eff = w_dist if dist_importa else 0.0
    
    # Final score
    df_filt["score_final"] = (
        w_cont * df_filt["sim_contenido"].fillna(0.0) +
        w_prec * df_filt["score_precio"].fillna(0.0) +
        w_dist_eff * df_filt["score_dist"].fillna(0.0) +
        w_fecha * df_filt["score_fecha"].fillna(0.0)
    )
    
    # Rank and deduplicate
    df_rank = (
        df_filt
        .drop_duplicates(subset=["title", "date_start_parsed"], keep="first")
        .sort_values(["score_final", "date_start_parsed"], ascending=[False, True])
    )
    
    # Dynamic Top-N
    if len(df_rank) <= 3:
        top_n = len(df_rank)
    elif len(df_rank) <= TOP_N_DEFAULT:
        top_n = len(df_rank)
    else:
        top_n = TOP_N_DEFAULT
    
    return df_rank.head(top_n)

