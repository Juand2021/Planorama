# src/geo_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import pandas as pd
from datetime import datetime, timedelta

# Parámetros por defecto para normalización de distancia
TAU_DEFAULT = 2.0   # km (cuanto menor, más “agresiva” la penalización por distancia)
R_DEFAULT   = 15.0  # km (radio razonable de proximidad para eventos urbanos)

# ──────────────────────────────────────────────────────────────────────────────
# Distancia y normalización
# ──────────────────────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distancia en km entre 2 coordenadas (lat/lon) usando fórmula de Haversine.
    """
    if None in (lat1, lon1, lat2, lon2):
        return float("nan")
    R = 6371.0  # km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def normalize_distance(d_km: float, mode: str = "exp", tau: float = TAU_DEFAULT, R: float = R_DEFAULT) -> float:
    """
    Convierte distancia en un score [0..1]. 
    - mode="exp": score = exp(-d/tau)
    - mode="linear": score = max(0, 1 - d/R)
    """
    if d_km is None or math.isnan(d_km):
        return 0.0
    d = max(0.0, float(d_km))
    if mode == "linear":
        return max(0.0, 1.0 - d / max(1e-6, R))
    # exponencial por defecto
    return math.exp(- d / max(1e-6, tau))

@dataclass
class DistScore:
    ok: bool
    dist_km: Optional[float]
    score: float

def compute_distance_score(
    dist_importa: bool,
    user_lat: float,
    user_lon: float,
    event_lat: float,
    event_lon: float,
    mode: str = "exp",
    tau: float = TAU_DEFAULT,
    R: float = R_DEFAULT,
) -> DistScore:
    """
    Si 'dist_importa' es False → score=0 (no influye).
    Si True y hay coords → calcula Haversine y normaliza.
    """
    if not dist_importa:
        return DistScore(ok=False, dist_km=None, score=0.0)
    try:
        dkm = haversine(user_lat, user_lon, event_lat, event_lon)
        sc = normalize_distance(dkm, mode=mode, tau=tau, R=R)
        return DistScore(ok=True, dist_km=dkm, score=sc)
    except Exception:
        return DistScore(ok=False, dist_km=None, score=0.0)

def require_user_point(dist_importa: bool, user_lat: Optional[float], user_lon: Optional[float]) -> bool:
    """
    Devuelve True si se requiere y EXISTE pin del usuario; False en otro caso.
    """
    if not dist_importa:
        return True
    return (user_lat is not None) and (user_lon is not None)

# ──────────────────────────────────────────────────────────────────────────────
# Fechas y preferencias
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DateRange:
    start: pd.Timestamp
    end: pd.Timestamp  # semiabierto [start, end)

def _today() -> pd.Timestamp:
    # sin tz para mantener consistencia con pandas naive en el CSV
    return pd.Timestamp.now(tz=None).normalize()

def parse_date_pref(pref: str) -> DateRange:
    """
    Convierte 'hoy' | 'mañana' | 'fin_de_semana' | 'YYYY-MM-DD' en rango [start, end).
    """
    s = (pref or "").strip().lower()
    now = _today()

    if s in {"hoy", "today"}:
        start = now
        end = start + pd.Timedelta(days=1)
        return DateRange(start=start, end=end)

    if s in {"mañana", "manana", "tomorrow"}:
        start = now + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=1)
        return DateRange(start=start, end=end)

    if s in {"fin_de_semana", "fin de semana", "finde", "weekend"}:
        # asumimos fin de semana próximo: sábado-domingo
        wd = now.weekday()  # lunes=0 ... domingo=6
        # siguiente sábado
        days_until_sat = (5 - wd) % 7
        start = (now + pd.Timedelta(days=days_until_sat)).normalize()
        end = start + pd.Timedelta(days=2)
        return DateRange(start=start, end=end)

    # Fecha explícita YYYY-MM-DD
    try:
        dt = pd.to_datetime(s, errors="raise").normalize()
        start = dt
        end = dt + pd.Timedelta(days=1)
        return DateRange(start=start, end=end)
    except Exception:
        # fallback: hoy
        return DateRange(start=now, end=now + pd.Timedelta(days=1))

def score_part_of_day(hour_start: Optional[float], desired: str) -> float:
    """
    Bonus por parte del día (0 o 1):
      - mañana: < 12
      - tarde:  13–18
      - noche:  > 18
    """
    if not desired or hour_start is None or math.isnan(hour_start):
        return 0.0
    h = float(hour_start)
    if desired == "mañana":
        return 1.0 if h < 12 else 0.0
    if desired == "tarde":
        return 1.0 if (13 <= h <= 18) else 0.0
    if desired == "noche":
        return 1.0 if h > 18 else 0.0
    return 0.0
