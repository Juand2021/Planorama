# src/ui_utils.py
# -*- coding: utf-8 -*-
"""
UI utilities for rendering maps and event results in Streamlit.
"""
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from typing import Dict


def render_location_map(user_lat=None, user_lon=None, center_lat=4.6486, center_lon=-74.0649):
    """
    Renders an interactive Folium map for user to select their location.
    
    Args:
        user_lat: Current user latitude (if already selected)
        user_lon: Current user longitude (if already selected)
        center_lat: Default map center latitude
        center_lon: Default map center longitude
    
    Returns:
        dict: Map data from st_folium (includes last_clicked coordinates)
    """
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)
    
    if user_lat is not None and user_lon is not None:
        folium.Marker(
            [user_lat, user_lon],
            tooltip="Tu zona",
            icon=folium.Icon(color="red", icon="home", prefix="fa")
        ).add_to(m)
    
    map_data = st_folium(m, width=900, height=420)
    return map_data


def render_results_map(df_rank: pd.DataFrame, user_lat: float, user_lon: float):
    """
    Renders a map showing user location and recommended events.
    
    Args:
        df_rank: DataFrame with ranked event recommendations
        user_lat: User latitude
        user_lon: User longitude
    """
    map_center = [user_lat, user_lon]
    result_map = folium.Map(location=map_center, zoom_start=12, control_scale=True)
    
    # Add user marker
    folium.Marker(
        [user_lat, user_lon],
        popup="Tu ubicación",
        tooltip="Tu ubicación",
        icon=folium.Icon(color="red", icon="home", prefix="fa")
    ).add_to(result_map)
    
    # Add event markers
    for idx, r in df_rank.iterrows():
        ev_lat = r.get("lat")
        ev_lon = r.get("lon")
        if pd.notna(ev_lat) and pd.notna(ev_lon):
            dist_km = r.get("dist_km")
            title = r.get("title", "Evento")
            venue = r.get("venue_name", "")
            popup_text = f"<b>{title}</b><br>{venue}"
            if dist_km is not None:
                popup_text += f"<br>~{dist_km:.1f} km"
            
            folium.Marker(
                [float(ev_lat), float(ev_lon)],
                popup=popup_text,
                tooltip=title[:50],
                icon=folium.Icon(color="blue", icon="music", prefix="fa")
            ).add_to(result_map)
    
    st_folium(result_map, width=900, height=500)


def render_event_card(event_row, dist_importa: bool = False):
    """
    Renders a single event card with all details.
    
    Args:
        event_row: Series/dict with event data
        dist_importa: Whether distance is relevant
    """
    title = event_row.get("title", "(sin título)") or "(sin título)"
    ts = event_row.get("date_start_parsed", "")
    date_txt = str(ts) if pd.notna(ts) else "—"
    zona = ", ".join([
        str(event_row.get("barrio", "") or ""),
        str(event_row.get("localidad", "") or "")
    ]).strip(", ").strip() or "—"
    
    # Price formatting
    is_free = bool(event_row.get("is_free") is True)
    price = event_row.get("price_min_cop", None)
    if is_free:
        price_txt = "Gratis"
    elif pd.notna(price):
        try:
            price_txt = f"{int(float(price)):,} COP".replace(",", ".")
        except Exception:
            price_txt = f"{price} COP"
    else:
        price_txt = "N/D"
    
    # Distance badge and text
    dist_txt = ""
    dist_badge = ""
    if dist_importa and event_row.get("dist_km") is not None:
        try:
            km = float(event_row['dist_km'])
            dist_txt = f" · 📏 ~{km:.1f} km de tu zona"
            if km < 2:
                dist_badge = " 🟢 **MUY CERCA**"
            elif km < 5:
                dist_badge = " 🟡 **CERCA**"
        except Exception:
            dist_txt = ""
    
    st.markdown(f"### {title}{dist_badge}")
    st.markdown(f"**Fecha:** {date_txt} · **Zona:** {zona} · **Desde:** {price_txt}{dist_txt}")
    
    # Image
    img = event_row.get("image_url", "")
    if isinstance(img, str) and img.startswith("http"):
        st.image(img, use_column_width=True)
    
    # Link
    url = event_row.get("organizer_url") or event_row.get("source_url") or ""
    if isinstance(url, str) and url.startswith("http"):
        st.link_button("Ver más / Comprar", url, use_container_width=True)


def render_results(df_rank: pd.DataFrame, perfil: Dict) -> None:
    """
    Main function to render all event results with optional map.
    
    Args:
        df_rank: DataFrame with ranked recommendations
        perfil: User profile dictionary
    """
    if df_rank.empty:
        st.warning("No encontré resultados con tus preferencias. Prueba ampliar categorías, fecha o presupuesto.")
        return
    
    dist_importa = (perfil.get("dist_importa") or "").lower() == "si"
    
    # Show map if distance matters and user has location
    if dist_importa and st.session_state.user_lat is not None and st.session_state.user_lon is not None:
        with st.expander("🗺️ Mapa de eventos recomendados", expanded=False):
            render_results_map(df_rank, st.session_state.user_lat, st.session_state.user_lon)
    
    # Render event cards
    for _, r in df_rank.iterrows():
        with st.container(border=True):
            render_event_card(r, dist_importa)
    
    # Details table
    with st.expander("Ver tabla (detalles)"):
        show_cols = [
            "title", "category", "date_start_parsed", "barrio", "localidad",
            "price_min_cop", "is_free", "age_min", "lat", "lon",
            "sim_contenido", "score_precio", "score_dist", "score_fecha", "score_final", "dist_km"
        ]
        show_cols = [c for c in show_cols if c in df_rank.columns]
        st.dataframe(df_rank[show_cols])

