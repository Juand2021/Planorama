PROFILE_TEMPLATE = {
    "categorias": [],            # ["concierto","stand-up","teatro", ...]
    "precio_max_cop": None,      # int o None
    "fecha": "",                 # "hoy" | "mañana" | "fin_de_semana" | "rango:YYYY-MM-DD..YYYY-MM-DD"
    "parte_del_dia": "",         # "mañana" | "tarde" | "noche" | "indiferente"
    "zona": [],                  # ["Chapinero","Usaquén", ...]
    "idioma": "indiferente",
    "indoor_outdoor": "indiferente",
    "grupo": ""                  # "solo" | "pareja" | "amigos" | "familia"
}
REQUIRED_KEYS = list(PROFILE_TEMPLATE.keys())