# src/llm_interviewer.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

@dataclass
class Profile:
    smalltalk: str
    categorias: List[str]
    fecha: str
    fecha_rango: Optional[Dict]
    parte_del_dia: str
    es_gratis: str
    precio_max_cop: Optional[int]
    dist_importa: str
    user_location_text: str
    edad_usuario: Optional[int]
    excluir_restriccion_edad: str
    zona: List[str]
    # progreso
    done_fecha: bool = False
    done_categoria: bool = False
    done_gratis_pago: bool = False
    done_presupuesto: bool = False
    done_cercania: bool = False
    done_edad: bool = False
    done_parte_dia: bool = False

def empty_profile() -> Profile:
    return Profile(
        smalltalk="",
        categorias=[],
        fecha="",
        fecha_rango=None,
        parte_del_dia="",
        es_gratis="indiferente",
        precio_max_cop=None,
        dist_importa="no",
        user_location_text="",
        edad_usuario=None,
        excluir_restriccion_edad="si",
        zona=[],
    )

def profile_to_dict(p: Profile) -> Dict:
    return asdict(p)

def _sync_done_flags(p: Profile) -> Profile:
    has_range = isinstance(p.fecha_rango, dict) and p.fecha_rango.get("start") and p.fecha_rango.get("end")
    has_fecha = bool(p.fecha)
    p.done_fecha       = has_range or has_fecha
    p.done_categoria   = bool(p.categorias)
    p.done_gratis_pago = p.es_gratis in {"gratis","pago","indiferente"}
    p.done_presupuesto = (p.es_gratis != "pago") or (p.precio_max_cop is not None)
    p.done_cercania    = p.dist_importa in {"si","no"}
    p.done_edad        = (p.edad_usuario is not None) or (p.excluir_restriccion_edad in {"si","no"})
    p.done_parte_dia   = bool(p.parte_del_dia)  # opcional
    return p

def _next_question(p: Profile) -> str:
    if not p.done_fecha:
        return ("¿Para **cuándo** te gustaría el plan? "
                "(responde: *hoy*, *mañana* o una fecha **YYYY-MM-DD**; "
                "también acepto *fin de semana*).")
    if not p.done_categoria:
        return ("¿Qué **tipo de evento** prefieres? "
                "(puedes elegir varias: *concierto*, *teatro* o *experiencia*).")
    if not p.done_gratis_pago:
        return "¿Lo quieres **gratis**, **de pago** o te es **indiferente**?"
    if p.es_gratis == "pago" and not p.done_presupuesto:
        return "¿Cuál es tu **presupuesto máximo**? (ej.: *120k*, *$120.000*, *150 mil*)."
    if not p.done_cercania:
        return ("¿Te **importa** que el evento esté **cerca** de ti? (responde *sí* o *no*). "
                "Si dices **sí**, después **debes** hacer clic en el mapa para marcar tu zona.")
    if not p.done_edad:
        return ("¿Tienes **18 años o más**? (responde *sí* o *no*). "
                "Si eres menor o vas con niños, dímelo para filtrar **todo público**.")
    if not p.done_parte_dia:
        return "¿Prefieres en la **mañana**, **tarde** o **noche**? (opcional)."
    return ""

def _final_summary(p: Profile) -> str:
    cats = ", ".join(p.categorias) if p.categorias else "—"
    presu = f"${p.precio_max_cop:,}".replace(",", ".") if p.precio_max_cop is not None else "—"
    cerc  = "sí (usaré tu mapa)" if p.dist_importa == "si" else "no"
    edad  = f"{p.edad_usuario} años" if p.edad_usuario is not None else "no indicada"
    restr = "sí" if p.excluir_restriccion_edad == "si" else "no"
    pdia  = p.parte_del_dia if p.parte_del_dia else "indiferente"
    return (
        "✅ Preferencias listas:\n"
        f"- **Fecha**: {p.fecha or '—'}\n"
        f"- **Categorías**: {cats}\n"
        f"- **Gratis/Pago**: {p.es_gratis}\n"
        f"- **Presupuesto máx**: {presu}\n"
        f"- **Cercanía importa**: {cerc}\n"
        f"- **Edad**: {edad} · **Excluir por restricción**: {restr}\n"
        f"- **Parte del día**: {pdia}"
    )

def process_turn(user_text: str, current_profile: Optional[Dict] = None):
    # reconstruye perfil y sincroniza flags
    if current_profile is None:
        p = empty_profile()
    else:
        base = asdict(empty_profile())
        base.update(current_profile or {})
        p = Profile(**base)
    p = _sync_done_flags(p)

    q = _next_question(p)
    if q:
        return q, profile_to_dict(p), False
    return _final_summary(p), profile_to_dict(p), True
