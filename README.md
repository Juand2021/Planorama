Planorama — README

Recomendador de planes y eventos en Bogotá con interfaz conversacional.
Técnica central: Recomendación basada en contenido usando TF-IDF + similitud del coseno, más reglas de negocio (fecha/rango, precio/presupuesto, edad) y cercanía con Haversine cuando el usuario lo pide. Gemini se usa solo para entender lenguaje natural y normalizar preferencias; no decide las recomendaciones.

1) Demo del flujo (alto nivel)
Usuario (chat) ──► Gemini (NLU → JSON) ──► Perfil normalizado
                                      │
                                      ▼
                              Filtros duros (fecha/rango, ciudad, edad, gratis/pago)
                                      │
                                      ▼
             TF-IDF (eventos) + consulta ► similitud del coseno
                                      │
                      Distancia (Haversine), Precio, Parte del día
                                      │
                                      ▼
                           Score final + Ranking Top-N
                                      │
                                      ▼
                     UI: tarjetas con fecha, precio, imagen, enlaces

2) Estructura del repo
PLANORAMA/
├─ data/
│  └─ Planorama_BD.csv                  # Dataset principal (usado por la app)
├─ src/
│  ├─ geo_utils.py                      # Haversine, normalización de distancia, fechas
│  ├─ llm_interviewer.py                # Orquestador de preguntas (usa perfil, no texto)
│  └─ schema.py (opcional)              # Si quieres tipar el perfil
├─ app.py                               # App Streamlit (UI + Gemini + TF-IDF + ranking)
├─ requirements.txt
└─ .streamlit/
   └─ secrets.toml                      # GOOGLE_API_KEY para Gemini

3) Requisitos e instalación

Python 3.9+ recomendado.

requirements.txt mínimo:

streamlit
pandas
numpy
scikit-learn
folium
streamlit-folium
google-generativeai


Instalación:

pip install -r requirements.txt


Configura tu clave de Gemini:

.streamlit/secrets.toml
-----------------------
GOOGLE_API_KEY = "TU_API_KEY"


Ejecutar:

streamlit run app.py

4) Dataset esperado (data/Planorama_BD.csv)

Columnas principales (texto/numérico; si falta alguna, la app rellena vacíos):

event_id, title, description, tags, category

date_start, time_start, date_end

lat, lon, city, barrio, localidad, venue_name

price_min_cop, price_max_cop, is_free

age_min

organizer_url, source_url, image_url

status

La app:

Parseará fechas/horas a date_start_parsed/hour_start.

Derivará city_norm (para filtrar Bogotá) y age_min_num.

Construirá text_blob (título+desc+tags+otros) para TF-IDF.

Opcional: normaliza category a concierto/teatro/experiencia.

5) Interfaz conversacional

Gemini convierte lo que diga el usuario en un JSON normalizado (contrato):

{
  "smalltalk": "texto cálido sin preguntas",
  "fecha": "hoy|mañana|fin_de_semana|YYYY-MM-DD",
  "fecha_rango": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
  "categorias": ["concierto|teatro|experiencia"],
  "es_gratis": "gratis|pago|indiferente",
  "precio_max_cop": 120000,
  "dist_importa": "si|no",
  "parte_del_dia": "mañana|tarde|noche",
  "edad_usuario": 21,
  "excluir_restriccion_edad": "si|no"
}


Mapa condicional: solo aparece si dist_importa = "si". Un clic fija user_lat/lon.

Panel “Tu perfil (solo para ti)” muestra el estado normalizado (no lo imprime en el chat).

Resultados automáticos: sin botón; se muestran cuando el perfil está completo (y si aplica, ya hay pin en el mapa).

6) Algoritmo de recomendación (core técnico)
6.1 Filtros duros (antes del scoring)

Ciudad: Bogotá (por city_norm).

Estado: activos (scheduled/active/…).

Futuro: date_start_parsed >= hoy.

Fecha:

Si fecha_rango → [start, end).
Corrección automática: si Gemini devuelve un rango en el pasado (p. ej., “próxima semana” de 2024), se normaliza al futuro relativo (siguiente semana/mes/año).

Si fecha (“hoy”, “mañana”, “fin_de_semana” o fecha ISO) → se convierte internamente a rango.

Categoría: canónicas concierto/teatro/experiencia.

Gratis/Pago:

gratis → is_free=True.

pago → is_free=False y price_min_cop ≤ presupuesto (si lo hay).

Edad: si excluir_restriccion_edad = "si" y edad_usuario dado → age_min_num ≤ edad_usuario.

6.2 Señal de contenido (IA clásica)

TF-IDF sobre text_blob del evento.

Consulta a partir de las categorías (y opcionalmente keywords).

Similitud del coseno → sim_contenido ∈ [0,1].

6.3 Señales contextuales (negocio)

Precio: score_precio (1.0 dentro de presupuesto; decae suavemente si excede).

Distancia: Haversine (km) y normalización exponencial exp(-d/τ)
τ (~2 km) y R (~15 km) regulan sensibilidad. Si dist_importa = "no", peso 0.

Parte del día: bonus (1/0) si la hora del evento cae en:

mañana <12:00, tarde 13–18, noche >18.

6.4 Score final y ranking

Ponderación (ajustable):

score_final = w_cont * sim_contenido
            + w_prec * score_precio
            + w_dist * score_dist      (w_dist = 0 si no importa cercanía)
            + w_fecha * score_fecha


Orden descendente por score_final (desempate por fecha más próxima).

Top-N dinámico: si hay pocos resultados, muestra 2–3; si hay suficientes, hasta 10.

7) Decisiones de diseño (justificación académica)

Enfoque “basado en contenido”: usa distancias/similitudes en un espacio vectorial (TF-IDF) para comparar la consulta con cada evento, devolviendo los más cercanos (coseno).

Es interpretable, eficiente y adecuado para frío-inicio (sin historial).

La noción de distancia/similitud es un principio central en métodos de aprendizaje clásico (KNN, clustering) — exactamente el marco teórico que sustenta esta solución.

8) Casos de prueba sugeridos

“Quiero un concierto gratis mañana.”

“Algo de comedia el próximo mes cerca de mí.” → debe aparecer el mapa.

“Teatro de pago máximo 120 mil.”

“Voy con hijos.” → filtrar todo público.

“Mañana en la noche.”

9) Limitaciones actuales

La afinidad semántica depende de TF-IDF (no capta sinónimos tan bien como embeddings).

No hay personalización por historial (aún).

Coordenadas faltantes en eventos anulan la señal de distancia para ese evento.

10) Roadmap (opcional, sin cambiar la técnica central)

Embeddings (sentence transformers) para mejorar similitud semántica (sigue siendo contenido).

Learning-to-Rank (LightGBM/XGBoost Ranker) cuando existan señales de clic/compra.

Filtros por barrios/localidades cuando el usuario lo mencione (“por Chapinero”).

11) Cómo contribuir

Añadir fuentes de eventos al CSV (con lat/lon).

Mejorar normalización de category a canónicas.

Ajustar pesos W_* del ranking (en app.py) según validación empírica.

PRs y issues bienvenidos.

12) Licencia

Uso académico/educativo. Ajusta según tu contexto.

TL;DR

Planorama aplica IA clásica basada en distancias: TF-IDF + coseno para similitud de contenido, más Haversine para cercanía y reglas de negocio. Gemini solo traduce el lenguaje natural del usuario al perfil que activa el recomendador. Con esto, el sistema es interpretable, eficiente y listo para evaluación académica.
