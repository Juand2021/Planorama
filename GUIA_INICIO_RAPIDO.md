# 🚀 GUÍA DE INICIO RÁPIDO - Planorama (FastAPI + HTML)

## ✅ ¿QUÉ SE CREÓ?

He migrado completamente tu proyecto de Streamlit a FastAPI + HTML. Ahora tienes:

1. ✅ **Backend API REST** (`api.py`) - Usa toda tu lógica actual de `src/`
2. ✅ **Frontend HTML/CSS/JS** (`frontend/`) - Interfaz web moderna y profesional
3. ✅ **Scripts de desarrollo** (`dev.bat`) - Para levantar todo con 1 comando
4. ✅ **Hot-reload automático** - Cambios visibles en segundos
5. ✅ **Documentación completa** (`README_FASTAPI.md`)

---

## 🎯 LO QUE TÚ DEBES HACER AHORA

### **Paso 1: Instalar nuevas dependencias**

Abre PowerShell/CMD en la carpeta del proyecto y ejecuta:

```bash
# Activa tu entorno virtual (si aún no lo hiciste)
venv\Scripts\activate

# Instala FastAPI y uvicorn
pip install fastapi uvicorn[standard]
```

**⏱️ Tiempo: ~30 segundos**

---

### **Paso 2: Configurar API Key de Gemini**

En la misma terminal, ejecuta:

```bash
set GOOGLE_API_KEY=TU_API_KEY_AQUI
```

> **💡 IMPORTANTE:** Reemplaza `TU_API_KEY_AQUI` con tu API key real de Gemini.
> 
> Si no tienes una, obtén una GRATIS en: https://makersuite.google.com/app/apikey

**⏱️ Tiempo: ~10 segundos**

---

### **Paso 3: Iniciar el entorno de desarrollo**

En la misma terminal, ejecuta:

```bash
dev.bat
```

**Esto hará automáticamente:**
1. ✅ Iniciará el backend (FastAPI) en `http://localhost:8000`
2. ✅ Iniciará el frontend (HTML) en `http://localhost:3000`
3. ✅ Abrirá tu navegador automáticamente

**⏱️ Tiempo: ~5 segundos**

---

### **Paso 4: ¡Prueba tu aplicación!**

Tu navegador se abrirá en `http://localhost:3000` y verás:

1. 🎟️ **Planorama** - Interfaz moderna
2. 💬 **Chat** - Escribe: "Quiero un concierto mañana"
3. 🧾 **Tu perfil** - Se actualiza automáticamente
4. 🎯 **Resultados** - Aparecen cuando completas el perfil

---

## 🔧 ¿CÓMO DESARROLLAR AHORA?

### **Opción A: Cambiar funcionalidad (Backend)**

1. Abre `src/recommender.py` (o cualquier archivo en `src/`)
2. Haz tus cambios (ej: agrega un filtro nuevo)
3. **Guarda el archivo** → El backend se reinicia automáticamente (~1 seg)
4. Refresca el navegador → ¡Cambios visibles!

**Ejemplo:**
```python
# En src/recommender.py
def compute_recommendations(...):
    # Agrega un nuevo filtro
    if perfil.get("solo_fines_de_semana"):
        df_filt = df_filt[df_filt["date_start_parsed"].dt.weekday >= 5]
    # ... resto del código
```

---

### **Opción B: Cambiar diseño (Frontend)**

1. Abre `frontend/css/style.css`
2. Cambia colores, fuentes, tamaños, etc.
3. **Guarda el archivo**
4. Refresca el navegador (F5) → ¡Cambios visibles!

**Ejemplo:**
```css
/* En frontend/css/style.css */
.header {
    background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
    /* Cambia el degradado del header */
}
```

---

### **Opción C: Cambiar textos (Frontend)**

1. Abre `frontend/index.html`
2. Cambia títulos, subtítulos, mensajes
3. **Guarda el archivo**
4. Refresca el navegador → ¡Cambios visibles!

---

## 🔍 HERRAMIENTAS ÚTILES

### **1. Ver logs del backend**

Mira la ventana de CMD que dice "Backend - FastAPI". Ahí verás:
- Requests que llegan
- Errores (si hay)
- Prints que agregues

### **2. Probar API directamente**

Abre en tu navegador: `http://localhost:8000/docs`

Verás una interfaz Swagger donde puedes:
- 📚 Ver todos los endpoints
- 🧪 Probar requests manualmente
- 📊 Ver respuestas JSON

### **3. Consola del navegador (para frontend)**

Presiona `F12` en el navegador y ve a la pestaña "Console". Ahí verás:
- Logs de JavaScript
- Errores (si hay)
- Requests al backend

---

## ❓ PREGUNTAS FRECUENTES

### **¿Cómo detengo todo?**

Cierra las 2 ventanas de CMD que se abrieron (Backend y Frontend).

O presiona `Ctrl + C` en cada una.

---

### **¿Cómo vuelvo a iniciar?**

Solo ejecuta de nuevo:
```bash
dev.bat
```

---

### **¿Qué pasó con Streamlit?**

Se hizo backup en `last_versions/app_streamlit_backup.py`.

Ya NO lo necesitas, pero está ahí por si acaso.

---

### **¿Puedo usar Streamlit y FastAPI a la vez?**

Sí, pero NO es recomendado. Es mejor enfocarte en FastAPI + HTML ahora.

Si REALMENTE quieres Streamlit, puedes ejecutar:
```bash
streamlit run last_versions/app_streamlit_backup.py
```

Pero recuerda: **NO desarrolles en Streamlit**, solo usa FastAPI + HTML.

---

### **¿Cómo agrego una nueva funcionalidad?**

**Ejemplo: Filtro por barrio**

1. **Backend** - Edita `src/recommender.py`:
```python
# En compute_recommendations()
if perfil.get("barrio"):
    df_filt = df_filt[df_filt["barrio"].str.contains(perfil["barrio"], case=False)]
```

2. **Frontend** - Edita `frontend/js/app.js`:
```javascript
// En updateProfileDisplay()
if (profile.barrio) {
    // Mostrar el barrio en el perfil
}
```

3. **Guarda ambos archivos**
4. Refresca el navegador
5. ¡Listo!

---

## 🆘 ¿PROBLEMAS?

### **Error: "Cannot connect to API"**

**Solución:**
1. Verifica que el backend esté corriendo (ventana CMD debe estar abierta)
2. Ve a `http://localhost:8000` en el navegador → Deberías ver JSON
3. Si no, ejecuta manualmente: `uvicorn api:app --reload`

---

### **Error: "GOOGLE_API_KEY not found"**

**Solución:**
1. Ejecuta de nuevo: `set GOOGLE_API_KEY=TU_API_KEY`
2. Verifica que sea correcta: `echo %GOOGLE_API_KEY%`
3. Reinicia `dev.bat`

---

### **Frontend no se ve bien**

**Solución:**
1. Presiona `Ctrl + Shift + R` (fuerza recarga)
2. Abre consola (F12) y busca errores
3. Verifica que `http://localhost:3000` esté abierto

---

## 📚 PRÓXIMOS PASOS

1. ✅ Lee `README_FASTAPI.md` completo
2. ✅ Prueba cambiar el color del header en `frontend/css/style.css`
3. ✅ Agrega un nuevo filtro en `src/recommender.py`
4. ✅ Explora Swagger docs en `http://localhost:8000/docs`
5. ✅ Prueba diferentes conversaciones en el chat

---

## 🎉 ¡FELICIDADES!

Has migrado exitosamente a una arquitectura **profesional** y **escalable**.

Ahora puedes:
- ✅ Desarrollar igual de rápido que con Streamlit
- ✅ Tener control total del diseño
- ✅ Desplegar a producción fácilmente
- ✅ Aprender tecnologías usadas en la industria

**¡A desarrollar! 🚀**

---

**¿Dudas?** Revisa `README_FASTAPI.md` o contacta a [tu email].

