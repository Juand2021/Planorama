// app.js - Main application logic
// Configuration
const API_URL = 'http://localhost:8000';

// Global state
const state = {
    profile: {
        smalltalk: "",
        fecha: "",
        fecha_rango: null,
        categorias: [],
        es_gratis: "",
        precio_max_cop: null,
        dist_importa: "",
        parte_del_dia: "",
        edad_usuario: null,
        excluir_restriccion_edad: "",
    },
    userLat: null,
    userLon: null,
    ready: false,
    chatHistory: []
};

// Initialize app on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎟️ Planorama Frontend initialized');
    checkAPIHealth();
    initializeApp();
});

// Check API health and Gemini status
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/`);
        const data = await response.json();
        
        const statusEl = document.getElementById('gemini-status');
        if (data.gemini_connected) {
            statusEl.textContent = '✅ Gemini conectado';
            statusEl.style.background = 'rgba(76, 175, 80, 0.3)';
        } else {
            statusEl.textContent = '⚠️ Gemini no disponible';
            statusEl.style.background = 'rgba(255, 152, 0, 0.3)';
        }
        
        console.log('✅ API Health:', data);
    } catch (error) {
        console.error('❌ API not available:', error);
        const statusEl = document.getElementById('gemini-status');
        statusEl.textContent = '❌ API no disponible';
        statusEl.style.background = 'rgba(244, 67, 54, 0.3)';
    }
}

// Initialize application
function initializeApp() {
    // Add initial bot message
    addBotMessage("¡Hola! 👋 Soy Planorama, tu asistente para encontrar planes en Bogotá. Cuéntame qué tipo de evento buscas y te ayudo a encontrar las mejores opciones. 😊");
    
    // Initialize chat functionality
    initializeChat();
    
    // Initialize map (if needed)
    initializeMap();
    
    // New search button
    document.getElementById('new-search-btn').addEventListener('click', resetSearch);
}

// Reset search
function resetSearch() {
    // Reset state
    state.profile = {
        smalltalk: "",
        fecha: "",
        fecha_rango: null,
        categorias: [],
        es_gratis: "",
        precio_max_cop: null,
        dist_importa: "",
        parte_del_dia: "",
        edad_usuario: null,
        excluir_restriccion_edad: "",
    };
    state.userLat = null;
    state.userLon = null;
    state.ready = false;
    state.chatHistory = [];
    
    // Clear UI
    document.getElementById('chat-messages').innerHTML = '';
    document.getElementById('results-container').innerHTML = '<div class="info-message"><p>Aún estoy reuniendo tus preferencias. Sigue respondiendo en el chat.</p></div>';
    document.getElementById('map-section').style.display = 'none';
    
    // Reset profile display
    updateProfileDisplay();
    
    // Add initial message
    addBotMessage("¡Hola! 👋 Soy Planorama, tu asistente para encontrar planes en Bogotá. Cuéntame qué tipo de evento buscas y te ayudo a encontrar las mejores opciones. 😊");
}

// Update profile display
function updateProfileDisplay() {
    const profile = state.profile;
    
    // Date
    let fechaText = "—";
    if (profile.fecha_rango && profile.fecha_rango.start && profile.fecha_rango.end) {
        fechaText = `${profile.fecha_rango.start} → ${profile.fecha_rango.end}`;
    } else if (profile.fecha) {
        fechaText = profile.fecha;
    }
    document.getElementById('profile-fecha').textContent = fechaText;
    
    // Categories
    const catsText = profile.categorias && profile.categorias.length > 0 
        ? profile.categorias.join(", ") 
        : "—";
    document.getElementById('profile-categorias').textContent = catsText;
    
    // Free/Paid
    const gratisText = profile.es_gratis || "—";
    document.getElementById('profile-gratis').textContent = gratisText;
    
    // Budget
    const presuText = profile.precio_max_cop 
        ? `$${parseInt(profile.precio_max_cop).toLocaleString('es-CO')}` 
        : "—";
    document.getElementById('profile-presupuesto').textContent = presuText;
    
    // Distance
    const cercText = profile.dist_importa || "—";
    document.getElementById('profile-cercania').textContent = cercText;
    
    // Age
    const edadText = profile.edad_usuario 
        ? `${profile.edad_usuario} años` 
        : "no especificada";
    document.getElementById('profile-edad').textContent = edadText;
    
    // Age restriction
    const restrText = profile.excluir_restriccion_edad || "—";
    document.getElementById('profile-restriccion').textContent = restrText;
    
    // Part of day
    let pdiaText = "—";
    if (Array.isArray(profile.parte_del_dia)) {
        pdiaText = profile.parte_del_dia.join(", ");
    } else if (profile.parte_del_dia) {
        pdiaText = profile.parte_del_dia;
    }
    document.getElementById('profile-parte-dia').textContent = pdiaText;
}

// Check if profile is complete
function isProfileComplete() {
    const p = state.profile;
    
    // Check date
    const hasRange = p.fecha_rango && p.fecha_rango.start && p.fecha_rango.end;
    const hasFecha = p.fecha && p.fecha.trim() !== "";
    if (!hasRange && !hasFecha) return false;
    
    // Check categories
    if (!p.categorias || p.categorias.length === 0) return false;
    
    // Check gratis/pago
    const eg = (p.es_gratis || "").toLowerCase();
    if (!["gratis", "pago", "indiferente"].includes(eg)) return false;
    
    // If pago, need budget
    if (eg === "pago" && (p.precio_max_cop === null || p.precio_max_cop === "")) return false;
    
    // Check distance
    const di = (p.dist_importa || "").toLowerCase();
    if (!["si", "no"].includes(di)) return false;
    
    // If distance matters, need location
    if (di === "si" && (state.userLat === null || state.userLon === null)) return false;
    
    // Check age policy
    const hasAge = p.edad_usuario !== null;
    const hasPolicy = ["si", "no", "indiferente"].includes((p.excluir_restriccion_edad || "").toLowerCase());
    if (!hasAge && !hasPolicy) return false;
    
    return true;
}

// Show/hide map based on distance preference
function updateMapVisibility() {
    const needsMap = (state.profile.dist_importa || "").toLowerCase() === "si";
    const mapSection = document.getElementById('map-section');
    
    if (needsMap) {
        mapSection.style.display = 'block';
        // Initialize map if not already done
        if (!window.planoramaMap) {
            initializeMap();
        }
    } else {
        mapSection.style.display = 'none';
    }
}

// Get recommendations
async function getRecommendations() {
    if (!isProfileComplete()) {
        console.log('Profile not complete yet');
        return;
    }
    
    // Check if we need location but don't have it
    const needsLocation = (state.profile.dist_importa || "").toLowerCase() === "si";
    if (needsLocation && (state.userLat === null || state.userLon === null)) {
        showWarning("Marcaste cercanía = sí. Falta que marques tu zona en el mapa para calcular distancias.");
        return;
    }
    
    // Show loading
    showLoading(true);
    
    try {
        const response = await fetch(`${API_URL}/api/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                profile: state.profile,
                user_lat: state.userLat,
                user_lon: state.userLon
            })
        });
        
        if (!response.ok) {
            throw new Error('API request failed');
        }
        
        const data = await response.json();
        console.log('📊 Recommendations:', data);
        
        // Display results
        displayResults(data.events);
        
    } catch (error) {
        console.error('❌ Error getting recommendations:', error);
        showError("Hubo un error al buscar recomendaciones. Por favor, intenta de nuevo.");
    } finally {
        showLoading(false);
    }
}

// Display results
function displayResults(events) {
    const container = document.getElementById('results-container');
    
    if (!events || events.length === 0) {
        container.innerHTML = '<div class="warning-message"><p>No encontré eventos que coincidan con tus preferencias. Intenta ajustar tus criterios.</p></div>';
        return;
    }
    
    container.innerHTML = '<div class="results-grid"></div>';
    const grid = container.querySelector('.results-grid');
    
    events.forEach(event => {
        const card = createEventCard(event);
        grid.appendChild(card);
    });
}

// Create event card
function createEventCard(event) {
    const card = document.createElement('div');
    card.className = 'event-card';
    
    // Price display
    let priceHTML = '';
    if (event.is_free) {
        priceHTML = '<span class="event-price free">GRATIS</span>';
    } else if (event.price_min_cop) {
        const price = parseInt(event.price_min_cop).toLocaleString('es-CO');
        priceHTML = `<span class="event-price">Desde $${price}</span>`;
    } else {
        priceHTML = '<span class="event-price">Precio no especificado</span>';
    }
    
    // Image
    const imageURL = event.image_url || 'https://via.placeholder.com/400x200?text=Sin+Imagen';
    
    // Description (truncate)
    const description = event.description 
        ? (event.description.length > 150 ? event.description.substring(0, 150) + '...' : event.description)
        : 'No hay descripción disponible.';
    
    // Links
    let linksHTML = '';
    if (event.source_url) {
        linksHTML += `<a href="${event.source_url}" target="_blank" class="event-link">🔗 Ver más</a>`;
    }
    if (event.organizer_url) {
        linksHTML += `<a href="${event.organizer_url}" target="_blank" class="event-link">📅 Comprar</a>`;
    }
    
    // Distance (if available)
    let distanceHTML = '';
    if (event.dist_km !== null && event.dist_km !== undefined) {
        distanceHTML = `<span class="event-meta-item">📍 ${event.dist_km.toFixed(1)} km</span>`;
    }
    
    card.innerHTML = `
        <img src="${imageURL}" alt="${event.title}" class="event-image" onerror="this.src='https://via.placeholder.com/400x200?text=Sin+Imagen'">
        <div class="event-content">
            <h3 class="event-title">${event.title}</h3>
            ${event.artist_name ? `<div class="event-artist">${event.artist_name}</div>` : ''}
            <div class="event-meta">
                <span class="event-meta-item">📅 ${event.date_start || 'Fecha no especificada'}</span>
                <span class="event-meta-item">🕐 ${event.time_start || 'Hora no especificada'}</span>
                ${distanceHTML}
            </div>
            <div class="event-meta">
                <span class="event-meta-item">📍 ${event.venue_name || 'Lugar no especificado'}</span>
                ${event.barrio ? `<span class="event-meta-item">${event.barrio}</span>` : ''}
            </div>
            <p class="event-description">${description}</p>
            <div class="event-footer">
                ${priceHTML}
                <div class="event-links">
                    ${linksHTML}
                </div>
            </div>
        </div>
    `;
    
    return card;
}

// Show warning message
function showWarning(message) {
    const container = document.getElementById('results-container');
    container.innerHTML = `<div class="warning-message"><p>${message}</p></div>`;
}

// Show error message
function showError(message) {
    const container = document.getElementById('results-container');
    container.innerHTML = `<div class="warning-message"><p>${message}</p></div>`;
}

// Show/hide loading overlay
function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    overlay.style.display = show ? 'flex' : 'none';
}

// Utility: scroll chat to bottom
function scrollChatToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

