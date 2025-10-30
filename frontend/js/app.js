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
        keywords: [],
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
    const statusEl = document.getElementById('gemini-status');
    
    if (!statusEl) {
        console.error('❌ Could not find gemini-status element');
        return;
    }
    
    try {
        // Try /health endpoint first, fallback to root with Accept header
        const healthUrl = `${API_URL}/health`;
        console.log(`🔍 Checking API health at ${healthUrl}`);
        const response = await fetch(healthUrl, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            },
            cache: 'no-cache'
        }).catch(async () => {
            // Fallback to root endpoint with Accept header
            console.log(`⚠️ /health failed, trying root endpoint...`);
            return fetch(`${API_URL}/`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json'
                },
                cache: 'no-cache'
            });
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ API Health Response:', data);
        
        if (data.gemini_connected) {
            statusEl.textContent = '✅ Gemini conectado';
            statusEl.style.background = 'rgba(76, 175, 80, 0.3)';
            console.log('✅ Gemini está conectado');
        } else {
            statusEl.textContent = '⚠️ Gemini no disponible';
            statusEl.style.background = 'rgba(255, 152, 0, 0.3)';
            console.warn('⚠️ Gemini no está disponible según el servidor');
        }
    } catch (error) {
        console.error('❌ Error checking API health:', error);
        console.error('❌ Error details:', {
            message: error.message,
            stack: error.stack,
            apiUrl: API_URL
        });
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
    
    // Map will be initialized lazily when needed (when user selects distance-based recommendations)
    
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
        keywords: [],
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
    document.getElementById('map-section').style.display = 'none';
    
    // Reset map
    if (planoramaMap) {
        planoramaMap.remove();
        planoramaMap = null;
        if (typeof userMarker !== 'undefined') {
            userMarker = null;
        }
    }
    
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
        if (!planoramaMap) {
            // Wait a tick for the browser to recalculate layout before initializing
            setTimeout(() => {
                initializeMap();
            }, 100);
        } else {
            // Map already exists, invalidate size to fix dimensions
            setTimeout(() => {
                planoramaMap.invalidateSize();
            }, 100);
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
        
        if (data.ai_enabled && data.top_recommendation) {
            displayTopRecommendationInChat(data.top_recommendation, data.count);
        } else {
            addBotMessage('No encontré un plan destacado con IA. Intenta ajustar tus preferencias.');
        }
        
    } catch (error) {
        console.error('❌ Error getting recommendations:', error);
        showError("Hubo un error al buscar recomendaciones. Por favor, intenta de nuevo.");
    } finally {
        showLoading(false);
    }
}

// Display a single top recommendation as a bot message in the chat
function displayTopRecommendationInChat(topRecommendation, totalCount) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    // Intro phrase
    const introDiv = document.createElement('div');
    introDiv.className = 'message message-bot';
    introDiv.innerHTML = `<strong>Planorama:</strong> Perfecto, ya tengo la información requerida y el plan que te recomiendo es el siguiente:`;
    chatMessages.appendChild(introDiv);
    
    // Explanation and probability with enhanced details
    const probability = topRecommendation.ai_probability || 85.0;
    const explainDiv = document.createElement('div');
    explainDiv.className = 'message message-bot recommendation-explanation-box';
    const countText = (typeof totalCount === 'number' && totalCount > 0) ? ` de <strong>${totalCount}</strong> recomendaciones posibles` : '';
    
    // Build detailed explanation
    let detailedExplanation = `
        <div class="recommendation-header">
            <div class="probability-badge">
                <span class="probability-number">${probability}%</span>
                <span class="probability-label">Compatibilidad</span>
            </div>
            <div class="recommendation-rank">
                <span class="rank-label">Recomendación Principal</span>
                ${countText ? `<span class="rank-count">Clasificada #1 ${countText}</span>` : ''}
            </div>
        </div>
        <div class="recommendation-reasons">
            <h4>🎯 Por qué esta es tu mejor opción:</h4>
            <p class="main-explanation">${escapeHtml(topRecommendation.explanation || 'Este evento coincide perfectamente con tus preferencias.')}</p>
    `;
    
    // Add detailed matching criteria if available
    const matchReasons = [];
    if (topRecommendation.category_match) {
        matchReasons.push(`✅ Coincide con tus categorías preferidas: <strong>${topRecommendation.category_match}</strong>`);
    }
    if (topRecommendation.date_match) {
        matchReasons.push(`📅 Fecha perfecta para tu disponibilidad`);
    }
    if (topRecommendation.price_match) {
        matchReasons.push(`💰 Dentro de tu presupuesto`);
    }
    if (topRecommendation.dist_km !== null && topRecommendation.dist_km !== undefined && topRecommendation.dist_km < 5) {
        matchReasons.push(`📍 Ubicación conveniente (${topRecommendation.dist_km.toFixed(1)} km de distancia)`);
    }
    if (topRecommendation.age_appropriate) {
        matchReasons.push(`👤 Apropiado para tu grupo de edad`);
    }
    
    if (matchReasons.length > 0) {
        detailedExplanation += `
            <ul class="match-reasons">
                ${matchReasons.map(reason => `<li>${reason}</li>`).join('')}
            </ul>
        `;
    }
    
    detailedExplanation += `</div>`;
    
    explainDiv.innerHTML = detailedExplanation;
    chatMessages.appendChild(explainDiv);
    
    // Card
    const cardWrapper = document.createElement('div');
    cardWrapper.className = 'message message-bot';
    cardWrapper.innerHTML = createEventCardHTML(topRecommendation, true);
    chatMessages.appendChild(cardWrapper);
    
    // Add button to see other options
    if (totalCount && totalCount > 1) {
        const optionsButtonDiv = document.createElement('div');
        optionsButtonDiv.className = 'message message-bot';
        optionsButtonDiv.innerHTML = `
            <div class="alternative-options-section">
                <p style="margin-bottom: 12px;">¿No es exactamente lo que buscas?</p>
                <button id="show-alternatives-btn" class="btn-show-alternatives">
                    🔄 Ver ${totalCount - 1} Otras Opciones Clasificadas
                </button>
            </div>
        `;
        chatMessages.appendChild(optionsButtonDiv);
        
        // Add event listener for the button
        setTimeout(() => {
            const showAltBtn = document.getElementById('show-alternatives-btn');
            if (showAltBtn) {
                showAltBtn.addEventListener('click', async () => {
                    showAltBtn.disabled = true;
                    showAltBtn.textContent = '⏳ Cargando alternativas...';
                    await showAlternativeRecommendations();
                });
            }
        }, 100);
    }
    
    scrollChatToBottom();
}

// Helper function to generate event card HTML (used for both regular cards and AI recommendation)
function createEventCardHTML(event, isAIRecommendation = false) {
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
    
    const cardClass = isAIRecommendation ? 'event-card ai-recommendation-event' : 'event-card';
    
    return `
        <div class="${cardClass}">
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
        </div>
    `;
}

// Create event card (returns DOM element)
function createEventCard(event) {
    const card = document.createElement('div');
    card.innerHTML = createEventCardHTML(event, false);
    return card.firstElementChild || card;
}

// Show warning message
function showWarning(message) {
    addBotMessage(message);
}

// Show error message
function showError(message) {
    addBotMessage(message);
}

// Show/hide loading overlay
function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    overlay.style.display = show ? 'flex' : 'none';
}

// Show alternative recommendations
async function showAlternativeRecommendations() {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    // Add loading message
    addBotMessage("Déjame mostrarte otras excelentes opciones basadas en tus preferencias...");
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_URL}/api/recommend/all`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                profile: state.profile,
                user_lat: state.userLat,
                user_lon: state.userLon,
                skip_top: true  // Skip the top recommendation since we already showed it
            })
        });
        
        if (!response.ok) {
            throw new Error('API request failed');
        }
        
        const data = await response.json();
        console.log('📊 Alternative recommendations:', data);
        
        if (data.recommendations && data.recommendations.length > 0) {
            // Display header for alternatives
            const headerDiv = document.createElement('div');
            headerDiv.className = 'message message-bot';
            headerDiv.innerHTML = `
                <div class="alternatives-header">
                    <h3>🎯 Recomendaciones Alternativas (Clasificadas por Puntuación)</h3>
                    <p>Aquí tienes ${data.recommendations.length} opciones más que coinciden con tus preferencias:</p>
                </div>
            `;
            chatMessages.appendChild(headerDiv);
            
            // Display each alternative recommendation
            data.recommendations.forEach((event, index) => {
                const rankNumber = index + 2; // Start from #2 since #1 was the top recommendation
                const probability = event.ai_probability || 0;
                
                const altDiv = document.createElement('div');
                altDiv.className = 'message message-bot alternative-recommendation';
                altDiv.innerHTML = `
                    <div class="alternative-rank-badge">
                        <span class="rank-number">#${rankNumber}</span>
                        <span class="rank-probability">${probability}% compatibilidad</span>
                    </div>
                    ${createEventCardHTML(event, false)}
                    ${event.explanation ? `<div class="alternative-explanation">💡 <em>${escapeHtml(event.explanation)}</em></div>` : ''}
                `;
                chatMessages.appendChild(altDiv);
            });
            
            scrollChatToBottom();
        } else {
            addBotMessage("No se encontraron recomendaciones adicionales que coincidan con tus criterios.");
        }
        
    } catch (error) {
        console.error('❌ Error getting alternative recommendations:', error);
        addBotMessage("Lo siento, no pude cargar las recomendaciones alternativas. Por favor, intenta de nuevo.");
    } finally {
        showLoading(false);
    }
}

// Utility: scroll chat to bottom
function scrollChatToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

