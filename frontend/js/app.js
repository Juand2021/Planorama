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
    chatHistory: [],
    alternativeRecommendations: [],  // Store alternative recommendations
    hasShownAlternatives: false      // Track if alternatives have been shown
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
            statusEl.style.background = '#00D9FF';
            statusEl.style.color = '#2C3E50';
            statusEl.style.borderColor = '#00D9FF';
            statusEl.style.fontWeight = '700';
            statusEl.style.boxShadow = '0 2px 12px rgba(0, 217, 255, 0.3)';
            console.log('✅ Gemini está conectado');
        } else {
            statusEl.textContent = '⚠️ Gemini no disponible';
            statusEl.style.background = '#FFB347';
            statusEl.style.color = '#2C3E50';
            statusEl.style.borderColor = '#FFB347';
            statusEl.style.fontWeight = '700';
            statusEl.style.boxShadow = '0 2px 12px rgba(255, 179, 71, 0.3)';
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
        statusEl.style.background = '#FF6B6B';
        statusEl.style.color = 'white';
        statusEl.style.borderColor = '#FF6B6B';
        statusEl.style.fontWeight = '700';
        statusEl.style.boxShadow = '0 2px 12px rgba(255, 107, 107, 0.3)';
    }
}

// Initialize application
function initializeApp() {
    // Initialize tab switching
    initializeTabSwitching();
    
    // Add initial bot message
    addBotMessage("¡Hola! 👋 Soy Planorama, tu asistente para encontrar planes en Bogotá. Cuéntame qué tipo de evento buscas y te ayudo a encontrar las mejores opciones. 😊");
    
    // Initialize chat functionality
    initializeChat();
    
    // Map will be initialized lazily when needed (when user selects distance-based recommendations)
    
    // New search button
    document.getElementById('new-search-btn').addEventListener('click', resetSearch);
}

// Initialize tab switching functionality
function initializeTabSwitching() {
    const tabButtons = document.querySelectorAll('.nav-menu-btn');
    
    if (!tabButtons || tabButtons.length === 0) {
        console.warn('⚠️ No tab buttons found');
        return;
    }
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;
            
            if (!targetTab) {
                console.warn('⚠️ Button has no data-tab attribute', button);
                return;
            }
            
            // Update active tab button
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Update active tab content
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => content.classList.remove('active'));
            
            const targetContent = document.getElementById(`tab-${targetTab}`);
            if (targetContent) {
                targetContent.classList.add('active');
            } else {
                console.warn(`⚠️ Tab content not found for: tab-${targetTab}`);
            }
            
            // Refresh map if switching to app tab
            if (targetTab === 'app' && typeof planoramaMap !== 'undefined' && planoramaMap) {
                setTimeout(() => {
                    planoramaMap.invalidateSize();
                }, 100);
            }
            
            // Refresh ScrollTrigger for GSAP animations
            if (typeof window.refreshScrollTrigger === 'function') {
                setTimeout(() => {
                    window.refreshScrollTrigger();
                }, 300);
            }
        });
    });
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
    state.alternativeRecommendations = [];
    state.hasShownAlternatives = false;
    
    // Clear UI
    document.getElementById('chat-messages').innerHTML = '';
    document.getElementById('map-section').style.display = 'none';
    
    // Hide recommendations loading and section
    const recommendationsLoading = document.getElementById('recommendations-loading');
    const recommendationsSection = document.getElementById('recommendations-section');
    if (recommendationsLoading) {
        recommendationsLoading.style.display = 'none';
    }
    if (recommendationsSection) {
        recommendationsSection.style.display = 'none';
    }
    
    // Reset map
    if (typeof planoramaMap !== 'undefined' && planoramaMap) {
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
    
    // Scroll to top of chat
    scrollChatToBottom();
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

// Get recommendations (optionally with selected tags)
async function getRecommendations(selectedTags = null) {
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
    
    // Add message in chat if this is initial request
    if (!selectedTags) {
        addBotMessage("Perfecto! Tengo toda la información necesaria. Voy a buscar los mejores planes para ti... 🎯");
    } else {
        addBotMessage(`Excelente elección! Refinando recomendaciones con tus intereses seleccionados... 🎯`);
    }
    
    // Show loading box below chat (5 seconds only on first load, 2 seconds for tag refinement)
    const loadingTime = selectedTags ? 2000 : 5000;
    showRecommendationsLoading(true);
    
    try {
        const requestBody = {
            profile: state.profile,
            user_lat: state.userLat,
            user_lon: state.userLon
        };
        
        // Add selected tags if provided
        if (selectedTags && selectedTags.length > 0) {
            requestBody.selected_tags = selectedTags;
        }
        
        const response = await fetch(`${API_URL}/api/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            throw new Error('API request failed');
        }
        
        const data = await response.json();
        console.log('📊 Recommendations:', data);
        
        // Store recommendations
        state.alternativeRecommendations = data.events || [];
        state.hasShownAlternatives = false;
        
        // Wait for loading time, then display recommendations or tag selection
        setTimeout(() => {
            showRecommendationsLoading(false);
            
            // Check if we need tag selection
            if (data.needs_tag_selection && data.available_tags && data.available_tags.length > 0) {
                // Show tag selection UI in chat
                showTagSelectionUI(data.available_tags, data.count);
            } else if (data.ai_enabled && data.top_recommendation) {
                // Show recommendations
                displayRecommendationsOutsideChat(data.top_recommendation, data.has_alternatives);
            } else {
                addBotMessage('No encontré planes que coincidan con tus preferencias. Intenta ajustar tus criterios.');
            }
        }, loadingTime);
        
    } catch (error) {
        console.error('❌ Error getting recommendations:', error);
        showRecommendationsLoading(false);
        showError("Hubo un error al buscar recomendaciones. Por favor, intenta de nuevo.");
    }
}

// Display a single top recommendation as a bot message in the chat
function displayTopRecommendationInChat(topRecommendation, totalCount, hasAlternatives) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    // Intro phrase in English as requested
    const introDiv = document.createElement('div');
    introDiv.className = 'message message-bot';
    introDiv.innerHTML = `<strong>Planorama:</strong> Perfect, I have the required information, and the plan I recommend is the following:`;
    chatMessages.appendChild(introDiv);
    
    // Explanation and probability
    const probability = topRecommendation.ai_probability || 85.0;
    const explainDiv = document.createElement('div');
    explainDiv.className = 'message message-bot';
    const countText = (typeof totalCount === 'number' && totalCount > 0) ? ` · Este plan es <strong>1/${totalCount}</strong> de las posibles recomendaciones.` : '';
    explainDiv.innerHTML = `<strong>🤖 ${probability}% de compatibilidad.</strong> ${escapeHtml(topRecommendation.explanation || '')}${countText}`;
    chatMessages.appendChild(explainDiv);
    
    // Card
    const cardWrapper = document.createElement('div');
    cardWrapper.className = 'message message-bot';
    cardWrapper.innerHTML = createEventCardHTML(topRecommendation, true);
    chatMessages.appendChild(cardWrapper);
    
    // Add button to show alternatives if they exist
    if (hasAlternatives && state.alternativeRecommendations.length > 0) {
        const alternativesPrompt = document.createElement('div');
        alternativesPrompt.className = 'message message-bot';
        alternativesPrompt.id = 'alternatives-prompt';
        alternativesPrompt.innerHTML = `
            <strong>Planorama:</strong> ¿No es lo que buscabas? 
            <div style="margin-top: 12px;">
                <button id="show-alternatives-btn" class="btn-show-alternatives" style="padding: 10px 24px; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; border: none; border-radius: 12px; cursor: pointer; font-weight: 600; font-size: 1rem; transition: all 0.3s;">
                    Ver otras opciones (${state.alternativeRecommendations.length})
                </button>
            </div>
        `;
        chatMessages.appendChild(alternativesPrompt);
        
        // Add click handler for the button
        setTimeout(() => {
            const showAlternativesBtn = document.getElementById('show-alternatives-btn');
            if (showAlternativesBtn) {
                showAlternativesBtn.addEventListener('click', function() {
                    showAlternativeRecommendations();
                    // Disable button after clicking
                    showAlternativesBtn.disabled = true;
                    showAlternativesBtn.style.opacity = '0.6';
                    showAlternativesBtn.style.cursor = 'not-allowed';
                    showAlternativesBtn.textContent = '✓ Mostrando alternativas...';
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

// Show/hide recommendations loading box
function showRecommendationsLoading(show) {
    const loadingBox = document.getElementById('recommendations-loading');
    if (loadingBox) {
        loadingBox.style.display = show ? 'block' : 'none';
        if (show) {
            // Scroll to show the loading box
            setTimeout(() => {
                loadingBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 100);
        }
    }
}

// Utility: scroll chat to bottom
function scrollChatToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Display recommendations outside the chat
function displayRecommendationsOutsideChat(topRecommendation, hasAlternatives) {
    const recommendationsSection = document.getElementById('recommendations-section');
    const recommendationsContent = document.getElementById('recommendations-content');
    
    if (!recommendationsSection || !recommendationsContent) {
        console.error('Recommendations section not found');
        return;
    }
    
    // Clear previous content
    recommendationsContent.innerHTML = '';
    
    // Display top recommendation
    const probability = topRecommendation.ai_probability || 0;
    const topRecommendationHTML = `
        <div class="recommendation-item top-recommendation">
            <div class="recommendation-header-inline">
                <span class="recommendation-label">🏆 Recomendación Principal</span>
                <span class="recommendation-probability">${probability}%</span>
            </div>
            <p class="recommendation-explanation">${escapeHtml(topRecommendation.explanation || '')}</p>
            ${createEventCardHTML(topRecommendation, false)}
        </div>
    `;
    recommendationsContent.innerHTML += topRecommendationHTML;
    
    // Show alternatives button if there are alternatives
    if (hasAlternatives && state.alternativeRecommendations.length > 0) {
        const alternativesButtonHTML = `
            <button id="show-alternatives-btn-outside" class="btn-show-alternatives">
                ¿No es lo que buscabas? Ver otras ${state.alternativeRecommendations.length} opciones
            </button>
        `;
        recommendationsContent.innerHTML += alternativesButtonHTML;
        
        // Add event listener after a delay
        setTimeout(() => {
            const btn = document.getElementById('show-alternatives-btn-outside');
            if (btn) {
                btn.addEventListener('click', function() {
                    showAlternativeRecommendationsOutside();
                    btn.disabled = true;
                    btn.textContent = '✓ Mostrando todas las opciones...';
                });
            }
        }, 100);
    }
    
    // Show the recommendations section
    recommendationsSection.style.display = 'block';
    
    // Scroll to show recommendations
    setTimeout(() => {
        recommendationsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 300);
}

// Show alternative recommendations outside chat
function showAlternativeRecommendationsOutside() {
    if (state.hasShownAlternatives || state.alternativeRecommendations.length === 0) {
        console.log('No alternatives to show or already shown');
        return;
    }
    
    const recommendationsContent = document.getElementById('recommendations-content');
    if (!recommendationsContent) return;
    
    // Mark as shown
    state.hasShownAlternatives = true;
    
    // Display each alternative
    state.alternativeRecommendations.forEach((event, index) => {
        const probability = event.ai_probability || 0;
        const alternativeHTML = `
            <div class="recommendation-item">
                <div class="recommendation-header-inline">
                    <span class="recommendation-label">Opción ${index + 2}</span>
                    <span class="recommendation-probability">${probability}%</span>
                </div>
                <p class="recommendation-explanation">${escapeHtml(event.explanation || '')}</p>
                ${createEventCardHTML(event, false)}
            </div>
        `;
        recommendationsContent.innerHTML += alternativeHTML;
    });
    
    // Scroll to show new alternatives
    setTimeout(() => {
        const lastItem = recommendationsContent.lastElementChild;
        if (lastItem) {
            lastItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }, 100);
}

// Show tag selection UI in chat
function showTagSelectionUI(availableTags, totalCount) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    console.log('🏷️ Showing tag selection UI with', availableTags.length, 'tags');
    
    // Add intro message
    const introDiv = document.createElement('div');
    introDiv.className = 'message message-bot';
    introDiv.innerHTML = `<strong>Planorama:</strong> Veo que hay bastantes planes que coinciden con tus preferencias (${totalCount} opciones). ¿Cuáles de estas actividades te interesan más? Selecciona las que más te atraigan:`;
    chatMessages.appendChild(introDiv);
    
    // Create unique ID for this tag selection UI
    const tagsId = `tags-${Date.now()}`;
    const selectedTags = new Set();
    
    const tagsDiv = document.createElement('div');
    tagsDiv.className = 'message message-bot tags-message';
    tagsDiv.id = tagsId;
    tagsDiv.innerHTML = `
        <div class="tags-list">
            ${availableTags.map(tagObj => `
                <button type="button" class="tag-btn" data-tag="${escapeHtml(tagObj.tag)}">
                    ${escapeHtml(tagObj.tag)} <span class="tag-count">(${tagObj.count})</span>
                </button>
            `).join('')}
        </div>
        <div class="tags-actions" style="margin-top: 15px; text-align: center;">
            <button type="button" class="btn-confirm-tags" disabled style="padding: 12px 32px; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; border: none; border-radius: 12px; cursor: not-allowed; font-weight: 700; font-size: 1rem; opacity: 0.5; transition: all 0.3s;">
                ✓ Confirmar selección
            </button>
        </div>
    `;
    
    chatMessages.appendChild(tagsDiv);
    
    // Wait a tick to ensure DOM is ready before adding event listeners
    setTimeout(() => {
        // Get references to buttons
        const tagButtons = tagsDiv.querySelectorAll('.tag-btn');
        const confirmButton = tagsDiv.querySelector('.btn-confirm-tags');
        
        if (!confirmButton) {
            console.error('❌ Confirm button not found!');
            return;
        }
        
        if (tagButtons.length === 0) {
            console.error('❌ No tag buttons found!');
            return;
        }
        
        // Add click handlers for tag buttons (toggle selection)
        tagButtons.forEach(btn => {
            // Remove any existing listeners by cloning
            const newBtn = btn.cloneNode(true);
            btn.parentNode.replaceChild(newBtn, btn);
            
            newBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                
                const tag = newBtn.dataset.tag;
                
                // Toggle selection
                if (selectedTags.has(tag)) {
                    selectedTags.delete(tag);
                    newBtn.classList.remove('selected');
                } else {
                    selectedTags.add(tag);
                    newBtn.classList.add('selected');
                }
                
                // Enable/disable confirm button based on selections
                if (selectedTags.size > 0) {
                    confirmButton.disabled = false;
                    confirmButton.style.opacity = '1';
                    confirmButton.style.cursor = 'pointer';
                } else {
                    confirmButton.disabled = true;
                    confirmButton.style.opacity = '0.5';
                    confirmButton.style.cursor = 'not-allowed';
                }
                
                console.log('🏷️ Tags selected:', Array.from(selectedTags));
            }, true);
        });
        
        // Add click handler for confirm button
        confirmButton.addEventListener('click', async function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            if (selectedTags.size === 0) {
                return;
            }
            
            const tagsArray = Array.from(selectedTags);
            console.log('✅ Confirming tag selection:', tagsArray);
            
            // Disable all buttons to prevent double-clicking
            tagButtons.forEach(btn => {
                btn.disabled = true;
                btn.style.pointerEvents = 'none';
            });
            confirmButton.disabled = true;
            confirmButton.style.pointerEvents = 'none';
            confirmButton.textContent = '✓ Enviando...';
            
            // Remove the tags UI to prevent further interaction
            setTimeout(() => {
                tagsDiv.style.opacity = '0.6';
            }, 100);
            
            // Create a message showing selected tags
            const tagsText = tagsArray.join(', ');
            addUserMessage(`He seleccionado: ${tagsText}`);
            
            // Get recommendations with selected tags
            await getRecommendations(tagsArray);
        }, true);
    }, 50);
    
    // Scroll to bottom
    scrollChatToBottom();
    
    console.log('✅ Tag selection UI displayed successfully');
}

