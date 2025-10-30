// chat.js - Chat functionality

// Initialize chat
function initializeChat() {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const userText = chatInput.value.trim();
        if (!userText) return;
        
        // Disable input while processing
        chatInput.disabled = true;
        sendBtn.disabled = true;
        
        // Add user message to chat
        addUserMessage(userText);
        
        // Clear input
        chatInput.value = '';
        
        // Send to API
        await sendChatMessage(userText);
        
        // Re-enable input
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
    });
}

// Add user message to chat
function addUserMessage(text) {
    const chatMessages = document.getElementById('chat-messages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message message-user';
    messageDiv.innerHTML = `<strong>Tú:</strong> ${escapeHtml(text)}`;
    
    chatMessages.appendChild(messageDiv);
    scrollChatToBottom();
    
    // Save to history
    state.chatHistory.push({ role: 'user', text: text });
}

// Add bot message to chat
function addBotMessage(text) {
    const chatMessages = document.getElementById('chat-messages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message message-bot';
    messageDiv.innerHTML = `<strong>Planorama:</strong> ${escapeHtml(text)}`;
    
    chatMessages.appendChild(messageDiv);
    scrollChatToBottom();
    
    // Save to history
    state.chatHistory.push({ role: 'bot', text: text });
}

// Send chat message to API
async function sendChatMessage(text) {
    try {
        const response = await fetch(`${API_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                profile: state.profile
            })
        });
        
        if (!response.ok) {
            throw new Error('API request failed');
        }
        
        const data = await response.json();
        console.log('💬 Chat response:', data);
        console.log('📋 show_categories flag:', data.show_categories);
        console.log('📋 profile categorias:', data.profile?.categorias);
        
        // Update profile
        state.profile = data.profile;
        state.ready = data.done;
        
        // Update profile display
        updateProfileDisplay();
        
        // Update map visibility
        updateMapVisibility();
        
        // Add bot reply
        addBotMessage(data.reply);
        
        // Show categories if requested OR if no categories in profile and profile not complete
        const hasCategories = data.profile?.categorias && Array.isArray(data.profile.categorias) && data.profile.categorias.length > 0;
        const shouldShowCategories = data.show_categories || (!data.done && !hasCategories);
        
        console.log('🔍 Category check:', {
            show_categories_flag: data.show_categories,
            done: data.done,
            hasCategories: hasCategories,
            categorias: data.profile?.categorias,
            shouldShowCategories: shouldShowCategories
        });
        
        if (shouldShowCategories) {
            console.log('✅ Showing categories list...');
            // Add a small delay to ensure the bot message is rendered first
            setTimeout(async () => {
                await showCategoriesList();
            }, 100);
        } else {
            console.log('❌ NOT showing categories');
        }
        
        // If profile is complete, get recommendations
        if (data.done && isProfileComplete()) {
            await getRecommendations();
        }
        
    } catch (error) {
        console.error('❌ Error sending message:', error);
        addBotMessage("Lo siento, hubo un error al procesar tu mensaje. Por favor, intenta de nuevo.");
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Format message with markdown-like syntax (optional enhancement)
function formatMessage(text) {
    // Bold: **text** -> <strong>text</strong>
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Italic: *text* -> <em>text</em>
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Line breaks
    text = text.replace(/\n/g, '<br>');
    
    return text;
}

// Load and show categories list
async function showCategoriesList() {
    console.log('🔄 showCategoriesList() called');
    try {
        const response = await fetch(`${API_URL}/api/categories`);
        if (!response.ok) {
            throw new Error('Failed to load categories');
        }
        
        const data = await response.json();
        console.log('📋 Categories loaded:', data);
        
        if (!data.categories || data.categories.length === 0) {
            console.warn('⚠️ No categories returned from API');
            addBotMessage("Lo siento, no hay categorías disponibles en este momento.");
            return;
        }
        
        // Create categories list UI
        const chatMessages = document.getElementById('chat-messages');
        if (!chatMessages) {
            console.error('❌ chat-messages element not found');
            return;
        }
        
        // Create unique ID for this categories selection UI
        const categoriesId = `categories-${Date.now()}`;
        const selectedCategories = new Set();
        
        const categoriesDiv = document.createElement('div');
        categoriesDiv.className = 'message message-bot categories-message';
        categoriesDiv.id = categoriesId;
        categoriesDiv.innerHTML = `
            <strong>Planorama:</strong> Selecciona las categorías que te interesan (puedes elegir varias):
            <div class="categories-list">
                ${data.categories.map(cat => `
                    <button type="button" class="category-btn" data-category="${escapeHtml(cat.name)}">
                        ${escapeHtml(cat.name)} <span class="category-count">(${cat.count} eventos)</span>
                    </button>
                `).join('')}
            </div>
            <div class="categories-actions" style="margin-top: 15px; text-align: center;">
                <button type="button" class="btn-confirm-categories" disabled style="padding: 10px 24px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: not-allowed; font-weight: 600; opacity: 0.5; transition: all 0.3s;">
                    ✓ Confirmar selección
                </button>
            </div>
        `;
        
        chatMessages.appendChild(categoriesDiv);
        
        // Wait a tick to ensure DOM is ready before adding event listeners
        setTimeout(() => {
            // Get references to buttons
            const categoryButtons = categoriesDiv.querySelectorAll('.category-btn');
            const confirmButton = categoriesDiv.querySelector('.btn-confirm-categories');
            
            if (!confirmButton) {
                console.error('❌ Confirm button not found!');
                return;
            }
            
            if (categoryButtons.length === 0) {
                console.error('❌ No category buttons found!');
                return;
            }
            
            // Add click handlers for category buttons (toggle selection)
            // Use capture phase and stopPropagation to prevent any other handlers
            categoryButtons.forEach(btn => {
                // Remove any existing listeners by cloning
                const newBtn = btn.cloneNode(true);
                btn.parentNode.replaceChild(newBtn, btn);
                
                newBtn.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    e.stopImmediatePropagation();
                    
                    const category = newBtn.dataset.category;
                    
                    // Toggle selection
                    if (selectedCategories.has(category)) {
                        selectedCategories.delete(category);
                        newBtn.classList.remove('selected');
                    } else {
                        selectedCategories.add(category);
                        newBtn.classList.add('selected');
                    }
                    
                    // Enable/disable confirm button based on selections
                    if (selectedCategories.size > 0) {
                        confirmButton.disabled = false;
                        confirmButton.style.opacity = '1';
                        confirmButton.style.cursor = 'pointer';
                    } else {
                        confirmButton.disabled = true;
                        confirmButton.style.opacity = '0.5';
                        confirmButton.style.cursor = 'not-allowed';
                    }
                    
                    console.log('🎯 Categories selected:', Array.from(selectedCategories));
                }, true); // Use capture phase
            });
            
            // Add click handler for confirm button
            confirmButton.addEventListener('click', async function(e) {
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();
                
                if (selectedCategories.size === 0) {
                    return;
                }
                
                const categoriesArray = Array.from(selectedCategories);
                console.log('✅ Confirming category selection:', categoriesArray);
                
                // Disable all buttons to prevent double-clicking
                categoryButtons.forEach(btn => {
                    btn.disabled = true;
                    btn.style.pointerEvents = 'none';
                });
                confirmButton.disabled = true;
                confirmButton.style.pointerEvents = 'none';
                confirmButton.textContent = '✓ Enviando...';
                
                // Remove the categories UI to prevent further interaction
                setTimeout(() => {
                    categoriesDiv.style.opacity = '0.6';
                }, 100);
                
                // Create a message showing selected categories
                const categoriesText = categoriesArray.join(', ');
                await sendChatMessage(`Quiero eventos de: ${categoriesText}`);
            }, true); // Use capture phase
        }, 50);
        
        // Scroll to bottom - check if function exists
        if (typeof scrollChatToBottom === 'function') {
            scrollChatToBottom();
        } else {
            // Fallback scroll
            setTimeout(() => {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }, 100);
        }
        
        console.log('✅ Categories list displayed successfully');
        
    } catch (error) {
        console.error('❌ Error loading categories:', error);
        addBotMessage("Lo siento, no pude cargar las categorías. Por favor, intenta de nuevo.");
    }
}

