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
        
        // Update profile
        state.profile = data.profile;
        state.ready = data.done;
        
        // Update profile display
        updateProfileDisplay();
        
        // Update map visibility
        updateMapVisibility();
        
        // Add bot reply
        addBotMessage(data.reply);
        
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

