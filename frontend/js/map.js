// map.js - Map functionality using Leaflet

let planoramaMap = null;
let userMarker = null;

// Initialize map
function initializeMap() {
    // Check if map is already initialized
    if (planoramaMap) return;
    
    // Default center: Bogotá
    const bogotaLat = 4.6533;
    const bogotaLon = -74.0836;
    
    // Create map
    planoramaMap = L.map('map').setView([bogotaLat, bogotaLon], 12);
    
    // Add OpenStreetMap tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(planoramaMap);
    
    // Add click handler
    planoramaMap.on('click', onMapClick);
    
    // If we already have user location, add marker
    if (state.userLat && state.userLon) {
        addUserMarker(state.userLat, state.userLon);
    }
    
    console.log('🗺️ Map initialized');
}

// Handle map click
function onMapClick(e) {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;
    
    console.log(`📍 Map clicked: ${lat}, ${lon}`);
    
    // Update state
    state.userLat = lat;
    state.userLon = lon;
    
    // Add/update marker
    addUserMarker(lat, lon);
    
    // Update status
    updateMapStatus(true);
    
    // If profile is complete, get new recommendations with location
    if (isProfileComplete()) {
        getRecommendations();
    }
}

// Add or update user marker
function addUserMarker(lat, lon) {
    // Remove existing marker if any
    if (userMarker) {
        planoramaMap.removeLayer(userMarker);
    }
    
    // Create custom icon (pin)
    const customIcon = L.icon({
        iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
        popupAnchor: [1, -34],
        shadowSize: [41, 41]
    });
    
    // Add marker
    userMarker = L.marker([lat, lon], { icon: customIcon })
        .addTo(planoramaMap)
        .bindPopup('📍 Tu ubicación')
        .openPopup();
    
    // Center map on marker
    planoramaMap.setView([lat, lon], 13);
}

// Update map status message
function updateMapStatus(success) {
    const statusDiv = document.getElementById('map-status');
    
    if (success) {
        statusDiv.className = 'map-status success';
        statusDiv.textContent = '✅ Zona marcada. Usaré tu ubicación para priorizar la cercanía.';
    } else {
        statusDiv.className = 'map-status error';
        statusDiv.textContent = '⚠️ Para continuar con cercanía, haz clic en el mapa y marca tu zona.';
    }
}

// Add events to map
function addEventsToMap(events) {
    if (!planoramaMap || !events || events.length === 0) return;
    
    // Create markers for events
    events.forEach(event => {
        if (event.lat && event.lon) {
            const eventIcon = L.icon({
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            });
            
            const marker = L.marker([event.lat, event.lon], { icon: eventIcon })
                .addTo(planoramaMap)
                .bindPopup(`
                    <strong>${event.title}</strong><br>
                    ${event.venue_name || ''}<br>
                    ${event.date_start || ''}
                `);
        }
    });
}

// Try to get user's location using browser geolocation
function tryGetUserLocation() {
    if (!navigator.geolocation) {
        console.log('Geolocation not supported');
        return;
    }
    
    navigator.geolocation.getCurrentPosition(
        (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            
            console.log(`📍 User location detected: ${lat}, ${lon}`);
            
            // Update state
            state.userLat = lat;
            state.userLon = lon;
            
            // Add marker
            if (planoramaMap) {
                addUserMarker(lat, lon);
                updateMapStatus(true);
            }
        },
        (error) => {
            console.log('Could not get user location:', error.message);
        }
    );
}

