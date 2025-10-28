// map.js - Map functionality using Leaflet

let planoramaMap = null;
let userMarker = null;

// Initialize map
function initializeMap() {
    // Check if map is already initialized
    if (planoramaMap) return;
    
    // Get map container
    const mapContainer = document.getElementById('map');
    if (!mapContainer) {
        console.error('Map container not found');
        return;
    }
    
    // Ensure map container is visible before initializing
    const mapSection = document.getElementById('map-section');
    if (mapSection && mapSection.style.display === 'none') {
        console.warn('Map section is hidden, aborting initialization');
        return;
    }
    
    // Default center: Bogotá (Zona T / Parque 93)
    const bogotaLat = 4.6738;
    const bogotaLon = -74.0530;
    
    // Create map with better zoom
    planoramaMap = L.map('map', {
        center: [bogotaLat, bogotaLon],
        zoom: 14,  // Closer zoom for better detail
        minZoom: 10,
        maxZoom: 19,
        zoomControl: true,
        attributionControl: true
    });
    
    // Add multiple tile layers with fallback for reliability
    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap',
        maxZoom: 19,
        minZoom: 11,
        tileSize: 256,
        crossOrigin: true
    });
    
    // Alternative tile server as fallback
    const esriLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
        attribution: '© Esri',
        maxZoom: 19,
        minZoom: 11
    });
    
    // Add the first layer
    osmLayer.addTo(planoramaMap);
    
    // Show loading message
    const loadingDiv = document.getElementById('map-status');
    if (loadingDiv) {
        loadingDiv.innerHTML = '⏳ Cargando mapa...';
        loadingDiv.className = 'map-status info-message';
    }
    
    // When tiles load, remove loading message
    planoramaMap.whenReady(() => {
        console.log('✅ Map tiles loaded');
        if (loadingDiv) {
            loadingDiv.innerHTML = '✅ Mapa cargado. Haz clic para marcar tu ubicación.';
            loadingDiv.className = 'map-status success';
            setTimeout(() => {
                loadingDiv.innerHTML = '';
                loadingDiv.className = 'map-status';
            }, 2000);
        }
    });
    
    // If tiles fail to load after 5 seconds, try alternative
    setTimeout(() => {
        if (!planoramaMap) return;
        // Check if tiles loaded
        const container = planoramaMap.getContainer();
        const tiles = container.querySelectorAll('.leaflet-tile-loaded');
        if (tiles.length < 4) {
            console.log('⚠️  OSM tiles failed, trying alternative');
            planoramaMap.removeLayer(osmLayer);
            esriLayer.addTo(planoramaMap);
        }
    }, 5000);
    
    // Set bounds to Bogotá area (allowing some movement outside)
    const bogotaBounds = [
        [4.3, -74.4],  // Southwest
        [5.0, -73.8]   // Northeast
    ];
    
    // Wait for tiles to load before enabling interaction
    setTimeout(() => {
        console.log('🗺️ Map tiles should be loaded');
    }, 1000);
    
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
    
    console.log(`📍 Map clicked at: ${lat.toFixed(4)}, ${lon.toFixed(4)}`);
    
    // Update state
    state.userLat = lat;
    state.userLon = lon;
    
    // Add/update marker
    addUserMarker(lat, lon);
    
    // Update status
    updateMapStatus(true);
    
    // Visual feedback
    console.log('✅ Location saved! Coordinates:', lat.toFixed(4), lon.toFixed(4));
    
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

