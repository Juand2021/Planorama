// animations.js - GSAP animations for Planorama

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', () => {
    // Ensure content is visible first, then animate
    ensureContentVisible();
    
    // Small delay to ensure DOM is fully rendered
    setTimeout(() => {
        initAnimations();
    }, 100);
});

// Ensure all content is visible by default (fallback if animations fail)
function ensureContentVisible() {
    const elements = [
        '.tab-content.active',
        '.chat-panel',
        '.sidebar-panel',
        '.panel-card'
    ];
    
    elements.forEach(selector => {
        const els = document.querySelectorAll(selector);
        els.forEach(el => {
            if (el) {
                el.style.opacity = '1';
                el.style.visibility = 'visible';
            }
        });
    });
}

function initAnimations() {
    // Check if GSAP is available
    if (typeof gsap === 'undefined') {
        console.warn('GSAP not loaded, skipping animations');
        return;
    }

    // Register GSAP plugins
    if (typeof ScrollTrigger !== 'undefined') {
        gsap.registerPlugin(ScrollTrigger);
    }

    // Animate navigation header on load
    animateNavHeader();

    // Animate app layout on tab switch
    setupTabAnimations();

    // Animate about page sections on scroll
    animateAboutSections();

    // Add micro-interactions
    addMicroInteractions();
}

// Animate navigation header
function animateNavHeader() {
    if (typeof gsap === 'undefined') return;

    gsap.from('.nav-header', {
        y: -100,
        opacity: 0,
        duration: 0.8,
        ease: 'power3.out'
    });

    gsap.from('.nav-brand', {
        x: -30,
        opacity: 0,
        duration: 0.6,
        delay: 0.3,
        ease: 'power2.out'
    });

    gsap.from('.nav-tab', {
        y: -20,
        opacity: 0,
        duration: 0.5,
        stagger: 0.1,
        delay: 0.4,
        ease: 'power2.out'
    });

    gsap.from('.nav-status', {
        x: 30,
        opacity: 0,
        duration: 0.6,
        delay: 0.5,
        ease: 'power2.out'
    });
}

// Setup tab switching animations
function setupTabAnimations() {
    if (typeof gsap === 'undefined') return;

    const tabs = document.querySelectorAll('.nav-tab');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;
            
            // Animate out current tab
            const currentContent = document.querySelector('.tab-content.active');
            if (currentContent) {
                gsap.to(currentContent, {
                    opacity: 0,
                    y: 20,
                    duration: 0.3,
                    ease: 'power2.in',
                    onComplete: () => {
                        currentContent.classList.remove('active');
                        
                        // Animate in new tab
                        const newContent = document.getElementById(`tab-${targetTab}`);
                        if (newContent) {
                            newContent.classList.add('active');
                            gsap.fromTo(newContent,
                                { opacity: 0, y: 20 },
                                { opacity: 1, y: 0, duration: 0.4, ease: 'power2.out' }
                            );
                        }
                    }
                });
            }
        });
    });

    // Animate initial app load - SAFE VERSION
    // Only animate if elements exist and are visible
    const chatPanel = document.querySelector('.chat-panel .panel-card');
    if (chatPanel) {
        gsap.fromTo('.chat-panel .panel-card',
            { x: -30, opacity: 0 },
            { x: 0, opacity: 1, duration: 0.6, delay: 0.3, ease: 'power2.out' }
        );
    }

    const sidebarCards = document.querySelectorAll('.sidebar-panel .panel-card');
    if (sidebarCards.length > 0) {
        gsap.fromTo('.sidebar-panel .panel-card',
            { x: 30, opacity: 0 },
            { x: 0, opacity: 1, duration: 0.6, delay: 0.4, stagger: 0.1, ease: 'power2.out' }
        );
    }
}

// Animate about page sections on scroll
function animateAboutSections() {
    if (typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') return;

    // Hero section
    gsap.from('.about-hero', {
        scrollTrigger: {
            trigger: '.about-hero',
            start: 'top 80%',
            toggleActions: 'play none none reverse'
        },
        scale: 0.9,
        opacity: 0,
        duration: 1,
        ease: 'power3.out'
    });

    // Animate each section
    gsap.utils.toArray('.about-section').forEach((section, index) => {
        gsap.from(section, {
            scrollTrigger: {
                trigger: section,
                start: 'top 80%',
                toggleActions: 'play none none reverse'
            },
            y: 60,
            opacity: 0,
            duration: 0.8,
            delay: index * 0.1,
            ease: 'power3.out'
        });
    });

    // Animate step cards
    gsap.utils.toArray('.step-card').forEach((card, index) => {
        gsap.from(card, {
            scrollTrigger: {
                trigger: card,
                start: 'top 85%',
                toggleActions: 'play none none reverse'
            },
            y: 40,
            opacity: 0,
            duration: 0.6,
            delay: index * 0.15,
            ease: 'back.out(1.2)'
        });
    });

    // Animate feature cards
    gsap.utils.toArray('.feature-card').forEach((card, index) => {
        gsap.from(card, {
            scrollTrigger: {
                trigger: card,
                start: 'top 85%',
                toggleActions: 'play none none reverse'
            },
            scale: 0.8,
            opacity: 0,
            duration: 0.5,
            delay: index * 0.1,
            ease: 'back.out(1.4)'
        });
    });

    // Animate tech items
    gsap.utils.toArray('.tech-item').forEach((item, index) => {
        gsap.from(item, {
            scrollTrigger: {
                trigger: item,
                start: 'top 85%',
                toggleActions: 'play none none reverse'
            },
            y: 30,
            opacity: 0,
            duration: 0.5,
            delay: index * 0.1,
            ease: 'power2.out'
        });
    });

    // Animate CTA section
    gsap.from('.about-cta', {
        scrollTrigger: {
            trigger: '.about-cta',
            start: 'top 80%',
            toggleActions: 'play none none reverse'
        },
        scale: 0.95,
        opacity: 0,
        duration: 0.8,
        ease: 'power3.out'
    });
}

// Add micro-interactions
function addMicroInteractions() {
    if (typeof gsap === 'undefined') return;

    // Animate messages when they're added
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === 1 && node.classList.contains('message')) {
                        // Ensure message is visible first
                        node.style.opacity = '1';
                        node.style.visibility = 'visible';
                        
                        // Then animate from slightly above
                        gsap.fromTo(node,
                            { y: 15, opacity: 0 },
                            { y: 0, opacity: 1, duration: 0.3, ease: 'power2.out' }
                        );
                    }
                });
            });
        });

        observer.observe(chatMessages, { childList: true });
    }

    // Animate profile value changes
    const profileValues = document.querySelectorAll('.profile-value');
    profileValues.forEach(value => {
        const observer = new MutationObserver(() => {
            gsap.fromTo(value,
                { scale: 1.2, color: '#6366f1' },
                { scale: 1, color: 'inherit', duration: 0.5, ease: 'power2.out' }
            );
        });

        observer.observe(value, { childList: true, characterData: true, subtree: true });
    });

    // Button hover effects
    document.querySelectorAll('.btn-send, .btn-icon').forEach(btn => {
        btn.addEventListener('mouseenter', () => {
            gsap.to(btn, {
                scale: 1.05,
                duration: 0.3,
                ease: 'power2.out'
            });
        });

        btn.addEventListener('mouseleave', () => {
            gsap.to(btn, {
                scale: 1,
                duration: 0.3,
                ease: 'power2.out'
            });
        });
    });

    // Animate event cards when they appear
    const observeEventCards = () => {
        const eventCards = document.querySelectorAll('.event-card');
        eventCards.forEach((card, index) => {
            if (!card.dataset.animated) {
                gsap.from(card, {
                    scrollTrigger: {
                        trigger: card,
                        start: 'top 90%',
                        toggleActions: 'play none none reverse'
                    },
                    y: 40,
                    opacity: 0,
                    duration: 0.6,
                    delay: index * 0.1,
                    ease: 'power3.out'
                });
                card.dataset.animated = 'true';
            }
        });
    };

    // Initial check
    observeEventCards();

    // Watch for new event cards
    const cardObserver = new MutationObserver(observeEventCards);
    if (chatMessages) {
        cardObserver.observe(chatMessages, { childList: true, subtree: true });
    }

    // Parallax effect for gradient orbs
    document.addEventListener('mousemove', (e) => {
        const orbs = document.querySelectorAll('.gradient-orb');
        const moveX = (e.clientX - window.innerWidth / 2) * 0.01;
        const moveY = (e.clientY - window.innerHeight / 2) * 0.01;

        orbs.forEach((orb, index) => {
            const speed = (index + 1) * 0.5;
            gsap.to(orb, {
                x: moveX * speed,
                y: moveY * speed,
                duration: 1,
                ease: 'power2.out'
            });
        });
    });
}

// Animate input focus
document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chat-input');
    if (chatInput && typeof gsap !== 'undefined') {
        chatInput.addEventListener('focus', () => {
            gsap.to('.input-wrapper', {
                scale: 1.02,
                duration: 0.3,
                ease: 'power2.out'
            });
        });

        chatInput.addEventListener('blur', () => {
            gsap.to('.input-wrapper', {
                scale: 1,
                duration: 0.3,
                ease: 'power2.out'
            });
        });
    }
});

// Export function for switching to app tab (used in about page CTA)
window.switchToAppTab = function() {
    const appTab = document.querySelector('.nav-tab[data-tab="app"]');
    if (appTab) {
        appTab.click();
        
        // Scroll to top smoothly
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    }
};

// Animate categories when they appear
window.animateCategories = function() {
    if (typeof gsap === 'undefined') return;

    const categoryButtons = document.querySelectorAll('.category-btn');
    gsap.from(categoryButtons, {
        scale: 0.8,
        opacity: 0,
        duration: 0.4,
        stagger: 0.05,
        ease: 'back.out(1.5)'
    });
};

// Refresh ScrollTrigger when map is shown/hidden
window.refreshScrollTrigger = function() {
    if (typeof ScrollTrigger !== 'undefined') {
        ScrollTrigger.refresh();
    }
};

