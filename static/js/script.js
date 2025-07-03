// Animated counters
document.addEventListener("DOMContentLoaded", function() {
    function animateCounter(el, end, duration=1600) {
        let start = 0, startTime = null;
        function animate(time) {
            if (!startTime) startTime = time;
            const progress = Math.min((time - startTime) / duration, 1);
            el.textContent = Math.floor(progress * (end - start) + start).toLocaleString();
            if (progress < 1) requestAnimationFrame(animate);
            else el.textContent = end.toLocaleString();
        }
        requestAnimationFrame(animate);
    }
    
    // Handle counter animations
    function handleCounters() {
        document.querySelectorAll('.counter:not(.counted)').forEach(el => {
            const rect = el.getBoundingClientRect();
            if (rect.top < window.innerHeight - 50) {
                el.classList.add('counted');
                animateCounter(el, parseInt(el.dataset.count));
            }
        });
    }

    // Simple smooth scroll for discover button
    function initSmoothScrolling() {
        const discoverBtn = document.querySelector('.discover-btn');
        if (discoverBtn) {
            discoverBtn.addEventListener('click', function(e) {
                e.preventDefault();
                const target = document.querySelector('#models');
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        }
    }
    
    // Staggered animations
    function handleStaggerItems() {
        document.querySelectorAll('.stagger-item:not(.animated)').forEach((el, index) => {
            const rect = el.getBoundingClientRect();
            if (rect.top < window.innerHeight - 50) {
                // Add a delay based on the index
                setTimeout(() => {
                    el.classList.add('animated');
                }, index * 150); // 150ms stagger for each item
            }
        });
    }
    
    // Parallax effect on scroll
    function handleParallax() {
        document.querySelectorAll('.parallax-element').forEach(el => {
            const scrollPosition = window.scrollY;
            const speed = el.dataset.speed || 0.1;
            el.style.transform = `translateY(${scrollPosition * speed}px)`;
        });
    }
    
    // Image reveal animation
    function handleImageReveal() {
        document.querySelectorAll('.image-reveal:not(.animated)').forEach(el => {
            const rect = el.getBoundingClientRect();
            if (rect.top < window.innerHeight - 50) {
                el.classList.add('animated');
            }
        });
    }

    // Animate .animate-reveal on scroll
    function revealOnScroll() {
        document.querySelectorAll('.animate-reveal:not(.animated)').forEach(el => {
            const rect = el.getBoundingClientRect();
            if (rect.top < window.innerHeight - 75) {
                el.classList.add('animated');
            }
        });
        
        // Handle other animations
        handleCounters();
        handleStaggerItems();
        handleImageReveal();
    }
    
    // Initialize smooth scrolling
    initSmoothScrolling();
    
    // Add scroll event listeners
    window.addEventListener('scroll', revealOnScroll);
    window.addEventListener('scroll', handleParallax);
    
    // Initial check for animations on page load
    setTimeout(() => {
        revealOnScroll();
        document.querySelectorAll('.animate-fadein, .animate-fadein-delay, .animate-fadein-delay2, .animate-fadeup')
            .forEach(el => el.style.opacity = 1);
    }, 100);

    // Testimonials carousel auto-slide
    const carousel = document.getElementById('testimonialCarousel');
    if (carousel) {
        let bsCarousel = new bootstrap.Carousel(carousel, { interval: 5000, ride: 'carousel' });
    }

    // Model navigation functions
    function showModel(index) {
        const modelCards = document.querySelectorAll('.model-detail-card');
        const dots = document.querySelectorAll('.indicator-dot');
        const modelNames = ['Brain Tumor Detection', 'Chest X-ray Analysis', 'Skin Cancer Detection', 'Fracture Detection'];
        const currentModelInfo = document.getElementById('currentModelInfo');
        
        // Hide all models
        modelCards.forEach(card => card.classList.remove('active'));
        dots.forEach(dot => dot.classList.remove('active'));
        
        // Show selected model
        modelCards[index].classList.add('active');
        dots[index].classList.add('active');
        
        // Update model info text
        currentModelInfo.textContent = `${modelNames[index]} (${index + 1} of ${modelNames.length})`;
        
        // Update current model index
        currentModel = index;
        
        // Enable/disable navigation buttons
        document.getElementById('prevBtn').disabled = currentModel === 0;
        document.getElementById('nextBtn').disabled = currentModel === modelNames.length - 1;
    }

    function nextModel() {
        const modelCards = document.querySelectorAll('.model-detail-card');
        if (currentModel < modelCards.length - 1) {
            showModel(currentModel + 1);
        }
    }

    function previousModel() {
        if (currentModel > 0) {
            showModel(currentModel - 1);
        }
    }

    // Initialize model navigation
    showModel(0);
});