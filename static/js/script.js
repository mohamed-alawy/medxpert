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

    // Split text for animated hero heading
    function splitTextIntoSpans(selector) {
        const elements = document.querySelectorAll(selector);
        
        elements.forEach(element => {
            const text = element.textContent;
            const words = text.split(' ');
            
            element.innerHTML = words.map(word => {
                return `<span style="animation-delay: ${Math.random() * 0.5}s">${word} </span>`;
            }).join('');
        });
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
    
    // Initialize hero animations
    splitTextIntoSpans('.hero-text-reveal h1');
    
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
});