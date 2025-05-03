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
    document.querySelectorAll('.counter').forEach(el => {
        animateCounter(el, parseInt(el.dataset.count));
    });

    // Animate .animate-reveal on scroll
    function revealOnScroll() {
        document.querySelectorAll('.animate-reveal:not(.animated)').forEach(el => {
            const rect = el.getBoundingClientRect();
            if (rect.top < window.innerHeight - 75) {
                el.classList.add('animated');
            }
        });
    }
    window.addEventListener('scroll', revealOnScroll);
    revealOnScroll();

    // Animate fadein/fadeup
    setTimeout(() => {
        document.querySelectorAll('.animate-fadein, .animate-fadein-delay, .animate-fadein-delay2, .animate-fadeup')
            .forEach(el => el.style.opacity = 1);
    }, 100);

    // Testimonials carousel auto-slide
    const carousel = document.getElementById('testimonialCarousel');
    if (carousel) {
        let bsCarousel = new bootstrap.Carousel(carousel, { interval: 5000, ride: 'carousel' });
    }
});