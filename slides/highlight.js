document.addEventListener('DOMContentLoaded', function() {
    Reveal.on('slidechanged', function(event) {
        const square1 = document.getElementById('highlight-square-1');
        const square2 = document.getElementById('highlight-square-2');
        
        // Clear all highlights
        square1.classList.remove('show');
        square2.classList.remove('show');
        
        // Highlight based on slide index
        if (event.indexh === 0) { // Example: Highlight first square on the first slide
            square1.classList.add('show');
        } else if (event.indexh === 1) { // Example: Highlight second square on the second slide
            square2.classList.add('show');
        }
    });
});