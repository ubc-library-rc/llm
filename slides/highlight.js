document.addEventListener('DOMContentLoaded', function() {
    Reveal.on('slidechanged', function(event) {
        if (event.indexh === YOUR_SLIDE_INDEX) {
            document.getElementById('highlight-square').classList.add('active');
        } else {
            document.getElementById('highlight-square').classList.remove('active');
        }
    });
});