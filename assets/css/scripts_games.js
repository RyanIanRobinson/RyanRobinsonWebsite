// Function to dynamically load external HTML files (header & footer)
function loadComponent(id, file) {
    fetch(file)
        .then(response => response.text())
        .then(data => document.getElementById(id).innerHTML = data)
        .catch(error => console.error(`Error loading ${file}:`, error));
}

// Load header and footer when the DOM is fully loaded
document.addEventListener("DOMContentLoaded", function () {
    loadComponent("header-container", "/includes/header.html"); // Adjust path as necessary
    loadComponent("footer-container", "/includes/footer.html"); // Adjust path as necessary
});
