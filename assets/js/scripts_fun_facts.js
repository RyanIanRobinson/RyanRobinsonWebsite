// Function to dynamically load external HTML files (header & footer)
function loadComponent(id, file) {
    fetch(file)
        .then(response => response.text())
        .then(data => document.getElementById(id).innerHTML = data)
        .catch(error => console.error(`Error loading ${file}:`, error));
}

// Load header and footer
document.addEventListener("DOMContentLoaded", function () {
    loadComponent("header-container", "/includes/header.html");
    loadComponent("footer-container", "/includes/footer.html");
});

const funFacts = [
    "Bananas are berries, but strawberries are not.",
    "A day on Venus is longer than a year on Venus.",
    "Honey never spoils. Archaeologists have found edible honey in ancient tombs!",
    "Octopuses have three hearts.",
    "Wombat poop is cube-shaped.",
    "There are more stars in the universe than grains of sand on Earth.",
    "Butterflies can taste with their feet.",
    "Sharks existed before trees.",
    "A cloud can weigh over a million pounds.",
    "A bolt of lightning is five times hotter than the sun's surface.",
    "Sloths can hold their breath longer than dolphins.",
    "Water can boil and freeze at the same time under the right conditions.",
    "There's a species of jellyfish that can live forever.",
    "Your brain generates enough electricity to power a small light bulb.",
    "The Eiffel Tower grows taller in the summer due to heat expansion.",
    "Some turtles can breathe through their butts.",
    "Cows have best friends and get stressed when separated.",
    "A group of flamingos is called a 'flamboyance.'",
    "An octopus has nine brains—one in its head and one in each arm.",
    "The world's smallest reptile was discovered in 2021 and can fit on a fingertip."
];

function fetchFunFact() {
    const randomIndex = Math.floor(Math.random() * funFacts.length);
    document.getElementById("funFactBox").textContent = funFacts[randomIndex];
}

// Fetch a fun fact on page load
document.addEventListener("DOMContentLoaded", fetchFunFact);