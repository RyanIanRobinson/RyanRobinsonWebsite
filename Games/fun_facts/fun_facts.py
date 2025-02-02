from flask import Flask, jsonify
from flask_cors import CORS
try:
    import funfact
    use_funfact = True
except ImportError:
    import random
    use_funfact = False

app = Flask(__name__)
CORS(app)  # Allow requests from different origins

# Hardcoded fallback fun facts
fun_facts = [
    "Bananas are berries, but strawberries are not.",
    "A day on Venus is longer than a year on Venus.",
    "Honey never spoils. Archaeologists have found edible honey in ancient tombs!",
    "Octopuses have three hearts.",
    "Wombat poop is cube-shaped."
]

@app.route('/funfact', methods=['GET'])
def get_fun_fact():
    if use_funfact:
        fact = funfact.get()  # Get fact from funfact library
    else:
        fact = random.choice(fun_facts)  # Use hardcoded facts if funfact is unavailable
    return jsonify({"fact": fact})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
