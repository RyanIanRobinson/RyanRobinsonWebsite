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

// Hardcoded script contents
const scriptContents = {
    "thermal_conduction_functions": 
`
import math
from thermal_conduction_params import *

def calculate_thermal_conductivity(rho, c, E, lambda_):
    """Calculate thermal conductivity of Aluminium using a molecular model."""
    v_s = math.sqrt(E / rho)  # Speed of sound in Aluminium
    k_Al = (1 / 3) * (rho * c) * v_s**2 * lambda_
    return k_Al

def calculate_reynolds_number(v, L, rho, mu):
    """Calculate Reynolds number."""
    return (rho * v * L) / mu

def calculate_nusselt_number(Re, flow_type):
    """Calculate Nusselt number based on the flow type."""
    if flow_type == "laminar":
        return 0.664 * Re**0.5 * Pr**(1 / 3)  # Laminar flow correlation
    else:  # Turbulent flow
        return 0.037 * Re**0.8 * Pr**(1 / 3)

def determine_flow_type(v):
    """Determine if the flow is laminar or turbulent based on air velocity."""
    if v < 2:
        return "laminar"
    else:
        return "turbulent"

def calculate_convective_coefficient(v, L_char):
    """Calculate convective heat transfer coefficient based on characteristic length."""
    Re = calculate_reynolds_number(v, L_char, rho_air, mu_air)
    flow_type = determine_flow_type(v)
    Nu = calculate_nusselt_number(Re, flow_type)
    h = (Nu * k_air) / L_char
    return h, Re, Nu, flow_type

def heat_transfer_iteration(T_torso, T_wing, h_torso, h_wing, A_torso, A_wing, time_step):
    """Perform a single iteration of the heat transfer calculation using exponential cooling."""
    # Calculate the rate of cooling for torso and wing (using Newton's Law of Cooling)
    dT_torso = -h_torso * A_torso * (T_torso - T_air) * time_step / (rho_Al * c_Al * A_torso)
    T_torso_new = T_torso + dT_torso

    dT_wing = -h_wing * A_wing * (T_wing - T_air) * time_step / (rho_Al * c_Al * A_wing)
    T_wing_new = T_wing + dT_wing

    return T_torso_new, T_wing_new

def log_flow_characteristics(Re, Nu, flow_type, label):
    """Log Reynolds number, Nusselt number, and flow type for debugging."""
    print(f"{label} Flow Characteristics:")
    print(f"  Reynolds Number (Re): {Re:.2f}")
    print(f"  Nusselt Number (Nu): {Nu:.2f}")
    print(f"  Flow Type: {flow_type.capitalize()}\n")

def print_assumptions():
    """Print the assumptions used in the calculations."""
    print("Assumptions:")
    print(f"  Torso radius: {R_torso} m, length: {L_torso} m, area: {A_torso:.3f} m², volume: {V_torso:.3f} m³, mass: {rho_Al*V_torso:.3f} kg")
    print(f"  Wing length: {L_wing} m, width: {W_wing} m, area: {A_wing:.3f} m², mass: {rho_Al * V_wing:.3f} kg")
    print(f"  Air velocity: {v_air} m/s")
    print(f"  Air temperature: {T_air} °C")
    print(f"  Initial duck temperature: {T_initial} °C")
    print(f"  Aluminium properties: Density={rho_Al} kg/m³, Specific heat={c_Al} J/kg·K\n")

# Perform cooling simulation and collect temperature data over time
def simulate_cooling(T_torso, T_wing, time_seconds, time_step):
    """Simulate cooling and return time and temperature data for plotting."""
    time_points = []
    torso_temps = []
    wing_temps = []

    # Calculate convective coefficients and log characteristics
    h_torso, Re_torso, Nu_torso, flow_torso = calculate_convective_coefficient(v_air, L_torso_char)
    log_flow_characteristics(Re_torso, Nu_torso, flow_torso, "Torso")

    h_wing, Re_wing, Nu_wing, flow_wing = calculate_convective_coefficient(v_air, L_wing_char)
    log_flow_characteristics(Re_wing, Nu_wing, flow_wing, "Wing")

    for t in range(0, time_seconds, time_step):
        time_points.append(t)
        torso_temps.append(T_torso)
        wing_temps.append(T_wing)

        # Update temperatures for the next iteration
        T_torso, T_wing = heat_transfer_iteration(
            T_torso, T_wing, h_torso, h_wing, A_torso, A_wing, time_step
        )

    return time_points, torso_temps, wing_temps
`
,
    "thermal_conduction_main": 
`
import matplotlib.pyplot as plt
from thermal_conduction_functions import *
from thermal_conduction_params import *

# Initial temperatures (in °C)
T_torso = T_initial
T_wing = T_initial

# Print assumptions
print_assumptions()

print("Starting Cooling Simulation...\n")
time_points, torso_temps, wing_temps = simulate_cooling(T_torso, T_wing, time_seconds, time_step)

# Plot the results
import matplotlib.pyplot as plt
plt.plot(time_points, torso_temps, label="Torso Temperature")
plt.plot(time_points, wing_temps, label="Wing Temperature")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Cooling of Aluminium Duck Over Time")
plt.legend()
plt.grid()
plt.show()
`
};

// Function to display script content
function showCode(scriptName) {
    const codeDisplay = document.getElementById('code-content');
    codeDisplay.textContent = scriptContents[scriptName] || "Error: Script content not found.";
}
