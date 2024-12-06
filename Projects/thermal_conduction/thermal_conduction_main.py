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