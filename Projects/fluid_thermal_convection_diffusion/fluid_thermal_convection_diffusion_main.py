import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
alpha = 1.43e-7  # Thermal diffusivity of water (m^2/s)
k = 0.2          # Thermal conductivity of the bottle material (W/m·K)
rho = 1000        # Density of water (kg/m^3)
cp = 4186         # Specific heat capacity of water (J/kg·K)
h = 10            # Convective heat transfer coefficient (W/m^2·K)
T_air = -18 + 273.15  # Freezer temperature in K
T_init = 20 + 273.15  # Initial temperature of the water (in K)
T_freeze = 273.15  # Freezing point of water in K
R = 0.05  # Radius of the bottle (m)
height = 0.2  # Height of the bottle (m)
L = 0.6  # Volume of the bottle (L)

# Grid Setup
Nr = 50  # Number of radial points
Nz = 50  # Number of vertical points (axial direction)
dr = R / Nr
dz = height / Nz
dt = 0.5  # Time step (seconds)
total_time = 600  # Total time to simulate (seconds)
steps = int(total_time / dt)  # Number of time steps

# Create the grid for r (radius) and z (height)
r = np.linspace(0, R, Nr)
z = np.linspace(0, height, Nz)
T = np.full((Nr, Nz), T_init)  # Initial temperature of water
T_new = np.copy(T)  # To store the updated temperature

# Define a function for velocity (simplified)
def get_velocity(T, beta=1e-4):
    """
    Compute the velocity field based on temperature gradient.
    Here, beta is a simplified constant for thermal expansion.
    """
    velocity = np.zeros_like(T)
    for i in range(1, Nr-1):
        for j in range(1, Nz-1):
            # Approximate velocity based on temperature difference
            temp_grad = T[i, j] - T[i, j-1]  # Simplified gradient in the vertical direction
            velocity[i, j] = -beta * temp_grad  # Use Boussinesq approximation
    return velocity

# Time-stepping loop
for step in range(steps):
    # Apply the heat diffusion equation with convection and freezing condition
    velocity = get_velocity(T)  # Calculate velocity due to convection
    
    for i in range(1, Nr-1):
        for j in range(1, Nz-1):
            # Finite difference for heat diffusion and convection
            dT_dr = (T[i+1, j] - T[i-1, j]) / (2 * dr)
            dT_dz = (T[i, j+1] - T[i, j-1]) / (2 * dz)
            T_new[i, j] = T[i, j] + dt * (alpha * (dT_dr / dr + dT_dz / dz) + velocity[i, j] * dT_dz)
    
    # Boundary conditions (symmetry at center and convection at the surface)
    T_new[0, :] = T_new[1, :]  # Symmetry condition at the center (r = 0)
    T_new[:, -1] = T_new[:, -2]  # Apply convection boundary at top (z = height)
    
    # Freezing condition: if water reaches 0°C, it freezes and releases latent heat
    T_new[T_new <= T_freeze] = T_freeze  # Set water to freezing point
    
    # Update the temperature grid
    T = np.copy(T_new)

    # Plot heatmap every 10 seconds
    if step % 20 == 0:  # 20 steps = 10 seconds
        plt.imshow(T, extent=[0, R, 0, height], origin='lower', aspect='auto', cmap='hot')
        plt.colorbar(label='Temperature (K)')
        plt.title(f'Temperature Profile at {step*dt}s')
        plt.xlabel('Radius (m)')
        plt.ylabel('Height (m)')
        plt.pause(0.1)

# Final heatmap at the end of the simulation
plt.imshow(T, extent=[0, R, 0, height], origin='lower', aspect='auto', cmap='hot')
plt.colorbar(label='Temperature (K)')
plt.title('Final Temperature Profile')
plt.xlabel('Radius (m)')
plt.ylabel('Height (m)')
plt.show()
