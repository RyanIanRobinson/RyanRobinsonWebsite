# Aluminium properties
rho_Al = 2700          # Density of Aluminium (kg/m³)
c_Al = 900             # Specific heat capacity (J/kg·K)
E_Al = 70e9            # Young's modulus (Pa)
lambda_Al = 2.5e-8     # Mean free path of phonons (m)

# Air properties
rho_air = 1.225        # Density of air (kg/m³)
c_air = 1005           # Specific heat capacity (J/kg·K)
k_air = 0.0257         # Thermal conductivity (W/m·K)
mu_air = 1.81e-5       # Dynamic viscosity (Pa·s)
Pr = 0.71              # Prandtl number

# Chaotic versus orderly airflow
C_turbulent = 0.037  # Empirical constant for turbulent flow (Chaotic)
C_laminar = 0.664  # Empirical constant for laminar flow (Orderly)

# Duck geometry
R_torso = 0.1          # Radius of the torso (m)
L_torso = 0.25         # Length of the torso (m)
A_torso = 2 * 3.1416 * R_torso * L_torso  # Surface area of the torso (m²)
V_torso = 3.1416 * R_torso**2 * L_torso  # Volume of the torso (m³)
L_torso_char = A_torso / (2 * R_torso + L_torso) # Characteristic Length

L_wing = 0.3           # Length of the wing (m)
W_wing = 0.1           # Width of the wing (m)
H_wing = 0.01          # Height of the wing (m)
A_wing = L_wing * W_wing  # Surface area of one wing (m²)
V_wing = L_wing * W_wing * H_wing # Colume of the wing (m³)
L_wing_char = A_wing / (W_wing + L_wing) # Characteristic Length

# Environmental conditions
T_air = 15             # Air temperature (°C)
v_air = 5              # Air velocity (m/s)

# Initial temperature of the duck
T_initial = 30         # °C

# Convergence criteria
time_step = 1         # Iterations for plot frequency (s)
time_seconds = 86400  # Duration of cooling (s)
