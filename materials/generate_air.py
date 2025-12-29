import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# Reference data (Cengel Table A-9, Air at 1 atm)
# Temperatures in °C
T_C_ref = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180,
                    200, 220, 240, 260, 280, 300, 320, 340,
                    360])  # °C

# Properties at these reference T (ρ in kg/m³, cp in kJ/kg·K, k in W/mK, μ in 1e-5 kg/ms)
rho_ref = np.array([1.293, 1.205, 1.128, 1.060, 1.000, 0.946, 0.898, 0.853, 0.813, 0.776,
                    0.742, 0.710, 0.681, 0.654, 0.629, 0.606, 0.584, 0.563, 0.544])
cp_ref  = np.array([1.003, 1.007, 1.009, 1.012, 1.015, 1.018, 1.020, 1.022, 1.025, 1.028,
                    1.030, 1.033, 1.035, 1.038, 1.040, 1.043, 1.045, 1.047, 1.050]) * 1000
k_ref   = np.array([0.02435, 0.02587, 0.02735, 0.02882, 0.03028, 0.03173, 0.03317, 0.03461, 0.03604, 0.03747,
                    0.03889, 0.04031, 0.04172, 0.04313, 0.04454, 0.04594, 0.04734, 0.04873, 0.05012])
mu_ref  = np.array([1.72, 1.81, 1.90, 1.98, 2.07, 2.16, 2.24, 2.32, 2.41, 2.49,
                    2.57, 2.65, 2.73, 2.81, 2.89, 2.97, 3.04, 3.12, 3.19]) * 1e-5

# Build interpolants (temperature input in K)
T_K_ref = T_C_ref + 273.15
interp_rho = PchipInterpolator(T_K_ref, rho_ref)
interp_cp  = PchipInterpolator(T_K_ref, cp_ref)
interp_k   = PchipInterpolator(T_K_ref, k_ref)
interp_mu  = PchipInterpolator(T_K_ref, mu_ref)

# Query points: 290–640 K in steps of 5
T_K = np.arange(290, 641, 5)
T_C = T_K - 273.15

rho = interp_rho(T_K)
cp = interp_cp(T_K)
k = interp_k(T_K)
mu = interp_mu(T_K)

# Derived properties
nu = mu / rho                # m²/s, kinematic viscosity
alpha = k / (rho * cp)       # m²/s, thermal diffusivity

# Build dataframe
df = pd.DataFrame({
    "T [K]": T_K,
    "T [°C]": T_C,
    "Density [kg/m3]": rho,
    "cp [J/kgK]": cp,
    "k [W/mK]": k,
    "mu [Pa s]": mu,
    "nu [m2/s]": nu,
    "alpha [m2/s]": alpha
})

# Save to CSV
df.to_csv("air_properties_Cengel.csv", index=False)

print(df.head())
