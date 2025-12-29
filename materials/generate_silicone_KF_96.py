# Estimated KF-96 (0.65 cSt) property table generator
# DISCLAIMER: estimates only — obtain manufacturer tables or measure for validation.
import numpy as np
import pandas as pd
from pathlib import Path

# Reference state (25°C)
T_ref = 298.15  # K
rho_ref = 965.0  # kg/m3  (typical for very low-viscosity PDMS)
cp_ref = 1460.0  # J/kg/K  (~0.35 cal/g/K)
k_ref = 0.13     # W/m/K   (Shin-Etsu: 0.10-0.15 W/mK for low-viscosity KF-96)
nu_ref = 0.65e-6 # m2/s    (0.65 cSt)
mu_ref = nu_ref * rho_ref  # Pa.s

# Simple model parameters
beta = 8.5e-4    # volumetric thermal expansion (1/K), PDMS typical 7.5-9.4e-4
C_visc = 0.002   # 1/K, weak viscosity sensitivity to T (tunable)

# Temperature grid
T_K = np.arange(290, 641, 5)
T_C = T_K - 273.15

# Property models (transparent, tunable)
rho_T = rho_ref / (1 + beta*(T_K - T_ref))        # approx density vs T
cp_T = np.full_like(T_K, cp_ref)                  # assume cp ~ constant
k_T = k_ref * (1 + 0.0006*(T_K - T_ref))          # small linear increase with T
mu_T = mu_ref * np.exp(-C_visc*(T_K - T_ref))     # weak exponential decay with T

# Derived
nu_T = mu_T / rho_T
alpha_T = k_T / (rho_T * cp_T)

# Build and save dataframe
df = pd.DataFrame({
    "T_K": np.round(T_K, 2),
    "T_C": np.round(T_C, 2),
    "rho_kg_m3": np.round(rho_T, 6),
    "cp_J_kgK": np.round(cp_T, 6),
    "k_W_mK": np.round(k_T, 6),
    "mu_Pa_s": np.round(mu_T, 9),
    "nu_m2_s": np.round(nu_T, 12),
    "alpha_m2_s": np.round(alpha_T, 12)
})
df.to_csv("KF96_estimated_properties_0.65cSt_290-640K_step5.csv", index=False)
print("Saved:", Path("KF96_estimated_properties_0.65cSt_290-640K_step5.csv").resolve())
