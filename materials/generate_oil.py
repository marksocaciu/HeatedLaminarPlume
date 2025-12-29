import numpy as np
import pandas as pd
import math
from pathlib import Path

def generate_spindle_oil_table(
    nu_ref_points,    # list of (T_ref, nu_ref) pairs, where nu_ref is kinematic viscosity in m²/s
    rho_ref,          # reference density in kg/m³ (at some T_ref)
    T_ref_for_rho,    # the temperature (K) at which rho_ref is defined
    beta,             # volumetric expansion coefficient (1/K)
    cp_ref,           # assumed (or reference) specific heat [J/(kg·K)] (can be constant or list)
    k_ref,            # reference thermal conductivity [W/(m·K)]
    k_slope=0.0,      # optional linear slope coefficient for k vs T (fractional change per K)
    T_min=290, T_max=640, dT=5
):
    """
    Generates a table of thermophysical properties for a spindle oil (or similar fluid).
    nu_ref_points: list of tuples (T_ref_i, nu_ref_i) in K and m²/s
    All other inputs are scalars or simple models (you may extend to arrays if data exist).
    Returns: pandas DataFrame with columns:
      T_K, T_C, rho, cp, k, nu, mu, alpha, Pr
    """

    # Prepare interpolation / fitting model for kinematic viscosity vs T
    # We'll fit a ln(nu) = A + B / T model (Andrade-type) using the given reference points
    # For robustness, we can do a least-squares fit if >2 points, or exact if 2 points
    T_refs = np.array([tp[0] for tp in nu_ref_points])
    nu_refs = np.array([tp[1] for tp in nu_ref_points])
    # Fit ln(nu) = A + B / T
    # If two points, solve exactly; else least squares
    if len(nu_ref_points) == 2:
        (T1, nu1), (T2, nu2) = nu_ref_points
        # Solve system:
        # ln(nu1) = A + B/T1
        # ln(nu2) = A + B/T2
        B = (math.log(nu1) - math.log(nu2)) / (1.0/T1 - 1.0/T2)
        A = math.log(nu1) - B / T1
    else:
        # More than 2 points: do least squares
        # x = [1, 1/T], y = ln(nu)
        X = np.vstack([np.ones_like(T_refs), 1.0 / T_refs]).T
        y = np.log(nu_refs)
        # Solve for [A, B]
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        A, B = coeffs[0], coeffs[1]

    # Create temperature grid
    T_K = np.arange(T_min, T_max + 1e-6, dT)
    T_C = T_K - 273.15

    # Density model (approx linear expansion)
    rho_T = rho_ref / (1.0 + beta * (T_K - T_ref_for_rho))

    # Specific heat: if cp_ref is scalar, assume constant; if list/array, you can do interpolation
    if np.isscalar(cp_ref):
        cp_T = np.full_like(T_K, cp_ref, dtype=float)
    else:
        # cp_ref could be list of (T, cp) pairs; here just fallback
        cp_T = np.full_like(T_K, cp_ref[0], dtype=float)

    # Thermal conductivity model: linear variation
    k_T = k_ref * (1.0 + k_slope * (T_K - T_ref_for_rho))

    # Kinematic viscosity via the fitted model
    nu_T = np.exp(A + B / T_K)

    # Dynamic viscosity
    mu_T = nu_T * rho_T

    # Thermal diffusivity
    alpha_T = k_T / (rho_T * cp_T)

    # Prandtl number
    Pr_T = nu_T / alpha_T

    # Assemble DataFrame
    df = pd.DataFrame({
        "T_K": np.round(T_K, 2),
        "T_C": np.round(T_C, 2),
        "rho_kg_m3": np.round(rho_T, 8),
        "cp_J_kgK": np.round(cp_T, 8),
        "k_W_mK": np.round(k_T, 8),
        "nu_m2_s": np.round(nu_T, 12),
        "mu_Pa_s": np.round(mu_T, 12),
        "alpha_m2_s": np.round(alpha_T, 12),
        "Pr": np.round(Pr_T, 4),
    })

    return df

if __name__ == "__main__":
    # Example usage: Hyspin 10 spindle oil (two viscosity data points from TDS)
    # From TDS: 10 cSt at 40 °C ; 2.6 cSt at 100 °C
    # Convert to SI: 10e-6 m²/s and 2.6e-6 m²/s
    nu_ref_points = [
        (40.0 + 273.15, 10.0e-6),
        (100.0 + 273.15, 2.6e-6)
    ]
    rho_ref = 0.89e3  # kg/m³
    T_ref_for_rho = 298.15  # K (25 °C)
    beta = 7.0e-4  # 1/K
    cp_ref = 1800.0  # J/(kg·K)
    k_ref = 0.129  # W/(m·K)
    k_slope = -0.00012  # linear coefficient (change per K)

    df = generate_spindle_oil_table(
        nu_ref_points, rho_ref, T_ref_for_rho,
        beta, cp_ref, k_ref, k_slope,
        T_min=290, T_max=640, dT=5
    )

    # Save to CSV
    out_path = Path("spindle_oil_estimated_table.csv")
    df.to_csv(out_path, index=False)
    print("Saved table to:", out_path)
    print(df.head(10))
