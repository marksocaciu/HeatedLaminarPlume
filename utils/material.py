from utils.imports import *

def k_of_T(T_K: float) -> float:
    _TK = np.array([300, 600, 800, 1000, 1200], dtype=float)
    _k  = np.array([10, 15, 19, 22, 25.5], dtype=float)  # W/m-K
    T = float(T_K)
    T = min(max(T, _TK[0]), _TK[-1])
    return float(np.interp(T, _TK, _k))


def mu_air_sutherland(T_K: float) -> float:
    # Sutherland equation: mu = b*T^(3/2)/(T+S)
    # For air: b = 1.458e-6 kg/(m*s*sqrt(K)), S = 110.4 K
    # Source values commonly used in aero/CFD references.
    b = 1.458e-6
    S = 110.4
    T = float(T_K)
    return b * (T ** 1.5) / (T + S)

def h_of_T(Ts_K: float, Tinf_K: float, D_m: float,
                                             p_Pa: float = 101325.0) -> float:
    g = 9.81
    R = 287.05          # J/kg-K, dry air
    cp = 1007.0         # J/kg-K (reasonable near ambient)
    Pr = 0.71           # typical for air

    Ts = float(Ts_K)
    Tinf = float(Tinf_K)
    D = float(D_m)

    Tf = 0.5*(Ts + Tinf)
    dT = max(Ts - Tinf, 0.0)  # correlation assumes heated surface in cooler fluid

    mu = mu_air_sutherland(Tf)
    rho = p_Pa/(R*Tf)
    nu = mu/rho
    beta = 1.0/Tf

    # Use k = mu*cp/Pr (common constant-Pr assumption)
    k_air = mu*cp/Pr

    Gr = g*beta*dT*D**3/(nu**2)
    Ra = Gr*Pr

    # Churchill–Chu form for horizontal cylinder (isothermal)
    Nu = (0.60 + (0.387*Ra**(1.0/6.0)) / (1.0 + (0.559/Pr)**(9.0/16.0))**(8.0/27.0))**2

    h = Nu*k_air/D
    return max(h, 0.0)
