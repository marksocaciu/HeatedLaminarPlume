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

class TemperatureDependentMaterial:
    """
    Temperature-dependent material model for air on a given mesh.
    Uses DG0 fields updated from the current temperature field.
    """

    def __init__(
        self,
        mesh,
        T_ref,
        mu_ref,
        cp_ref,
        k_ref,
        beta_ref,
        rho_ref,
        table_file=None
    ):
        self.mesh = mesh
        self.T_ref = float(T_ref)

        # DG0 space
        self.V0 = fenics.FunctionSpace(mesh, "DG", 0)

        # DG0 material fields (used directly in UFL)
        self.mu   = fenics.Function(self.V0, name="mu")
        self.Pr   = fenics.Function(self.V0, name="Pr")
        self.k    = fenics.Function(self.V0, name="k")
        self.rho  = fenics.Function(self.V0, name="rho")
        self.beta = fenics.Function(self.V0, name="beta")

        # temporary DG0 temperature
        self.T_DG0 = fenics.Function(self.V0)

        # store reference constants
        self.mu_ref   = mu_ref
        self.cp_ref   = cp_ref
        self.k_ref    = k_ref
        self.beta_ref = beta_ref
        self.rho_ref  = rho_ref

        # optional lookup table
        self.use_table = table_file is not None
        if self.use_table:
            self._load_table(table_file)

        # initialize with reference values
        self._initialize_fields()

    # -------------------------------------------------------

    def _load_table(self, filename):
        data = np.loadtxt(filename, delimiter=",", skiprows=1)

        self.T_tab    = data[:, 0]
        self.rho_tab  = data[:, 1]
        self.cp_tab   = data[:, 2]
        self.k_tab    = data[:, 3]
        self.mu_tab   = data[:, 4]
        self.nu_tab   = data[:, 5]
        self.alpha_tab = data[:, 6]


    # -------------------------------------------------------

    def _initialize_fields(self):
        self.mu.vector()[:]   = self.mu_ref
        self.k.vector()[:]    = self.k_ref
        self.rho.vector()[:]  = self.rho_ref
        self.beta.vector()[:] = self.beta_ref
        self.Pr.vector()[:]   = self.cp_ref * self.mu_ref / self.k_ref

    # -------------------------------------------------------

    def update(self, T):
        """
        Update material properties from temperature field T
        using CSV lookup tables.
        """

        # Project temperature to DG0 (cellwise)
        self.T_DG0.assign(fenics.project(T, self.V0))
        Tvals = self.T_DG0.vector().get_local()

        # --- safeguard: clip to table range ---
        Tvals_clipped = np.clip(
            Tvals,
            self.T_tab[0],
            self.T_tab[-1]
        )

        # --- interpolate from tables ---
        rho_vals = np.interp(Tvals_clipped, self.T_tab, self.rho_tab)
        cp_vals  = np.interp(Tvals_clipped, self.T_tab, self.cp_tab)
        k_vals   = np.interp(Tvals_clipped, self.T_tab, self.k_tab)
        mu_vals  = np.interp(Tvals_clipped, self.T_tab, self.mu_tab)

        # --- derived quantities ---
        Pr_vals = cp_vals * mu_vals / k_vals

        # --- assign to DG0 fields ---
        self.rho.vector()[:] = rho_vals
        self.k.vector()[:]  = k_vals
        self.mu.vector()[:] = mu_vals
        self.Pr.vector()[:] = Pr_vals

