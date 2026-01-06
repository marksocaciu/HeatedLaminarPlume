from utils.imports import *

@dataclass(frozen=True)
class NondimScales:
    Lref: float       # [m]
    dTref: float      # [K]
    Uref: float       # [m/s]
    Pr: float         # [-]
    Ra: float         # [-]
    nu: float         # [m^2/s]
    alpha: float      # [m^2/s]
    qsurf: Optional[float] = None     # [W/m^2] if available/derived
    QL: Optional[float] = None        # [W/m]   if provided
    qstar: Optional[float] = None     # [-] nondimensional flux on interface (often 1.0)

def compute_nondimensional_scales(experiment) -> NondimScales:
    """
    Compute reference scales for nondimensional Boussinesq/ABS-style plume solver.

    Assumptions (consistent with your current nondimensional weak form):
      - Lref = wire radius
      - Uref = nu / Lref  (viscous velocity scale)
      - theta = (T_dim - T_inf) / dTref
      - Energy diffusion coefficient becomes 1/Pr.

    Temperature scale dTref:
      - if heat_surface q'' is provided: dTref = q''*Lref/k_inf
      - if heat_length QL is provided:  q'' = QL/(pi*d), dTref = q''*Lref/k_inf = QL/(2*pi*k_inf)
      - if heat_volume q''' is provided: derive an equivalent q'' by matching power per unit length:
            QL = q''' * A, q'' = QL/(pi*d)
        (still uses k_inf)
    """
    # --- geometry ---
    d = float(experiment.dimensions.wire.diameter)  # [m]
    if d <= 0:
        raise ValueError("Wire diameter must be positive.")
    Lref = 0.5 * d  # radius

    # --- fluid props (ambient/reference) ---
    props = experiment.fluid.properties
    k = float(props["k"])       # [W/m/K]
    rho = float(props["rho"])   # [kg/m^3]
    mu = float(props["mu"])     # [Pa*s]
    cp = float(props["cp"])     # [J/kg/K]
    beta = float(props["beta"]) # [1/K]
    g = float(props.get("g", 9.81))  # [m/s^2] use from props if you store it

    nu = mu / rho
    alpha = k / (rho * cp)
    Pr = nu / alpha

    # --- heating: determine an equivalent surface flux q'' [W/m^2] ---
    ic = experiment.initial_conditions
    qsurf = None
    QL = None

    if getattr(ic, "heat_surface", None) is not None:
        qsurf = float(ic.heat_surface)  # [W/m^2]
        # With dTref choice below, the nondimensional boundary flux will be 1
    elif getattr(ic, "heat_length", None) is not None:
        QL = float(ic.heat_length)      # [W/m]
        qsurf = QL / (math.pi * d)
    elif getattr(ic, "heat_volume", None) is not None:
        qvol = float(ic.heat_volume)    # [W/m^3]
        A = math.pi * (0.5 * d) ** 2
        QL = qvol * A                   # [W/m]
        qsurf = QL / (math.pi * d)
    else:
        raise ValueError("No heating specified: set heat_surface, heat_length, or heat_volume.")

    # --- temperature scale ---
    # Choose dTref so that nondimensional imposed flux on the wire surface is q* = 1:
    #   q* = q'' * Lref / (k * dTref)  -> choose dTref = q''*Lref/k
    dTref = qsurf * Lref / k
    if dTref <= 0:
        raise ValueError("Computed dTref is non-positive; check heating/k/geometry.")

    # --- velocity scale (matches your energy diffusion coefficient 1/Pr) ---
    Uref = nu / Lref

    # --- Rayleigh number based on dTref and Lref ---
    Ra = g * beta * dTref * (Lref ** 3) / (nu * alpha)

    # nondimensional interface flux under this dTref choice
    qstar = qsurf * Lref / (k * dTref)  # should be 1.0 (up to roundoff)

    return NondimScales(
        Lref=Lref, dTref=dTref, Uref=Uref, Pr=Pr, Ra=Ra,
        nu=nu, alpha=alpha, qsurf=qsurf, QL=QL, qstar=qstar
    )


def dimensionalize_fields(sub_mesh, u_star, p_star, theta, Uref, dTref, T_inf, rho):
    # Uref_c  = fenics.Constant(float(Uref))
    # dTref_c = fenics.Constant(float(dTref))
    # Tinf_c  = fenics.Constant(float(T_inf))
    # rho_c   = fenics.Constant(float(rho))

    # # Collapse spaces (this avoids the subspace creation error)
    # Vu, _ = u_star.function_space().collapse()
    # Vp, _ = p_star.function_space().collapse()
    # Vt, _ = theta.function_space().collapse()

    # u_dim = fenics.Function(Vu, name="u_dim")
    # T_dim = fenics.Function(Vt, name="T_dim")
    # p_dim = fenics.Function(Vp, name="p_dim")

    # # Build UFL expressions
    # u_expr = Uref_c * u_star
    # T_expr = Tinf_c + dTref_c * theta
    # p_expr = rho_c * Uref_c * Uref_c * p_star

    # # Interpolate if possible; otherwise project (more robust)
    # try:
    #     u_dim.interpolate(u_expr)
    # except RuntimeError:
    #     u_dim.assign(fenics.project(u_expr, Vu))

    # try:
    #     T_dim.interpolate(T_expr)
    # except RuntimeError:
    #     T_dim.assign(fenics.project(T_expr, Vt))

    # try:
    #     p_dim.interpolate(p_expr)
    # except RuntimeError:
    #     p_dim.assign(fenics.project(p_expr, Vp))

    # return u_dim, p_dim, T_dim

    # p0 = fenics.assemble(p_dim * fenics.dx) / fenics.assemble(1.0 * fenics.dx(domain=p_dim.function_space().mesh()))
    # p_dim.vector().axpy(-float(p0), fenics.Vector(p_dim.vector()))  # subtract mean

    Uref = float(Uref)
    dTref = float(dTref)
    T_inf = float(T_inf)
    rho = float(rho)

    # Velocity: u_dim = Uref * u_star
    u_dim = u_star.copy(deepcopy=True)
    u_dim.rename("u_dim", "u_dim")
    u_dim.vector()[:] *= Uref

    # Pressure: p_dim = rho * Uref^2 * p_star
    p_dim = p_star.copy(deepcopy=True)
    p_dim.rename("p_dim", "p_dim")
    p_dim.vector()[:] *= (rho * Uref * Uref)

    # Temperature: T_dim = T_inf + dTref * theta
    T_dim = theta.copy(deepcopy=True)
    T_dim.rename("T_dim", "T_dim")
    T_dim.vector()[:] *= dTref

    # Add T_inf * 1 (create "ones" by copying theta and setting nodal values to 1)
    ones = theta.copy(deepcopy=True)
    ones.vector()[:] = 1.0
    T_dim.vector().axpy(T_inf, ones.vector())

    return u_dim, p_dim, T_dim
    