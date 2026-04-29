from utils.imports import *

# @dataclass(frozen=True)
# class NondimScales:
#     Lref: float       # [m]
#     dTref: float      # [K]
#     Uref: float       # [m/s]
#     Pr: float         # [-]
#     Ra: float         # [-]
#     nu: float         # [m^2/s]
#     alpha: float      # [m^2/s]
#     qsurf: Optional[float] = None     # [W/m^2] if available/derived
#     QL: Optional[float] = None        # [W/m]   if provided
#     qstar: Optional[float] = None     # [-] nondimensional flux on interface (often 1.0)
#     e: float = 0.0    # Kis' perturbation param
#     Ec: float = 0.0   # Eq19 Kis
#     fEc: float = 0.0  # Eq23 Kis

# def compute_nondimensional_scales(experiment) -> NondimScales:
#     """
#     Compute reference scales for nondimensional Boussinesq/ABS-style plume solver.

#     Assumptions (consistent with your current nondimensional weak form):
#       - Lref = wire radius
#       - Uref = nu / Lref  (viscous velocity scale)
#       - theta = (T_dim - T_inf) / dTref
#       - Energy diffusion coefficient becomes 1/Pr.

#     Temperature scale dTref:
#       - if heat_surface q'' is provided: dTref = q''*Lref/k_inf
#       - if heat_length QL is provided:  q'' = QL/(pi*d), dTref = q''*Lref/k_inf = QL/(2*pi*k_inf)
#       - if heat_volume q''' is provided: derive an equivalent q'' by matching power per unit length:
#             QL = q''' * A, q'' = QL/(pi*d)
#         (still uses k_inf)
#     """
#     # --- geometry ---
#     d = float(experiment.dimensions.wire.diameter)  # [m]
#     if d <= 0:
#         raise ValueError("Wire diameter must be positive.")
#     Lref = 0.5 * d  # radius

#     # --- fluid props (ambient/reference) ---
#     props = experiment.fluid.properties
#     k = float(props["k"])       # [W/m/K]
#     rho = float(props["rho"])   # [kg/m^3]
#     mu = float(props["mu"])     # [Pa*s]
#     cp = float(props["cp"])     # [J/kg/K]
#     beta = float(props["beta"]) # [1/K]
#     g = float(props.get("g", 9.81))  # [m/s^2] use from props if you store it

#     nu = mu / rho
#     alpha = k / (rho * cp)
#     Pr = nu / alpha

#     # --- heating: determine an equivalent surface flux q'' [W/m^2] ---
#     ic = experiment.initial_conditions
#     qsurf = None
#     QL = None

#     if getattr(ic, "heat_surface", None) is not None:
#         qsurf = float(ic.heat_surface)  # [W/m^2]
#         # With dTref choice below, the nondimensional boundary flux will be 1
#     elif getattr(ic, "heat_length", None) is not None:
#         QL = float(ic.heat_length)      # [W/m]
#         qsurf = QL / (math.pi * d)
#     elif getattr(ic, "heat_volume", None) is not None:
#         qvol = float(ic.heat_volume)    # [W/m^3]
#         A = math.pi * (0.5 * d) ** 2
#         QL = qvol * A                   # [W/m]
#         qsurf = QL / (math.pi * d)
#     else:
#         raise ValueError("No heating specified: set heat_surface, heat_length, or heat_volume.")

#     # --- temperature scale ---
#     # Choose dTref so that nondimensional imposed flux on the wire surface is q* = 1:
#     #   q* = q'' * Lref / (k * dTref)  -> choose dTref = q''*Lref/k
#     dTref = qsurf * Lref / k
#     if dTref <= 0:
#         raise ValueError("Computed dTref is non-positive; check heating/k/geometry.")

#     # --- velocity scale (matches your energy diffusion coefficient 1/Pr) ---
#     Uref = nu / Lref

#     # --- Rayleigh number based on dTref and Lref ---
#     Ra = g * beta * dTref * (Lref ** 3) / (nu * alpha)

#     # nondimensional interface flux under this dTref choice
#     qstar = qsurf * Lref / (k * dTref)  # should be 1.0 (up to roundoff)

#     Tref = float(experiment.initial_conditions.temperature)  # ambient/reference temperature [K]

#     e = dTref / Tref
#     Ec = (Uref**2) / (cp * dTref)
#     fEc = Ec / e   # = Uref^2 * Tref / (cp * dTref^2)


#     return NondimScales(
#         Lref=Lref, dTref=dTref, Uref=Uref, Pr=Pr, Ra=Ra,
#         nu=nu, alpha=alpha, qsurf=qsurf, QL=QL, qstar=qstar,
#         e=e, Ec=Ec, fEc=fEc
#     )


# def dimensionalize_fields(sub_mesh, u_star, p_star, theta, Uref, dTref, T_inf, rho):
#     # Uref_c  = fenics.Constant(float(Uref))
#     # dTref_c = fenics.Constant(float(dTref))
#     # Tinf_c  = fenics.Constant(float(T_inf))
#     # rho_c   = fenics.Constant(float(rho))

#     # # Collapse spaces (this avoids the subspace creation error)
#     # Vu, _ = u_star.function_space().collapse()
#     # Vp, _ = p_star.function_space().collapse()
#     # Vt, _ = theta.function_space().collapse()

#     # u_dim = fenics.Function(Vu, name="u_dim")
#     # T_dim = fenics.Function(Vt, name="T_dim")
#     # p_dim = fenics.Function(Vp, name="p_dim")

#     # # Build UFL expressions
#     # u_expr = Uref_c * u_star
#     # T_expr = Tinf_c + dTref_c * theta
#     # p_expr = rho_c * Uref_c * Uref_c * p_star

#     # # Interpolate if possible; otherwise project (more robust)
#     # try:
#     #     u_dim.interpolate(u_expr)
#     # except RuntimeError:
#     #     u_dim.assign(fenics.project(u_expr, Vu))

#     # try:
#     #     T_dim.interpolate(T_expr)
#     # except RuntimeError:
#     #     T_dim.assign(fenics.project(T_expr, Vt))

#     # try:
#     #     p_dim.interpolate(p_expr)
#     # except RuntimeError:
#     #     p_dim.assign(fenics.project(p_expr, Vp))

#     # return u_dim, p_dim, T_dim

#     # p0 = fenics.assemble(p_dim * fenics.dx) / fenics.assemble(1.0 * fenics.dx(domain=p_dim.function_space().mesh()))
#     # p_dim.vector().axpy(-float(p0), fenics.Vector(p_dim.vector()))  # subtract mean

#     Uref = float(Uref)
#     dTref = float(dTref)
#     T_inf = float(T_inf)
#     rho = float(rho)

#     # Velocity: u_dim = Uref * u_star
#     u_dim = u_star.copy(deepcopy=True)
#     u_dim.rename("u_dim", "u_dim")
#     u_dim.vector()[:] *= Uref

#     # Pressure: p_dim = rho * Uref^2 * p_star
#     p_dim = p_star.copy(deepcopy=True)
#     p_dim.rename("p_dim", "p_dim")
#     p_dim.vector()[:] *= (rho * Uref * Uref)

#     # Temperature: T_dim = T_inf + dTref * theta
#     T_dim = theta.copy(deepcopy=True)
#     T_dim.rename("T_dim", "T_dim")
#     T_dim.vector()[:] *= dTref

#     # Add T_inf * 1 (create "ones" by copying theta and setting nodal values to 1)
#     ones = theta.copy(deepcopy=True)
#     ones.vector()[:] = 1.0
#     T_dim.vector().axpy(T_inf, ones.vector())

#     return u_dim, p_dim, T_dim

@dataclass(frozen=True)
class NondimScales:
    # --- solver scales actually used by the current legacy FEniCS weak forms ---
    Lref: float       # [m] solver length scale (kept as wire radius for compatibility)
    dTref: float      # [K] solver temperature scale (kept as wall-flux scale for compatibility)
    Uref: float       # [m/s] solver velocity scale = nu / Lref (kept for compatibility)
    Pr: float         # [-]
    Ra: float         # [-]
    nu: float         # [m^2/s]
    alpha: float      # [m^2/s]
    Gr: float                  # [-] Grashof number based on current solver Lref,dTref
    Gr_wire: Optional[float] = None   # [-] diameter-based Grashof number

    # --- heating bookkeeping ---
    qsurf: Optional[float] = None     # [W/m^2] equivalent wire surface heat flux
    QL: Optional[float] = None        # [W/m]   total heat input per unit wire length
    qstar: Optional[float] = None     # [-] nondimensional flux under current solver scaling

    # --- physically meaningful plume scales for diagnostics / reporting ---
    Lplume: Optional[float] = None    # [m] l_h from line-source plume scaling
    Uplume: Optional[float] = None    # [m/s] v_h = alpha / l_h = (alpha*g*beta*QL/k)^(1/3)
    dTline: Optional[float] = None    # [K] line-source temperature scale QL / k
    Uref_over_Uplume: Optional[float] = None  # [-] how much larger the solver velocity scale is

    # --- weakly non-Boussinesq diagnostics already used in your code ---
    e: float = 0.0
    Ec: float = 0.0
    fEc: float = 0.0


def compute_nondimensional_scales(experiment) -> NondimScales:
    """
    IMPORTANT
    ---------
    This function keeps the CURRENT solver nondimensionalization unchanged so that
    the rest of the legacy FEniCS code still works exactly as before:

        Lref  = wire radius
        dTref = q'' * Lref / k
        Uref  = nu / Lref

    That means:
      * mu = nu / (Uref * Lref) = 1
      * energy diffusion coefficient is 1 / Pr
      * buoyancy coefficient is Ra / Pr

    In addition, we now compute physically meaningful plume scales for diagnostics:

        dTline = QL / k
        Lplume = (g * beta * QL / (k * alpha^2))^(-1/3)
        Uplume = alpha / Lplume = (alpha * g * beta * QL / k)^(1/3)

    These plume scales should be used when you interpret the solution, compare to
    line-source theory, or sanity-check expected velocity magnitudes.

    NOTE
    ----
    Merely changing Uref in post-processing would NOT be correct unless the weak
    forms are re-derived and updated consistently. Therefore Uref is kept as-is for
    solver compatibility, and Uplume is exposed separately for diagnostics.
    """
    # --- geometry ---
    d = float(experiment.dimensions.wire.diameter)  # [m]
    if d <= 0:
        raise ValueError("Wire diameter must be positive.")
    r = 0.5 * d
    Lref = r

    # --- ambient/reference fluid properties ---
    props = experiment.fluid.properties
    k = float(props["k"])       # [W/m/K]
    rho = float(props["rho"])   # [kg/m^3]
    mu = float(props["mu"])     # [Pa*s]
    cp = float(props["cp"])     # [J/kg/K]
    beta = float(props["beta"]) # [1/K]
    g = float(props.get("g", 9.81))

    nu = mu / rho
    alpha = k / (rho * cp)
    Pr = nu / alpha

    # --- heating: convert everything to an equivalent surface flux q'' and line input QL ---
    ic = experiment.initial_conditions
    qsurf = None
    QL = None

    if getattr(ic, "heat_surface", None) is not None:
        qsurf = float(ic.heat_surface)          # [W/m^2]
        QL = qsurf * math.pi * d                # [W/m]
    elif getattr(ic, "heat_length", None) is not None:
        QL = float(ic.heat_length)              # [W/m]
        qsurf = QL / (math.pi * d)              # [W/m^2]
    elif getattr(ic, "heat_volume", None) is not None:
        qvol = float(ic.heat_volume)            # [W/m^3]
        A = math.pi * r**2
        QL = qvol * A                           # [W/m]
        qsurf = QL / (math.pi * d)              # [W/m^2]
    else:
        raise ValueError("No heating specified: set heat_surface, heat_length, or heat_volume.")

    if QL is None or QL <= 0.0:
        raise ValueError("Computed QL is non-positive; check heating specification.")

    # --- current solver temperature scale (kept unchanged) ---
    dTref = qsurf * experiment.dimensions.domain.x_max / k                    # = QL / (2*pi*k)
    if dTref <= 0:
        raise ValueError("Computed dTref is non-positive; check heating/k/geometry.")

    # --- current solver velocity scale (kept unchanged for compatibility) ---
    Uref = nu / Lref

    # --- current solver Rayleigh number (based on wire radius and dTref) ---
    Ra = g * beta * dTref * (Lref ** 3) / (nu * alpha)

    # --- current nondimensional imposed interface flux ---
    qstar = qsurf * Lref / (k * dTref)          # should be 1.0

    # --- plume / line-source scales for interpretation ---
    dTline = QL / k
    Lplume = (g * beta * QL / (k * alpha * alpha)) ** (-1.0 / 3.0)
    Uplume = alpha / Lplume
    Uref_over_Uplume = Uref / Uplume

    # --- weakly non-Boussinesq diagnostics ---
    Tref = float(experiment.initial_conditions.temperature)
    e = dTref / Tref
    Ec = (Uref**2) / (cp * dTref)
    fEc = Ec / e

    Gr = Ra / Pr
    Gr_wire = g * beta * dTref * (d ** 3) / (nu ** 2)

    return NondimScales(
        Lref=Lref,
        dTref=dTref,
        Uref=Uref,
        Pr=Pr,
        Ra=Ra,
        nu=nu,
        alpha=alpha,
        Gr=Gr,
        Gr_wire=Gr_wire,
        qsurf=qsurf,
        QL=QL,
        qstar=qstar,
        Lplume=Lplume,
        Uplume=Uplume,
        dTline=dTline,
        Uref_over_Uplume=Uref_over_Uplume,
        e=e,
        Ec=Ec,
        fEc=fEc,
    )


def dimensionalize_fields(sub_mesh, u_star, p_star, theta, Uref, dTref, T_inf, rho):
    """
    Convert the CURRENT solver unknowns back to dimensional fields.

    Because u_star was computed with the legacy solver scale Uref = nu/Lref,
    the physical velocity is and remains

        u_dim = Uref * u_star.

    This function therefore stays unchanged.
    """
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

    ones = theta.copy(deepcopy=True)
    ones.vector()[:] = 1.0
    T_dim.vector().axpy(T_inf, ones.vector())

    return u_dim, p_dim, T_dim


def plume_velocity_from_solver_velocity(u_star_value: float, scales: NondimScales) -> float:
    """Physical velocity [m/s] from a solver-nondimensional velocity value."""
    return float(scales.Uref) * float(u_star_value)


def plume_normalized_velocity_from_solver_velocity(u_star_value: float, scales: NondimScales) -> float:
    """
    Convert the current solver nondimensional velocity to a plume-scale nondimensional value:

        u_hat = u_dim / Uplume = (Uref / Uplume) * u_star
    """
    if scales.Uplume is None or scales.Uplume <= 0.0:
        raise ValueError("Uplume is not available or non-positive.")
    return float(scales.Uref_over_Uplume) * float(u_star_value)
