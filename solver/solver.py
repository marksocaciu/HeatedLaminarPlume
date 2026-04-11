from utils.imports import *
from solver.params_bcs import *
from utils.material import *
from solver.scales import *
from utils.plot import *
from utils.geometry import *

class RestrictToAir(fenics.UserExpression):
    def __init__(self, T_full, **kwargs):
        super().__init__(**kwargs)
        self.T_full = T_full

    def eval(self, values, x):
        # Evaluate full-mesh temperature at point x
        values[0] = self.T_full(x)

    def value_shape(self):
        return ()

def solve_thermal_sign_check(
    experiment: Experiment,
    W: fenics.FunctionSpace,
    w: fenics.Function,
    mu, Pr,
    sub_dx, sub_ds, sub_ft,
    qn_air,
    T_c,
    T_air_bc,
    w_n: fenics.Function,
):
    """
    Solves only the conduction part in the air with the same mixed space.
    Velocity is forced to zero by the BCs that act on W_u; there is no buoyancy
    and no advection in the residual. Use this to verify the interface heat-flux sign.
    """
    w.vector()[:] = w_n.vector()
    w.vector().apply("insert")

    theta = w.sub(2, deepcopy=True)
    print("Thermal sign check:")
    print(f"  theta min/max = {theta.vector().min():.6e}, {theta.vector().max():.6e}")

    return w

def solve_buoyancy_sign_check(
    experiment: Experiment,
    W: fenics.FunctionSpace,
    w: fenics.Function,
    mu, Pr,
    sub_dx, sub_ds, sub_ft,
    qn_air,
    T_c,
    T_air_bc,
    w_n: fenics.Function,
):

    w.vector()[:] = w_n.vector()
    w.vector().apply("insert")

    u_chk = w.sub(1, deepcopy=True)
    Vscal = fenics.FunctionSpace(u_chk.function_space().mesh(), "CG", 1)
    uy = fenics.project(u_chk[1], Vscal,solver_type="mumps")

    print("Buoyancy sign check:")
    print(f"  uy min/max = {uy.vector().min():.6e}, {uy.vector().max():.6e}")

    r = (experiment.dimensions.wire.diameter / 2) / compute_nondimensional_scales(experiment).Lref
    x_probe = 0.5 * r
    y0 = 11.0 * r

    probe_points = [
        (x_probe, y0 + 1.5 * r),
        (x_probe, y0 + 3.0 * r),
        (x_probe, y0 + 6.0 * r),
        (x_probe, y0 + 8.0 * r),
    ]

    print("Buoyancy sign check probes (uy):")
    for xp, yp in probe_points:
        try:
            val = uy(xp, yp)
            print(f"  uy({xp:.6e}, {yp:.6e}) = {val:.6e}")
        except RuntimeError:
            print(f"  probe failed at ({xp:.6e}, {yp:.6e})")


    return w

def thermal_galerkin_supg_form(
    mesh,
    T,
    psi_T,
    u,
    Pr,
    sub_dx,
    convection_scale=1.0,
    T_prev=None,
    dtau=None,
):
    """
    Temperature equation:
        dT/dtau + c * u·grad(T) - div((1/Pr) grad(T)) = 0

    Weak form:
        (Galerkin) + (SUPG)

    Notes:
    - Omitted boundary terms imply homogeneous natural diffusive flux
      on boundaries where no Dirichlet BC is imposed.
    - SUPG is switched by the streamline derivative dot(u, grad(psi_T)).
    """
    kappa = 1.0 / Pr
    c = fenics.Constant(float(convection_scale))

    # Standard Galerkin part
    galerkin = (
        kappa * fenics.dot(fenics.grad(psi_T), fenics.grad(T))
        + c * psi_T * fenics.dot(u, fenics.grad(T))
    )

    # Strong residual of the temperature equation
    r_T = c * fenics.dot(u, fenics.grad(T)) - fenics.div(kappa * fenics.grad(T))
    if (T_prev is not None) and (dtau is not None):
        r_T += (T - T_prev) / dtau

    # SUPG tau
    h = fenics.CellDiameter(mesh)
    u_mag = fenics.sqrt(fenics.inner(u, u) + fenics.DOLFIN_EPS)

    tau = 1.0 / fenics.sqrt(
        (2.0 * u_mag / h)**2
        + (4.0 * kappa / (h * h))**2
        + fenics.DOLFIN_EPS
    )

    supg = tau * c * fenics.dot(u, fenics.grad(psi_T)) * r_T

    return (galerkin + supg) * sub_dx

def solver(sub_mesh: fenics.Mesh, T_full: fenics.Function, T_ambient: float,
           rho_air: float, beta_air: float, experiment: Experiment):
    P1 = fenics.FiniteElement('P', sub_mesh.ufl_cell(), 1)
    P2 = fenics.VectorElement('P', sub_mesh.ufl_cell(), 2)
    mixed_element = fenics.MixedElement([P1, P2, P1]) 
    W = fenics.FunctionSpace(sub_mesh, mixed_element)

    psi_p, psi_u, psi_T = fenics.TestFunctions(W)

    w = fenics.Function(W)
    p, u, T = fenics.split(w)

    mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc = set_param(sub_mesh, T_full, T, T_ambient, rho_air, beta_air, experiment)

    # Build mixed initial state
    w_n = fenics.Function(W)

    # Collapsed subspaces (these are the canonical source spaces for assignment)
    Vp, p_to_W = W.sub(0).collapse(True)   # scalar
    Vu, u_to_W = W.sub(1).collapse(True)   # vector
    VT, T_to_W = W.sub(2).collapse(True)   # scalar

    # Source functions (must live in the collapsed spaces)
    p0 = fenics.Function(Vp)
    u0 = fenics.Function(Vu)
    T0 = fenics.Function(VT)

    p0.vector().zero()
    u0.vector().zero()

    # Temperature initial guess:
    # If T_full is already a scalar Function on the *same sub_mesh* (your current pipeline),
    # interpolate it onto VT (safe even if VT is a different object).
    T0.interpolate(T_full)

    # If you ever pass a function on a different mesh, you cannot do this; you'd need restriction/projection.

    # Assign into mixed function using FunctionAssigners that match spaces exactly
    assign_p = fenics.FunctionAssigner(W.sub(0), Vp)
    assign_u = fenics.FunctionAssigner(W.sub(1), Vu)
    assign_T = fenics.FunctionAssigner(W.sub(2), VT)

    assign_p.assign(w_n.sub(0), p0)
    assign_u.assign(w_n.sub(1), u0)
    assign_T.assign(w_n.sub(2), T0)

    w_n.vector().apply("insert")

    print("Init T min/max:",
        w_n.sub(2).vector().min(),
        w_n.sub(2).vector().max())

    # Now split for convenience (these are UFL objects / views; OK for variational forms)
    p_n, u_n, T_n = fenics.split(w_n)

    print(f"Initial guess max theta (air): {w_n.sub(2).vector().max():.6e}")
    print(f"Initial guess min theta (air): {w_n.sub(2).vector().min():.6e}")

    return W, w, p, u, T, w_n, p_n, u_n, T_n, psi_p, psi_u, psi_T, mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc

def base_solver(F, w: fenics.Function, boundary_conditions, JF,
                relaxation=0.5, maxit=20, atol=1e-9, rtol=1e-8,
                return_meta: bool = False):
    problem = fenics.NonlinearVariationalProblem(F, w, boundary_conditions, JF)
    solver = fenics.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["nonlinear_solver"] = "newton"

    nprm = prm["newton_solver"]
    nprm["linear_solver"] = "mumps"
    nprm["absolute_tolerance"] = atol
    nprm["relative_tolerance"] = rtol
    nprm["maximum_iterations"] = maxit
    nprm["report"] = True
    nprm["error_on_nonconvergence"] = True
    nprm["relaxation_parameter"] = relaxation

    result = solver.solve()

    n_iter = None
    converged = True
    if isinstance(result, tuple):
        if len(result) >= 1:
            n_iter = result[0]
        if len(result) >= 2:
            converged = bool(result[1])
    elif isinstance(result, (int, float)):
        n_iter = int(result)

    if return_meta:
        return w, n_iter, converged
    return w

def nonlinear_solver(experiment: Experiment,u_n: fenics.Function, u: fenics.Function, T_n: fenics.Function, T: fenics.Function, p: fenics.Function,
                     W: fenics.FunctionSpace, w: fenics.Function,
                     psi_p, psi_u, psi_T,
                     mu, Pr, f_b, T_c, T_air_bc,
                     sub_dx, sub_ds, sub_ft, qn_air,
                     w_n: fenics.Function,
                     buoyancy_scale=1.0,
                     qn_scale=1.0,
                     include_convection=True,
                     convection_scale=1.0,
                     _SUPG=False):

    inner, dot, grad, div, sym = \
        fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
        
    mass = -psi_p*div(u)
            
    buoyancy_scale = fenics.Constant(float(buoyancy_scale))
    convection_scale = fenics.Constant(float(convection_scale))
    convection_term = convection_scale * dot(grad(u), u) if include_convection else fenics.Constant((0.0, 0.0))

    momentum = (
        dot(psi_u, convection_term + buoyancy_scale * f_b)
        - div(psi_u)*p
        + 2.0 * mu * inner(sym(grad(psi_u)), sym(grad(u)))
    )

    if _SUPG:
        energy = thermal_galerkin_supg_form(
            mesh=W.mesh(),
            T=T,
            psi_T=psi_T,
            u=u,
            Pr=Pr,
            sub_dx=sub_dx,
            convection_scale=float(convection_scale),
        )

        F = (mass + momentum) * sub_dx + energy
    else:
        energy = (1.0/Pr) * dot(grad(psi_T), grad(T)) + psi_T * dot(u, grad(T))
        F = (mass + momentum + energy) * sub_dx


    qn_scale_c = fenics.Constant(float(qn_scale))

    print("Max qn_air:", qn_air.vector().max())
    print(f"Applied qn_scale: {float(qn_scale):.4f}")

    F += - qn_scale_c * qn_air * psi_T * sub_ds(INTERFACE_TAG)

    scales = compute_nondimensional_scales(experiment)
    k_inf = float(experiment.fluid.properties["k"])
    qn_dim = qn_scale_c * qn_air * fenics.Constant(
        k_inf * float(scales.dTref) / float(scales.Lref)
    )
    QL_half = fenics.assemble(qn_dim * sub_ds(INTERFACE_TAG)) * scales.Lref
    print(f"Heat flux from wire to fluid (half wire): QL_half = {QL_half:.6e} W/m")

    JF = fenics.derivative(F, w, fenics.TrialFunction(W))

    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

    w.leaf_node().vector()[:] = w_n.leaf_node().vector()

    return F,w, boundary_conditions, JF, w_n

def build_temperature_assigner(W: fenics.FunctionSpace):
    VT, _ = W.sub(2).collapse(True)
    assign_T = fenics.FunctionAssigner(W.sub(2), VT)
    return VT, assign_T

def assign_mixed_temperature(
    w_mixed: fenics.Function,
    theta_src: fenics.Function,
    VT: fenics.FunctionSpace,
    assign_T: fenics.FunctionAssigner,
    theta_tmp: fenics.Function = None,
):
    if theta_tmp is None:
        theta_tmp = fenics.Function(VT)

    theta_tmp.interpolate(theta_src)
    assign_T.assign(w_mixed.sub(2), theta_tmp)
    w_mixed.vector().apply("insert")
    return theta_tmp

def update_material_from_mixed_temperature(
    fluid_material: TemperatureDependentMaterial,
    w_mixed: fenics.Function,
    scales,
    T_ambient: float,
):
    """
    Update temperature-dependent material fields from the mixed-state temperature.

    The mixed variable stores nondimensional theta on the star-scaled air mesh.
    We convert it back to dimensional temperature before calling
    ``fluid_material.update(...)``.
    """
    theta = w_mixed.sub(2, deepcopy=True)
    T_dim = fenics.Function(theta.function_space())
    T_dim.vector()[:] = float(T_ambient) + float(scales.dTref) * theta.vector()[:]
    T_dim.vector().apply("insert")
    fluid_material.update(T_dim)
    return T_dim

def build_stokes_only_startup_problem(
    W_pu: fenics.FunctionSpace,
    mu,
    frozen_buoyancy_temperature,
    sub_dx,
    scales,
):
    """
    Linear startup on reduced mixed space (p, u) only.
    Temperature is frozen and only appears in the RHS buoyancy term.
    """
    p, u = fenics.TrialFunctions(W_pu)
    q, v = fenics.TestFunctions(W_pu)

    gvec = fenics.Constant((0.0, -1.0))
    buoyancy_coeff = fenics.Constant(float(scales.Ra / scales.Pr))

    a = (
        (-q * fenics.div(u))
        + (
            -fenics.div(v) * p
            + 2.0 * mu * fenics.inner(
                fenics.sym(fenics.grad(v)),
                fenics.sym(fenics.grad(u)),
            )
        )
    ) * sub_dx

    L = - fenics.dot(v, buoyancy_coeff * frozen_buoyancy_temperature * gvec) * sub_dx

    return a, L

def set_bcs_stokes_only(W_pu, experiment: Experiment, scales: NondimScales):
    r = (experiment.dimensions.wire.diameter / 2) / scales.Lref

    class Hot_wall(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(
                (x[0]**2) + ((x[1] - (experiment.dimensions.domain.y_max / scales.Lref / 10. + 11.*r))**2)
                - 1.*r*r, 0., eps=1.e-1*r
            ) and \
            x[1] >= experiment.dimensions.domain.y_max / scales.Lref / 10. + 10.*r - 1e-12 and \
            x[1] <= experiment.dimensions.domain.y_max / scales.Lref / 10. + 12.*r + 1e-12

    class PressurePin(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return (
                fenics.near(x[0], experiment.dimensions.domain.x_max / scales.Lref, 1.0e-8)
                and fenics.near(x[1], experiment.dimensions.domain.y_max / scales.Lref, 1.0e-8)
            )

    hot_wall = Hot_wall()
    p_pin = PressurePin()

    W_p = W_pu.sub(0)
    W_u = W_pu.sub(1)

    boundary_conditions = [
        fenics.DirichletBC(W_u, (0.0, 0.0), hot_wall),   # keep no-slip on wire
        fenics.DirichletBC(W_p, fenics.Constant(0.0), p_pin, method="pointwise"),
    ]
    return boundary_conditions

def build_linear_startup_problem(
    experiment: Experiment,
    W: fenics.FunctionSpace,
    mu, Pr,
    sub_dx, sub_ds,
    qn_air,
    qn_scale=1.0,
    frozen_buoyancy_temperature=None,
    scales=None,
):
    """
    Return bilinear form a and linear form L for the startup solve.

    Unknowns: (p, u, T)
    Test functions: (q, v, s)
    """
    p, u, T = fenics.TrialFunctions(W)
    q, v, s = fenics.TestFunctions(W)

    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym

    if scales is None:
        scales = compute_nondimensional_scales(experiment)

    gvec = fenics.Constant((0.0, -1.0))
    buoyancy_coeff = fenics.Constant(float(scales.Ra / scales.Pr))

    a = (
        (-q * fenics.div(u))
        + (-fenics.div(v) * p + 2.0 * mu * fenics.inner(fenics.sym(fenics.grad(v)),
                                                        fenics.sym(fenics.grad(u))))
        + fenics.dot(fenics.grad(s), (1.0 / Pr) * fenics.grad(T))
    ) * sub_dx

    L = fenics.Constant(float(qn_scale)) * qn_air * s * sub_ds(INTERFACE_TAG)

    if frozen_buoyancy_temperature is not None:
        L += - dot(v, buoyancy_coeff * frozen_buoyancy_temperature * gvec) * sub_dx

    return a, L

def solve_linear_problem(a, L, w, boundary_conditions, linear_solver="mumps"):
    problem = fenics.LinearVariationalProblem(a, L, w, boundary_conditions)
    solver = fenics.LinearVariationalSolver(problem)

    prm = solver.parameters
    prm["linear_solver"] = linear_solver

    solver.solve()
    w.vector().apply("insert")
    return w

def solve_linear_problem_temp(a, L, w, boundary_conditions, linear_solver="mumps"):
    A, b = fenics.assemble_system(a, L, boundary_conditions)

    print(f"  matrix size: {A.size(0)} x {A.size(1)}")
    print(f"  rhs l2 norm: {b.norm('l2'):.6e}")

    if fenics.has_lu_solver_method(linear_solver):
        print(f"  using LU solver: {linear_solver}")
        solver = fenics.LUSolver(A, linear_solver)
    else:
        print(f"  requested LU solver '{linear_solver}' not available")
        print(f"  available LU solvers: {fenics.lu_solver_methods()}")
        solver = fenics.LUSolver(A, "default")

    solver.solve(w.vector(), b)
    w.vector().apply("insert")
    return w

def stokes_initial_guess(
    experiment: Experiment,
    u_n: fenics.Function, u: fenics.Function, T_n: fenics.Function, T: fenics.Function, p: fenics.Function,
    W: fenics.FunctionSpace, w: fenics.Function,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b, T_c, T_air_bc,
    sub_dx, sub_ds, sub_ft, qn_air,
    w_n: fenics.Function,
    lambdas=(0.10, 0.25, 0.50, 1.00),
):
    """
    Fast linear startup:
      1) conduction-only solve
      2) frozen-temperature Stokes solve
    for each continuation lambda.
    """
    scales = compute_nondimensional_scales(experiment)
    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

    # cache temperature assignment machinery
    VT, assign_T = build_temperature_assigner(W)
    theta_tmp = fenics.Function(VT)

    # reference temperature field for continuation scaling
    theta_ref = fenics.Function(VT)
    theta_ref.vector()[:] = w_n.sub(2, deepcopy=True).vector()
    theta_ref.vector().apply("insert")

    # working storage
    theta_lam = fenics.Function(VT)

    w.vector()[:] = w_n.vector()
    w.vector().apply("insert")

    for lam in lambdas:
        print(f"\n=== Linear startup lambda = {lam:.2f} ===")

        # scale reference thermal field for the current continuation level
        theta_lam.vector()[:] = theta_ref.vector()
        theta_lam.vector()[:] *= float(lam)
        theta_lam.vector().apply("insert")

        assign_mixed_temperature(w_n, theta_lam, VT, assign_T, theta_tmp)

        # start each stage from latest accepted mixed state
        w.vector()[:] = w_n.vector()
        w.vector().apply("insert")

        print("  -> Stage A: conduction-only solve")
        a_cond, L_cond = build_linear_startup_problem(
            experiment=experiment,
            W=W,
            mu=mu,
            Pr=Pr,
            sub_dx=sub_dx,
            sub_ds=sub_ds,
            qn_air=qn_air,
            qn_scale=lam,
            frozen_buoyancy_temperature=None,
            scales=scales,
        )
        w = solve_linear_problem(a_cond, L_cond, w, boundary_conditions)

        theta_cond = w.sub(2, deepcopy=True)

        w_n.assign(w)
        w_n.vector().apply("insert")

        print("  -> Stage B: frozen-temperature Stokes solve")
        a_stokes, L_stokes = build_linear_startup_problem(
            experiment=experiment,
            W=W,
            mu=mu,
            Pr=Pr,
            sub_dx=sub_dx,
            sub_ds=sub_ds,
            qn_air=qn_air,
            qn_scale=lam,
            frozen_buoyancy_temperature=theta_cond,
            scales=scales,
        )
        w = solve_linear_problem(a_stokes, L_stokes, w, boundary_conditions)

        w_n.assign(w)
        w_n.vector().apply("insert")

    return w_n

def build_temperature_bcs(
    VT: fenics.FunctionSpace,
    sub_ft,
    T_air_bc,
    T_c,
    experiment: Experiment,
    scales,
):
    """
    Temperature-only BCs for the startup conduction solve.

    This mirrors the temperature part of set_bcs(...):
      - theta = 0 on the east boundary
    """
    class EastBoundary(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(
                x[0], experiment.dimensions.domain.x_max / scales.Lref, eps=1.0e-10
            )

    east = EastBoundary()

    T_bcs = [
        fenics.DirichletBC(VT, fenics.Constant(0.0), east),
    ]
    return T_bcs


def solve_temperature_only_startup(
    VT: fenics.FunctionSpace,
    Pr,
    sub_dx,
    sub_ds,
    qn_air,
    qn_scale=1.0,
    T_bcs=None,
    linear_solver="mumps",
):
    """
    Solve only the scalar conduction problem on the temperature space VT:

        -div((1/Pr) grad(theta)) = 0   in air
        (1/Pr) grad(theta)·n = qn_scale * qn_air   on interface

    in weak form:
        ∫ grad(s)·((1/Pr) grad(T)) dx = ∫ qn_scale * qn_air * s ds
    """
    T = fenics.TrialFunction(VT)
    s = fenics.TestFunction(VT)

    aT = fenics.dot(fenics.grad(s), (1.0 / Pr) * fenics.grad(T)) * sub_dx
    LT = fenics.Constant(float(qn_scale)) * qn_air * s * sub_ds(INTERFACE_TAG)

    theta = fenics.Function(VT)
    problem = fenics.LinearVariationalProblem(aT, LT, theta, T_bcs or [])
    solver = fenics.LinearVariationalSolver(problem)

    prm = solver.parameters
    prm["linear_solver"] = linear_solver

    solver.solve()
    theta.vector().apply("insert")
    return theta

# def stokes_initial_guess_temp(
#     experiment: Experiment,
#     u_n: fenics.Function, u: fenics.Function, T_n: fenics.Function, T: fenics.Function, p: fenics.Function,
#     W: fenics.FunctionSpace, w: fenics.Function,
#     psi_p, psi_u, psi_T,
#     mu, Pr, f_b, T_c, T_air_bc,
#     sub_dx, sub_ds, sub_ft, qn_air,
#     fluid_material: TemperatureDependentMaterial,
#     w_n: fenics.Function,
#     T_ambient: float,
#     lambdas=(0.10, 0.25, 0.50, 1.00),
# ):
#     """
#     Temperature-dependent startup.

#     Coefficients are frozen during each linear startup solve, but updated from the
#     current mixed-state temperature before each lambda stage and again after the
#     accepted startup state is written back into ``w_n``.
#     """
#     scales = compute_nondimensional_scales(experiment)
#     boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

#     VT, assign_T = build_temperature_assigner(W)
#     theta_tmp = fenics.Function(VT)

#     theta_ref = fenics.Function(VT)
#     theta_ref.vector()[:] = w_n.sub(2, deepcopy=True).vector()
#     theta_ref.vector().apply("insert")

#     theta_lam = fenics.Function(VT)

#     w.vector()[:] = w_n.vector()
#     w.vector().apply("insert")

#     for lam in lambdas:
#         print(f"\n=== Linear startup lambda = {lam:.2f} (temp-dependent) ===")

#         theta_lam.vector()[:] = theta_ref.vector()
#         theta_lam.vector()[:] *= float(lam)
#         theta_lam.vector().apply("insert")

#         assign_mixed_temperature(w_n, theta_lam, VT, assign_T, theta_tmp)
#         update_material_from_mixed_temperature(fluid_material, w_n, scales, T_ambient)

#         w.vector()[:] = w_n.vector()
#         w.vector().apply("insert")

#         print("  -> Stage A: conduction-only solve")
#         a_cond, L_cond = build_linear_startup_problem(
#             experiment=experiment,
#             W=W,
#             mu=fluid_material.mu,
#             Pr=fluid_material.Pr,
#             sub_dx=sub_dx,
#             sub_ds=sub_ds,
#             qn_air=qn_air,
#             qn_scale=lam,
#             frozen_buoyancy_temperature=None,
#             scales=scales,
#         )
#         w = solve_linear_problem(a_cond, L_cond, w, boundary_conditions)

#         theta_cond = w.sub(2, deepcopy=True)

#         w_n.assign(w)
#         w_n.vector().apply("insert")
#         update_material_from_mixed_temperature(fluid_material, w_n, scales, T_ambient)

#         print("  -> Stage B: frozen-temperature Stokes solve")
#         a_stokes, L_stokes = build_linear_startup_problem(
#             experiment=experiment,
#             W=W,
#             mu=fluid_material.mu,
#             Pr=fluid_material.Pr,
#             sub_dx=sub_dx,
#             sub_ds=sub_ds,
#             qn_air=qn_air,
#             qn_scale=lam,
#             frozen_buoyancy_temperature=theta_cond,
#             scales=scales,
#         )
#         w = solve_linear_problem(a_stokes, L_stokes, w, boundary_conditions)

#         w_n.assign(w)
#         w_n.vector().apply("insert")
#         update_material_from_mixed_temperature(fluid_material, w_n, scales, T_ambient)

#     return w_n

def stokes_initial_guess_temp(
    experiment: Experiment,
    u_n: fenics.Function, u: fenics.Function, T_n: fenics.Function, T: fenics.Function, p: fenics.Function,
    W: fenics.FunctionSpace, w: fenics.Function,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b, T_c, T_air_bc,
    sub_dx, sub_ds, sub_ft, qn_air,
    fluid_material: TemperatureDependentMaterial,
    w_n: fenics.Function,
    T_ambient: float,
    lambdas=(0.10, 0.25, 0.50, 1.00),
):
    """
    Temperature-dependent startup.

    Stage A:
        solve temperature only on VT
    Stage B:
        solve mixed frozen-temperature Stokes problem on W

    Material coefficients are updated from the accepted mixed temperature state.
    """
    scales = compute_nondimensional_scales(experiment)

    # Mixed BCs for Stage B
    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)
    print("\nBC dof counts:")
    for i, bc in enumerate(boundary_conditions):
        vals = bc.get_boundary_values()
        print(f"  bc[{i}] -> {len(vals)} dofs")

    # Temperature-only space and assigner
    VT, assign_T = build_temperature_assigner(W)
    theta_tmp = fenics.Function(VT)
    # W_pu = fenics.FunctionSpace(W.mesh(), fenics.MixedElement([psi_p, psi_u]))
    mixed_el = W.ufl_element()
    p_el, u_el, T_el = mixed_el.sub_elements()
    W_pu = fenics.FunctionSpace(W.mesh(), fenics.MixedElement([p_el, u_el]))
    bcs_stokes = set_bcs_stokes_only(W_pu, experiment, scales)

    print("\nStokes-only BC dof counts:")
    for i, bc in enumerate(bcs_stokes):
        vals = bc.get_boundary_values()
        print(f"  bc[{i}] -> {len(vals)} dofs")

    # Temperature-only BCs for Stage A
    T_bcs = build_temperature_bcs(VT, sub_ft, T_air_bc, T_c, experiment, scales)

    # Reference temperature field from initial mixed state
    theta_ref = fenics.Function(VT)
    theta_ref.vector()[:] = w_n.sub(2, deepcopy=True).vector()
    theta_ref.vector().apply("insert")

    theta_lam = fenics.Function(VT)

    # working state
    w.vector()[:] = w_n.vector()
    w.vector().apply("insert")

    for lam in lambdas:
        print(f"\n=== Linear startup lambda = {lam:.2f} (temp-dependent) ===")

        # scale the reference temperature for continuation
        theta_lam.vector()[:] = theta_ref.vector()
        theta_lam.vector()[:] *= float(lam)
        theta_lam.vector().apply("insert")

        # write scaled theta into mixed state and update materials
        assign_mixed_temperature(w_n, theta_lam, VT, assign_T, theta_tmp)
        update_material_from_mixed_temperature(fluid_material, w_n, scales, T_ambient)

        # restart current iterate from last accepted state
        w.vector()[:] = w_n.vector()
        w.vector().apply("insert")

        # --------------------------------------------------
        # Stage A: scalar temperature-only conduction solve
        # --------------------------------------------------
        print("  -> Stage A: temperature-only conduction solve")

        theta_cond = solve_temperature_only_startup(
            VT=VT,
            Pr=fluid_material.Pr,
            sub_dx=sub_dx,
            sub_ds=sub_ds,
            qn_air=qn_air,
            qn_scale=lam,
            T_bcs=T_bcs,
            linear_solver="mumps",
        )

        # inject temperature result into mixed state
        assign_T.assign(w.sub(2), theta_cond)
        w.vector().apply("insert")

        # accept Stage A into w_n and update material fields from it
        w_n.assign(w)
        w_n.vector().apply("insert")
        update_material_from_mixed_temperature(fluid_material, w_n, scales, T_ambient)

        # --------------------------------------------------
        # Stage B: frozen-temperature Stokes solve on W
        # --------------------------------------------------
        # print("  -> Stage B: frozen-temperature Stokes solve")
        print("  -> Stage B: frozen-temperature Stokes solve (reduced p-u system)")

        a_stokes, L_stokes = build_stokes_only_startup_problem(
            W_pu=W_pu,
            mu=fluid_material.mu,
            frozen_buoyancy_temperature=theta_cond,
            sub_dx=sub_dx,
            scales=scales,
        )

        w_pu = fenics.Function(W_pu)
        w_pu = solve_linear_problem(a_stokes, L_stokes, w_pu, bcs_stokes, linear_solver="mumps")

        # copy reduced solve back into full mixed state
        fenics.assign(w.sub(0), w_pu.sub(0))
        fenics.assign(w.sub(1), w_pu.sub(1))
        assign_T.assign(w.sub(2), theta_cond)
        w.vector().apply("insert")

        w_n.assign(w)
        w_n.vector().apply("insert")
        update_material_from_mixed_temperature(fluid_material, w_n, scales, T_ambient)

        try:
            print(f"  mu min/max: {fluid_material.mu.vector().min():.6e}, {fluid_material.mu.vector().max():.6e}")
        except Exception:
            pass

        try:
            print(f"  Pr min/max: {fluid_material.Pr.vector().min():.6e}, {fluid_material.Pr.vector().max():.6e}")
        except Exception:
            pass

        print(f"  theta_cond min/max: {theta_cond.vector().min():.6e}, {theta_cond.vector().max():.6e}")

        a_stokes, L_stokes = build_linear_startup_problem(
            experiment=experiment,
            W=W,
            mu=fluid_material.mu,
            Pr=fluid_material.Pr,
            sub_dx=sub_dx,
            sub_ds=sub_ds,
            qn_air=qn_air,
            qn_scale=lam,
            frozen_buoyancy_temperature=theta_cond,
            scales=scales,
        )

        w = solve_linear_problem(a_stokes, L_stokes, w, boundary_conditions)

        # accept Stage B
        w_n.assign(w)
        w_n.vector().apply("insert")
        update_material_from_mixed_temperature(fluid_material, w_n, scales, T_ambient)

    return w_n

def weak_open_boundary_momentum_term(
    mesh,
    u,
    psi_u,
    experiment,
    scales,
    outlet_penalty=1.0e-3,
    backflow_beta=5.0e-1,
):
    """
    Mild open-boundary stabilization on EAST + TOP only.

    - outlet_penalty * (u·n)(v·n) :
        weakly discourages large normal motion
    - backflow_beta * <-(u·n)> (u·v) :
        damps only inflow/backflow, not clean outflow

    SOUTH is intentionally excluded.
    """
    if outlet_penalty <= 0.0 and backflow_beta <= 0.0:
        return fenics.Constant(0.0) * fenics.dx(domain=mesh)

    ds_open, EAST_ID, TOP_ID, SOUTH_ID = build_open_boundary_measure(mesh, experiment, scales)
    n = fenics.FacetNormal(mesh)
    un = fenics.dot(u, n)

    ds_out = ds_open(EAST_ID) + ds_open(TOP_ID)

    term = 0

    if outlet_penalty > 0.0:
        term += fenics.Constant(float(outlet_penalty)) * fenics.dot(u, n) * fenics.dot(psi_u, n) * ds_out

    if backflow_beta > 0.0:
        un_in = 0.5 * (abs(un) - un)   # positive only when u·n < 0
        term += fenics.Constant(float(backflow_beta)) * un_in * fenics.dot(u, psi_u) * ds_out

    return term

def build_nonlinear_problem(
    W, w,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b,
    sub_dx, sub_ds, qn_air,
    buoyancy_scale=1.0,
    qn_scale=1.0,
    include_convection=True,
    convection_scale=1.0,
    SUPG=False,
    buoyancy_prefactor=None,
    experiment=None,
    scales=None,
    outlet_penalty=0.0,
    backflow_beta=0.0,
):
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym

    p, u, T = fenics.split(w)

    buoyancy_scale_c = fenics.Constant(float(buoyancy_scale))
    convection_scale_c = fenics.Constant(float(convection_scale))
    qn_scale_c = fenics.Constant(float(qn_scale))

    mass = -psi_p * div(u)

    convection_term = (
        convection_scale_c * dot(grad(u), u)
        if include_convection else fenics.Constant((0.0, 0.0))
    )

    if buoyancy_prefactor is not None:
        gvec = fenics.Constant((0.0, -1.0))
        buoyancy_force = buoyancy_prefactor * T * gvec
    else:
        buoyancy_force = f_b

    momentum = (
        dot(psi_u, convection_term + buoyancy_scale_c * buoyancy_force)
        - div(psi_u) * p
        + 2.0 * mu * inner(sym(grad(psi_u)), sym(grad(u)))
    )

    if SUPG:
        print("Using SUPG stabilization for thermal equation.")
        energy = thermal_galerkin_supg_form(
            mesh=W.mesh(),
            T=T,
            psi_T=psi_T,
            u=u,
            Pr=Pr,
            sub_dx=sub_dx,
            convection_scale=convection_scale,
        )
        F = (mass + momentum) * sub_dx + energy
    else:
        energy = dot(grad(psi_T), (1.0 / Pr) * grad(T) - T * u * convection_scale_c)
        F = (mass + momentum + energy) * sub_dx

    F += -qn_scale_c * qn_air * psi_T * sub_ds(INTERFACE_TAG)

    if experiment is not None and scales is not None:
        F += weak_open_boundary_momentum_term(
            mesh=W.mesh(),
            u=u,
            psi_u=psi_u,
            experiment=experiment,
            scales=scales,
            outlet_penalty=outlet_penalty,
            backflow_beta=backflow_beta,
        )

    JF = fenics.derivative(F, w, fenics.TrialFunction(W))
    return F, JF

def copy_state(dst: fenics.Function, src: fenics.Function):
    dst.vector()[:] = src.vector()
    dst.vector().apply("insert")

def build_stage_grid(lam: float):
    """
    Base convection-ramp grid for one continuation lambda.

    Keep this relatively coarse. If a step fails, it will be bisected automatically.
    """
    grid = [0.0, 0.10, 0.30, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00]
    if lam < 0.05:
        grid = [0.0, 0.10, 0.30, 0.50, 0.70, 0.90, 1.00]
    return grid

def try_newton_stage_FJF_outside(
    F,
    JF,
    w: fenics.Function,
    w_n: fenics.Function,
    boundary_conditions,
    relaxation_schedule=(0.9, 0.7, 0.5),
    maxit=20,
    atol=1e-9,
    rtol=1e-8,
    stage_name="",
):
    """
    Try one Newton solve from the last accepted state w_n.
    Returns:
        success: bool
        used_relaxation: float | None
        last_error: Exception | None
    """
    last_error = None

    for relaxation in relaxation_schedule:
        print(f"    Newton attempt [{stage_name}] with relaxation={relaxation:.3f}")

        # always restart from last accepted state
        w.vector()[:] = w_n.vector()
        w.vector().apply("insert")

        try:
            base_solver(
                F, w, boundary_conditions, JF,
                relaxation=relaxation,
                maxit=maxit,
                atol=atol,
                rtol=rtol,
            )
            return True, relaxation, None

        except RuntimeError as err:
            last_error = err
            print(f"    failed [{stage_name}] with relaxation={relaxation:.3f}")

    return False, None, last_error

def accept_current_state(w: fenics.Function, w_n: fenics.Function):
    """
    Promote current iterate to last accepted state.
    """
    w_n.vector()[:] = w.vector()
    w_n.vector().apply("insert")

def conv_stage_sequence():
    """
    Coarse monotone convection ladder.
    Adjust this if needed, but keep it increasing.
    """
    return [0.20, 0.40, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00]

def refine_conv_interval(left, right, min_interval=0.01):
    """
    Return midpoint if interval is still worth refining, else None.
    """
    mid = 0.5 * (left + right)
    if (right - left) < min_interval:
        return None
    return mid

def advance_convection_monotone(
    W, w,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b,
    sub_dx, sub_ds, qn_air,
    boundary_conditions,
    w_n: fenics.Function,
    buoyancy_scale: float,
    start_conv: float,
    target_conv: float,
    buoyancy_prefactor=None,
    fluid_material=None,
    scales=None,
    T_ambient: float = 0.0,
    relaxation_schedule=(0.9, 0.7, 0.5),
    min_interval=0.01,
    max_local_bisections=3,
):
    """
    Advance monotonically from start_conv to target_conv.
    If target fails, bisect only that interval until either:
      - target is reached
      - interval becomes too small
      - local bisection limit is reached

    Returns:
        accepted_conv: float
    """
    accepted_conv = float(start_conv)
    pending_targets = [float(target_conv)]
    n_bisect = 0
    F_accepted, JF_accepted = None, None

    while pending_targets:
        trial_conv = pending_targets.pop(0)

        if trial_conv <= accepted_conv + 1e-14:
            continue

        print(
            f"  -> convection advance: accepted={accepted_conv:.4f} "
            f"target={trial_conv:.4f}"
        )

        if fluid_material is not None and scales is not None and T_ambient is not None:
            update_material_from_mixed_temperature(w_n, fluid_material, scales, T_ambient)

        F, JF = build_nonlinear_problem(
            W=W, w=w,
            psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
            mu=mu, Pr=Pr, f_b=f_b,
            sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
            buoyancy_scale=buoyancy_scale,
            qn_scale=buoyancy_scale,
            include_convection=True,
            convection_scale=trial_conv,
            buoyancy_prefactor=buoyancy_prefactor,
            experiment=experiment,
            scales=scales,
            outlet_penalty=1.0e-3,
            backflow_beta=5.0e-1,
        )

        ok, used_relax, last_error = try_newton_stage_FJF_outside(
            F=F,
            JF=JF,
            w=w,
            w_n=w_n,
            boundary_conditions=boundary_conditions,
            relaxation_schedule=relaxation_schedule,
            stage_name=f"conv_{trial_conv:.4f}",
        )

        if ok:
            accept_current_state(w, w_n)

            if fluid_material is not None and scales is not None and T_ambient is not None:
                update_material_from_mixed_temperature(w_n, fluid_material, scales, T_ambient)

            accepted_conv = trial_conv
            F_accepted, JF_accepted = F, JF

            print(
                f"    accepted convection scale {accepted_conv:.4f} "
                f"(relaxation={used_relax:.3f})"
            )
            continue

        midpoint = refine_conv_interval(accepted_conv, trial_conv, min_interval=min_interval)

        if midpoint is None or n_bisect >= max_local_bisections:
            raise RuntimeError(
                f"Could not advance convection beyond {accepted_conv:.4f}; "
                f"failed target={trial_conv:.4f}. Last error: {last_error}"
            )

        n_bisect += 1
        print(
            f"    target {trial_conv:.4f} failed; "
            f"bisecting interval [{accepted_conv:.4f}, {trial_conv:.4f}] "
            f"-> {midpoint:.4f}"
        )

        pending_targets = [midpoint, trial_conv] + pending_targets

    return accepted_conv, F_accepted, JF_accepted

def build_ptc_problem(
    W, w, w_prev,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b,
    sub_dx, sub_ds, qn_air,
    dtau,
    buoyancy_scale=1.0,
    qn_scale=1.0,
    include_convection=True,
    convection_scale=1.0,
    SUPG=False,
    buoyancy_prefactor=None,
    experiment=None,
    scales=None,
    outlet_penalty=0.0,
    backflow_beta=0.0,
):
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym

    p, u, T = fenics.split(w)
    _, u_prev, T_prev = fenics.split(w_prev)

    buoyancy_scale_c = fenics.Constant(float(buoyancy_scale))
    convection_scale_c = fenics.Constant(float(convection_scale))
    qn_scale_c = fenics.Constant(float(qn_scale))
    dtau_c = fenics.Constant(float(dtau))

    convection_term = (
        convection_scale_c * dot(grad(u), u)
        if include_convection else fenics.Constant((0.0, 0.0))
    )

    mass = -psi_p * div(u)

    pseudo_velocity = (1.0 / dtau_c) * inner(psi_u, u - u_prev)
    pseudo_temperature = (1.0 / dtau_c) * psi_T * (T - T_prev)

    if buoyancy_prefactor is not None:
        gvec = fenics.Constant((0.0, -1.0))
        buoyancy_force = buoyancy_prefactor * T * gvec
    else:
        buoyancy_force = f_b

    momentum = (
        dot(psi_u, convection_term + buoyancy_scale_c * buoyancy_force)
        - div(psi_u) * p
        + 2.0 * mu * inner(sym(grad(psi_u)), sym(grad(u)))
    )

    if SUPG:
        print("  building SUPG-stabilized energy form for PTC problem")
        energy = thermal_galerkin_supg_form(
            mesh=W.mesh(),
            T=T,
            psi_T=psi_T,
            u=u,
            Pr=Pr,
            sub_dx=sub_dx,
            convection_scale=convection_scale,
            T_prev=T_prev,
            dtau=dtau_c,
        )
        F = (mass + pseudo_velocity + momentum) * sub_dx + pseudo_temperature * sub_dx + energy
    else:
        energy = dot(grad(psi_T), (1.0 / Pr) * grad(T) - T * u * convection_scale_c)
        F = (mass + pseudo_velocity + momentum + pseudo_temperature + energy) * sub_dx

    F += -qn_scale_c * qn_air * psi_T * sub_ds(INTERFACE_TAG)

    if experiment is not None and scales is not None:
        F += weak_open_boundary_momentum_term(
            mesh=W.mesh(),
            u=u,
            psi_u=psi_u,
            experiment=experiment,
            scales=scales,
            outlet_penalty=outlet_penalty,
            backflow_beta=backflow_beta,
        )

    JF = fenics.derivative(F, w, fenics.TrialFunction(W))
    return F, JF

def vector_relative_update(w_new: fenics.Function, w_old: fenics.Function) -> float:
    dw = w_new.vector().copy()
    dw.axpy(-1.0, w_old.vector())
    return dw.norm("l2") / (w_new.vector().norm("l2") + 1e-14)

def steady_residual_norm(
    W,
    w,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b,
    sub_dx, sub_ds, qn_air,
    boundary_conditions,
    buoyancy_scale=1.0,
    qn_scale=1.0,
    include_convection=True,
    convection_scale=1.0,
    experiment=None,
    scales=None,
    outlet_penalty=1.0e-3,
    backflow_beta=5.0e-1,

) -> float:
    F_steady, _ = build_nonlinear_problem(
        W=W, w=w,
        psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
        mu=mu, Pr=Pr, f_b=f_b,
        sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
        buoyancy_scale=buoyancy_scale,
        qn_scale=qn_scale,
        include_convection=include_convection,
        convection_scale=convection_scale,
        experiment=experiment,
        scales=scales,
        outlet_penalty=1.0e-3,
        backflow_beta=5.0e-1,
    )

    r = fenics.assemble(F_steady)
    for bc in boundary_conditions:
        bc.apply(r)

    return r.norm("l2")

def collect_observables(w: fenics.Function) -> dict:
    p_f, u_f, T_f = w.split(deepcopy=True)
    u_vec = u_f.vector().get_local()
    T_vec = T_f.vector().get_local()
    p_vec = p_f.vector().get_local()

    return {
        "u_l2": u_f.vector().norm("l2"),
        "T_l2": T_f.vector().norm("l2"),
        "p_l2": p_f.vector().norm("l2"),
        "u_max_abs": float(np.max(np.abs(u_vec))) if u_vec.size else 0.0,
        "T_max": float(np.max(T_vec)) if T_vec.size else 0.0,
        "T_min": float(np.min(T_vec)) if T_vec.size else 0.0,
        "p_max_abs": float(np.max(np.abs(p_vec))) if p_vec.size else 0.0,
    }

def history_window_increasing(values, window=5):
    if len(values) < window:
        return False
    tail = values[-window:]
    return all(tail[i] > tail[i - 1] for i in range(1, len(tail)))

def history_window_nondecreasing(values, window=5):
    if len(values) < window:
        return False
    tail = values[-window:]
    return all(tail[i] >= tail[i - 1] for i in range(1, len(tail)))

def assembled_residual_norm(F_form, boundary_conditions):
    r = fenics.assemble(F_form)
    for bc in boundary_conditions:
        bc.apply(r)
    return r.norm("l2")

def safe_eval_scalar(f, x, y):
    try:
        val = f(x, y)
        if isinstance(val, (tuple, list)):
            return float(val[0])
        return float(val)
    except Exception:
        return float("nan")

def safe_eval_vector_component(f, x, y, comp=1):
    try:
        val = f(x, y)
        return float(val[comp])
    except Exception:
        return float("nan")

def collect_ptc_probe_diagnostics(w, sub_dx, probe_ys, x_probe=0.0):
    """
    Collect a compact set of scalar diagnostics for PTC monitoring.
    """
    p_f, u_f, T_f = w.split(deepcopy=True)

    u_l2 = float(u_f.vector().norm("l2"))
    T_l2 = float(T_f.vector().norm("l2"))
    kinetic_energy = float(fenics.assemble(0.5 * fenics.inner(u_f, u_f) * sub_dx))

    data = {
        "u_l2": u_l2,
        "T_l2": T_l2,
        "kinetic_energy": kinetic_energy,
    }

    for y in probe_ys:
        tag = str(y).replace(".", "p")
        data[f"uy_y{tag}"] = safe_eval_vector_component(u_f, 1e-4, y, comp=1)
        data[f"T_y{tag}"] = safe_eval_scalar(T_f, 1e-4, y)

    return data

def init_ptc_csv(log_path, probe_ys):
    fieldnames = [
        "step",
        "pseudo_time",
        "dtau",
        "rel_update",
        "steady_residual",
        "u_l2",
        "T_l2",
        "kinetic_energy",
    ]

    for y in probe_ys:
        tag = str(y).replace(".", "p")
        fieldnames.append(f"uy_y{tag}")
    for y in probe_ys:
        tag = str(y).replace(".", "p")
        fieldnames.append(f"T_y{tag}")

    # os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

    return fieldnames

def append_ptc_csv(log_path, fieldnames, row):
    with open(log_path, "a", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writerow(row)

def compute_cfl(sub_mesh, u, dt):
    """
    u  : velocity Function on the current mesh
    dt : current nondimensional timestep
    """
    V0 = fenics.FunctionSpace(sub_mesh, "DG", 0)

    h = fenics.project(fenics.CellDiameter(sub_mesh), V0)
    umag = fenics.project(fenics.sqrt(fenics.inner(u, u)), V0)

    h_loc = h.vector().get_local()
    u_loc = umag.vector().get_local()

    cfl_loc = dt * u_loc / np.maximum(h_loc, 1e-14)
    cfl_max = float(np.max(cfl_loc))
    cfl_mean = float(np.mean(cfl_loc))

    cfl_fun = fenics.Function(V0, name="CFL")
    cfl_fun.vector()[:] = cfl_loc
    cfl_fun.vector().apply("insert")

    return cfl_max, cfl_mean, cfl_fun

def cfl_limited_dt(sub_mesh, u, cfl_target=1.0, safety=0.9, dt_min=1e-5, dt_max=1.0):
    V0 = fenics.FunctionSpace(sub_mesh, "DG", 0)
    h = fenics.project(fenics.CellDiameter(sub_mesh), V0)
    umag = fenics.project(fenics.sqrt(fenics.inner(u, u)), V0)

    h_loc = h.vector().get_local()
    u_loc = umag.vector().get_local()

    speed_over_h = u_loc / np.maximum(h_loc, 1e-14)
    denom = np.max(speed_over_h)

    if denom < 1e-14:
        return dt_max

    dt_new = safety * cfl_target / denom
    return max(dt_min, min(dt_max, float(dt_new)))
