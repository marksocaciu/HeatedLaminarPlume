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
                relaxation=0.5, maxit=80, atol=1e-7, rtol=1e-6):
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

    solver.solve()
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
                     convection_scale=1.0):

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

    energy = (dot(grad(psi_T), (1.0/Pr) * grad(T) - T * u))

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

def _build_temperature_assigner(W: fenics.FunctionSpace):
    VT, _ = W.sub(2).collapse(True)
    assign_T = fenics.FunctionAssigner(W.sub(2), VT)
    return VT, assign_T

def _assign_mixed_temperature(
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

def _build_linear_startup_problem(
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
    VT, assign_T = _build_temperature_assigner(W)
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

        _assign_mixed_temperature(w_n, theta_lam, VT, assign_T, theta_tmp)

        # start each stage from latest accepted mixed state
        w.vector()[:] = w_n.vector()
        w.vector().apply("insert")

        print("  -> Stage A: conduction-only solve")
        a_cond, L_cond = _build_linear_startup_problem(
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
        a_stokes, L_stokes = _build_linear_startup_problem(
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

def build_nonlinear_problem(
    W, w,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b,
    sub_dx, sub_ds, qn_air,
    buoyancy_scale=1.0,
    qn_scale=1.0,
    include_convection=True,
    convection_scale=1.0,
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

    momentum = (
        dot(psi_u, convection_term + buoyancy_scale_c * f_b)
        - div(psi_u) * p
        + 2.0 * mu * inner(sym(grad(psi_u)), sym(grad(u)))
    )

    energy = dot(grad(psi_T), (1.0 / Pr) * grad(T) - T * u * convection_scale_c)

    F = (mass + momentum + energy) * sub_dx
    F += -qn_scale_c * qn_air * psi_T * sub_ds(INTERFACE_TAG)

    JF = fenics.derivative(F, w, fenics.TrialFunction(W))
    return F, JF


def _copy_state(dst: fenics.Function, src: fenics.Function):
    dst.vector()[:] = src.vector()
    dst.vector().apply("insert")


def _build_stage_grid(lam: float):
    """
    Base convection-ramp grid for one continuation lambda.

    Keep this relatively coarse. If a step fails, it will be bisected automatically.
    """
    grid = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    if lam < 0.05:
        grid = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00]
    return grid


# def _try_newton_stage(
#     w: fenics.Function,
#     w_seed: fenics.Function,
#     boundary_conditions,
#     W,
#     psi_p, psi_u, psi_T,
#     mu, Pr, f_b,
#     sub_dx, sub_ds, qn_air,
#     lam: float,
#     include_convection: bool,
#     conv_scale: float,
#     relaxation_schedule,
#     maxit: int = 60,
#     atol: float = 1e-9,
#     rtol: float = 1e-8,
# ):
#     """
#     Try a steady Newton solve for a single stage starting from w_seed.
#     Returns (success, last_error, used_relaxation).
#     """
#     F, JF = build_nonlinear_problem(
#         W=W, w=w,
#         psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
#         mu=mu, Pr=Pr, f_b=f_b,
#         sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
#         buoyancy_scale=lam,
#         qn_scale=lam,
#         include_convection=include_convection,
#         convection_scale=conv_scale,
#     )

#     last_error = None
#     for relaxation in relaxation_schedule:
#         print(f"    Newton try: conv_scale={conv_scale:.6f}, relaxation={relaxation:.3f}")
#         _copy_state(w, w_seed)
#         try:
#             base_solver(
#                 F, w, boundary_conditions, JF,
#                 relaxation=relaxation,
#                 maxit=maxit,
#                 atol=atol,
#                 rtol=rtol,
#             )
#             return True, None, relaxation
#         except RuntimeError as err:
#             last_error = err
#             print(f"    Newton failed at conv_scale={conv_scale:.6f}, relaxation={relaxation:.3f}")

#     return False, last_error, None

def _clone_state(src: fenics.Function) -> fenics.Function:
    dst = fenics.Function(src.function_space())
    _copy_state(dst, src)
    return dst


def _try_newton_stage(
    F,
    JF,
    w: fenics.Function,
    w_n: fenics.Function,
    boundary_conditions,
    relaxation_schedule=(0.9, 0.7, 0.5),
    maxit=60,
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

def _accept_current_state(w: fenics.Function, w_n: fenics.Function):
    """
    Promote current iterate to last accepted state.
    """
    w_n.vector()[:] = w.vector()
    w_n.vector().apply("insert")

def _conv_stage_sequence():
    """
    Coarse monotone convection ladder.
    Adjust this if needed, but keep it increasing.
    """
    return [0.20, 0.40, 0.60, 0.70, 0.80, 0.90, 1.00]

def _refine_conv_interval(left, right, min_interval=0.01):
    """
    Return midpoint if interval is still worth refining, else None.
    """
    mid = 0.5 * (left + right)
    if (right - left) < min_interval:
        return None
    return mid

def _advance_convection_monotone(
    W, w,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b,
    sub_dx, sub_ds, qn_air,
    boundary_conditions,
    w_n: fenics.Function,
    buoyancy_scale: float,
    start_conv: float,
    target_conv: float,
    relaxation_schedule=(0.9, 0.7, 0.5),
    min_interval=0.01,
    max_local_bisections=20,
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

    while pending_targets:
        trial_conv = pending_targets.pop(0)

        if trial_conv <= accepted_conv + 1e-14:
            continue

        print(
            f"  -> convection advance: accepted={accepted_conv:.4f} "
            f"target={trial_conv:.4f}"
        )

        F, JF = build_nonlinear_problem(
            W=W, w=w,
            psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
            mu=mu, Pr=Pr, f_b=f_b,
            sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
            buoyancy_scale=buoyancy_scale,
            qn_scale=buoyancy_scale,
            include_convection=True,
            convection_scale=trial_conv,
        )

        ok, used_relax, last_error = _try_newton_stage(
            F=F,
            JF=JF,
            w=w,
            w_n=w_n,
            boundary_conditions=boundary_conditions,
            relaxation_schedule=relaxation_schedule,
            stage_name=f"conv_{trial_conv:.4f}",
        )

        if ok:
            _accept_current_state(w, w_n)
            accepted_conv = trial_conv
            print(
                f"    accepted convection scale {accepted_conv:.4f} "
                f"(relaxation={used_relax:.3f})"
            )
            continue

        midpoint = _refine_conv_interval(accepted_conv, trial_conv, min_interval=min_interval)

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

        # try midpoint first, then come back to original target later
        pending_targets = [midpoint, trial_conv] + pending_targets

    return accepted_conv

# def _adaptive_convection_path(
#     *,
#     experiment: Experiment,
#     W,
#     w: fenics.Function,
#     w_start: fenics.Function,
#     psi_p, psi_u, psi_T,
#     mu, Pr, f_b, T_c, T_air_bc,
#     sub_dx, sub_ds, sub_ft, qn_air,
#     lam: float,
#     boundary_conditions,
#     relaxation_schedule=(0.2, 0.1, 0.05),
#     maxit: int = 40,
#     atol: float = 1e-9,
#     rtol: float = 1e-8,
#     start_conv: float = 0.0,
#     target_conv: float = 1.0,
#     initial_step: float = 0.2,
#     max_step: float = 0.2,
#     min_step: float = 0.005,
# ):
#     """
#     Adaptive local continuation in convection_scale from start_conv to target_conv.

#     Strategy:
#       - try a jump current -> current + step
#       - if it converges, accept and maybe enlarge the next step
#       - if it fails, try PTC from the failed iterate
#       - if still not enough, cut the step in half
#     """
#     current_conv = float(start_conv)
#     step = float(initial_step)
#     current_state = _clone_state(w_start)

#     while current_conv < target_conv - 1e-14:
#         trial_conv = min(target_conv, current_conv + step)

#         ok, accepted_state, failed_state, used_relax, err = _try_newton_stage(
#             W=W,
#             w=w,
#             w_init=current_state,
#             psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
#             mu=mu, Pr=Pr, f_b=f_b,
#             sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
#             lam=lam,
#             include_convection=(trial_conv > 0.0),
#             conv_scale=trial_conv,
#             boundary_conditions=boundary_conditions,
#             relaxation_schedule=relaxation_schedule,
#             maxit=maxit,
#             atol=atol,
#             rtol=rtol,
#             stage_name=f"adaptive_conv_{trial_conv:.6f}",
#         )

#         if ok:
#             current_conv = trial_conv
#             current_state = accepted_state
#             step = min(max_step, 1.35 * step)
#             print(
#                 f"  accepted adaptive step to conv_scale={current_conv:.6f} "
#                 f"with relaxation={used_relax:.3f}"
#             )
#             continue

#         print(
#             f"  Newton stalled between conv_scale={current_conv:.6f} "
#             f"and {trial_conv:.6f}; trying PTC rescue from failed iterate"
#         )

#         ptc_success = pseudo_transient_rescue(
#             experiment=experiment,
#             W=W,
#             w=w,
#             w_n=failed_state,   # important: rescue from failed iterate, not old accepted state
#             psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
#             mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
#             sub_dx=sub_dx, sub_ds=sub_ds, sub_ft=sub_ft, qn_air=qn_air,
#             lam=lam,
#             conv_scale=trial_conv,
#             dtau_schedule=(1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
#             steps_per_dtau=6,
#             update_tol=1e-8,
#             retry_newton_every=3,
#             relaxation=1.0,
#         )

#         if ptc_success:
#             rescued_seed = _clone_state(failed_state)

#             ok2, accepted_state2, failed_state2, used_relax2, err2 = _try_newton_stage(
#                 W=W,
#                 w=w,
#                 w_init=rescued_seed,
#                 psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
#                 mu=mu, Pr=Pr, f_b=f_b,
#                 sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
#                 lam=lam,
#                 include_convection=(trial_conv > 0.0),
#                 conv_scale=trial_conv,
#                 boundary_conditions=boundary_conditions,
#                 relaxation_schedule=relaxation_schedule,
#                 maxit=maxit,
#                 atol=atol,
#                 rtol=rtol,
#                 stage_name=f"adaptive_conv_{trial_conv:.6f}_after_ptc",
#             )

#             if ok2:
#                 current_conv = trial_conv
#                 current_state = accepted_state2
#                 step = min(max_step, 1.25 * step)
#                 print(
#                     f"  accepted adaptive step after PTC to conv_scale={current_conv:.6f} "
#                     f"with relaxation={used_relax2:.3f}"
#                 )
#                 continue

#         step *= 0.5
#         print(
#             f"  shrinking continuation step; current={current_conv:.6f}, "
#             f"trial={trial_conv:.6f}, new_step={step:.6f}"
#         )

#         if step < min_step:
#             raise RuntimeError(
#                 f"Adaptive continuation failed at lambda={lam:.2f} "
#                 f"near conv_scale={current_conv:.6f}."
#             )

#     return current_state

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
):
    """
    Backward-Euler pseudo-transient problem for the mixed steady system.

    The pseudo-time mass is added only to velocity and temperature.
    Pressure remains algebraic, which is appropriate for incompressible flow.
    """
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

    momentum = (
        dot(psi_u, convection_term + buoyancy_scale_c * f_b)
        - div(psi_u) * p
        + 2.0 * mu * inner(sym(grad(psi_u)), sym(grad(u)))
    )

    energy = dot(grad(psi_T), (1.0 / Pr) * grad(T) - T * u * convection_scale_c)

    F = (mass + pseudo_velocity + momentum + pseudo_temperature + energy) * sub_dx
    F += -qn_scale_c * qn_air * psi_T * sub_ds(INTERFACE_TAG)

    JF = fenics.derivative(F, w, fenics.TrialFunction(W))
    return F, JF


def pseudo_transient_rescue(
    experiment: Experiment,
    W: fenics.FunctionSpace,
    w: fenics.Function,
    w_n: fenics.Function,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b, T_c, T_air_bc,
    sub_dx, sub_ds, sub_ft, qn_air,
    lam: float,
    conv_scale: float,
    dtau_schedule=(1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    steps_per_dtau: int = 6,
    update_tol: float = 1e-8,
    retry_newton_every: int = 3,
    relaxation: float = 1.0,
):
    """
    Pseudo-transient rescue started from the provided seed state w_n.

    In the adaptive continuation workflow, w_n should usually be the
    failed Newton iterate at the difficult continuation stage.
    """
    scales = compute_nondimensional_scales(experiment)
    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

    w_prev = fenics.Function(W)
    _copy_state(w, w_n)
    _copy_state(w_prev, w_n)

    ptc_step = 0
    for dtau in dtau_schedule:
        print(f"    -> PTC block with dtau={dtau:.3e}")
        for _ in range(steps_per_dtau):
            ptc_step += 1
            _copy_state(w_prev, w)

            F_ptc, JF_ptc = build_ptc_problem(
                W=W, w=w, w_prev=w_prev,
                psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
                mu=mu, Pr=Pr, f_b=f_b,
                sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
                dtau=dtau,
                buoyancy_scale=lam,
                qn_scale=lam,
                include_convection=(conv_scale > 0.0),
                convection_scale=conv_scale,
            )

            try:
                base_solver(
                    F_ptc, w, boundary_conditions, JF_ptc,
                    relaxation=relaxation,
                    maxit=40,
                    atol=1e-9,
                    rtol=1e-8,
                )
            except RuntimeError as err:
                print(f"    PTC step failed at dtau={dtau:.3e}: {err}")
                return False

            delta = w.vector().copy()
            delta.axpy(-1.0, w_prev.vector())
            rel_update = delta.norm("l2") / (w.vector().norm("l2") + 1e-14)
            print(f"    PTC step {ptc_step:03d}: rel_update={rel_update:.3e}")

            _copy_state(w_n, w)

            if rel_update < update_tol:
                print("    PTC update is small; handing control back to steady Newton.")
                return True

            if (ptc_step % retry_newton_every) == 0:
                print("    PTC block produced a new seed; retry steady Newton now.")
                return True

    return True

def solve_steady_newton_continuation(
    experiment: Experiment,
    u_n: fenics.Function, u: fenics.Function, T_n: fenics.Function, T: fenics.Function, p: fenics.Function,
    W: fenics.FunctionSpace, w: fenics.Function,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b, T_c, T_air_bc,
    sub_dx, sub_ds, sub_ft, qn_air,
    w_n: fenics.Function,
    lambdas=None,
    relaxation_schedule=(0.9, 0.7, 0.5),
    stokes_startup=True,
    sub_mesh_star=None,
    sub_mesh_dim=None,
    p_path: str = "",
    u_path: str = "",
    T_path: str = "",
):
    """
    Steady continuation solve with monotone convection ramping.

    For each lambda:
      1) solve Stokes / no-momentum-convection stage
      2) ramp convection monotonically with a coarse ladder
      3) if a target convection scale fails, bisect only that interval

    This avoids the wasteful pattern:
      direct full solve -> many failed relaxations -> fallback continuation
    """
    if lambdas is None:
        lambdas = [0.05, 0.10, 0.20, 0.40, 0.70, 1.00]

    w.vector()[:] = w_n.vector()
    w.vector().apply("insert")

    scales = compute_nondimensional_scales(experiment)
    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

    conv_targets = _conv_stage_sequence()

    for lam in lambdas:
        print(f"\n=== Newton continuation lambda = {lam:.2f} ===")

        # optional output of current accepted state before advancing lambda
        if sub_mesh_star is not None and sub_mesh_dim is not None:
            p_star, u_star, theta = w_n.split(deepcopy=True)

            u_dim, p_dim, T_dim = dimensionalize_fields(
                sub_mesh_star, u_star, p_star, theta,
                scales.Uref, scales.dTref, T_ambient,
                experiment.fluid.properties["rho"]
            )

            p_out = p_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
            u_out = u_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
            t_out = T_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"

            save_experiment(p_out, sub_mesh_dim, [p_dim])
            save_experiment(u_out, sub_mesh_dim, [u_dim])
            save_experiment(t_out, sub_mesh_dim, [T_dim])

        # ------------------------------------------------------------
        # Stage 1: Stokes / zero momentum convection
        # ------------------------------------------------------------
        print("  --- stage: stokes ---")

        F_stokes, JF_stokes = build_nonlinear_problem(
            W=W, w=w,
            psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
            mu=mu, Pr=Pr, f_b=f_b,
            sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
            buoyancy_scale=lam,
            qn_scale=lam,
            include_convection=False,
            convection_scale=0.0,
        )

        ok, used_relax, last_error = _try_newton_stage(
            F=F_stokes,
            JF=JF_stokes,
            w=w,
            w_n=w_n,
            boundary_conditions=boundary_conditions,
            relaxation_schedule=relaxation_schedule,
            stage_name=f"lambda_{lam:.2f}_stokes",
        )

        if not ok:
            raise RuntimeError(
                f"Continuation Newton failed at lambda={lam:.2f} during stokes stage. "
                f"Last error: {last_error}"
            )

        _accept_current_state(w, w_n)
        print(f"  stokes accepted at lambda={lam:.2f} (relaxation={used_relax:.3f})")

        # ------------------------------------------------------------
        # Stage 2: monotone convection advance
        # ------------------------------------------------------------
        accepted_conv = 0.0
        for target_conv in conv_targets:
            accepted_conv = _advance_convection_monotone(
                W=W, w=w,
                psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
                mu=mu, Pr=Pr, f_b=f_b,
                sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
                boundary_conditions=boundary_conditions,
                w_n=w_n,
                buoyancy_scale=lam,
                start_conv=accepted_conv,
                target_conv=target_conv,
                relaxation_schedule=relaxation_schedule,
                min_interval=0.005,          # tighten/loosen as needed
                max_local_bisections=25,
            )

        print(f"  lambda={lam:.2f} completed with accepted conv_scale={accepted_conv:.4f}")

        # keep working function synchronized with accepted state
        w.vector()[:] = w_n.vector()
        w.vector().apply("insert")

    return w

# def solve_steady_newton_continuation(
#     experiment: Experiment,
#     u_n: fenics.Function, u: fenics.Function, T_n: fenics.Function, T: fenics.Function, p: fenics.Function,
#     W: fenics.FunctionSpace, w: fenics.Function,
#     psi_p, psi_u, psi_T,
#     mu, Pr, f_b, T_c, T_air_bc,
#     sub_dx, sub_ds, sub_ft, qn_air,
#     w_n: fenics.Function,
#     lambdas=None,
#     relaxation_schedule=(0.9, 0.7, 0.5),
#     stokes_startup=True,
#     sub_mesh_star=None,
#     sub_mesh_dim=None,
#     p_path: str = "",
#     u_path: str = "",
#     T_path: str = "",
#     initial_conv_step: float = 0.2,
#     max_conv_step: float = 0.2,
#     min_conv_step: float = 0.005,
# ):
#     """
#     Adaptive steady continuation:

#     For each lambda:
#       1) try the full stage directly
#       2) if that fails, solve a zero-convection stage
#       3) continue convection adaptively from 0 -> 1
#       4) if a local stage stalls, use PTC only as a rescue device
#     """
#     if lambdas is None:
#         lambdas = [0.05, 0.10, 0.20, 0.40, 0.70, 1.00]

#     _copy_state(w, w_n)
#     scales = compute_nondimensional_scales(experiment)
#     boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

#     p_base = p_path.split(".xdmf")[0] if p_path else ""
#     u_base = u_path.split(".xdmf")[0] if u_path else ""
#     T_base = T_path.split(".xdmf")[0] if T_path else ""

#     for lam in lambdas:
#         print(f"\n=== Newton continuation lambda = {lam:.2f} ===")

#         accepted_lambda_state = _clone_state(w_n)

#         if sub_mesh_star is not None and sub_mesh_dim is not None and p_base and u_base and T_base:
#             p_star, u_star, theta = accepted_lambda_state.split(deepcopy=True)
#             u_dim, p_dim, T_dim = dimensionalize_fields(
#                 sub_mesh_star, u_star, p_star, theta,
#                 scales.Uref, scales.dTref, T_ambient,
#                 experiment.fluid.properties["rho"],
#             )
#             tag = int(round(100.0 * lam))
#             save_experiment(f"{p_base}_lambda_{tag:03d}.xdmf", sub_mesh_dim, [p_dim])
#             save_experiment(f"{u_base}_lambda_{tag:03d}.xdmf", sub_mesh_dim, [u_dim])
#             save_experiment(f"{T_base}_lambda_{tag:03d}.xdmf", sub_mesh_dim, [T_dim])

#         # 1) Try the full stage immediately
#         print("  -> direct full-stage attempt")
#         ok_full, full_state, failed_state, used_relax, err = _try_newton_stage(
#             W=W,
#             w=w,
#             w_init=accepted_lambda_state,
#             psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
#             mu=mu, Pr=Pr, f_b=f_b,
#             sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
#             lam=lam,
#             include_convection=True,
#             conv_scale=1.0,
#             boundary_conditions=boundary_conditions,
#             relaxation_schedule=relaxation_schedule,
#             stage_name=f"lambda_{lam:.2f}_full_direct",
#         )

#         if ok_full:
#             _copy_state(w_n, full_state)
#             print(f"  accepted lambda={lam:.2f} directly")
#             continue

#         # 2) Fallback: robust Stokes / zero-convection stage
#         print("  -> direct full stage failed; solving zero-convection stage")
#         ok_stokes, stokes_state, failed_state, used_relax_s, err_s = _try_newton_stage(
#             W=W,
#             w=w,
#             w_init=accepted_lambda_state,
#             psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
#             mu=mu, Pr=Pr, f_b=f_b,
#             sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
#             lam=lam,
#             include_convection=False,
#             conv_scale=0.0,
#             boundary_conditions=boundary_conditions,
#             relaxation_schedule=relaxation_schedule,
#             stage_name=f"lambda_{lam:.2f}_stokes",
#         )

#         if not ok_stokes:
#             raise RuntimeError(
#                 f"Continuation failed already at zero-convection stage for lambda={lam:.2f}."
#             )

#         # 3) Adaptive local continuation only where needed
#         final_state = _adaptive_convection_path(
#             experiment=experiment,
#             W=W,
#             w=w,
#             w_start=stokes_state,
#             psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
#             mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
#             sub_dx=sub_dx, sub_ds=sub_ds, sub_ft=sub_ft, qn_air=qn_air,
#             lam=lam,
#             boundary_conditions=boundary_conditions,
#             relaxation_schedule=relaxation_schedule,
#             start_conv=0.0,
#             target_conv=1.0,
#             initial_step=initial_conv_step,
#             max_step=max_conv_step,
#             min_step=min_conv_step,
#         )

#         _copy_state(w_n, final_state)
#         print(f"  accepted lambda={lam:.2f} after adaptive fallback")

#     _copy_state(w, w_n)
#     return w

# def solve_steady_newton_continuation(
#     experiment: Experiment,
#     u_n: fenics.Function, u: fenics.Function, T_n: fenics.Function, T: fenics.Function, p: fenics.Function,
#     W: fenics.FunctionSpace, w: fenics.Function,
#     psi_p, psi_u, psi_T,
#     mu, Pr, f_b, T_c, T_air_bc,
#     sub_dx, sub_ds, sub_ft, qn_air,
#     w_n: fenics.Function,
#     lambdas=None,
#     relaxation_schedule=(0.2, 0.1, 0.05),
#     stokes_startup=True,
#     sub_mesh_star=None,
#     sub_mesh_dim=None,
#     p_path: str = "",
#     u_path: str = "",
#     T_path: str = "",
# ):
#     """
#       Hybrid steady solve:
#       - primary method: damped Newton continuation
#       - on a failed convection substage: bisect the failed step
#       - if the bisected step still fails: pseudo-transient rescue
#       - then retry steady Newton from the rescued state

#     This keeps the target solution steady, while using pseudo-time marching only as a
#     globalization device near difficult continuation points.

#     For each lambda:
#       1) solve a no-momentum-convection stage
#       2) solve the full nonlinear stage
#     and promote the converged solution after each successful stage.
#     """
#     if lambdas is None:
#         lambdas = [0.05, 0.10, 0.20, 0.40, 0.70, 1.00]

#     # w.vector()[:] = w_n.vector()
#     # w.vector().apply("insert")
#     _copy_state(w, w_n)
#     scales = compute_nondimensional_scales(experiment)
#     boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)
#     max_bisection_depth = 6

#     for lam in lambdas:
#         print(f"\n=== Newton continuation lambda = {lam:.2f} ===")
        
#         if sub_mesh_star is not None and sub_mesh_dim is not None and p_path and u_path and T_path:
#             p_star, u_star, theta = w.split(deepcopy=True)
#             u_dim, p_dim, T_dim = dimensionalize_fields(
#                 sub_mesh_star, u_star, p_star, theta,
#                 scales.Uref, scales.dTref, T_ambient,
#                 experiment.fluid.properties["rho"],
#             )
#             p_path_l = p_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
#             v_path_l = u_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
#             t_path_l = T_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
#             save_experiment(p_path_l, sub_mesh_dim, [p_dim])
#             save_experiment(v_path_l, sub_mesh_dim, [u_dim])
#             save_experiment(t_path_l, sub_mesh_dim, [T_dim])

#         # if lam < 0.05:
#         #         stage_attempts = [
#         #         ("stokes",   False, 0.00),
#         #         ("conv_005", True,  0.05),
#         #         ("conv_010", True,  0.10),
#         #         ("conv_020", True,  0.20),
#         #         ("conv_030", True,  0.30),
#         #         ("conv_040", True,  0.40),
#         #         ("conv_050", True,  0.50),
#         #         # ("conv_060", True,  0.60),
#         #         ("conv_070", True,  0.70),
#         #         # ("conv_080", True,  0.80),
#         #         ("conv_090", True,  0.90),
#         #         ("full",     True,  1.00),
#         #     ]
#         # else:
#         #     stage_attempts = [
#         #         ("stokes",   False, 0.00),
#         #         ("conv_005", True,  0.05),
#         #         ("conv_010", True,  0.10),
#         #         ("conv_020", True,  0.20),
#         #         ("conv_030", True,  0.30),
#         #         ("conv_040", True,  0.40),
#         #         ("conv_050", True,  0.50),
#         #         ("conv_055", True,  0.55),
#         #         ("conv_060", True,  0.60),
#         #         ("conv_062", True,  0.62),
#         #         ("conv_064", True,  0.64),
#         #         ("conv_0645", True, 0.645),
#         #         ("conv_0650", True, 0.650),
#         #         ("conv_0655", True, 0.655),
#         #         ("conv_0660", True, 0.660),
#         #         ("conv_0665", True, 0.665),
#         #         ("conv_0670", True, 0.670),
#         #         ("conv_0680", True, 0.680),
#         #         ("conv_0700", True, 0.700),
#         #         ("conv_072", True,  0.72),
#         #         ("conv_074", True,  0.74),
#         #         ("conv_076", True,  0.76),
#         #         ("conv_078", True,  0.78),
#         #         ("conv_080", True,  0.80),
#         #         ("conv_082", True,  0.82),
#         #         ("conv_084", True,  0.84),
#         #         ("conv_085", True,  0.85),
#         #         ("conv_086", True,  0.86),
#         #         ("conv_087", True,  0.87),
#         #         ("conv_088", True,  0.88),
#         #         ("conv_089", True,  0.89),
#         #         ("conv_090", True,  0.90),
#         #         ("conv_091", True,  0.91),
#         #         ("conv_092", True,  0.92),
#         #         ("conv_093", True,  0.93),
#         #         ("conv_094", True,  0.94),
#         #         ("conv_095", True,  0.95),
#         #         ("conv_096", True,  0.96),
#         #         ("conv_097", True,  0.97),
#         #         ("conv_098", True,  0.98),
#         #         ("conv_099", True,  0.99),
#         #         ("full",     True,  1.00),
#         #     ]

#         stage_grid = _build_stage_grid(lam)
#         accepted_conv_scale = 0.0
#         i = 0
        
#         while i < len(stage_grid):
#             target_conv_scale = stage_grid[i]
#             include_convection = target_conv_scale > 0.0

#             if i == 0 and target_conv_scale == 0.0:
#                 print("  --- stage: Stokes / zero-convection stage ---")
#             else:
#                 print(f"  --- stage: convection_scale = {target_conv_scale:.6f} ---")

#             stage_seed = fenics.Function(W)
#             _copy_state(stage_seed, w_n)

#             ok, err, used_relax = _try_newton_stage(
#                 w=w,
#                 w_seed=stage_seed,
#                 boundary_conditions=boundary_conditions,
#                 W=W,
#                 psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
#                 mu=mu, Pr=Pr, f_b=f_b,
#                 sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
#                 lam=lam,
#                 include_convection=include_convection,
#                 conv_scale=target_conv_scale,
#                 relaxation_schedule=relaxation_schedule,
#             )

#             if ok:
#                 _copy_state(w_n, w)
#                 accepted_conv_scale = target_conv_scale
#                 print(
#                     f"  accepted stage at lambda={lam:.2f}, conv_scale={target_conv_scale:.6f}, "
#                     f"relaxation={used_relax:.3f}"
#                 )
#                 i += 1
#                 continue

#             print(
#                 f"  steady Newton stalled between conv_scale={accepted_conv_scale:.6f} "
#                 f"and {target_conv_scale:.6f} at lambda={lam:.2f}."
#             )

#             inserted = False
#             depth = 0
#             left = accepted_conv_scale
#             right = target_conv_scale
#             while depth < max_bisection_depth and (right - left) > 1e-6:
#                 mid = 0.5 * (left + right)
#                 print(f"  trying adaptive midpoint conv_scale={mid:.6f}")

#                 ok_mid, err_mid, used_relax_mid = _try_newton_stage(
#                     w=w,
#                     w_seed=stage_seed,
#                     boundary_conditions=boundary_conditions,
#                     W=W,
#                     psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
#                     mu=mu, Pr=Pr, f_b=f_b,
#                     sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
#                     lam=lam,
#                     include_convection=(mid > 0.0),
#                     conv_scale=mid,
#                     relaxation_schedule=relaxation_schedule,
#                 )

#                 if ok_mid:
#                     _copy_state(w_n, w)
#                     accepted_conv_scale = mid
#                     stage_grid.insert(i, mid)
#                     print(
#                         f"  inserted successful midpoint conv_scale={mid:.6f} "
#                         f"with relaxation={used_relax_mid:.3f}"
#                     )
#                     inserted = True
#                     break

#                 right = mid
#                 depth += 1

#             if inserted:
#                 continue

#             print("  launching pseudo-transient rescue...")
#             rescued = pseudo_transient_rescue(
#                 experiment=experiment,
#                 W=W,
#                 w=w,
#                 w_n=w_n,
#                 psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
#                 mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
#                 sub_dx=sub_dx, sub_ds=sub_ds, sub_ft=sub_ft, qn_air=qn_air,
#                 lam=lam,
#                 conv_scale=accepted_conv_scale,
#                 dtau_schedule=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
#                 steps_per_dtau=6,
#                 update_tol=1e-8,
#                 retry_newton_every=3,
#                 relaxation=1.0,
#             )

#             if not rescued:
#                 raise RuntimeError(
#                     f"Pseudo-transient rescue failed at lambda={lam:.2f} near conv_scale={accepted_conv_scale:.6f}."
#                 )

#             rescue_seed = fenics.Function(W)
#             _copy_state(rescue_seed, w_n)
#             ok_retry, err_retry, used_relax_retry = _try_newton_stage(
#                 w=w,
#                 w_seed=rescue_seed,
#                 boundary_conditions=boundary_conditions,
#                 W=W,
#                 psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
#                 mu=mu, Pr=Pr, f_b=f_b,
#                 sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
#                 lam=lam,
#                 include_convection=include_convection,
#                 conv_scale=target_conv_scale,
#                 relaxation_schedule=relaxation_schedule,
#             )

#             if ok_retry:
#                 _copy_state(w_n, w)
#                 accepted_conv_scale = target_conv_scale
#                 print(
#                     f"  stage recovered after PTC at lambda={lam:.2f}, "
#                     f"conv_scale={target_conv_scale:.6f}, relaxation={used_relax_retry:.3f}"
#                 )
#                 i += 1
#                 continue

#             raise RuntimeError(
#                 f"Hybrid continuation failed at lambda={lam:.2f}, conv_scale={target_conv_scale:.6f}. "
#                 f"Last steady Newton error: {err_retry if err_retry is not None else err}"
#             )


#         # for stage_name, include_convection, conv_scale in stage_attempts:
#         #     print(f"  --- stage: {stage_name} ---")
#         #     stage_success = False
#         #     last_error = None

#         #     F, JF = build_nonlinear_problem(
#         #         W=W, w=w,
#         #         psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
#         #         mu=mu, Pr=Pr, f_b=f_b,
#         #         sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
#         #         buoyancy_scale=lam,
#         #         qn_scale=lam,
#         #         include_convection=include_convection,
#         #         convection_scale=conv_scale
#         #     )
#         #     relax = relaxation_schedule[:]
#         #     if conv_scale > 0.6 :
#         #         relax = relax[:]
#         #         # relax[0] = 0.7
#         #     for relaxation in relax:
#         #         print(f"  attempt={stage_name}, relaxation={relaxation:.3f}")
                
#         #         # restart from last accepted continuation state
#         #         w.vector()[:] = w_n.vector()
#         #         w.vector().apply("insert")

#         #         try:
#         #             w = base_solver(
#         #                 F, w, boundary_conditions, JF,
#         #                 relaxation=relaxation,
#         #                 maxit=60,
#         #                 atol=1e-9,
#         #                 rtol=1e-8,
#         #             )

#         #             # accept stage result
#         #             w_n.vector()[:] = w.vector()
#         #             w_n.vector().apply("insert")

#         #             print(
#         #                 f"  converged at lambda={lam:.2f} "
#         #                 f"with relaxation={relaxation:.3f} ({stage_name})"
#         #             )
#         #             stage_success = True
#         #             break

#         #         except RuntimeError as err:
#         #             last_error = err
#         #             print(
#         #                 f"  failed at lambda={lam:.2f} "
#         #                 f"with relaxation={relaxation:.3f} ({stage_name})"
#         #             )

#         #     if not stage_success:
#         #         raise RuntimeError(
#         #             f"Continuation Newton failed at lambda={lam:.2f} "
#         #             f"during stage '{stage_name}'. Last error: {last_error}"
#         #         )

#     return w

def solve_temp_newton_continuation(
    experiment: Experiment,
    u_n: fenics.Function, u: fenics.Function, T_n: fenics.Function, T: fenics.Function, p: fenics.Function,
    W: fenics.FunctionSpace, w: fenics.Function,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b, T_c, T_air_bc,
    sub_dx, sub_ds, sub_ft, qn_air,
    fluid_material: TemperatureDependentMaterial,
    w_n: fenics.Function,
    lambdas=None,
    relaxation_schedule=(0.2, 0.1, 0.05),
    stokes_startup=True,
    sub_mesh_star=None,
    sub_mesh_dim=None,
    p_path: str = "",
    u_path: str = "",
    T_path: str = "",
):
    """
    Steady continuation solve with damped Newton + MUMPS.

    For each lambda:
      1) solve a no-momentum-convection stage
      2) solve the full nonlinear stage
    and promote the converged solution after each successful stage.
    """
    if lambdas is None:
        lambdas = [0.05, 0.10, 0.20, 0.40, 0.70, 1.00]

    w.vector()[:] = w_n.vector()
    w.vector().apply("insert")
    scales = compute_nondimensional_scales(experiment)
    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

    for lam in lambdas:
        print(f"\n=== Newton continuation lambda = {lam:.2f} ===")
        # Split nondimensional solution
        p_star, u_star, theta = w.split(deepcopy=True)

        # Dimensionalize fields (note: mesh is star; dimensionalize handles scaling)
        u_dim, p_dim, T_dim = dimensionalize_fields(
            sub_mesh_star, u_star, p_star, theta,
            scales.Uref, scales.dTref, T_ambient,
            experiment.fluid.properties["rho"]
        )

        p_path = p_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
        v_path = u_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
        t_path = T_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
        
        save_experiment(p_path, sub_mesh_dim, [p_dim])
        save_experiment(v_path, sub_mesh_dim, [u_dim])
        save_experiment(t_path, sub_mesh_dim, [T_dim])

        if lam < 0.05:
                stage_attempts = [
                ("stokes",   False, 0.00),
                ("conv_005", True,  0.05),
                ("conv_010", True,  0.10),
                ("conv_020", True,  0.20),
                ("conv_030", True,  0.30),
                ("conv_040", True,  0.40),
                ("conv_050", True,  0.50),
                ("conv_060", True,  0.60),
                ("conv_070", True,  0.70),
                ("conv_080", True,  0.80),
                ("conv_090", True,  0.90),
                ("full",     True,  1.00),
            ]
        else:
            stage_attempts = [
                ("stokes",   False, 0.00),
                ("conv_005", True,  0.05),
                ("conv_010", True,  0.10),
                ("conv_020", True,  0.20),
                ("conv_030", True,  0.30),
                ("conv_040", True,  0.40),
                ("conv_050", True,  0.50),
                ("conv_055", True,  0.55),
                ("conv_060", True,  0.60),
                ("conv_062", True,  0.62),
                ("conv_064", True,  0.64),
                ("conv_066", True,  0.66),
                ("conv_068", True,  0.68),
                ("conv_070", True,  0.70),
                ("conv_072", True,  0.72),
                ("conv_074", True,  0.74),
                ("conv_076", True,  0.76),
                ("conv_078", True,  0.78),
                ("conv_080", True,  0.80),
                ("conv_082", True,  0.82),
                ("conv_084", True,  0.84),
                ("conv_085", True,  0.85),
                ("conv_086", True,  0.86),
                ("conv_087", True,  0.87),
                ("conv_088", True,  0.88),
                ("conv_089", True,  0.89),
                ("conv_090", True,  0.90),
                ("conv_091", True,  0.91),
                ("conv_092", True,  0.92),
                ("conv_093", True,  0.93),
                ("conv_094", True,  0.94),
                ("conv_095", True,  0.95),
                ("conv_096", True,  0.96),
                ("conv_097", True,  0.97),
                ("conv_098", True,  0.98),
                ("conv_099", True,  0.99),
                ("full",     True,  1.00),
            ]

        for stage_name, include_convection, conv_scale in stage_attempts:
            print(f"  --- stage: {stage_name} ---")
            stage_success = False
            last_error = None

            F, JF = build_nonlinear_problem(
                W=W, w=w,
                psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
                mu=mu, Pr=Pr, f_b=f_b,
                sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
                buoyancy_scale=lam,
                qn_scale=lam,
                include_convection=include_convection,
                convection_scale=conv_scale,
            )
            relax = relaxation_schedule[:]
            if conv_scale > 0.6 :
                relax = relax[:]
                # relax[0] = 0.7
            for relaxation in relax:
                print(f"  attempt={stage_name}, relaxation={relaxation:.3f}")
                
                # restart from last accepted continuation state
                w.vector()[:] = w_n.vector()
                w.vector().apply("insert")

                try:
                    w = base_solver(
                        F, w, boundary_conditions, JF,
                        relaxation=relaxation,
                        maxit=60,
                        atol=1e-9,
                        rtol=1e-8,
                    )

                    # accept stage result
                    w_n.vector()[:] = w.vector()
                    w_n.vector().apply("insert")

                    print(
                        f"  converged at lambda={lam:.2f} "
                        f"with relaxation={relaxation:.3f} ({stage_name})"
                    )
                    stage_success = True
                    break

                except RuntimeError as err:
                    last_error = err
                    print(
                        f"  failed at lambda={lam:.2f} "
                        f"with relaxation={relaxation:.3f} ({stage_name})"
                    )

            if not stage_success:
                raise RuntimeError(
                    f"Continuation Newton failed at lambda={lam:.2f} "
                    f"during stage '{stage_name}'. Last error: {last_error}"
                )
            elif stage_name == "full":
                # Initialize
                w.vector()[:] = w_n.vector()

                # Outer loop: update materials from last temperature, then Newton solve
                _, _, T_old = w.split(True)

                for it in range(max_it):
                    fluid_material.update(T_old)   # updates DG0 mu/Pr/... on sub_mesh

                    # Newton solve with frozen coefficients
                    w = base_solver(
                        F, w, boundary_conditions, JF,
                        relaxation=0.9,
                        maxit=60,
                        atol=1e-9,
                        rtol=1e-8,
                    )

                    _, _, T_new = w.split(True)

                    # convergence check on temperature (choose your norm)
                    diff = (T_new.vector() - T_old.vector()).norm("l2")
                    norm = T_old.vector().norm("l2") + 1e-14
                    rel  = diff / norm

                    print(f"[material loop {it}] rel ||ΔT|| = {rel:.3e}")

                    if rel < rtol:
                        break

                    T_old.assign(T_new)

    return w

def build_nonlinear_ABE_problem(
    W, w,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b,
    sub_dx, sub_ds, qn_air,
    fEc: fenics.Constant,
    buoyancy_scale=1.0,
    qn_scale=1.0,
    include_convection=True,
    convection_scale=1.0,
):
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym

    p, u, T = fenics.split(w)

    buoyancy_scale_c = fenics.Constant(float(buoyancy_scale))
    convection_scale_c = fenics.Constant(float(convection_scale))
    qn_scale_c = fenics.Constant(float(qn_scale))
    gvec = fenics.Constant((0.0, -1.0))

    mass = -psi_p * div(u)

    convection_term = (
        convection_scale_c * dot(grad(u), u)
        if include_convection else fenics.Constant((0.0, 0.0))
    )

    momentum = (
        dot(psi_u, convection_term + buoyancy_scale_c * f_b)
        - div(psi_u) * p
        + 2.0 * mu * inner(sym(grad(psi_u)), sym(grad(u)))
    )

    energy = dot(grad(psi_T), (1.0 / Pr) * grad(T) - T * u * convection_scale_c) \
        - psi_T * fEc * dot(gvec, u)                    # extra thermal coupling therm

    F = (mass + momentum + energy) * sub_dx
    F += -qn_scale_c * qn_air * psi_T * sub_ds(INTERFACE_TAG)

    JF = fenics.derivative(F, w, fenics.TrialFunction(W))
    return F, JF

def solve_ABE_newton_continuation(
    experiment: Experiment,
    u_n: fenics.Function, u: fenics.Function, T_n: fenics.Function, T: fenics.Function, p: fenics.Function,
    W: fenics.FunctionSpace, w: fenics.Function,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b, T_c, T_air_bc,
    sub_dx, sub_ds, sub_ft, qn_air,
    w_n: fenics.Function,
    lambdas=None,
    relaxation_schedule=(0.2, 0.1, 0.05),
    stokes_startup=True,
    sub_mesh_star=None,
    sub_mesh_dim=None,
    p_path: str = "",
    u_path: str = "",
    T_path: str = "",
):
    """
    Steady continuation solve with damped Newton + MUMPS.

    For each lambda:
      1) solve a no-momentum-convection stage
      2) solve the full nonlinear stage
    and promote the converged solution after each successful stage.
    """
    if lambdas is None:
        lambdas = [0.05, 0.10, 0.20, 0.40, 0.70, 1.00]

    w.vector()[:] = w_n.vector()
    w.vector().apply("insert")
    scales = compute_nondimensional_scales(experiment)
    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

    for lam in lambdas:
        print(f"\n=== Newton continuation lambda = {lam:.2f} ===")
        # Split nondimensional solution
        p_star, u_star, theta = w.split(deepcopy=True)

        # Dimensionalize fields (note: mesh is star; dimensionalize handles scaling)
        u_dim, p_dim, T_dim = dimensionalize_fields(
            sub_mesh_star, u_star, p_star, theta,
            scales.Uref, scales.dTref, T_ambient,
            experiment.fluid.properties["rho"]
        )

        p_path = p_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
        v_path = u_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
        t_path = T_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
        
        save_experiment(p_path, sub_mesh_dim, [p_dim])
        save_experiment(v_path, sub_mesh_dim, [u_dim])
        save_experiment(t_path, sub_mesh_dim, [T_dim])

        if lam < 0.05:
                stage_attempts = [
                ("stokes",   False, 0.00),
                ("conv_005", True,  0.05),
                ("conv_010", True,  0.10),
                ("conv_020", True,  0.20),
                ("conv_030", True,  0.30),
                ("conv_040", True,  0.40),
                ("conv_050", True,  0.50),
                ("conv_060", True,  0.60),
                ("conv_070", True,  0.70),
                ("conv_080", True,  0.80),
                ("conv_090", True,  0.90),
                ("full",     True,  1.00),
            ]
        else:
            stage_attempts = [
                ("stokes",   False, 0.00),
                ("conv_005", True,  0.05),
                ("conv_010", True,  0.10),
                ("conv_020", True,  0.20),
                ("conv_030", True,  0.30),
                ("conv_040", True,  0.40),
                ("conv_050", True,  0.50),
                ("conv_055", True,  0.55),
                ("conv_060", True,  0.60),
                ("conv_062", True,  0.62),
                ("conv_064", True,  0.64),
                ("conv_066", True,  0.66),
                ("conv_068", True,  0.68),
                ("conv_070", True,  0.70),
                ("conv_072", True,  0.72),
                ("conv_074", True,  0.74),
                ("conv_076", True,  0.76),
                ("conv_078", True,  0.78),
                ("conv_080", True,  0.80),
                ("conv_082", True,  0.82),
                ("conv_084", True,  0.84),
                ("conv_085", True,  0.85),
                ("conv_086", True,  0.86),
                ("conv_087", True,  0.87),
                ("conv_088", True,  0.88),
                ("conv_089", True,  0.89),
                ("conv_090", True,  0.90),
                ("conv_091", True,  0.91),
                ("conv_092", True,  0.92),
                ("conv_093", True,  0.93),
                ("conv_094", True,  0.94),
                ("conv_095", True,  0.95),
                ("conv_096", True,  0.96),
                ("conv_097", True,  0.97),
                ("conv_098", True,  0.98),
                ("conv_099", True,  0.99),
                ("full",     True,  1.00),
            ]

        for stage_name, include_convection, conv_scale in stage_attempts:
            print(f"  --- stage: {stage_name} ---")
            stage_success = False
            last_error = None

            F, JF = build_nonlinear_ABE_problem(
                W=W, w=w,
                psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
                mu=mu, Pr=Pr, f_b=f_b,
                sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
                buoyancy_scale=lam,
                qn_scale=lam,
                include_convection=include_convection,
                convection_scale=conv_scale,
                fEc=fenics.Constant(scales.fEc)
            )
            relax = relaxation_schedule[:]
            if conv_scale > 0.6 :
                relax = relax[:]
                # relax[0] = 0.7
            for relaxation in relax:
                print(f"  attempt={stage_name}, relaxation={relaxation:.3f}")
                
                # restart from last accepted continuation state
                w.vector()[:] = w_n.vector()
                w.vector().apply("insert")

                try:
                    w = base_solver(
                        F, w, boundary_conditions, JF,
                        relaxation=relaxation,
                        maxit=60,
                        atol=1e-9,
                        rtol=1e-8,
                    )

                    # accept stage result
                    w_n.vector()[:] = w.vector()
                    w_n.vector().apply("insert")

                    print(
                        f"  converged at lambda={lam:.2f} "
                        f"with relaxation={relaxation:.3f} ({stage_name})"
                    )
                    stage_success = True
                    break

                except RuntimeError as err:
                    last_error = err
                    print(
                        f"  failed at lambda={lam:.2f} "
                        f"with relaxation={relaxation:.3f} ({stage_name})"
                    )

            if not stage_success:
                raise RuntimeError(
                    f"Continuation Newton failed at lambda={lam:.2f} "
                    f"during stage '{stage_name}'. Last error: {last_error}"
                )

    return w
