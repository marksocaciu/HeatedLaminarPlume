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
                relaxation=0.5, maxit=20, atol=1e-9, rtol=1e-8):
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
    fenics.parameters["form_compiler"]["cpp_optimize"] = True
    fenics.parameters["form_compiler"]["optimize"] = True
    fenics.parameters["form_compiler"]["representation"] = "uflacs"

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
    return [
        (0.00, 0.10),
        (0.00, 0.20),
        (0.10, 0.30),
        (0.10, 0.50),
        (0.20, 0.70),
        (0.20, 0.90),
        (0.20, 1.00),
        (0.30, 1.00),
        (0.40, 1.00),
        (0.50, 1.00),
        (0.60, 1.00),
        (0.65, 1.00),
        (0.70, 1.00),
        (0.75, 1.00),
        (0.80, 1.00),
        (0.85, 1.00),
        (0.90, 1.00),
        (0.95, 1.00),
        (1.00, 1.00),
        ]

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
    F_accepted,JF_accepted = None, None

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
            accepted_conv = trial_conv
            F_accepted,JF_accepted = F, JF
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

        # try midpoint first, then come back to original target later
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
):
    """
    Backward-Euler pseudo-transient problem for the mixed steady system.

    The pseudo-time mass is added only to velocity and temperature.
    Pressure remains algebraic.
    """
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym

    p, u, T = fenics.split(w)
    _, u_prev, T_prev = fenics.split(w_prev)

    # IMPORTANT:
    # dtau, buoyancy_scale, qn_scale, convection_scale are assumed to be
    # either plain scalars or already-created FEniCS Constants/UFL objects.
    # Do NOT wrap them again in fenics.Constant(...) here.
    dtau_c = dtau
    buoyancy_scale_c = buoyancy_scale
    qn_scale_c = qn_scale
    convection_scale_c = convection_scale

    zero_vec = fenics.Constant((0.0, 0.0))
    convection_term = (
        convection_scale_c * dot(grad(u), u)
        if include_convection else zero_vec
    )

    continuity = -psi_p * div(u)

    pseudo_velocity = (1.0 / dtau_c) * inner(psi_u, u - u_prev)
    pseudo_temperature = (1.0 / dtau_c) * psi_T * (T - T_prev)

    momentum = (
        dot(psi_u, convection_term + buoyancy_scale_c * f_b)
        - div(psi_u) * p
        + 2.0 * mu * inner(sym(grad(psi_u)), sym(grad(u)))
    )

    energy = dot(grad(psi_T), (1.0 / Pr) * grad(T) - convection_scale_c * T * u)

    F = (continuity + pseudo_velocity + momentum + pseudo_temperature + energy) * sub_dx
    F += -qn_scale_c * qn_air * psi_T * sub_ds(INTERFACE_TAG)

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
    )

    r = fenics.assemble(F_steady)
    for bc in boundary_conditions:
        bc.apply(r)

    return r.norm("l2")


def collect_observables(w: fenics.Function):
    p_f, u_f, T_f = w.split(deepcopy=True)
    return {
        "u_l2": u_f.vector().norm("l2"),
        "T_l2": T_f.vector().norm("l2"),
        "p_l2": p_f.vector().norm("l2"),
        "u_max_abs": np.max(np.abs(u_f.vector().get_local())) if u_f.vector().local_size() > 0 else 0.0,
        "T_max": T_f.vector().max(),
        "T_min": T_f.vector().min(),
    }
