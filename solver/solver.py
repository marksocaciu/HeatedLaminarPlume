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
    
def solver(sub_mesh: fenics.Mesh, T_full: fenics.Function, T_ambient: float,
           rho_air: float, beta_air: float, experiment: Experiment):
    P1 = fenics.FiniteElement('P', sub_mesh.ufl_cell(), 1)
    P2 = fenics.VectorElement('P', sub_mesh.ufl_cell(), 2)
    mixed_element = fenics.MixedElement([P1, P2, P2]) 
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
    # fenics.plot(T_n)
    # plt.title("$T^0$")
    # plt.xlabel("$x$")
    # plt.ylabel("$y$")

    return W, w, p, u, T, w_n, p_n, u_n, T_n, psi_p, psi_u, psi_T, mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc

def nonlinear_solver_ABE(experiment: Experiment,u_n: fenics.Function, u: fenics.Function, T_n: fenics.Function, T: fenics.Function, p: fenics.Function,
                     W: fenics.FunctionSpace, w: fenics.Function,
                     psi_p, psi_u, psi_T,
                     mu, Pr, f_b, T_c, T_air_bc,
                     sub_dx, sub_ds, sub_ft, qn_air,
                     w_n: fenics.Function,
                     fEc: fenics.Constant
                     ):
    timestep_size = 0.001

    Delta_t = fenics.Constant(timestep_size)

    u_t = (u - u_n)/Delta_t

    T_t = (T - T_n)/Delta_t

    inner, dot, grad, div, sym = \
        fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
        
    mass = -psi_p*div(u)
            
    momentum = (
        dot(psi_u, u_t + dot(grad(u), u) + f_b)
        - div(psi_u)*p
        + 2.0 * mu * inner(sym(grad(psi_u)), sym(grad(u)))
    )
 
    gvec = fenics.Constant((0.0, -1.0))

    energy = (
        psi_T*T_t
        + dot(grad(psi_T), (1.0/Pr) * grad(T) - T*u)
        - psi_T * fEc * dot(gvec, u)                    # extra thermal coupling therm

    )

    F = (mass + momentum + energy) * sub_dx
    # F = (mass + momentum + energy)*fenics.dx


    penalty_stabilization_parameter = 1.e-7

    gamma = fenics.Constant(penalty_stabilization_parameter)

    print("Max qn_air:", qn_air.vector().max())

    F += -psi_p * gamma * p * sub_dx
    F += qn_air * psi_T * sub_ds(INTERFACE_TAG)
    # F += -psi_p*gamma*p*fenics.dx

    scales = compute_nondimensional_scales(experiment)
    k_inf = float(experiment.fluid.properties["k"])  # use experiment value (not global)
    qn_dim = qn_air * fenics.Constant(k_inf * float(scales.dTref) / float(scales.Lref))
    QL_half = fenics.assemble(qn_dim * sub_ds(INTERFACE_TAG))
    print(f"Heat flux from wire to fluid (half wire): QL_half = {QL_half:.6e} W/m")

    JF = fenics.derivative(F, w, fenics.TrialFunction(W))

    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

    w.leaf_node().vector()[:] = w_n.leaf_node().vector()


    return F,w, boundary_conditions, JF, w_n


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

# def _assign_mixed_temperature(W: fenics.FunctionSpace, w_mixed: fenics.Function, theta_src: fenics.Function):
#     """Assign a scalar temperature field into W.sub(2)."""
#     VT, _ = W.sub(2).collapse(True)
#     theta_tmp = fenics.Function(VT)
#     theta_tmp.interpolate(theta_src)
#     assign_T = fenics.FunctionAssigner(W.sub(2), VT)
#     assign_T.assign(w_mixed.sub(2), theta_tmp)
#     w_mixed.vector().apply("insert")

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

# def _build_linear_startup_problem(
#     experiment: Experiment,
#     W: fenics.FunctionSpace,
#     w: fenics.Function,
#     mu, Pr,
#     sub_dx, sub_ds, sub_ft,
#     qn_air,
#     T_c,
#     T_air_bc,
#     qn_scale=1.0,
#     frozen_buoyancy_temperature=None,
# ):
#     """
#     Linear startup problem used to generate a robust initial guess.
#     - momentum convection is removed
#     - thermal advection is removed
#     - buoyancy may be frozen from a prescribed temperature field
#     """
#     q, v, s = fenics.TestFunctions(W)
#     p_trial, u_trial, T_trial = fenics.split(w)
#     inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym

#     scales = compute_nondimensional_scales(experiment)
#     gvec = fenics.Constant((0.0, -1.0))
#     buoyancy_coeff = fenics.Constant(float(scales.Ra / scales.Pr))

#     mass = -q * div(u_trial)

#     momentum = (
#         - div(v) * p_trial
#         + 2.0 * mu * inner(sym(grad(v)), sym(grad(u_trial)))
#     )

#     if frozen_buoyancy_temperature is not None:
#         momentum += dot(v, buoyancy_coeff * frozen_buoyancy_temperature * gvec)

#     energy = dot(grad(s), (1.0 / Pr) * grad(T_trial))

#     F = (mass + momentum + energy) * sub_dx
#     F += - fenics.Constant(float(qn_scale)) * qn_air * s * sub_ds(INTERFACE_TAG)

#     JF = fenics.derivative(F, w, fenics.TrialFunction(W))
#     boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)
#     return F, boundary_conditions, JF

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

# def stokes_initial_guess(
#     experiment: Experiment,
#     u_n: fenics.Function, u: fenics.Function, T_n: fenics.Function, T: fenics.Function, p: fenics.Function,
#     W: fenics.FunctionSpace, w: fenics.Function,
#     psi_p, psi_u, psi_T,
#     mu, Pr, f_b, T_c, T_air_bc,
#     sub_dx, sub_ds, sub_ft, qn_air,
#     w_n: fenics.Function,
#     lambdas=(0.10, 0.25, 0.50, 1.00),
#     relaxation=0.2,
#     maxit=60,
#     atol=4.5e-8,
#     rtol=3.2e-7,
# ):
#     """
#     Build a genuinely linear startup state before the full nonlinear solve.

#     For each continuation parameter lambda:
#     1) solve pure conduction with q'' scaled by lambda
#     2) solve linear Stokes with buoyancy frozen from that conduction field

#     The same BC construction is reused, so the pointwise pressure pin stays active.
#     """
#     # Reference conduction field corresponding to full heating (lambda = 1).
#     theta_ref = w_n.sub(2, deepcopy=True)

#     w.vector()[:] = w_n.vector()
#     w.vector().apply("insert")

#     for lam in lambdas:
#         print(f"\n=== Linear startup lambda = {lam:.2f} ===")

#         # Keep the initial temperature guess consistent with the current continuation step.
#         theta_lam = fenics.Function(theta_ref.function_space())
#         theta_lam.vector()[:] = float(lam) * theta_ref.vector().get_local()
#         theta_lam.vector().apply("insert")
#         _assign_mixed_temperature(W, w_n, theta_lam)
#         w.vector()[:] = w_n.vector()
#         w.vector().apply("insert")

#         print("  -> Stage A: conduction-only solve")
#         F_cond, boundary_conditions, JF_cond = _build_linear_startup_problem(
#             experiment=experiment,
#             W=W,
#             w=w,
#             mu=mu,
#             Pr=Pr,
#             sub_dx=sub_dx,
#             sub_ds=sub_ds,
#             sub_ft=sub_ft,
#             qn_air=qn_air,
#             T_c=T_c,
#             T_air_bc=T_air_bc,
#             qn_scale=lam,
#             frozen_buoyancy_temperature=None,
#         )
#         w = base_solver(
#             F_cond, w, boundary_conditions, JF_cond,
#             relaxation=1.0,
#             maxit=maxit,
#             atol=atol,
#             rtol=rtol,
#         )

#         theta_cond = w.sub(2, deepcopy=True)
#         w_n.assign(w)
#         w_n.vector().apply("insert")

#         print("  -> Stage B: frozen-temperature Stokes solve")
#         F_stokes, boundary_conditions, JF_stokes = _build_linear_startup_problem(
#             experiment=experiment,
#             W=W,
#             w=w,
#             mu=mu,
#             Pr=Pr,
#             sub_dx=sub_dx,
#             sub_ds=sub_ds,
#             sub_ft=sub_ft,
#             qn_air=qn_air,
#             T_c=T_c,
#             T_air_bc=T_air_bc,
#             qn_scale=lam,
#             frozen_buoyancy_temperature=theta_cond,
#         )
#         w = base_solver(
#             F_stokes, w, boundary_conditions, JF_stokes,
#             relaxation=1.0,
#             maxit=maxit,
#             atol=atol,
#             rtol=rtol,
#         )

#         w_n.assign(w)
#         w_n.vector().apply("insert")

#     return w_n

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

def solve_steady_newton_continuation(
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
        # plot_mesh(T_dim, title="Temperature field", label="Temperature (K)",
        #             cmap="coolwarm", colorbar=True)
        # plot_mesh(theta, title="Temperature field nondimensional", label="Temperature (nondim)",
        #             cmap="coolwarm", colorbar=True)
        # plot_mesh(u_dim, title="Velocity magnitude", label="Velocity (m/s)",
        #             cmap="coolwarm", colorbar=True, mode="glyphs")
        # plot_mesh(p_dim, title="Pressure field", label="Pressure (Pa)",
        #             cmap="coolwarm", colorbar=True)
        p_path = p_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
        v_path = u_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
        t_path = T_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
        
        save_experiment(p_path, sub_mesh_dim, [p_dim])
        save_experiment(v_path, sub_mesh_dim, [u_dim])
        save_experiment(t_path, sub_mesh_dim, [T_dim])

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
            # F, w, boundary_conditions, JF, _ = nonlinear_solver(
            #     experiment, u_n, u, T_n, T, p, W, w,
            #     psi_p, psi_u, psi_T,
            #     mu, Pr, f_b, T_c, T_air_bc,
            #     sub_dx, sub_ds, sub_ft, qn_air,
            #     w_n,
            #     buoyancy_scale=lam,
            #     qn_scale=lam,
            #     include_convection=include_convection,
            #     convection_scale=conv_scale,
            # )

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

    return w

def temp_dep_solver(F,w, boundary_conditions, JF, w_n: fenics.Function, fluid_material: TemperatureDependentMaterial):
    problem = fenics.NonlinearVariationalProblem(F, w, boundary_conditions, JF)

    solver = fenics.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["nonlinear_solver"] = "newton"

    prm["newton_solver"]["absolute_tolerance"] = 5e-10
    prm["newton_solver"]["relative_tolerance"] = 5e-10
    prm["newton_solver"]["maximum_iterations"] = 100

    nprm = solver.parameters["newton_solver"]
    nprm["linear_solver"] = "petsc"
    nprm["preconditioner"] = "none"
    # prm["preconditioner"] = "none"

    # Initialize
    w.vector()[:] = w_n.vector()

    # Outer loop: update materials from last temperature, then Newton solve
    p_old, u_old, T_old = w.split(True)

    for it in range(max_it):
        fluid_material.update(T_old)   # updates DG0 mu/Pr/... on sub_mesh

        solver.solve()               # Newton solve with frozen coefficients

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
    # F, bcs, JF = _build_linear_startup_problem(
    #     experiment=experiment,
    #     W=W,
    #     w=w,
    #     mu=mu,
    #     Pr=Pr,
    #     sub_dx=sub_dx,
    #     sub_ds=sub_ds,
    #     sub_ft=sub_ft,
    #     qn_air=qn_air,
    #     T_c=T_c,
    #     T_air_bc=T_air_bc,
    #     qn_scale=1.0,
    #     frozen_buoyancy_temperature=None,
    # )

    w.vector()[:] = w_n.vector()
    w.vector().apply("insert")

    # w = base_solver(
    #     F, w, bcs, JF,
    #     relaxation=1.0,
    #     maxit=50,
    #     atol=5e-6,
    #     rtol=4e-5,
    # )

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
    # theta_ref = w_n.sub(2, deepcopy=True)

    # qn_zero = fenics.Function(qn_air.function_space())
    # qn_zero.vector().zero()
    # qn_zero.vector().apply("insert")

    # F, bcs, JF = _build_linear_startup_problem(
    #     experiment=experiment,
    #     W=W,
    #     w=w,
    #     mu=mu,
    #     Pr=Pr,
    #     sub_dx=sub_dx,
    #     sub_ds=sub_ds,
    #     sub_ft=sub_ft,
    #     qn_air=qn_zero,
    #     T_c=T_c,
    #     T_air_bc=T_air_bc,
    #     qn_scale=0.0,
    #     frozen_buoyancy_temperature=theta_ref,
    # )

    w.vector()[:] = w_n.vector()
    w.vector().apply("insert")

    # w = base_solver(
    #     F, w, bcs, JF,
    #     relaxation=1.0,
    #     maxit=50,
    #     atol=5e-6,
    #     rtol=4e-5,
    # )

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
