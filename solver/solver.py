from utils.imports import *
from solver.params_bcs import *
from utils.material import *
from solver.scales import *


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
    # fenics.plot(T_n)
    # plt.title("$T^0$")
    # plt.xlabel("$x$")
    # plt.ylabel("$y$")

    return W, w, p, u, T, w_n, p_n, u_n, T_n, psi_p, psi_u, psi_T, mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc

def nonlinear_solver(experiment: Experiment,u_n: fenics.Function, u: fenics.Function, T_n: fenics.Function, T: fenics.Function, p: fenics.Function,
                     W: fenics.FunctionSpace, w: fenics.Function,
                     psi_p, psi_u, psi_T,
                     mu, Pr, f_b, T_c, T_air_bc,
                     sub_dx, sub_ds, sub_ft, qn_air,
                     w_n: fenics.Function):

    inner, dot, grad, div, sym = \
        fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
        
    mass = -psi_p*div(u)

    # momentum = dot(psi_u, u_t + dot(grad(u), u) + f_b) - div(psi_u)*p \
    #     + 2.*mu*inner(sym(grad(psi_u)), sym(grad(u)))

    # energy = psi_T*T_t + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
            
    momentum = (
        dot(psi_u, dot(grad(u), u) + f_b)
        - div(psi_u)*p
        + 2.0 * mu * inner(sym(grad(psi_u)), sym(grad(u)))
    )

    energy = (dot(grad(psi_T), (1.0/Pr) * grad(T) - T * u))

    F = (mass + momentum + energy) * sub_dx
    # F = (mass + momentum + energy)*fenics.dx


    penalty_stabilization_parameter = 1.e-7

    gamma = fenics.Constant(penalty_stabilization_parameter)

    print("Max qn_air:", qn_air.vector().max())

    # F += -psi_p * gamma * p * sub_dx
    F += qn_air * psi_T * sub_ds(INTERFACE_TAG)
    # F += -psi_p*gamma*p*fenics.dx

    scales = compute_nondimensional_scales(experiment)
    k_inf = float(experiment.fluid.properties["k"])  # use experiment value (not global)
    qn_dim = qn_air * fenics.Constant(k_inf * float(scales.dTref) / float(scales.Lref))
    QL_half = fenics.assemble(qn_dim * sub_ds(INTERFACE_TAG)) * scales.Lref
    print(f"Heat flux from wire to fluid (half wire): QL_half = {QL_half:.6e} W/m")

    JF = fenics.derivative(F, w, fenics.TrialFunction(W))

    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

    w.leaf_node().vector()[:] = w_n.leaf_node().vector()

    return F,w, boundary_conditions, JF, w_n

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

    # momentum = dot(psi_u, u_t + dot(grad(u), u) + f_b) - div(psi_u)*p \
    #     + 2.*mu*inner(sym(grad(psi_u)), sym(grad(u)))

    # energy = psi_T*T_t + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
            
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


def base_solver(F, w: fenics.Function, boundary_conditions, JF):
    problem = fenics.NonlinearVariationalProblem(F, w, boundary_conditions, JF)
    solver = fenics.NonlinearVariationalSolver(problem)
    nprm = solver.parameters["newton_solver"]
    # nprm["linear_solver"] = "petsc"
    # nprm["preconditioner"] = "none"
    nprm["linear_solver"] = "mumps"
    
    # damping
    nprm["relaxation_parameter"] = 0.1
    nprm["maximum_iterations"] = 200

    # nprm = solver.parameters["newton_solver"]
    # nprm["absolute_tolerance"] = 1e-10
    # nprm["relative_tolerance"] = 1e-9
    # nprm["report"] = True
    # nprm["error_on_nonconvergence"] = True

    # # Linear solver inside Newton
    # nprm["linear_solver"] = "gmres"
    # # nprm["linear_solver"] = "mumps"
    # nprm["preconditioner"] = "ilu"  # robust baseline

    # kprm = nprm["krylov_solver"]
    # kprm["relative_tolerance"] = 1e-8
    # kprm["maximum_iterations"] = 200
    # kprm["monitor_convergence"] = True
    solver.solve()
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

def steady_picard_forms(
    experiment: Experiment,
    W: fenics.FunctionSpace,
    w_k: fenics.Function,
    mu, Pr, Ra,
    sub_dx, sub_ds,
    qn_air
):
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym

    # unknown of current Picard step
    dw = fenics.TrialFunction(W)
    p, u, T = fenics.split(dw)

    # test functions
    q, v, s = fenics.TestFunctions(W)

    # previous iterate (known / frozen)
    p_k, u_k, T_k = fenics.split(w_k)

    gvec = fenics.Constant((0.0, -1.0))
    f_b = (Ra / Pr) * T * gvec

    # continuity
    a_mass = (-q * div(u)) * sub_dx
    L_mass = fenics.Constant(0.0) * q * sub_dx

    # Oseen / Picard momentum
    a_mom = (
        0.5 * (dot(v, dot(grad(u), u_k)) - dot(u, dot(grad(v), u_k)))
        - div(v) * p
        + 2.0 * mu * inner(sym(grad(v)), sym(grad(u)))
        + dot(v, f_b)
        + fenics.Constant(0.05) * div(u) * div(v)
    ) * sub_dx
    L_mom = fenics.Constant(0.0) * q * sub_dx

    # Picard energy
    a_eng = (
        (1.0 / Pr) * dot(grad(s), grad(T))
        + 0.5 * (s * dot(u_k, grad(T)) - T * dot(u_k, grad(s)))
    ) * sub_dx

    L_eng = (-qn_air * s) * sub_ds(INTERFACE_TAG)

    # optional mild SUPG for temperature
    h = fenics.CellDiameter(W.mesh())
    unorm = fenics.sqrt(dot(u_k, u_k) + fenics.Constant(1.0e-12))
    tau_T = h / (2.0 * unorm)

    RT = dot(u_k, grad(T)) - (1.0 / Pr) * div(grad(T))
    a_supg = (tau_T * dot(u_k, grad(s)) * RT) * sub_dx

    a = a_mass + a_mom + a_eng + a_supg
    L = L_mass + L_mom + L_eng

    bcs = set_bcs(W, None, None, None, experiment, compute_nondimensional_scales(experiment))
    return a, L, bcs

def solve_steady_picard(
    experiment,
    W,
    w,
    w_n,
    psi_p, psi_u, psi_T,   # kept for compatibility, not used here
    mu, Pr, Ra,
    sub_dx, sub_ds, sub_ft, qn_air
):
    w_k = fenics.Function(W)
    w_k.vector()[:] = w_n.vector()

    # continuation in forcing
    lambdas = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.00]

    qn_air_full = fenics.Function(qn_air.function_space())
    qn_air_full.vector()[:] = qn_air.vector()

    for lam in lambdas:
        print(f"\n=== continuation lambda = {lam:.2f} ===")

        qn_lam = fenics.Function(qn_air.function_space())
        qn_lam.vector()[:] = lam * qn_air_full.vector().get_local()
        qn_lam.vector().apply("insert")

        Ra_lam = fenics.Constant(float(Ra) * lam)

        picard_tol = 1.0e-8
        picard_max = 500

        for k in range(picard_max):
            a, L, bcs = steady_picard_forms(
                experiment=experiment,
                W=W,
                w_k=w_k,
                mu=mu, Pr=Pr, Ra=Ra_lam,
                sub_dx=sub_dx, sub_ds=sub_ds,
                qn_air=qn_lam
            )

            fenics.solve(
                a == L, w, bcs,
                solver_parameters={
                    "linear_solver": "mumps"
                }
            )

            # diff = w.vector().copy()
            # diff.axpy(-1.0, w_k.vector())
            # rel = diff.norm("l2") / (w.vector().norm("l2") + 1.0e-14)

            p_new, u_new, T_new = w.split(deepcopy=True)
            p_old, u_old, T_old = w_k.split(deepcopy=True)

            du = u_new.vector().copy()
            du.axpy(-1.0, u_old.vector())

            dT = T_new.vector().copy()
            dT.axpy(-1.0, T_old.vector())

            num = (du.norm("l2")**2 + dT.norm("l2")**2)**0.5
            den = (u_new.vector().norm("l2")**2 + T_new.vector().norm("l2")**2)**0.5 + 1e-14
            rel = num / den

            print(f"  Picard iter {k:02d}: rel = {rel:.3e}")

            # w_k.vector()[:] = w.vector()
            omega = 0.3   # start conservative
            w_k.vector()[:] = (1.0 - omega) * w_k.vector().get_local() + omega * w.vector().get_local()
            w_k.vector().apply("insert")

            if rel < picard_tol:
                break
        else:
            print(f"Picard failed to converge at lambda={lam:.2f}")

    return w