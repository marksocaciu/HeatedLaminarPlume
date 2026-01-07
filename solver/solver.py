from utils.imports import *
from solver.params_bcs import *
from utils.material import *
from solver.scales import *

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

    w_n = fenics.interpolate(
    fenics.Expression(("0.", "0.", "0.", "T_full"), 
                      T_full = T_full,
                      element = mixed_element),
                        W)
    
    p_n, u_n, T_n = fenics.split(w_n)
    
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

    energy = (
        psi_T*T_t
        + dot(grad(psi_T), (1.0/Pr) * grad(T) - T*u)
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

    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment)

    w.leaf_node().vector()[:] = w_n.leaf_node().vector()


    return F,w, boundary_conditions, JF, w_n

def base_solver(F, w: fenics.Function, boundary_conditions, JF):
    problem = fenics.NonlinearVariationalProblem(F, w, boundary_conditions, JF)
    solver = fenics.NonlinearVariationalSolver(problem)
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
