from utils.imports import *
from utils.parser import *

def set_param(sub_mesh: fenics.Mesh, T_full: fenics.Function, T: fenics.Function, T_ambient: float,
              rho_air: float, beta_air: float):
    # DG0 fields for mu, PR, Ra, and f_B
    V0 = fenics.FunctionSpace(sub_mesh, "DG", 0)
    mu_func = fenics.Function(V0)
    Pr_func = fenics.Function(V0)
    Ra_func = fenics.Function(V0)
    f_b_func = fenics.Function(V0)


    # -----------------------------------------
    # Fill cellwise values (using MeshFunction ct)
    # -----------------------------------------
    # ct: MeshFunction("size_t", mesh, mesh.topology().dim())
    # ct.array() gives tag per cell in order of mesh.cells()

    dynamic_viscosity = 1.
    prandtl_number = 0.71
    rayleigh_number = 10

    mu = fenics.Constant(dynamic_viscosity)

    Pr = fenics.Constant(prandtl_number)

    Ra = fenics.Constant(rayleigh_number)

    gvec = fenics.Constant((0.0, -1.0))

    T_ref = fenics.Constant(T_ambient)


    f_b = rho_air * beta_air * (T - T_ref) * gvec
    # f_b = (Ra/Pr) * T * gvec
    # f_b_vals = Ra_func/Pr_func*T*fenics.Constant((0., -1.))
    # f_b_func.vector()[:] = f_b_vals

    VTa = fenics.FunctionSpace(sub_mesh, "CG", 1)
    T_air_bc = fenics.Function(VTa)
    T_air_bc.interpolate(T_full)

    hot_wall_temperature = float(T_air_bc.vector().max())
    # hot_wall_temperature = T_air_bc.vector()
    T_h = fenics.Constant(hot_wall_temperature)

    cold_wall_temperature = T_ambient

    T_c = fenics.Constant(cold_wall_temperature)
    return mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc

class Hot_wall(fenics.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fenics.near((x[0]**2)+((x[1]-11.)**2)-1., 0., eps= 1.e-1) and x[1] >= 10. and x[1] <= 12.
# hot_wall = " near((x[0]*x[0])+((x[1]-11.)*(x[1]-11.))-1., 0.)"

def set_bcs(W, sub_ft, T_air_bc, cold_wall_temperature):
    hot_wall=Hot_wall()

    cold_wall = "near(x[0],  1.) | near(x[1], 0.) | near(x[1], 1.)"

    adiabatic_walls = "near(x[0],  0.)"

    # walls = hot_wall + " | " + cold_wall + " | " + adiabatic_walls

    W_u = W.sub(1)

    W_T = W.sub(2)

    boundary_conditions = [
        fenics.DirichletBC(W_u, (0., 0.), hot_wall),
        # fenics.DirichletBC(W_T, hot_wall_temperature, hot_wall),
        fenics.DirichletBC(W_T,T_air_bc,sub_ft,INTERFACE_TAG),
        fenics.DirichletBC(W_T, cold_wall_temperature, cold_wall)]
    
    return boundary_conditions


def volume_heat_source(experiment: Experiment):
    if experiment.initial_conditions.heat_length is not None:
        heat_volume = (math.pi *(experiment.dimensions.wire.diameter / 2)**2)* experiment.initial_conditions.heat_length
    elif experiment.initial_conditions.heat_volume is not None:
        heat_volume = experiment.initial_conditions.heat_volume
    elif experiment.initial_conditions.heat_surface is not None:
        heat_volume = 4.0 / experiment.dimensions.wire.diameter * (experiment.initial_conditions.heat_surface )
    return heat_volume