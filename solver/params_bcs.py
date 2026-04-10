from matplotlib.pyplot import sca

from utils.imports import *
from utils.parser import *
from solver.scales import *

# def set_param(sub_mesh: fenics.Mesh, T_full: fenics.Function, T: fenics.Function, T_ambient: float,
#               rho_air: float, beta_air: float, experiment: Experiment):
#     sc = compute_nondimensional_scales(experiment)
#     Pr = fenics.Constant(sc.Pr)
#     Ra = fenics.Constant(sc.Ra)
#     gvec = fenics.Constant((0.0, -1.0))          # direction only
#     T_ref = fenics.Constant(0.0)                 # theta-reference

#     f_b = (Ra/Pr) * T * gvec                      # here T == theta
#     mu = fenics.Constant(sc.nu / (sc.Uref * sc.Lref))  # nondim viscosity

#     # -----------------------------------------


#     VTa = fenics.FunctionSpace(sub_mesh, "CG", 1)
#     T_air_bc = fenics.Function(VTa)
#     T_air_bc.interpolate(T_full)

#     hot_wall_temperature = float(T_air_bc.vector().max())
#     # hot_wall_temperature = T_air_bc.vector()
#     T_h = fenics.Constant(hot_wall_temperature)

#     cold_wall_temperature =fenics.Constant(0.0)

#     # cold_wall_temperature = T_ambient

#     T_c = fenics.Constant(cold_wall_temperature)
#     return mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc

# def set_bcs(W, sub_ft, T_air_bc, cold_wall_temperature, experiment: Experiment, scales: NondimScales):
#     # r = experiment.dimensions.wire.diameter / 2.
#     r = (experiment.dimensions.wire.diameter/2) / scales.Lref  # == 1.0
#     class Hot_wall(fenics.SubDomain):
#         def inside(self, x, on_boundary):
#             return on_boundary and fenics.near(
#                 (x[0]**2)+((x[1]-(experiment.dimensions.domain.y_max / scales.Lref / 10. + 11.*r))**2)\
#                     -1.*r*r, 0., eps= 1.e-1*r
#                 ) \
#                     and x[1] >= experiment.dimensions.domain.y_max / scales.Lref / 10. + 10.*r - 1e-12 \
#                     and x[1] <= experiment.dimensions.domain.y_max / scales.Lref / 10. + 12.*r + 1e-12
        
#     class Cold_wall_preset(fenics.SubDomain):
#         def inside(self, x, on_boundary):
#             east = on_boundary and fenics.near(x[0],  40.*r, eps = 1.e-1*r)
#             south = on_boundary and fenics.near(x[1],  0.0, eps = 1.e-1*r)
#             north = on_boundary and fenics.near(x[1],  100.*r, eps = 1.e-1*r)
#             return east or south or north
        
#     class Cold_wall_modified(fenics.SubDomain):
#         def inside(self, x, on_boundary):
#             east = on_boundary and fenics.near(x[0],  experiment.dimensions.domain.x_max / scales.Lref, eps = 1.e-10)
#             south = on_boundary and fenics.near(x[1],  experiment.dimensions.domain.y_min / scales.Lref, eps = 1.e-10)
#             north = on_boundary and fenics.near(x[1],  experiment.dimensions.domain.y_max / scales.Lref, eps = 1.e-10)
#             return east or south or north
    
#     class EastBoundary(fenics.SubDomain):
#         def inside(self, x, on_boundary):
#             return on_boundary and fenics.near(
#                 x[0], experiment.dimensions.domain.x_max / scales.Lref, eps=1.0e-10
#             )

#     # class PressurePin(fenics.SubDomain):
#     #     def inside(self, x, on_boundary):
#     #         return (
#     #             fenics.near(x[0], experiment.dimensions.domain.x_max / scales.Lref, 1.0e-10)
#     #             and fenics.near(x[1], experiment.dimensions.domain.y_max / scales.Lref, 1.0e-10)
#             # )
        
#     class PressurePin(fenics.SubDomain):
#         def inside(self, x, on_boundary):
#             xR = experiment.dimensions.domain.x_max / scales.Lref
#             yT = experiment.dimensions.domain.y_max / scales.Lref
#             return fenics.near(x[0], xR, 1.0e-8) and fenics.near(x[1], yT, 1.0e-8)
    
#     hot_wall=Hot_wall()
#     east = EastBoundary()
#     p_pin = PressurePin()
#     cold_wall=Cold_wall_modified()

#     # x[0] - x coordinate
#     # x[1] - y coordinate

#     adiabatic_walls = f"near(x[0],  {experiment.dimensions.domain.x_min})"

#     # walls = hot_wall + " | " + cold_wall + " | " + adiabatic_walls
#     W_p = W.sub(0)
#     W_u = W.sub(1)
#     W_T = W.sub(2)

#     print("Setting boundary conditions...")
#     boundary_conditions = [
#         fenics.DirichletBC(W_u, (0., 0.), hot_wall),                    # no-slip on wire
#         # fenics.DirichletBC(W_u, (0., 0.), east),                       # no-slip on cold walls
#         fenics.DirichletBC(W_u.sub(0), fenics.Constant(0.0), east),     # free-slip in y on east
#         # fenics.DirichletBC(W_T, hot_wall_temperature, hot_wall),
#         # fenics.DirichletBC(W_T,T_air_bc,sub_ft,INTERFACE_TAG),
#         # fenics.DirichletBC(W_T, fenics.Constant(0.0), cold_wall),
#         # fenics.DirichletBC(W_T, fenics.Constant(0.0), east),
#         fenics.DirichletBC(W_p, fenics.Constant(0.0), p_pin, method="pointwise")
#         ]
    
#     return boundary_conditions

from utils.imports import *
from utils.parser import *
from solver.scales import *

def set_param(sub_mesh: fenics.Mesh, T_full: fenics.Function, T: fenics.Function, T_ambient: float,
              rho_air: float, beta_air: float, experiment: Experiment):
    sc = compute_nondimensional_scales(experiment)
    Pr = fenics.Constant(sc.Pr)
    Ra = fenics.Constant(sc.Ra)
    gvec = fenics.Constant((0.0, -1.0))
    T_ref = fenics.Constant(0.0)

    f_b = (Ra / Pr) * T * gvec
    mu = fenics.Constant(sc.nu / (sc.Uref * sc.Lref))

    VTa = fenics.FunctionSpace(sub_mesh, "CG", 1)
    T_air_bc = fenics.Function(VTa)
    T_air_bc.interpolate(T_full)

    hot_wall_temperature = float(T_air_bc.vector().max())
    T_h = fenics.Constant(hot_wall_temperature)
    T_c = fenics.Constant(0.0)

    return mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc


def set_bcs(W, sub_ft, T_air_bc, cold_wall_temperature, experiment: Experiment, scales: NondimScales):
    class Hot_wall(fenics.SubDomain):
        def inside(self, x, on_boundary):
            r = (experiment.dimensions.wire.diameter / 2.0) / scales.Lref
            yc = experiment.dimensions.domain.y_max / scales.Lref / 10.0 + 11.0 * r
            return (
                on_boundary
                and fenics.near(x[0]**2 + (x[1] - yc)**2 - r*r, 0.0, eps=1.0e-1 * r)
                and x[1] >= yc - r - 1e-12
                and x[1] <= yc + r + 1e-12
            )

    class WestBoundary(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(
                x[0], experiment.dimensions.domain.x_min / scales.Lref, eps=1.0e-10
            )

    class EastBoundary(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(
                x[0], experiment.dimensions.domain.x_max / scales.Lref, eps=1.0e-10
            )

    class SouthBoundary(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(
                x[1], experiment.dimensions.domain.y_min / scales.Lref, eps=1.0e-10
            )

    class PressurePin(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return (
                fenics.near(x[0], experiment.dimensions.domain.x_max / scales.Lref, 1.0e-10)
                and fenics.near(x[1], experiment.dimensions.domain.y_max / scales.Lref, 1.0e-10)
            )

    hot_wall = Hot_wall()
    west = WestBoundary()
    east = EastBoundary()
    south = SouthBoundary()
    p_pin = PressurePin()

    W_p = W.sub(0)
    W_u = W.sub(1)
    W_ux = W.sub(1).sub(0)   # x-component only
    W_uy = W.sub(1).sub(1)   # y-component only
    W_T = W.sub(2)

    print("Setting boundary conditions...")

    boundary_conditions = [
        fenics.DirichletBC(W_u, fenics.Constant((0.0, 0.0)), hot_wall),   # wire no-slip
        fenics.DirichletBC(W_ux, fenics.Constant(0.0), west),             # symmetry: u_x = 0
        # fenics.DirichletBC(W_ux, fenics.Constant(0.0), east),             # far-field lateral: no penetration
        fenics.DirichletBC(W_uy, fenics.Constant(0.0), south),            # far-field bottom: no penetration
        fenics.DirichletBC(W_T, fenics.Constant(0.0), east),              # ambient anchor on east
        fenics.DirichletBC(W_p, fenics.Constant(0.0), p_pin, method="pointwise"),
    ]

    return boundary_conditions

def volume_heat_source(experiment: Experiment):
    if experiment.initial_conditions.heat_length is not None:
        heat_volume = experiment.initial_conditions.heat_length / (math.pi *(experiment.dimensions.wire.diameter / 2)**2) 
    elif experiment.initial_conditions.heat_volume is not None:
        heat_volume = experiment.initial_conditions.heat_volume
    elif experiment.initial_conditions.heat_surface is not None:
        heat_volume = 4.0 / experiment.dimensions.wire.diameter * (experiment.initial_conditions.heat_surface )
    return heat_volume


