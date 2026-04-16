from matplotlib.pyplot import sca

from utils.imports import *
from utils.parser import *
from solver.scales import *
from utils.imports import *
from utils.parser import *
from solver.scales import *
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
                x[0], -1.0 * experiment.dimensions.domain.x_max / scales.Lref, eps=1.0e-10
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

    class NorthBoundary(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(
                x[1], experiment.dimensions.domain.y_max / scales.Lref, eps=1.0e-10
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
    north = NorthBoundary()
    p_pin = PressurePin()

    W_p = W.sub(0)
    W_u = W.sub(1)
    W_T = W.sub(2)

    print("Setting boundary conditions...")

    boundary_conditions = [
        fenics.DirichletBC(W_u, fenics.Constant((0.0, 0.0)), hot_wall),   # wire no-slip
        fenics.DirichletBC(W_u, fenics.Constant((0.0, 0.0)), east),       # east no-slip
        fenics.DirichletBC(W_u, fenics.Constant((0.0, 0.0)), west),       # west no-slip
        fenics.DirichletBC(W_u, fenics.Constant((0.0, 0.0)), south),      # south no-slip
        fenics.DirichletBC(W_u, fenics.Constant((0.0, 0.0)), north),      # north no-slip
        fenics.DirichletBC(W_T, fenics.Constant(0.0), west),              # ambient anchor on west
        fenics.DirichletBC(W_T, fenics.Constant(0.0), east),              # ambient anchor on east
        fenics.DirichletBC(W_p, fenics.Constant(0.0), p_pin, method="pointwise"),
    ]

    return boundary_conditions


def build_open_boundary_measure(mesh: fenics.Mesh, experiment: Experiment, scales: NondimScales):
    """
    Build geometric facet markers for open-boundary control.

    ids:
      1 -> east
      2 -> top
      3 -> south
    """
    tdim = mesh.topology().dim()
    facet_markers = fenics.MeshFunction("size_t", mesh, tdim - 1, 0)

    x_min = experiment.dimensions.domain.x_min / scales.Lref
    x_max = experiment.dimensions.domain.x_max / scales.Lref
    y_min = experiment.dimensions.domain.y_min / scales.Lref
    y_max = experiment.dimensions.domain.y_max / scales.Lref

    class EastBoundary(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[0], x_max, 1.0e-10)

    class TopBoundary(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[1], y_max, 1.0e-10)

    class SouthBoundary(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[1], y_min, 1.0e-10)

    EastBoundary().mark(facet_markers, 1)
    TopBoundary().mark(facet_markers, 2)
    SouthBoundary().mark(facet_markers, 3)

    ds_open = fenics.Measure("ds", domain=mesh, subdomain_data=facet_markers)
    return ds_open, 1, 2, 3


def volume_heat_source(experiment: Experiment):
    if experiment.initial_conditions.heat_length is not None:
        heat_volume = experiment.initial_conditions.heat_length / (math.pi *(experiment.dimensions.wire.diameter / 2)**2) 
    elif experiment.initial_conditions.heat_volume is not None:
        heat_volume = experiment.initial_conditions.heat_volume
    elif experiment.initial_conditions.heat_surface is not None:
        heat_volume = 4.0 / experiment.dimensions.wire.diameter * (experiment.initial_conditions.heat_surface )
    return heat_volume


def set_bcs_flow_only(W_pu, sub_ft, experiment):
    """
    Flow-only boundary conditions for the AIR submesh.

    Assumes W_pu = [P1, P2] = [pressure, velocity].
    Adjust boundary tags to your exact project markers.
    """
    bcs = []

    # Subspaces
    P_sub = W_pu.sub(0)
    U_sub = W_pu.sub(1)

    zero_vec = fenics.Constant((0.0, 0.0))
    zero_p = fenics.Constant(0.0)

    # ------------------------------------------------------------------
    # 1) No-slip on the wire interface as seen from the air side
    # ------------------------------------------------------------------
    bcs.append(
        fenics.DirichletBC(U_sub, zero_vec, sub_ft, INTERFACE_TAG)
    )

    # ------------------------------------------------------------------
    # 3) Pressure pin
    #
    # Best to pin a single point to remove nullspace.
    # Reuse your existing pressure pin pattern if you already have one.
    # ------------------------------------------------------------------
    class PressurePin(fenics.SubDomain):
        def inside(self, x, on_boundary):
            # Adjust pin location if needed
            return on_boundary and fenics.near(x[0], 0.0, 1.0e-12) and fenics.near(x[1], 0.0, 1.0e-12)

    pin = PressurePin()
    bcs.append(fenics.DirichletBC(P_sub, zero_p, pin, method="pointwise"))

    return bcs
