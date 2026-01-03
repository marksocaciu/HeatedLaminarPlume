from utils.imports import *
from utils.geometry import *
from utils.plot import *
from utils.material import *

def biot(sub_mesh: fenics.Mesh, sub_ft: fenics.MeshFunction, T_full: fenics.Function,
         qn_air: fenics.Function, T_ambient: float,
         k_wire: float, D_wire: float):
    V0_air = FunctionSpace(sub_mesh, "DG", 0)

    h_eff_air = Function(V0_air, name="h_eff_air")
    Bi_air    = Function(V0_air, name="Bi_air")

    h_eff_air.vector().zero()
    Bi_air.vector().zero()

    counts = h_eff_air.vector().copy()
    counts.zero()

    T_inf = float(T_ambient)
    k_wire_val = k_wire
    Lc = D_wire/2  # R/2 with R=1
    tol = 1e-12

    for f in fenics.facets(sub_mesh):
        if sub_ft[f] != INTERFACE_TAG:
            continue

        c_air = list(fenics.cells(f))[0]
        idx = c_air.index()

        x = f.midpoint().array()

        # Surface temperature (from conduction trace or air solution)
        Ts = T_full(fenics.Point(*x))      # preferred: solid-side temperature
        # Ts = T(Point(*x))         # alternative: air-side temperature

        dT = Ts - T_inf
        if abs(dT) < tol:
            continue

        q = qn_air.vector()[idx]

        h_eff = q / dT
        Bi = h_eff * Lc / k_wire_val

        h_eff_air.vector()[idx] += h_eff
        Bi_air.vector()[idx]    += Bi
        counts[idx] += 1.0

    counts_arr = counts.get_local()
    counts_arr[counts_arr == 0.0] = 1.0

    h_eff_air.vector()[:] /= counts_arr
    Bi_air.vector()[:]    /= counts_arr

    print("Biot number between air and solid: ", Bi_air.vector().min(), " to ", Bi_air.vector().max())

    # save_experiment(
    #     "/base/air_biot.xdmf",
    #     sub_mesh,
    #     [h_eff_air, Bi_air]
    # )
    
    return h_eff_air, Bi_air
