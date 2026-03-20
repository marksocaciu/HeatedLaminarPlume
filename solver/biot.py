from utils.imports import *
from utils.geometry import *
from utils.plot import *
from utils.material import *
from solver.scales import *

def biot(sub_mesh: fenics.Mesh, sub_ft: fenics.MeshFunction, T_full: fenics.Function,
         qn_air: fenics.Function, T_ambient: float,
         k_wire: float, D_wire: float, scale: NondimScales = None):
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


def _biot_length_from_wire_diameter(wire_diameter, mode="radius"):
    d = float(wire_diameter)
    if d <= 0.0:
        raise ValueError("wire_diameter must be positive.")

    mode = str(mode).lower()
    if mode in ("radius", "r"):
        return 0.5 * d
    if mode in ("diameter", "d"):
        return d
    if mode in ("volume_over_area", "v/a", "va", "lumped"):
        return 0.25 * d   # long cylinder: V/A = r/2 = d/4
    raise ValueError(f"Unknown characteristic_length mode: {mode}")


def compute_local_biot_on_air_submesh(
    sub_mesh,
    sub_ft,
    T_air_dim,
    qn_air,
    scales,
    T_ref,
    k_wire,
    Lc,
    interface_tag=INTERFACE_TAG,
):
    """
    Build a DG0 field with local Biot numbers on air cells touching the interface.

    Parameters
    ----------
    T_air_dim : scalar Function on the air submesh [K]
    qn_air    : DG0 scalar Function on the air submesh [nondim heat flux]
    scales    : NondimScales, used to dimensionalize qn_air
    T_ref     : reference temperature [K], usually ambient T_infty
    k_wire    : wire conductivity [W/m/K]
    Lc        : characteristic length [m]
    """
    V0 = fenics.FunctionSpace(sub_mesh, "DG", 0)
    Bi = fenics.Function(V0, name="Bi_local")
    Bi.vector().zero()

    counts = Bi.vector().copy()
    counts.zero()

    qscale = float(scales.qsurf)  # = k_inf * dTref / Lref
    T_ref = float(T_ref)
    k_wire = float(k_wire)
    Lc = float(Lc)

    if k_wire <= 0.0:
        raise ValueError("k_wire must be positive.")
    if Lc <= 0.0:
        raise ValueError("Lc must be positive.")

    tdim = sub_mesh.topology().dim()
    sub_mesh.init(tdim - 1, tdim)

    for f in fenics.facets(sub_mesh):
        if sub_ft[f] != interface_tag:
            continue

        adjacent_cells = list(fenics.cells(f))
        if not adjacent_cells:
            continue
        c = adjacent_cells[0]
        ci = c.index()

        x = f.midpoint()
        qn_dim = qscale * qn_air.vector()[ci]   # [W/m^2]
        Ts = float(T_air_dim(x))                # [K]
        dT = Ts - T_ref

        if abs(dT) > 1.0e-14:
            Bi.vector()[ci] += qn_dim * Lc / (k_wire * dT)
            counts[ci] += 1.0

    vals = Bi.vector().get_local()
    cnts = counts.get_local()
    vals = vals / np.maximum(cnts, 1.0)
    Bi.vector()[:] = vals
    Bi.vector().apply("insert")
    return Bi


def compute_average_biot(
    sub_ds,
    qn_air,
    T_air_dim,
    scales,
    T_ref,
    k_wire,
    Lc,
    interface_tag=INTERFACE_TAG,
):
    """
    Compute an area-weighted effective heat-transfer coefficient and average Bi.

        h_eff = int q'' ds / int (Ts - T_ref) ds
        Bi    = h_eff * Lc / k_wire
    """
    T_ref_c = fenics.Constant(float(T_ref))
    qn_dim = qn_air * fenics.Constant(float(scales.qsurf))

    num = fenics.assemble(qn_dim * sub_ds(interface_tag))
    den = fenics.assemble((T_air_dim - T_ref_c) * sub_ds(interface_tag))
    Lint = fenics.assemble(fenics.Constant(1.0) * sub_ds(interface_tag))

    if abs(den) <= 1.0e-14:
        return float("nan"), float("nan"), float(Lint)

    h_eff = float(num) / float(den)
    Bi_avg = h_eff * float(Lc) / float(k_wire)
    return h_eff, Bi_avg, float(Lint)


def biot_wrap(
    sub_mesh,
    sub_ft,
    sub_ds,
    T_air_dim,
    qn_air,
    scales,
    T_ref,
    k_wire,
    wire_diameter,
    characteristic_length="radius",
    interface_tag=INTERFACE_TAG,
    return_local_field=False,
):
    """
    Convenience wrapper for Biot-number diagnostics on the air submesh.

    Returns
    -------
    h_eff, Bi_avg
    or
    h_eff, Bi_avg, Bi_local
    """
    Lc = _biot_length_from_wire_diameter(wire_diameter, characteristic_length)

    h_eff, Bi_avg, Lint = compute_average_biot(
        sub_ds=sub_ds,
        qn_air=qn_air,
        T_air_dim=T_air_dim,
        scales=scales,
        T_ref=T_ref,
        k_wire=k_wire,
        Lc=Lc,
        interface_tag=interface_tag,
    )

    print("=== Biot diagnostic on air submesh ===")
    print(f"Characteristic length Lc = {Lc:.6e} m ({characteristic_length})")
    print(f"Interface length         = {Lint:.6e} m")
    print(f"Effective h              = {h_eff:.6e} W/m^2/K")
    print(f"Average Biot number      = {Bi_avg:.6e}")
    print("=====================================")

    if not return_local_field:
        return h_eff, Bi_avg

    Bi_local = compute_local_biot_on_air_submesh(
        sub_mesh=sub_mesh,
        sub_ft=sub_ft,
        T_air_dim=T_air_dim,
        qn_air=qn_air,
        scales=scales,
        T_ref=T_ref,
        k_wire=k_wire,
        Lc=Lc,
        interface_tag=interface_tag,
    )
    return h_eff, Bi_avg, Bi_local
