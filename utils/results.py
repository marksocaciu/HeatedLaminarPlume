import fenics
import numpy as np
import os
import csv


def _mark_horizontal_slab_cells(mesh, y0_star, eps_star, marker_id=1):
    cell_markers = fenics.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for cell in fenics.cells(mesh):
        y_mid = cell.midpoint().y()
        if abs(y_mid - y0_star) <= eps_star:
            cell_markers[cell] = marker_id
    return cell_markers

# def plane_fluxes_slab_star(
#     mesh_star,
#     u_star,          # vector Function on mesh_star (nondimensional)
#     theta,           # scalar Function on mesh_star (nondimensional)
#     y0_m_list,       # list of physical heights [m] above wire center
#     scales,          # has Lref, Uref, dTref
#     rho, cp, k,      # physical properties for the air
#     eps_m=None       # slab half-thickness in meters (optional)
# ):
#     Lref = float(scales.Lref)
#     Uref = float(scales.Uref)
#     dTref = float(scales.dTref)

#     # Choose slab half-thickness: ~1–2 local cell heights is typical.
#     # If user doesn't provide eps_m, take a small fraction of Lref.
#     if eps_m is None:
#         n_eps = 2.0
#         h_char_star = mesh_star.hmin()   # or local estimate per plane
#         eps_m = n_eps * h_char_star * Lref
#     eps_star = eps_m / Lref

#     # Prefactors to dimensionalize the star-integrals
#     pref_Qconv = rho * cp * Uref * dTref * Lref     # [W/m]
#     pref_Qcond = k * dTref                          # [W/m]  (note sign in integrand)
#     pref_mdot  = rho * Uref * Lref                  # [kg/(s·m)]

#     u_y = u_star[1]
#     dtheta_dy = fenics.Dx(theta, 1)

#     results = []
#     for y0_m in y0_m_list:
#         y0_star = y0_m / Lref

#         cell_markers = _mark_horizontal_slab_cells(mesh_star, y0_star, eps_star, marker_id=1)
#         dx_slab = fenics.Measure("dx", domain=mesh_star, subdomain_data=cell_markers)

#         slab_area_star = fenics.assemble(1.0 * dx_slab(1))  # in star units
#         if slab_area_star <= 0.0:
#             results.append((y0_m, np.nan, np.nan, np.nan, np.nan))
#             continue

#         # Slab average per thickness: divide by (2*eps_star)
#         inv_thickness = 1.0 / (2.0 * eps_star)

#         I_conv_star = inv_thickness * fenics.assemble(theta * u_y * dx_slab(1))
#         I_cond_star = inv_thickness * fenics.assemble(dtheta_dy * dx_slab(1))
#         I_mdot_star = inv_thickness * fenics.assemble(u_y * dx_slab(1))

#         Qconv = pref_Qconv * float(I_conv_star)
#         Qcond = -pref_Qcond * float(I_cond_star)
#         Qtot  = Qconv + Qcond
#         mdot  = pref_mdot * float(I_mdot_star)

#         results.append((y0_m, Qconv, Qcond, Qtot, mdot))

#     return results

def _build_multi_slab_markers(mesh, y0_star_list, eps_star):
    """
    Mark all requested horizontal slabs in a single MeshFunction.

    Marker 0: unmarked
    Marker i: slab for y0_star_list[i-1]
    """
    cell_markers = fenics.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

    # store once as plain floats for faster Python-side access
    y_targets = [float(y) for y in y0_star_list]

    for cell in fenics.cells(mesh):
        y_mid = cell.midpoint().y()

        # assign the first matching slab
        # if slabs overlap, earlier y0 entries take precedence
        for i, y0_star in enumerate(y_targets, start=1):
            if abs(y_mid - y0_star) <= eps_star:
                cell_markers[cell] = i
                break

    return cell_markers


def plane_fluxes_slab_star(
    mesh_star,
    u_star,          # vector Function on mesh_star (nondimensional)
    theta,           # scalar Function on mesh_star (nondimensional)
    y0_m_list,       # list of physical heights [m] above wire center
    scales,          # has Lref, Uref, dTref
    rho, cp, k,      # physical properties for the air
    eps_m=None       # slab half-thickness in meters (optional)
):
    Lref = float(scales.Lref)
    Uref = float(scales.Uref)
    dTref = float(scales.dTref)

    if eps_m is None:
        n_eps = 2.0
        h_char_star = mesh_star.hmin()
        eps_m = n_eps * h_char_star * Lref

    eps_star = float(eps_m) / Lref

    # dimensional prefactors
    pref_Qconv = rho * cp * Uref * dTref * Lref
    pref_Qcond = k * dTref
    pref_mdot = rho * Uref * Lref

    y0_star_list = [float(y0_m) / Lref for y0_m in y0_m_list]

    # mark all slabs once
    cell_markers = _build_multi_slab_markers(mesh_star, y0_star_list, eps_star)
    dx_slab = fenics.Measure("dx", domain=mesh_star, subdomain_data=cell_markers)

    u_y = u_star[1]
    dtheta_dy = fenics.Dx(theta, 1)

    inv_thickness = 1.0 / (2.0 * eps_star)

    results = []
    for i, y0_m in enumerate(y0_m_list, start=1):
        slab_area_star = fenics.assemble(fenics.Constant(1.0) * dx_slab(i))

        if slab_area_star <= 0.0:
            results.append((y0_m, np.nan, np.nan, np.nan, np.nan))
            continue

        I_conv_star = inv_thickness * fenics.assemble(theta * u_y * dx_slab(i))
        I_cond_star = inv_thickness * fenics.assemble(dtheta_dy * dx_slab(i))
        I_mdot_star = inv_thickness * fenics.assemble(u_y * dx_slab(i))

        Qconv = pref_Qconv * float(I_conv_star)
        Qcond = -pref_Qcond * float(I_cond_star)
        Qtot = Qconv + Qcond
        mdot = pref_mdot * float(I_mdot_star)

        results.append((y0_m, Qconv, Qcond, Qtot, mdot))

    return results

def compute_horizontal_plane_heat_fluxes(
    u_dim,
    T_dim,
    sub_mesh_dim,
    experiment,
    y_planes_m,
    nx=400,
    T_ref=None,
    half_domain_symmetric=True,
):
    """
    Returns dict with total heat flux through horizontal planes y = const.

    Flux definition:
        Q(y) = ∫ [rho*cp*u_y*(T - T_ref) - k*dT/dy] dx

    Units: W/m  (per unit out-of-plane depth)
    """
    rho = float(experiment.fluid.properties["rho"])
    cp  = float(experiment.fluid.properties["cp"])
    k   = float(experiment.fluid.properties["k"])

    if T_ref is None:
        T_ref = float(experiment.initial_conditions.temperature)

    coords = sub_mesh_dim.coordinates()
    x_min = float(np.min(coords[:, 0]))
    x_max = float(np.max(coords[:, 0]))
    y_min = float(np.min(coords[:, 1]))
    y_max = float(np.max(coords[:, 1]))

    u_dim.set_allow_extrapolation(True)
    T_dim.set_allow_extrapolation(True)

    fluxes = {}

    # small offset for numerical dT/dy
    dy_fd = max(5.0 * sub_mesh_dim.hmin(), 1.0e-6)

    for y in y_planes_m:
        if not (y_min + dy_fd < y < y_max - dy_fd):
            fluxes[f"Qy_{y:.4f}m"] = float("nan")
            continue

        xs = np.linspace(x_min, x_max, nx)

        conv_vals = []
        cond_vals = []

        for x in xs:
            try:
                u_val = u_dim(x, y)
                T0    = T_dim(x, y)
                Tp    = T_dim(x, y + dy_fd)
                Tm    = T_dim(x, y - dy_fd)

                uy = float(u_val[1])
                dTdy = (Tp - Tm) / (2.0 * dy_fd)

                q_conv = rho * cp * uy * (T0 - T_ref)
                q_cond = -k * dTdy

                conv_vals.append(q_conv)
                cond_vals.append(q_cond)
            except Exception:
                conv_vals.append(np.nan)
                cond_vals.append(np.nan)

        conv_vals = np.array(conv_vals, dtype=float)
        cond_vals = np.array(cond_vals, dtype=float)

        mask = np.isfinite(conv_vals) & np.isfinite(cond_vals)
        if np.count_nonzero(mask) < 2:
            fluxes[f"Qy_{y:.4f}m"] = float("nan")
            fluxes[f"Qy_conv_{y:.4f}m"] = float("nan")
            fluxes[f"Qy_cond_{y:.4f}m"] = float("nan")
            continue

        q_conv_int = np.trapz(conv_vals[mask], xs[mask])
        q_cond_int = np.trapz(cond_vals[mask], xs[mask])
        q_total    = q_conv_int + q_cond_int

        if half_domain_symmetric:
            q_conv_int *= 2.0
            q_cond_int *= 2.0
            q_total    *= 2.0

        fluxes[f"Qy_{y:.4f}m"] = float(q_total)
        fluxes[f"Qy_conv_{y:.4f}m"] = float(q_conv_int)
        fluxes[f"Qy_cond_{y:.4f}m"] = float(q_cond_int)

    return fluxes


def append_plane_flux_csv(csv_path, row):
    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
