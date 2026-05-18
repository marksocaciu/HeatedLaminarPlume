#!/usr/bin/env python3
"""
Steady plume post-processing for legacy-FEniCS XDMF/HDF5 exports.

Reads nodal temperature, nodal velocity, and optionally cell-centred heat-flux files
written in the simple XDMF/HDF5 format used by the current plume project, then computes:

  * horizontal-plane total vertical energy flux [W/m]
  * mass, volume, vertical-momentum and kinetic-energy fluxes at selected heights
  * centreline temperature/velocity decay and virtual-origin fits
  * one scalar near-wire thermal boundary-layer thickness based on an angular average
    of radial 1-percent temperature-excess distances around the cylinder
  * approximate cumulative buoyancy and vertical-momentum-flux balance diagnostics
  * CSV files and diagnostic plots, including combined profile plots across all requested heights

The script intentionally does not depend on FEniCS. It uses h5py + matplotlib.tri
linear interpolation so it can be run after the solver on saved fields.

Typical use for your nondimensional mesh coordinates:

python postprocess_steady_plume_v6.py \
  --temperature-xdmf air_temperature_steady_from_transient_step_11860.xdmf \
  --velocity-xdmf air_velocity_steady_from_transient_step_11860.xdmf \
  --heatflux-xdmf air_temperature_heatflux_final_steady_-0001.xdmf \
  --outdir steady_postprocess \
  --coords-are-dimensionless --lref 0.001 \
  --T-inf 292.95 \
  --rho 1.1614 --cp 1007.0 --k 0.0257 --mu 1.85e-5 --beta 0.0034 \
  --q-input-per-length 1.0

Important conventions:
  * coordinate-scale converts mesh coordinates to metres.
    - if --coords-are-dimensionless, coordinate_scale = --lref
    - otherwise coordinate_scale = --coordinate-scale, default 1.0
  * heights supplied by --planes are physical heights above the wire centre, in metres.
  * the wire centre is inferred automatically as y_min + H/10 + 11*r, with r = --lref.
  * all integral outputs are per unit out-of-plane depth, i.e. W/m, kg/(s m), etc.
  * if q_heat_dim is supplied, it is used internally to form q_total = rho cp uy (T-T_inf) + q_y.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import matplotlib.tri as mtri


@dataclass
class FieldData:
    points_mesh: np.ndarray          # (n, 2), original mesh coordinates
    points_m: np.ndarray             # (n, 2), physical coordinates [m]
    cells: np.ndarray                # (m, 3), triangle connectivity
    values: np.ndarray               # nodal or cell values
    name: str
    center: str                      # "Node" or "Cell"
    source_h5: Path


def parse_xdmf(xdmf_path: Path) -> Dict[str, str]:
    text = xdmf_path.read_text()

    def grab(pattern: str) -> str:
        m = re.search(pattern, text, flags=re.S)
        if not m:
            raise ValueError(f"Could not parse {pattern!r} from {xdmf_path}")
        return m.group(1).strip()

    topo_ref = grab(r'<Topology[^>]*>\s*<DataItem[^>]*>(.*?)</DataItem>\s*</Topology>')
    geom_ref = grab(r'<Geometry[^>]*>\s*<DataItem[^>]*>(.*?)</DataItem>\s*</Geometry>')
    attr_match = re.search(
        r'<Attribute\s+Name="([^"]+)"\s+AttributeType="([^"]+)"\s+Center="([^"]+)"[^>]*>\s*<DataItem[^>]*>(.*?)</DataItem>\s*</Attribute>',
        text,
        flags=re.S,
    )
    if not attr_match:
        raise ValueError(f"Could not parse Attribute block from {xdmf_path}")

    attr_name, attr_type, center, attr_ref = attr_match.groups()
    return {
        "topology_ref": topo_ref,
        "geometry_ref": geom_ref,
        "attribute_ref": attr_ref.strip(),
        "attribute_name": attr_name,
        "attribute_type": attr_type,
        "center": center,
    }


def split_hdf_ref(ref: str, xdmf_path: Path) -> Tuple[Path, str]:
    if ":" not in ref:
        raise ValueError(f"Expected HDF reference 'file.h5:/path', got {ref!r}")
    fname, dset = ref.split(":", 1)
    h5_path = Path(fname)
    if not h5_path.is_absolute():
        h5_path = xdmf_path.parent / h5_path
    return h5_path, dset


def load_xdmf_field(xdmf_path: str | Path, coordinate_scale: float) -> FieldData:
    xdmf_path = Path(xdmf_path)
    info = parse_xdmf(xdmf_path)
    h5_topo, topo_dset = split_hdf_ref(info["topology_ref"], xdmf_path)
    h5_geom, geom_dset = split_hdf_ref(info["geometry_ref"], xdmf_path)
    h5_attr, attr_dset = split_hdf_ref(info["attribute_ref"], xdmf_path)
    if h5_topo != h5_geom or h5_topo != h5_attr:
        raise ValueError("This script expects topology, geometry, and attribute in the same HDF5 file.")

    with h5py.File(h5_topo, "r") as h5:
        cells = np.asarray(h5[topo_dset], dtype=np.int64)
        points_mesh = np.asarray(h5[geom_dset], dtype=float)
        values = np.asarray(h5[attr_dset], dtype=float)

    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim == 2 and values.shape[1] == 3:
        # FEniCS writes 2D vectors as 3 columns; keep x and y.
        values = values[:, :2]

    return FieldData(
        points_mesh=points_mesh,
        points_m=points_mesh * float(coordinate_scale),
        cells=cells,
        values=values,
        name=info["attribute_name"],
        center=info["center"],
        source_h5=h5_topo,
    )


def check_same_mesh(a: FieldData, b: FieldData, tol: float = 1e-12) -> None:
    if a.cells.shape != b.cells.shape or a.points_m.shape != b.points_m.shape:
        raise ValueError("The fields do not have the same mesh shape.")
    if not np.array_equal(a.cells, b.cells):
        raise ValueError("The fields do not use the same triangle connectivity.")
    if np.max(np.abs(a.points_m - b.points_m)) > tol:
        raise ValueError("The fields do not use the same coordinates.")


def finite_or_nan(x):
    arr = np.asarray(x)
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype=float)


def make_interpolators(tri: mtri.Triangulation, T: np.ndarray, u: np.ndarray):
    Ti = mtri.LinearTriInterpolator(tri, T)
    uxi = mtri.LinearTriInterpolator(tri, u[:, 0])
    uyi = mtri.LinearTriInterpolator(tri, u[:, 1])
    return Ti, uxi, uyi


def cell_centres(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    return points[cells].mean(axis=1)


def cell_areas(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    p0 = points[cells[:, 0]]
    p1 = points[cells[:, 1]]
    p2 = points[cells[:, 2]]
    return 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))


def sample_line(Ti, uxi, uyi, x: np.ndarray, y: float):
    yy = np.full_like(x, y, dtype=float)
    T = finite_or_nan(Ti(x, yy))
    ux = finite_or_nan(uxi(x, yy))
    uy = finite_or_nan(uyi(x, yy))
    return T, ux, uy


def robust_trapz(y: np.ndarray, x: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(x)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.trapezoid(y[mask], x[mask]))


def first_crossing_half_width(x: np.ndarray, f: np.ndarray, threshold: float, side: str) -> float:
    """Return distance from x=0 to first |x| where f <= threshold on one side."""
    mask = np.isfinite(x) & np.isfinite(f)
    x = x[mask]
    f = f[mask]
    if x.size < 2:
        return float("nan")
    if side == "right":
        m = x >= 0
        xs = x[m]
        fs = f[m]
        order = np.argsort(xs)
    else:
        m = x <= 0
        xs = -x[m]  # positive distance from centreline
        fs = f[m]
        order = np.argsort(xs)
    xs = xs[order]
    fs = fs[order]
    if xs.size < 2:
        return float("nan")

    # Use normalized absolute value where appropriate; threshold is already in same units as f.
    below = np.where(fs <= threshold)[0]
    if below.size == 0:
        return float("nan")
    j = int(below[0])
    if j == 0:
        return float(xs[0])
    x0, x1 = xs[j - 1], xs[j]
    f0, f1 = fs[j - 1], fs[j]
    if f1 == f0:
        return float(x1)
    w = (threshold - f0) / (f1 - f0)
    return float(x0 + w * (x1 - x0))


def fit_virtual_origin_powerlaw(y: np.ndarray, a: np.ndarray, exponent: float, min_points: int = 4) -> Dict[str, float]:
    """
    Fit a(y) = C * (y - y0)^(-exponent).

    Transform a^(-1/exponent) = C^(-1/exponent) * (y - y0) = A*y + B,
    so y0 = -B/A. Requires positive centreline amplitude a.
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=float)
    mask = np.isfinite(y) & np.isfinite(a) & (a > 0.0)
    y = y[mask]
    a = a[mask]
    if y.size < min_points:
        return {"C": np.nan, "y0": np.nan, "r2": np.nan, "npoints": int(y.size)}

    z = a ** (-1.0 / exponent)
    A, B = np.polyfit(y, z, 1)
    zhat = A * y + B
    ss_res = float(np.sum((z - zhat) ** 2))
    ss_tot = float(np.sum((z - np.mean(z)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    y0 = -B / A if A != 0 else np.nan
    C = A ** (-exponent) if A > 0 else np.nan
    return {"C": float(C), "y0": float(y0), "r2": float(r2), "npoints": int(y.size)}


def write_csv(path: Path, rows: List[Dict[str, float | str]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_xy(path: Path, x, ys: Sequence[Tuple[str, np.ndarray]], xlabel: str, ylabel: str, title: str, semilogy: bool = False) -> None:
    plt.figure(figsize=(7.2, 4.8))
    for label, y in ys:
        if semilogy:
            plt.semilogy(x, y, label=label)
        else:
            plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.35)
    if len(ys) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--temperature-xdmf", required=True)
    ap.add_argument("--velocity-xdmf", required=True)
    ap.add_argument("--heatflux-xdmf", default=None, help="Optional cell-centred q_heat_dim XDMF; assumed q=-k grad(T)")
    ap.add_argument("--outdir", default="steady_plume_postprocess")

    coord = ap.add_mutually_exclusive_group()
    coord.add_argument("--coords-are-dimensionless", action="store_true", help="Multiply coordinates by --lref")
    coord.add_argument("--coords-are-dimensional", action="store_true", help="Use coordinates as metres unless --coordinate-scale is given")
    ap.add_argument("--lref", type=float, default=None, help="Reference length [m], e.g. wire radius. Required for dimensionless coordinates.")
    ap.add_argument("--coordinate-scale", type=float, default=None, help="Generic multiplier from mesh coordinates to metres.")

    # The wire centre is not a user input for this case family.
    # Geometry convention: source centre at y_min + H/10 + 11*r, with r = lref.
    ap.add_argument("--T-inf", type=float, required=True, help="Ambient/reference temperature [K]")
    ap.add_argument("--rho", type=float, required=True, help="Density [kg/m^3]")
    ap.add_argument("--cp", type=float, required=True, help="Specific heat [J/(kg K)]")
    ap.add_argument("--k", type=float, required=True, help="Thermal conductivity [W/(m K)]")
    ap.add_argument("--mu", type=float, default=None, help="Dynamic viscosity [Pa s]; used for Re-like diagnostics")
    ap.add_argument("--beta", type=float, default=None, help="Thermal expansion coefficient [1/K]; used for buoyancy diagnostics")
    ap.add_argument("--g", type=float, default=9.81, help="Gravity magnitude [m/s^2]")
    ap.add_argument("--q-input-per-length", type=float, default=None, help="Known heat input per unit length [W/m] for energy-balance error")

    ap.add_argument("--planes", type=float, nargs="+", default=[0.01, 0.02, 0.04, 0.08], help="Physical heights above wire [m]")
    ap.add_argument("--fit-y-min", type=float, default=None, help="Minimum physical height above wire [m] used for virtual-origin fits")
    ap.add_argument("--fit-y-max", type=float, default=None, help="Maximum physical height above wire [m] used for virtual-origin fits")
    ap.add_argument("--profile-half-width", type=float, default=None, help="Sample only |x|<=this physical half-width [m]. Default: full mesh width.")
    ap.add_argument("--nx", type=int, default=1601, help="Number of x samples per horizontal profile")
    ap.add_argument("--ny-balance", type=int, default=300, help="Number of y levels for balance/fit curves")
    ap.add_argument("--threshold", type=float, default=0.01, help="Boundary-layer threshold fraction of local near-surface temperature excess")
    ap.add_argument("--bl-angles", type=int, default=181, help="Number of radial directions used for angular-average near-wire boundary-layer thickness")
    ap.add_argument("--bl-r-max", type=float, default=None, help="Maximum radial distance from cylinder surface [m] for boundary-layer search. Default: largest distance fitting in domain")
    ap.add_argument("--bl-nr", type=int, default=600, help="Number of radial samples per angular direction for boundary-layer search")
    args = ap.parse_args()

    if args.coords_are_dimensionless:
        if args.lref is None:
            raise SystemExit("--lref is required when --coords-are-dimensionless is used.")
        coordinate_scale = float(args.lref)
    elif args.coordinate_scale is not None:
        coordinate_scale = float(args.coordinate_scale)
    else:
        coordinate_scale = 1.0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Tdata = load_xdmf_field(args.temperature_xdmf, coordinate_scale)
    Udata = load_xdmf_field(args.velocity_xdmf, coordinate_scale)
    check_same_mesh(Tdata, Udata)
    if Tdata.center != "Node" or Udata.center != "Node":
        raise ValueError("Temperature and velocity must be nodal fields.")
    if Udata.values.ndim != 2 or Udata.values.shape[1] != 2:
        raise ValueError("Velocity field must be a two-component vector field.")

    Qdata = None
    if args.heatflux_xdmf:
        Qdata = load_xdmf_field(args.heatflux_xdmf, coordinate_scale)
        check_same_mesh(Tdata, Qdata)
        if Qdata.center != "Cell":
            raise ValueError("Heat-flux field is expected to be cell-centred.")
        if Qdata.values.ndim != 2 or Qdata.values.shape[1] != 2:
            raise ValueError("Heat-flux field must be a two-component vector field.")

    x = Tdata.points_m[:, 0]
    y = Tdata.points_m[:, 1]
    T = np.asarray(Tdata.values, dtype=float)
    u = np.asarray(Udata.values, dtype=float)
    tri = mtri.Triangulation(x, y, Tdata.cells)
    Ti, uxi, uyi = make_interpolators(tri, T, u)
    dTdx_i, dTdy_i = Ti.gradient(tri.x, tri.y)
    dTdy_i = finite_or_nan(dTdy_i)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    domain_h_m = ymax - ymin
    if args.lref is None:
        raise SystemExit("--lref is required because the wire radius is needed to infer wire_y = y_min + H/10 + 11*r.")
    wire_radius_m = float(args.lref)
    wire_y_m = ymin + domain_h_m / 10.0 + 11.0 * wire_radius_m
    wire_top_y_m = wire_y_m + wire_radius_m
    if not (ymin <= wire_y_m <= ymax):
        raise SystemExit(f"Inferred wire centre y={wire_y_m:g} m lies outside mesh bounds [{ymin:g}, {ymax:g}] m.")

    if args.profile_half_width is None:
        xs = np.linspace(xmin, xmax, args.nx)
    else:
        hw = float(args.profile_half_width)
        xs = np.linspace(max(xmin, -hw), min(xmax, hw), args.nx)

    # Plane diagnostics at requested heights.
    plane_rows = []
    profile_rows = []

    # If heat flux is cell-centred, interpolate it from triangle centres onto plane samples.
    qx_i = qy_i = None
    if Qdata is not None:
        cc = cell_centres(Tdata.points_m, Tdata.cells)
        # Triangulate cell centres for interpolation. This is a diagnostic interpolation only.
        qtri = mtri.Triangulation(cc[:, 0], cc[:, 1])
        qx_i = mtri.LinearTriInterpolator(qtri, Qdata.values[:, 0])
        qy_i = mtri.LinearTriInterpolator(qtri, Qdata.values[:, 1])

    for h in args.planes:
        yp = float(wire_y_m + h)
        Tline, uxline, uyline = sample_line(Ti, uxi, uyi, xs, yp)
        theta = Tline - args.T_inf
        speed2 = uxline**2 + uyline**2

        if qy_i is not None:
            qcond_y = finite_or_nan(qy_i(xs, np.full_like(xs, yp)))
        else:
            dTdx_line, dTdy_line = Ti.gradient(xs, np.full_like(xs, yp))
            qcond_y = -args.k * finite_or_nan(dTdy_line)

        qconv_y = args.rho * args.cp * uyline * theta
        qtot_y = qconv_y + qcond_y

        # Upward-only and downward-only convective split is useful for closed boxes / recirculation.
        up = uyline > 0.0
        down = uyline < 0.0

        Qconv = robust_trapz(qconv_y, xs)
        Qcond = robust_trapz(qcond_y, xs)
        Qtot = robust_trapz(qtot_y, xs)
        Qconv_up = robust_trapz(np.where(up, qconv_y, 0.0), xs)
        Qconv_down = robust_trapz(np.where(down, qconv_y, 0.0), xs)

        mdot = robust_trapz(args.rho * uyline, xs)
        mdot_up = robust_trapz(np.where(up, args.rho * uyline, 0.0), xs)
        mdot_down = robust_trapz(np.where(down, args.rho * uyline, 0.0), xs)
        vdot = robust_trapz(uyline, xs)
        mom_y = robust_trapz(args.rho * uyline * uyline, xs)
        mom_y_signed = robust_trapz(args.rho * uyline * np.abs(uyline), xs)
        ke_flux = robust_trapz(0.5 * args.rho * speed2 * uyline, xs)
        ke_flux_abs = robust_trapz(0.5 * args.rho * speed2 * np.abs(uyline), xs)

        idx0 = int(np.nanargmin(np.abs(xs)))
        Tc = float(Tline[idx0])
        Tce = float(theta[idx0])
        uyc = float(uyline[idx0])

        th_threshold = args.threshold * max(Tce, 0.0)
        # For velocity thickness, use centreline upward velocity as reference. If centreline is bad, use max upward velocity.
        uy_ref = uyc if uyc > 0 else np.nanmax(uyline)
        uy_threshold = args.threshold * max(uy_ref, 0.0)
        deltaT_r = first_crossing_half_width(xs, theta, th_threshold, side="right")
        deltaT_l = first_crossing_half_width(xs, theta, th_threshold, side="left")
        deltaU_r = first_crossing_half_width(xs, uyline, uy_threshold, side="right")
        deltaU_l = first_crossing_half_width(xs, uyline, uy_threshold, side="left")

        rel_err = (Qtot - args.q_input_per_length) / args.q_input_per_length if args.q_input_per_length else np.nan

        plane_rows.append({
            "height_m": h,
            "y_m": yp,
            "T_center_K": Tc,
            "DeltaT_center_K": Tce,
            "uy_center_m_per_s": uyc,
            "Q_total_W_per_m": Qtot,
            "Q_conv_up_W_per_m": Qconv_up,
            "Q_conv_down_W_per_m": Qconv_down,
            "energy_balance_rel_error": rel_err,
            "volume_flux_m2_per_s": vdot,
            "mass_flux_kg_per_s_per_m": mdot,
            "mass_flux_up_kg_per_s_per_m": mdot_up,
            "mass_flux_down_kg_per_s_per_m": mdot_down,
            "vertical_momentum_flux_N_per_m": mom_y,
            "signed_vertical_momentum_flux_N_per_m": mom_y_signed,
            "kinetic_energy_flux_W_per_m": ke_flux,
            "kinetic_energy_flux_abs_W_per_m": ke_flux_abs,
            "thermal_1pct_halfwidth_left_m": deltaT_l,
            "thermal_1pct_halfwidth_right_m": deltaT_r,
            "thermal_1pct_fullwidth_m": deltaT_l + deltaT_r if np.isfinite(deltaT_l) and np.isfinite(deltaT_r) else np.nan,
            "velocity_1pct_halfwidth_left_m": deltaU_l,
            "velocity_1pct_halfwidth_right_m": deltaU_r,
            "velocity_1pct_fullwidth_m": deltaU_l + deltaU_r if np.isfinite(deltaU_l) and np.isfinite(deltaU_r) else np.nan,
        })

        for xi, Ti_, uxi_, uyi_, qt_ in zip(xs, Tline, uxline, uyline, qtot_y):
            profile_rows.append({
                "height_m": h,
                "y_m": yp,
                "x_m": xi,
                "T_K": Ti_,
                "DeltaT_K": Ti_ - args.T_inf,
                "ux_m_per_s": uxi_,
                "uy_m_per_s": uyi_,
                "qtotal_y_W_per_m2": qt_,
            })

    write_csv(outdir / "plane_integrals.csv", plane_rows)
    write_csv(outdir / "plane_profiles.csv", profile_rows)

    # Continuous centreline / balance curves from wire to top.
    y_start = max(float(wire_y_m + min(args.planes) * 0.25), ymin)
    y_end = min(ymax, float(wire_y_m + max(args.planes) * 1.25))
    yy = np.linspace(y_start, y_end, args.ny_balance)
    x0 = np.zeros_like(yy)
    T_c = finite_or_nan(Ti(x0, yy))
    ux_c = finite_or_nan(uxi(x0, yy))
    uy_c = finite_or_nan(uyi(x0, yy))
    dT_c = T_c - args.T_inf

    center_rows = []
    for yi, Ti_, uxi_, uyi_ in zip(yy, T_c, ux_c, uy_c):
        center_rows.append({
            "y_m": yi,
            "height_above_wire_m": yi - wire_y_m,
            "T_center_K": Ti_,
            "DeltaT_center_K": Ti_ - args.T_inf,
            "ux_center_m_per_s": uxi_,
            "uy_center_m_per_s": uyi_,
        })
    write_csv(outdir / "centerline.csv", center_rows)

    # Angular-average near-wire thermal boundary-layer thickness.
    # Definition: for each ray starting at the cylinder surface, find the radial distance
    # where DeltaT falls to threshold * DeltaT_near_surface on that same ray. The reported
    # scalar is the angular mean of these finite distances. This is a practical diagnostic
    # for a finite cylinder; it is not the same as a boundary-layer-theory local delta(theta).
    if args.bl_angles < 8:
        raise SystemExit("--bl-angles should be at least 8 for a meaningful angular average.")
    if args.bl_nr < 20:
        raise SystemExit("--bl-nr should be at least 20 for a meaningful radial crossing search.")

    # Default radial search length: stay inside the rectangular computational bounds in most directions.
    # Use a conservative upper bound so rays do not immediately leave the mesh.
    if args.bl_r_max is None:
        bl_r_max = 0.95 * min(
            wire_y_m - ymin,
            ymax - wire_y_m,
            0.5 * (xmax - xmin),
        ) - wire_radius_m
        bl_r_max = max(bl_r_max, 5.0 * wire_radius_m)
    else:
        bl_r_max = float(args.bl_r_max)
    bl_r_max = min(bl_r_max, max(ymax - ymin, xmax - xmin))
    if bl_r_max <= 0.0:
        raise SystemExit(f"Invalid boundary-layer radial search length: {bl_r_max:g} m")

    angles = np.linspace(0.0, 2.0 * np.pi, args.bl_angles, endpoint=False)
    s_ray = np.linspace(0.0, bl_r_max, args.bl_nr)  # distance from cylinder surface
    ray_rows = []
    delta_values = []
    theta_reference_values = []

    for ang in angles:
        ca, sa = math.cos(float(ang)), math.sin(float(ang))
        rr = wire_radius_m + s_ray
        xr = rr * ca
        yr = wire_y_m + rr * sa
        Tr = finite_or_nan(Ti(xr, yr))
        dTr = Tr - args.T_inf
        valid = np.isfinite(dTr)
        # Reference is the first finite positive sample just outside the cylinder.
        positive = np.where(valid & (dTr > 0.0))[0]
        if positive.size == 0:
            delta = np.nan
            dTref = np.nan
            dTthr = np.nan
        else:
            i0 = int(positive[0])
            dTref = float(dTr[i0])
            dTthr = args.threshold * dTref
            # Search only after the reference point. This allows a tiny masked/invalid region near the surface.
            hit = np.where(valid & (np.arange(dTr.size) > i0) & (dTr <= dTthr))[0]
            if hit.size == 0:
                delta = np.nan
            else:
                j = int(hit[0])
                j0 = max(i0, j - 1)
                s0_, s1_ = s_ray[j0], s_ray[j]
                f0_, f1_ = dTr[j0], dTr[j]
                if np.isfinite(f0_) and np.isfinite(f1_) and f1_ != f0_:
                    delta = float(s0_ + (dTthr - f0_) * (s1_ - s0_) / (f1_ - f0_))
                else:
                    delta = float(s1_)
        if np.isfinite(delta):
            delta_values.append(delta)
        if np.isfinite(dTref):
            theta_reference_values.append(dTref)
        ray_rows.append({
            "angle_rad": float(ang),
            "angle_deg": float(np.degrees(ang)),
            "direction_x": ca,
            "direction_y": sa,
            "DeltaT_reference_near_surface_K": dTref,
            "DeltaT_threshold_K": dTthr,
            "thermal_boundary_layer_thickness_m": delta,
            "thermal_boundary_layer_thickness_over_r": delta / wire_radius_m if np.isfinite(delta) and wire_radius_m else np.nan,
        })

    delta_arr = np.array(delta_values, dtype=float)
    dTref_arr = np.array(theta_reference_values, dtype=float)
    thermal_bl_mean_m = float(np.mean(delta_arr)) if delta_arr.size else np.nan
    thermal_bl_std_m = float(np.std(delta_arr, ddof=1)) if delta_arr.size > 1 else np.nan
    thermal_bl_min_m = float(np.min(delta_arr)) if delta_arr.size else np.nan
    thermal_bl_max_m = float(np.max(delta_arr)) if delta_arr.size else np.nan
    thermal_bl_median_m = float(np.median(delta_arr)) if delta_arr.size else np.nan

    write_csv(outdir / "near_wire_boundary_layer_by_angle.csv", ray_rows)
    write_csv(outdir / "near_wire_boundary_layer.csv", [{
        "definition": "angular_mean_radial_distance_from_cylinder_surface_to_1pct_local_DeltaT",
        "wire_radius_m": wire_radius_m,
        "wire_center_y_m": wire_y_m,
        "threshold_fraction": args.threshold,
        "radial_search_max_m": bl_r_max,
        "n_angles_requested": int(args.bl_angles),
        "n_angles_valid": int(delta_arr.size),
        "valid_angle_fraction": float(delta_arr.size / args.bl_angles),
        "mean_DeltaT_reference_near_surface_K": float(np.mean(dTref_arr)) if dTref_arr.size else np.nan,
        "thermal_boundary_layer_thickness_mean_m": thermal_bl_mean_m,
        "thermal_boundary_layer_thickness_std_m": thermal_bl_std_m,
        "thermal_boundary_layer_thickness_min_m": thermal_bl_min_m,
        "thermal_boundary_layer_thickness_median_m": thermal_bl_median_m,
        "thermal_boundary_layer_thickness_max_m": thermal_bl_max_m,
        "thermal_boundary_layer_thickness_mean_over_r": thermal_bl_mean_m / wire_radius_m if np.isfinite(thermal_bl_mean_m) and wire_radius_m else np.nan,
        "thermal_boundary_layer_thickness_std_over_r": thermal_bl_std_m / wire_radius_m if np.isfinite(thermal_bl_std_m) and wire_radius_m else np.nan,
        "thermal_boundary_layer_thickness_min_over_r": thermal_bl_min_m / wire_radius_m if np.isfinite(thermal_bl_min_m) and wire_radius_m else np.nan,
        "thermal_boundary_layer_thickness_median_over_r": thermal_bl_median_m / wire_radius_m if np.isfinite(thermal_bl_median_m) and wire_radius_m else np.nan,
        "thermal_boundary_layer_thickness_max_over_r": thermal_bl_max_m / wire_radius_m if np.isfinite(thermal_bl_max_m) and wire_radius_m else np.nan,
    }])

    # Virtual origin fits. Classical laminar line-source similarity suggests DeltaT_c ~ (y-y0)^(-3/5), uy_c ~ (y-y0)^(-1/5).
    fit_mask = np.isfinite(yy) & np.isfinite(dT_c) & np.isfinite(uy_c)
    if args.fit_y_min is not None:
        fit_mask &= (yy - wire_y_m) >= args.fit_y_min
    if args.fit_y_max is not None:
        fit_mask &= (yy - wire_y_m) <= args.fit_y_max
    fitT = fit_virtual_origin_powerlaw(yy[fit_mask], dT_c[fit_mask], exponent=3.0/5.0)
    fitU = fit_virtual_origin_powerlaw(yy[fit_mask], uy_c[fit_mask], exponent=1.0/5.0)
    fit_rows = [
        {"field": "temperature_centerline", "assumed_decay_exponent": 3.0/5.0, **fitT},
        {"field": "velocity_centerline", "assumed_decay_exponent": 1.0/5.0, **fitU},
    ]
    write_csv(outdir / "virtual_origin_fits.csv", fit_rows)

    # Approximate balance curves at many y-levels using the same x sampling.
    balance_rows = []
    for yp in yy:
        Tline, uxline, uyline = sample_line(Ti, uxi, uyi, xs, yp)
        theta = Tline - args.T_inf
        speed2 = uxline**2 + uyline**2
        if qy_i is not None:
            qcond_y = finite_or_nan(qy_i(xs, np.full_like(xs, yp)))
        else:
            dTdx_line, dTdy_line = Ti.gradient(xs, np.full_like(xs, yp))
            qcond_y = -args.k * finite_or_nan(dTdy_line)
        qconv_y = args.rho * args.cp * uyline * theta
        Qconv = robust_trapz(qconv_y, xs)
        Qcond = robust_trapz(qcond_y, xs)
        mom_y = robust_trapz(args.rho * uyline * uyline, xs)
        ke_flux = robust_trapz(0.5 * args.rho * speed2 * uyline, xs)
        mass_flux = robust_trapz(args.rho * uyline, xs)
        buoy_line = np.nan
        if args.beta is not None:
            buoy_line = robust_trapz(args.rho * args.g * args.beta * theta, xs)  # N/m^2 integrated over x -> N/m^2? per unit height per depth
        balance_rows.append({
            "y_m": yp,
            "height_above_wire_m": yp - wire_y_m,
            "Q_total_W_per_m": Qconv + Qcond,
            "mass_flux_kg_per_s_per_m": mass_flux,
            "vertical_momentum_flux_N_per_m": mom_y,
            "kinetic_energy_flux_W_per_m": ke_flux,
            "buoyancy_force_density_integral_N_per_m2": buoy_line,
        })
    # cumulative buoyancy over y, useful proxy for vertical momentum source.
    if args.beta is not None:
        b = np.array([r["buoyancy_force_density_integral_N_per_m2"] for r in balance_rows], dtype=float)
        cum = np.full_like(b, np.nan)
        valid = np.isfinite(b)
        if np.count_nonzero(valid) > 1:
            # cumulative trapezoid without scipy
            cum[0] = 0.0
            for i in range(1, len(b)):
                if np.isfinite(b[i]) and np.isfinite(b[i - 1]):
                    cum[i] = (cum[i - 1] if np.isfinite(cum[i - 1]) else 0.0) + 0.5 * (b[i] + b[i - 1]) * (yy[i] - yy[i - 1])
            for r, c in zip(balance_rows, cum):
                r["cumulative_buoyancy_N_per_m"] = c
    write_csv(outdir / "balance_curves.csv", balance_rows)

    # Plots.
    plane_h = np.array([r["height_m"] for r in plane_rows], dtype=float)
    energy_series = [("total", np.array([r["Q_total_W_per_m"] for r in plane_rows]))]
    if args.q_input_per_length:
        energy_series.append(("input heat per length", np.full_like(plane_h, args.q_input_per_length, dtype=float)))
    plot_xy(outdir / "energy_flux_vs_height.png", plane_h, energy_series,
            "height above wire [m]", "total vertical energy flux [W/m]",
            "Total vertical energy flux through horizontal planes")
    plot_xy(outdir / "mass_momentum_vs_height.png", plane_h,
            [("mass flux", np.array([r["mass_flux_kg_per_s_per_m"] for r in plane_rows])),
             ("vertical momentum flux", np.array([r["vertical_momentum_flux_N_per_m"] for r in plane_rows]))],
            "height above wire [m]", "integral", "Mass and vertical-momentum flux")
    # Centreline plots with fitted virtual-source curves.
    h_center = yy - wire_y_m
    dT_fit = np.full_like(yy, np.nan, dtype=float)
    uy_fit = np.full_like(yy, np.nan, dtype=float)
    if np.isfinite(fitT["C"]) and np.isfinite(fitT["y0"]):
        m = yy > fitT["y0"]
        dT_fit[m] = fitT["C"] * (yy[m] - fitT["y0"]) ** (-(3.0/5.0))
    if np.isfinite(fitU["C"]) and np.isfinite(fitU["y0"]):
        m = yy > fitU["y0"]
        uy_fit[m] = fitU["C"] * (yy[m] - fitU["y0"]) ** (-(1.0/5.0))

    plot_xy(outdir / "centerline_temperature_virtual_origin.png", h_center,
            [("Delta T center", dT_c), ("line-plume fit", dT_fit)],
            "height above wire centre [m]", "Delta T centre [K]",
            "Centreline temperature decay and virtual-origin fit", semilogy=True)
    plot_xy(outdir / "centerline_velocity_virtual_origin.png", h_center,
            [("uy center", uy_c), ("line-plume fit", uy_fit)],
            "height above wire centre [m]", "uy centre [m/s]",
            "Centreline vertical velocity decay and virtual-origin fit", semilogy=True)

    # Linearized virtual-origin convergence plots. Intercept with zero gives y0.
    def plot_linearized_virtual_origin(path, yy_, amp_, exponent, fit, ylabel, title):
        mask = np.isfinite(yy_) & np.isfinite(amp_) & (amp_ > 0)
        plt.figure(figsize=(7.2, 4.8))
        z = np.full_like(yy_, np.nan, dtype=float)
        z[mask] = amp_[mask] ** (-1.0 / exponent)
        plt.plot(yy_[mask] - wire_y_m, z[mask], label="transformed centreline")
        if np.isfinite(fit["y0"]):
            A = fit["C"] ** (-1.0 / exponent) if np.isfinite(fit["C"]) and fit["C"] > 0 else np.nan
            if np.isfinite(A):
                zfit = A * (yy_ - fit["y0"])
                plt.plot(yy_ - wire_y_m, zfit, label=f"linear fit; y0={fit['y0']:.4e} m")
                plt.axvline(fit["y0"] - wire_y_m, linestyle="--", label="virtual origin")
        plt.xlabel("height above wire centre [m]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    plot_linearized_virtual_origin(outdir / "virtual_origin_temperature_linearized.png", yy, dT_c, 3.0/5.0, fitT,
                                   r"$\Delta T_c^{-5/3}$", "Linearized temperature virtual-origin fit")
    plot_linearized_virtual_origin(outdir / "virtual_origin_velocity_linearized.png", yy, uy_c, 1.0/5.0, fitU,
                                   r"$u_{y,c}^{-5}$", "Linearized velocity virtual-origin fit")

    # Angular near-wire boundary-layer plot.
    angle_deg = np.array([r["angle_deg"] for r in ray_rows], dtype=float)
    delta_ang = np.array([r["thermal_boundary_layer_thickness_m"] for r in ray_rows], dtype=float)
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(angle_deg, delta_ang / wire_radius_m)
    if np.isfinite(thermal_bl_mean_m):
        plt.axhline(thermal_bl_mean_m / wire_radius_m, linestyle="--", label=f"mean={thermal_bl_mean_m / wire_radius_m:.4g} r")
    plt.xlabel("angle around cylinder [deg]; 0=right, 90=up")
    plt.ylabel("1% thermal thickness / r")
    plt.title("Near-wire angular thermal boundary-layer thickness")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "near_wire_boundary_layer_by_angle.png", dpi=180)
    plt.close()

    # Polar-style visualization of the same radial thickness values.
    plt.figure(figsize=(6.2, 6.2))
    ax = plt.subplot(111, projection="polar")
    ax.plot(angles, delta_ang / wire_radius_m)
    if np.isfinite(thermal_bl_mean_m):
        ax.plot(angles, np.full_like(angles, thermal_bl_mean_m / wire_radius_m), linestyle="--", label="mean")
    ax.set_title("Angular 1% thermal thickness / r")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outdir / "near_wire_boundary_layer_polar.png", dpi=180)
    plt.close()

    # Combined profile plots: one figure per quantity, with all requested heights overlaid.
    def combined_profile_plot(filename: str, quantity_key: str, ylabel: str, title: str) -> None:
        plt.figure(figsize=(7.2, 4.8))
        for h in args.planes:
            rows = [r for r in profile_rows if abs(r["height_m"] - h) < 1e-15]
            if not rows:
                continue
            xp = np.array([r["x_m"] for r in rows], dtype=float)
            qp = np.array([r[quantity_key] for r in rows], dtype=float)
            order = np.argsort(xp)
            plt.plot(xp[order], qp[order], label=f"h={h:g} m")
        plt.xlabel("x [m]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=180)
        plt.close()

    combined_profile_plot(
        "profiles_temperature_all_heights.png",
        "DeltaT_K",
        r"$T-T_\infty$ [K]",
        "Temperature profiles at requested heights",
    )
    combined_profile_plot(
        "profiles_uy_all_heights.png",
        "uy_m_per_s",
        r"$u_y$ [m/s]",
        "Vertical-velocity profiles at requested heights",
    )
    combined_profile_plot(
        "profiles_ux_all_heights.png",
        "ux_m_per_s",
        r"$u_x$ [m/s]",
        "Horizontal-velocity profiles at requested heights",
    )
    combined_profile_plot(
        "profiles_qtotal_all_heights.png",
        "qtotal_y_W_per_m2",
        r"$q_{y,total}$ [W/m$^2$]",
        "Total vertical energy-flux-density profiles at requested heights",
    )

    summary_path = outdir / "README_summary.txt"
    with summary_path.open("w") as f:
        f.write("Steady plume post-processing summary\n")
        f.write("====================================\n\n")
        f.write(f"Temperature file: {args.temperature_xdmf}\n")
        f.write(f"Velocity file:    {args.velocity_xdmf}\n")
        f.write(f"Heat-flux file:   {args.heatflux_xdmf}\n")
        f.write(f"Coordinate scale to metres: {coordinate_scale:.16e}\n")
        f.write(f"Wire/source y-coordinate inferred as y_min + H/10 + 11*r: {wire_y_m:.16e} m\n")
        f.write(f"Wire radius r = lref: {wire_radius_m:.16e} m\n")
        f.write(f"Near-wire 1% thermal boundary-layer thickness, angular mean: {thermal_bl_mean_m:.8e} m ({thermal_bl_mean_m / wire_radius_m if np.isfinite(thermal_bl_mean_m) else np.nan:.6g} r)\n")
        f.write(f"Near-wire 1% thickness angular std/min/median/max: {thermal_bl_std_m:.8e}, {thermal_bl_min_m:.8e}, {thermal_bl_median_m:.8e}, {thermal_bl_max_m:.8e} m\n")
        f.write(f"Valid boundary-layer ray crossings: {delta_arr.size}/{args.bl_angles}\n")
        f.write(f"Physical mesh bounds: x=[{xmin:.6e}, {xmax:.6e}], y=[{ymin:.6e}, {ymax:.6e}] m\n")
        f.write(f"T_inf: {args.T_inf:.12g} K\n")
        if args.q_input_per_length is not None:
            f.write(f"Input heat per length: {args.q_input_per_length:.12g} W/m\n")
        f.write("\nVirtual-origin fits use line-plume exponents:\n")
        f.write("  DeltaT_c ~ C_T * (y - y0_T)^(-3/5)\n")
        f.write("  uy_c     ~ C_U * (y - y0_U)^(-1/5)\n")
        f.write(f"Temperature virtual origin y0 = {fitT['y0']:.8e} m, R2={fitT['r2']:.6f}, n={fitT['npoints']}\n")
        f.write(f"Velocity virtual origin    y0 = {fitU['y0']:.8e} m, R2={fitU['r2']:.6f}, n={fitU['npoints']}\n")
        f.write("\nMain outputs:\n")
        f.write("  plane_integrals.csv\n")
        f.write("  plane_profiles.csv\n")
        f.write("  centerline.csv\n")
        f.write("  virtual_origin_fits.csv\n")
        f.write("  near_wire_boundary_layer.csv\n")
        f.write("  near_wire_boundary_layer_by_angle.csv\n")
        f.write("  balance_curves.csv\n")
        f.write("  *.png diagnostic plots\n")

    print(f"Wrote outputs to: {outdir}")
    print(f"Temperature virtual origin y0 = {fitT['y0']:.6e} m, R2={fitT['r2']:.4f}")
    print(f"Velocity virtual origin    y0 = {fitU['y0']:.6e} m, R2={fitU['r2']:.4f}")
    print(f"Near-wire 1% thermal boundary-layer thickness angular mean = {thermal_bl_mean_m:.6e} m")
    print(f"Valid boundary-layer ray crossings = {delta_arr.size}/{args.bl_angles}")


if __name__ == "__main__":
    main()
