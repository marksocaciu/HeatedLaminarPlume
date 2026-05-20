#!/usr/bin/env python3
"""
Steady plume post-processing for legacy-FEniCS XDMF/HDF5 exports.

Reads nodal temperature, nodal velocity, and optionally cell-centred heat-flux files
written in the simple XDMF/HDF5 format used by the current plume project, then computes:

  * horizontal-plane total vertical energy flux [W/m] and boundary heat escape [W/m]
  * mass, volume, vertical-momentum and kinetic-energy fluxes at selected heights
  * centreline temperature/velocity decay and automatic/user-window virtual-origin fits
  * one scalar near-wire thermal boundary-layer thickness based on an angular average
    of radial 1-percent temperature-excess distances around the cylinder
  * approximate cumulative buoyancy and vertical-momentum-flux balance diagnostics
  * optional full rectangular-control-volume vertical momentum balance when pressure is supplied
  * CSV files and diagnostic plots, including combined profile plots across all requested heights

The script intentionally does not depend on FEniCS. It uses h5py + matplotlib.tri
linear interpolation so it can be run after the solver on saved fields.

Typical use for your nondimensional mesh coordinates:

python postprocess_steady_plume_v10.py \
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
  * --velocity-scale-factor is applied immediately after reading the velocity field, before all plots, fits, and flux integrals.
  * if q_heat_dim is supplied, it is used internally to form q_total = rho cp uy (T-T_inf) + q_y and to integrate heat escaping the domain boundaries.
  * if --pressure-xdmf is supplied, a full vertical momentum balance is computed over rectangular control volumes.
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


def make_scalar_interpolator(tri: mtri.Triangulation, values: np.ndarray):
    return mtri.LinearTriInterpolator(tri, np.asarray(values, dtype=float))


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
    Fit a(y) = C * (y - y0)^exponent.

    Signed exponent convention:
      * centreline temperature excess: exponent = -3/5
      * centreline vertical velocity:  exponent = +1/5

    Linearization:
        a^(1/exponent) = C^(1/exponent) * (y - y0) = A*y + B,
        y0 = -B/A.

    Requires positive centreline amplitude a.
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=float)
    mask = np.isfinite(y) & np.isfinite(a) & (a > 0.0)
    y = y[mask]
    a = a[mask]
    if y.size < min_points or exponent == 0.0:
        return {"C": np.nan, "y0": np.nan, "r2": np.nan, "npoints": int(y.size)}

    z = a ** (1.0 / exponent)
    if not np.all(np.isfinite(z)):
        return {"C": np.nan, "y0": np.nan, "r2": np.nan, "npoints": int(y.size)}

    A, B = np.polyfit(y, z, 1)
    zhat = A * y + B
    ss_res = float(np.sum((z - zhat) ** 2))
    ss_tot = float(np.sum((z - np.mean(z)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    y0 = -B / A if A != 0 else np.nan
    C = A ** exponent if A > 0 else np.nan
    return {"C": float(C), "y0": float(y0), "r2": float(r2), "npoints": int(y.size)}



def select_virtual_origin_window(
    y: np.ndarray,
    amp: np.ndarray,
    exponent: float,
    wire_y: float,
    y_min_above_wire: Optional[float],
    y_max_above_wire: Optional[float],
    min_points: int,
    min_span: float,
    require_monotone: bool = True,
    require_y0_below_wire: bool = True,
    field_label: str = "field",
) -> Dict[str, float]:
    """
    Conservative automatic fit-window selector for the linearized virtual-origin fit.

    It deliberately avoids short-window overfitting. Candidate windows must be
    positive, reasonably long, decaying, nearly linear after transformation, and
    have a virtual origin below the selected window. For this cylinder-in-cavity
    case, the default also requires the virtual origin to be below the wire centre.
    """
    y = np.asarray(y, dtype=float)
    amp = np.asarray(amp, dtype=float)
    h = y - wire_y
    mask = np.isfinite(y) & np.isfinite(amp) & (amp > 0.0)
    if y_min_above_wire is not None:
        mask &= h >= y_min_above_wire
    if y_max_above_wire is not None:
        mask &= h <= y_max_above_wire
    yy = y[mask]
    aa = amp[mask]
    hh = h[mask]

    if yy.size < min_points:
        return {
            "C": np.nan, "y0": np.nan, "r2": np.nan, "npoints": int(yy.size),
            "fit_y_min_m": np.nan, "fit_y_max_m": np.nan,
            "fit_height_min_m": np.nan, "fit_height_max_m": np.nan,
            "auto_score": np.nan,
            "fit_mode": f"auto_failed_too_few_points_for_{field_label}",
        }

    best = None
    n = yy.size
    for i in range(0, n - min_points + 1):
        for j in range(i + min_points, n + 1):
            ywin = yy[i:j]
            awin = aa[i:j]
            hwin = hh[i:j]
            span = ywin[-1] - ywin[0]
            if span < min_span:
                continue
            if np.any(~np.isfinite(awin)) or np.nanmin(awin) <= 0:
                continue

            if require_monotone:
                da = np.diff(awin)
                if da.size:
                    if exponent < 0.0:
                        monotone_fraction = np.mean(da <= 0.0)
                        net_change_ok = (awin[0] - awin[-1]) > 0.0
                    else:
                        monotone_fraction = np.mean(da >= 0.0)
                        net_change_ok = (awin[-1] - awin[0]) > 0.0
                else:
                    monotone_fraction = 0.0
                    net_change_ok = False
                if monotone_fraction < 0.80 or not net_change_ok:
                    continue

            z = awin ** (1.0 / exponent)
            if not np.all(np.isfinite(z)):
                continue
            A, B = np.polyfit(ywin, z, 1)
            if not np.isfinite(A) or A <= 0.0:
                continue
            y0 = -B / A
            if not np.isfinite(y0) or y0 >= ywin[0]:
                continue
            if require_y0_below_wire and y0 > wire_y:
                continue

            zhat = A * ywin + B
            ss_res = float(np.sum((z - zhat) ** 2))
            ss_tot = float(np.sum((z - np.mean(z)) ** 2))
            if ss_tot <= 0.0:
                continue
            r2 = 1.0 - ss_res / ss_tot
            if r2 < 0.90:
                continue

            C = A ** exponent
            span_bonus = min(0.05, 0.05 * span / max(min_span, 1e-30))
            # Mildly prefer longer, cleaner windows. Do not reward tiny local fits.
            score = r2 + span_bonus
            candidate = {
                "C": float(C),
                "y0": float(y0),
                "r2": float(r2),
                "npoints": int(ywin.size),
                "fit_y_min_m": float(ywin[0]),
                "fit_y_max_m": float(ywin[-1]),
                "fit_height_min_m": float(hwin[0]),
                "fit_height_max_m": float(hwin[-1]),
                "auto_score": float(score),
                "fit_mode": "auto_window_conservative",
            }
            if best is None or candidate["auto_score"] > best["auto_score"]:
                best = candidate

    if best is not None:
        return best

    return {
        "C": np.nan, "y0": np.nan, "r2": np.nan, "npoints": int(yy.size),
        "fit_y_min_m": np.nan, "fit_y_max_m": np.nan,
        "fit_height_min_m": np.nan, "fit_height_max_m": np.nan,
        "auto_score": np.nan,
        "fit_mode": f"auto_failed_no_physical_line_plume_window_for_{field_label}",
    }



def boundary_edges_with_cells(cells: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return boundary edges as (node_a, node_b, adjacent_cell_index)."""
    edge_map: Dict[Tuple[int, int], List[int]] = {}
    for ci, tri in enumerate(cells):
        a, b, c = map(int, tri)
        for e in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(e))
            edge_map.setdefault(key, []).append(ci)
    out: List[Tuple[int, int, int]] = []
    for (a, b), owners in edge_map.items():
        if len(owners) == 1:
            out.append((a, b, owners[0]))
    return out


def triangle_temperature_gradient(points: np.ndarray, cells: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Piecewise-constant gradient of a nodal P1 temperature field in each triangle.
    Returns array (n_cells, 2) with dT/dx, dT/dy in physical coordinates.
    """
    grads = np.full((cells.shape[0], 2), np.nan, dtype=float)
    for ci, tri in enumerate(cells):
        pts = points[tri]
        vals = T[tri]
        A = np.column_stack([np.ones(3), pts[:, 0], pts[:, 1]])
        try:
            coeff = np.linalg.solve(A, vals)
            grads[ci, 0] = coeff[1]
            grads[ci, 1] = coeff[2]
        except np.linalg.LinAlgError:
            pass
    return grads


def integrate_boundary_heat_escape_from_facets(
    points: np.ndarray,
    cells: np.ndarray,
    T: np.ndarray,
    k: float,
    Qdata: Optional[FieldData] = None,
) -> Tuple[List[Dict[str, float | str]], Dict[str, float]]:
    """
    Integrate outward heat flux over actual exterior mesh facets.

    If a cell-centred q_heat field is supplied, the adjacent boundary cell value is
    used directly. This avoids extrapolating a cell-centred field to the rectangular
    boundary, which was the reason v7 produced NaNs. If q_heat is not supplied,
    q=-k*grad(T) is reconstructed in each boundary cell from the P1 temperature.
    """
    edges = boundary_edges_with_cells(cells)
    centroids = cell_centres(points, cells)

    if Qdata is not None and Qdata.center == "Cell" and Qdata.values.ndim == 2 and Qdata.values.shape[0] == cells.shape[0]:
        q_cell = np.asarray(Qdata.values[:, :2], dtype=float)
        q_source = "cell_centered_q_heat_field"
    else:
        gradT = triangle_temperature_gradient(points, cells, np.asarray(T, dtype=float))
        q_cell = -float(k) * gradT
        q_source = "reconstructed_from_temperature_gradient"

    xmin, xmax = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
    ymin, ymax = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
    tol = 1e-8 * max(xmax - xmin, ymax - ymin, 1.0)

    sums = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0, "other": 0.0}
    counts = {key: 0 for key in sums}
    lengths = {key: 0.0 for key in sums}

    for a, b, ci in edges:
        p0 = points[a]
        p1 = points[b]
        mid = 0.5 * (p0 + p1)
        edge_vec = p1 - p0
        L = float(np.linalg.norm(edge_vec))
        if L <= 0.0:
            continue

        # Two candidate normals. Choose the one pointing away from the owning cell centroid.
        nvec = np.array([edge_vec[1], -edge_vec[0]], dtype=float)
        nvec /= np.linalg.norm(nvec)
        outward_hint = mid - centroids[ci]
        if np.dot(nvec, outward_hint) < 0.0:
            nvec *= -1.0

        q = q_cell[ci]
        if not np.all(np.isfinite(q)):
            continue

        flux = float(np.dot(q, nvec) * L)

        if abs(mid[1] - ymax) <= tol:
            name = "top"
        elif abs(mid[1] - ymin) <= tol:
            name = "bottom"
        elif abs(mid[0] - xmin) <= tol:
            name = "left"
        elif abs(mid[0] - xmax) <= tol:
            name = "right"
        else:
            name = "other"

        sums[name] += flux
        counts[name] += 1
        lengths[name] += L

    rows: List[Dict[str, float | str]] = []
    for name in ["top", "bottom", "left", "right", "other"]:
        rows.append({
            "boundary": name,
            "heat_escape_W_per_m": sums[name],
            "positive_means": "out_of_domain",
            "edge_count": counts[name],
            "integrated_length_m": lengths[name],
            "flux_source": q_source,
        })

    outer_total = float(sums["top"] + sums["bottom"] + sums["left"] + sums["right"])
    all_total = float(sum(sums.values()))
    source_inferred = float(-sums["other"])  # positive means heat enters fluid from the cylinder/inner boundary
    totals = {
        "Q_escape_total_W_per_m": outer_total,
        "Q_escape_outer_W_per_m": outer_total,
        "Q_escape_all_boundaries_W_per_m": all_total,
        "Q_source_inferred_from_inner_boundary_W_per_m": source_inferred,
        "boundary_edge_count": int(sum(counts.values())),
        "boundary_integrated_length_m": float(sum(lengths.values())),
        "flux_source": q_source,
    }
    return rows, totals




def integrate_control_volume_vertical_momentum(
    Ti,
    uxi,
    uyi,
    pi,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
    rho: float,
    mu: float,
    beta: float,
    g: float,
    T_inf: float,
    n_side: int = 801,
    n_bottom_top: int = 1201,
    n_area_y: int = 101,
) -> Dict[str, float]:
    """
    Integrated steady vertical momentum balance over a rectangular 2D control volume.

    Equation used:
      ∮ rho*u_y*(u·n) ds = ∮[-p*n_y + tau_yj*n_j] ds + ∬ rho*g*beta*(T-T_inf) dA.

    Residual is advective - pressure - viscous - buoyancy. All quantities are per unit
    out-of-plane depth. Positive vertical direction is upward.
    """
    if pi is None:
        return {}
    if y_top <= y_bottom or x_right <= x_left:
        return {}

    def line_terms(xv, yv, nx, ny):
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)
        Tline = finite_or_nan(Ti(xv, yv))
        ux = finite_or_nan(uxi(xv, yv))
        uy = finite_or_nan(uyi(xv, yv))
        pp = finite_or_nan(pi(xv, yv))
        dux_dx, dux_dy = uxi.gradient(xv, yv)
        duy_dx, duy_dy = uyi.gradient(xv, yv)
        dux_dx = finite_or_nan(dux_dx)
        dux_dy = finite_or_nan(dux_dy)
        duy_dx = finite_or_nan(duy_dx)
        duy_dy = finite_or_nan(duy_dy)
        un = ux * nx + uy * ny
        adv = rho * uy * un
        tau_yx = mu * (duy_dx + dux_dy)
        tau_yy = 2.0 * mu * duy_dy
        pressure = -pp * ny
        viscous = tau_yx * nx + tau_yy * ny
        return adv, pressure, viscous

    # Top and bottom horizontal faces.
    xb = np.linspace(x_left, x_right, n_bottom_top)
    xt = xb.copy()
    adv_t, pres_t, visc_t = line_terms(xt, np.full_like(xt, y_top), 0.0, 1.0)
    adv_b, pres_b, visc_b = line_terms(xb, np.full_like(xb, y_bottom), 0.0, -1.0)
    adv_top = robust_trapz(adv_t, xt)
    adv_bottom = robust_trapz(adv_b, xb)
    pres_top = robust_trapz(pres_t, xt)
    pres_bottom = robust_trapz(pres_b, xb)
    visc_top = robust_trapz(visc_t, xt)
    visc_bottom = robust_trapz(visc_b, xb)

    # Left and right vertical faces.
    ys = np.linspace(y_bottom, y_top, n_side)
    adv_l, pres_l, visc_l = line_terms(np.full_like(ys, x_left), ys, -1.0, 0.0)
    adv_r, pres_r, visc_r = line_terms(np.full_like(ys, x_right), ys, 1.0, 0.0)
    adv_left = robust_trapz(adv_l, ys)
    adv_right = robust_trapz(adv_r, ys)
    pres_left = robust_trapz(pres_l, ys)
    pres_right = robust_trapz(pres_r, ys)
    visc_left = robust_trapz(visc_l, ys)
    visc_right = robust_trapz(visc_r, ys)

    adv_total = adv_top + adv_bottom + adv_left + adv_right
    pressure_total = pres_top + pres_bottom + pres_left + pres_right
    viscous_total = visc_top + visc_bottom + visc_left + visc_right

    # Area buoyancy integral by repeated trapezoid integration.
    ya = np.linspace(y_bottom, y_top, n_area_y)
    bx = np.full_like(ya, np.nan, dtype=float)
    for i, yy_ in enumerate(ya):
        Tline = finite_or_nan(Ti(xb, np.full_like(xb, yy_)))
        theta = Tline - T_inf
        bx[i] = robust_trapz(rho * g * beta * theta, xb)
    buoyancy_total = robust_trapz(bx, ya)

    residual = adv_total - pressure_total - viscous_total - buoyancy_total
    rhs_total = pressure_total + viscous_total + buoyancy_total
    scale = max(abs(adv_total), abs(pressure_total), abs(viscous_total), abs(buoyancy_total), abs(rhs_total), 1e-300)
    return {
        "y_top_m": float(y_top),
        "height_top_above_wire_m": float("nan"),
        "y_bottom_m": float(y_bottom),
        "x_left_m": float(x_left),
        "x_right_m": float(x_right),
        "advective_vertical_momentum_flux_N_per_m": float(adv_total),
        "advective_top_N_per_m": float(adv_top),
        "advective_bottom_N_per_m": float(adv_bottom),
        "advective_left_N_per_m": float(adv_left),
        "advective_right_N_per_m": float(adv_right),
        "pressure_vertical_force_N_per_m": float(pressure_total),
        "pressure_top_N_per_m": float(pres_top),
        "pressure_bottom_N_per_m": float(pres_bottom),
        "pressure_left_N_per_m": float(pres_left),
        "pressure_right_N_per_m": float(pres_right),
        "viscous_vertical_force_N_per_m": float(viscous_total),
        "viscous_top_N_per_m": float(visc_top),
        "viscous_bottom_N_per_m": float(visc_bottom),
        "viscous_left_N_per_m": float(visc_left),
        "viscous_right_N_per_m": float(visc_right),
        "buoyancy_vertical_force_N_per_m": float(buoyancy_total),
        "rhs_pressure_plus_viscous_plus_buoyancy_N_per_m": float(rhs_total),
        "momentum_balance_residual_N_per_m": float(residual),
        "momentum_balance_residual_relative": float(residual / scale),
    }

def write_csv(path: Path, rows: List[Dict[str, float | str]]) -> None:
    if not rows:
        return
    # Preserve first-row order, then include any later keys.
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
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



def read_profile_overlay_csv(csv_path: Path, T_inf: float, default_label: str) -> List[Dict[str, float]]:
    """
    Read optional experimental/theory profile data for overlay plots.

    Accepted columns:
      required: x_m plus one of height_m, height_above_wire_m, or y_m
      temperature: DeltaT_K, DeltaT, theta_K, or T_K
      velocity: uy_m_per_s, uy, v_m_per_s, or v
      optional: label

    Rows may contain only temperature or only velocity. Non-numeric missing
    entries are converted to NaN.
    """
    def get_float(row: Dict[str, str], names: Sequence[str], default: float = np.nan) -> float:
        for name in names:
            if name in row and str(row[name]).strip() != "":
                try:
                    return float(row[name])
                except ValueError:
                    return default
        return default

    rows: List[Dict[str, float]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no CSV header.")
        for row in reader:
            x = get_float(row, ["x_m", "x", "x_coord_m"])
            h = get_float(row, ["height_m", "height_above_wire_m", "h_m", "z_m"])
            y = get_float(row, ["y_m", "y"])
            dT = get_float(row, ["DeltaT_K", "DeltaT", "theta_K", "temperature_excess_K"])
            T = get_float(row, ["T_K", "T", "temperature_K"])
            if not np.isfinite(dT) and np.isfinite(T):
                dT = T - T_inf
            uy = get_float(row, ["uy_m_per_s", "uy", "v_m_per_s", "v", "vertical_velocity_m_per_s"])
            label = str(row.get("label", "")).strip() or default_label
            if np.isfinite(x) and (np.isfinite(h) or np.isfinite(y)):
                rows.append({
                    "x_m": x,
                    "height_m": h,
                    "y_m": y,
                    "DeltaT_K": dT,
                    "uy_m_per_s": uy,
                    "label": label,
                })
    return rows


def group_overlay_rows_by_label_and_height(
    rows: List[Dict[str, float]],
    requested_height: float,
    height_tol: float,
) -> Dict[str, List[Dict[str, float]]]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for r in rows:
        h = r.get("height_m", np.nan)
        if not np.isfinite(h):
            continue
        if abs(h - requested_height) <= height_tol:
            grouped.setdefault(str(r.get("label", "overlay")), []).append(r)
    return grouped


def fit_loglog_powerlaw(h: np.ndarray, a: np.ndarray, hmin: float, hmax: float) -> Dict[str, float]:
    """
    Fit a = C h^n on a selected h-window in log-log space.
    This is separate from the virtual-origin fit and is mainly a diagnostic
    for the straight portion of the log-log centreline curve.
    """
    h = np.asarray(h, dtype=float)
    a = np.asarray(a, dtype=float)
    mask = np.isfinite(h) & np.isfinite(a) & (h > 0.0) & (a > 0.0)
    mask &= h >= hmin
    mask &= h <= hmax
    if np.count_nonzero(mask) < 4:
        return {"C": np.nan, "exponent": np.nan, "r2": np.nan, "npoints": int(np.count_nonzero(mask)),
                "fit_height_min_m": hmin, "fit_height_max_m": hmax}
    x = np.log(h[mask])
    y = np.log(a[mask])
    n, logC = np.polyfit(x, y, 1)
    yhat = n * x + logC
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"C": float(np.exp(logC)), "exponent": float(n), "r2": float(r2),
            "npoints": int(np.count_nonzero(mask)), "fit_height_min_m": float(hmin), "fit_height_max_m": float(hmax)}

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--temperature-xdmf", required=True)
    ap.add_argument("--velocity-xdmf", required=True)
    ap.add_argument("--pressure-xdmf", default=None, help="Optional nodal pressure XDMF. If supplied, compute full rectangular-control-volume vertical momentum balance with pressure and viscous tractions.")
    ap.add_argument("--pressure-scale", type=float, default=1.0, help="Multiplicative factor applied to the exported pressure field before momentum-balance diagnostics.")
    ap.add_argument("--heatflux-xdmf", default=None, help="Optional cell-centred q_heat_dim XDMF; assumed q=-k grad(T)")
    ap.add_argument("--heatflux-scale", type=float, default=None,
                    help="Multiplicative scale applied to q_heat field before integration. "
                         "Default: 1/lref when --coords-are-dimensionless, otherwise 1. "
                         "This corrects q fields computed as -k*grad(T) on a nondimensional mesh.")
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
    ap.add_argument("--velocity-scale-factor", type=float, default=1.0,
                    help="Multiplicative factor applied to the exported velocity field before all diagnostics. Use this when the saved dimensional velocity was reconstructed with Uref but should be reported with Uplume, or vice versa.")

    ap.add_argument("--planes", type=float, nargs="+", default=[0.01, 0.02, 0.04, 0.08], help="Physical heights above wire [m]")
    ap.add_argument("--fit-y-min", type=float, default=None, help="Minimum physical height above wire [m] used for virtual-origin fits. If omitted, an automatic window is selected.")
    ap.add_argument("--fit-y-max", type=float, default=None, help="Maximum physical height above wire [m] used for virtual-origin fits. If omitted, an automatic window is selected.")
    ap.add_argument("--auto-fit-min-points", type=int, default=30, help="Minimum number of centreline samples in automatic virtual-origin fit window")
    ap.add_argument("--auto-fit-min-span", type=float, default=None, help="Minimum physical height span [m] of automatic virtual-origin fit window. Default: max(0.025 m, 25*r)")
    ap.add_argument("--auto-fit-lower-cutoff", type=float, default=None, help="Minimum height above wire [m] for automatic fits. Default: max(12*r, 0.010 m)")
    ap.add_argument("--auto-fit-upper-cutoff", type=float, default=None,
                    help="Maximum height above wire [m] allowed for automatic virtual-origin fits. Default: stop at 90% of the distance from wire centre to the top boundary, to avoid fitting the cooled top-wall region.")
    ap.add_argument("--boundary-n", type=int, default=2001, help="Number of samples per rectangular boundary side for boundary heat-escape integration")
    ap.add_argument("--profile-half-width", type=float, default=None, help="Sample only |x|<=this physical half-width [m]. Default: full mesh width.")
    ap.add_argument("--comparison-profile-half-width", type=float, default=None,
                    help="Half-width [m] used only for profile-comparison plots. Default: profile-half-width if given, otherwise full sampled profile.")
    ap.add_argument("--comparison-height-tol", type=float, default=5e-5,
                    help="Height tolerance [m] for matching experimental/theory CSV rows to requested --planes.")
    ap.add_argument("--comparison-x-scale", type=float, default=1000.0,
                    help="Multiplier for x-axis in comparison plots. Default 1000 gives mm.")
    ap.add_argument("--experiment-profile-csv", action="append", default=[],
                    help="Optional CSV with experimental profile data. May be supplied multiple times.")
    ap.add_argument("--theory-profile-csv", action="append", default=[],
                    help="Optional CSV with boundary-layer/self-similar profile data. May be supplied multiple times.")
    ap.add_argument("--nx", type=int, default=1601, help="Number of x samples per horizontal profile")
    ap.add_argument("--ny-balance", type=int, default=300, help="Number of y levels for balance/fit curves")
    ap.add_argument("--momentum-cv-half-width", type=float, default=None, help="Half-width [m] of the rectangular momentum-balance control volume. Default: same as profile-half-width, or full mesh half-width.")
    ap.add_argument("--momentum-cv-y0", type=float, default=None, help="Lower height above wire [m] for cumulative control-volume momentum balance. Default: balance-y-min.")
    ap.add_argument("--balance-y-max", type=float, default=None,
                    help="Maximum height above wire [m] for continuous centreline and balance diagnostics. Default: 90% of available height from wire to top boundary.")
    ap.add_argument("--balance-y-min", type=float, default=None,
                    help="Minimum height above wire [m] for continuous centreline and balance diagnostics. Default: just above the wire/source region.")
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

    Pdata = None
    p_field = None
    if args.pressure_xdmf:
        Pdata = load_xdmf_field(args.pressure_xdmf, coordinate_scale)
        check_same_mesh(Tdata, Pdata)
        if Pdata.center != "Node":
            raise ValueError("Pressure field must be nodal for the control-volume momentum balance.")
        p_field = np.asarray(Pdata.values, dtype=float) * float(args.pressure_scale)
        if p_field.ndim != 1:
            raise ValueError("Pressure field must be scalar-valued.")

    Qdata = None
    if args.heatflux_xdmf:
        Qdata = load_xdmf_field(args.heatflux_xdmf, coordinate_scale)
        check_same_mesh(Tdata, Qdata)
        if Qdata.center != "Cell":
            raise ValueError("Heat-flux field is expected to be cell-centred.")
        if Qdata.values.ndim != 2 or Qdata.values.shape[1] != 2:
            raise ValueError("Heat-flux field must be a two-component vector field.")
        heatflux_scale = args.heatflux_scale
        if heatflux_scale is None:
            heatflux_scale = (1.0 / float(args.lref)) if args.coords_are_dimensionless else 1.0
        Qdata.values = np.asarray(Qdata.values, dtype=float) * float(heatflux_scale)

    x = Tdata.points_m[:, 0]
    y = Tdata.points_m[:, 1]
    T = np.asarray(Tdata.values, dtype=float)
    u = np.asarray(Udata.values, dtype=float) * float(args.velocity_scale_factor)
    tri = mtri.Triangulation(x, y, Tdata.cells)
    Ti, uxi, uyi = make_interpolators(tri, T, u)
    pi = make_scalar_interpolator(tri, p_field) if p_field is not None else None
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

        vertical_heat_transport_fraction = Qtot / args.q_input_per_length if args.q_input_per_length else np.nan

        plane_rows.append({
            "height_m": h,
            "y_m": yp,
            "T_center_K": Tc,
            "DeltaT_center_K": Tce,
            "uy_center_m_per_s": uyc,
            "Q_total_W_per_m": Qtot,
            "Q_conv_up_W_per_m": Qconv_up,
            "Q_conv_down_W_per_m": Qconv_down,
            "vertical_heat_transport_fraction": vertical_heat_transport_fraction,
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

    # Continuous centreline / balance curves.  By default this extends well above
    # the requested profile planes, because virtual-origin and momentum-balance
    # diagnostics need the developing/far-field region, not only the sampled cuts.
    default_balance_y_min_h = max(2.5 * wire_radius_m, 0.001)
    default_balance_y_max_h = 0.90 * max(ymax - wire_y_m, 0.0)
    balance_y_min_h = args.balance_y_min if args.balance_y_min is not None else default_balance_y_min_h
    balance_y_max_h = args.balance_y_max if args.balance_y_max is not None else default_balance_y_max_h
    # Always include at least the requested plane range when possible.
    if args.planes:
        balance_y_max_h = max(balance_y_max_h, max(args.planes) * 1.25)
    y_start = max(float(wire_y_m + balance_y_min_h), ymin)
    y_end = min(float(wire_y_m + balance_y_max_h), ymax)
    if y_end <= y_start:
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

    # Virtual origin fits. Classical laminar line-source similarity suggests
    # DeltaT_c ~ (y-y0)^(-3/5), uy_c ~ (y-y0)^(+1/5).
    if args.fit_y_min is not None or args.fit_y_max is not None:
        fit_mask = np.isfinite(yy) & np.isfinite(dT_c) & np.isfinite(uy_c)
        if args.fit_y_min is not None:
            fit_mask &= (yy - wire_y_m) >= args.fit_y_min
        if args.fit_y_max is not None:
            fit_mask &= (yy - wire_y_m) <= args.fit_y_max
        fitT = fit_virtual_origin_powerlaw(yy[fit_mask], dT_c[fit_mask], exponent=-(3.0/5.0))
        fitU = fit_virtual_origin_powerlaw(yy[fit_mask], uy_c[fit_mask], exponent=1.0/5.0)
        for fit in (fitT, fitU):
            fit.update({
                "fit_y_min_m": float(np.nanmin(yy[fit_mask])) if np.any(fit_mask) else np.nan,
                "fit_y_max_m": float(np.nanmax(yy[fit_mask])) if np.any(fit_mask) else np.nan,
                "fit_height_min_m": float(np.nanmin(yy[fit_mask] - wire_y_m)) if np.any(fit_mask) else np.nan,
                "fit_height_max_m": float(np.nanmax(yy[fit_mask] - wire_y_m)) if np.any(fit_mask) else np.nan,
                "auto_score": np.nan,
                "fit_mode": "user_window",
            })
    else:
        lower = args.auto_fit_lower_cutoff
        if lower is None:
            lower = max(12.0 * wire_radius_m, 0.010)

        min_span = args.auto_fit_min_span
        if min_span is None:
            min_span = max(0.025, 25.0 * wire_radius_m)

        upper = args.auto_fit_upper_cutoff
        if upper is None:
            upper = 0.90 * max(ymax - wire_y_m, 0.0)

        # Velocity theory has uy_c ~ (y-y0)^(+1/5), so the automatic fit should
        # use the increasing line-plume-like region before the upper-wall pressure
        # deceleration. Avoid the immediate source region with the same lower cutoff
        # as the temperature fit; if a clear maximum exists, cap the fit above it.
        vel_lower = lower
        vel_upper = upper
        if np.any(np.isfinite(uy_c)):
            imax_u = int(np.nanargmax(uy_c))
            h_umax = float(yy[imax_u] - wire_y_m)
            candidate_upper = h_umax - 3.0 * wire_radius_m
            if candidate_upper > vel_lower + min_span:
                vel_upper = min(vel_upper, candidate_upper)

        fitT = select_virtual_origin_window(
            yy, dT_c, -(3.0/5.0), wire_y_m,
            y_min_above_wire=lower,
            y_max_above_wire=upper,
            min_points=args.auto_fit_min_points,
            min_span=min_span,
            require_monotone=True,
            require_y0_below_wire=True,
            field_label="temperature",
        )
        fitU = select_virtual_origin_window(
            yy, uy_c, 1.0/5.0, wire_y_m,
            y_min_above_wire=vel_lower,
            y_max_above_wire=vel_upper,
            min_points=args.auto_fit_min_points,
            min_span=min_span,
            require_monotone=True,
            require_y0_below_wire=True,
            field_label="velocity",
        )
    fit_rows = [
        {"field": "temperature_centerline", "assumed_powerlaw_exponent": -(3.0/5.0), **fitT},
        {"field": "velocity_centerline", "assumed_powerlaw_exponent": 1.0/5.0, **fitU},
    ]
    write_csv(outdir / "virtual_origin_fits.csv", fit_rows)

    # Boundary heat escape: integrate outward heat flux over actual exterior mesh facets.
    boundary_rows, boundary_totals = integrate_boundary_heat_escape_from_facets(
        Tdata.points_m, Tdata.cells, T, args.k, Qdata=Qdata
    )
    if args.q_input_per_length:
        for r in boundary_rows:
            r["fraction_of_input"] = r["heat_escape_W_per_m"] / args.q_input_per_length
        boundary_totals["Q_escape_total_fraction_of_input"] = boundary_totals["Q_escape_total_W_per_m"] / args.q_input_per_length
        boundary_totals["Q_escape_minus_input_W_per_m"] = boundary_totals["Q_escape_total_W_per_m"] - args.q_input_per_length
        boundary_totals["Q_escape_minus_input_fraction"] = boundary_totals["Q_escape_minus_input_W_per_m"] / args.q_input_per_length
    write_csv(outdir / "boundary_heat_escape.csv", boundary_rows + [{
        "boundary": "total",
        "heat_escape_W_per_m": boundary_totals["Q_escape_total_W_per_m"],
        "positive_means": "out_of_domain",
        "fraction_of_input": boundary_totals.get("Q_escape_total_fraction_of_input", np.nan),
        "edge_count": boundary_totals.get("boundary_edge_count", np.nan),
        "integrated_length_m": boundary_totals.get("boundary_integrated_length_m", np.nan),
        "flux_source": boundary_totals.get("flux_source", ""),
        "outer_boundary_escape_W_per_m": boundary_totals.get("Q_escape_outer_W_per_m", np.nan),
        "all_boundary_net_W_per_m": boundary_totals.get("Q_escape_all_boundaries_W_per_m", np.nan),
        "inferred_source_input_from_inner_boundary_W_per_m": boundary_totals.get("Q_source_inferred_from_inner_boundary_W_per_m", np.nan),
    }, {
        "boundary": "inferred_source_from_inner_boundary",
        "heat_escape_W_per_m": boundary_totals.get("Q_source_inferred_from_inner_boundary_W_per_m", np.nan),
        "positive_means": "into_fluid_from_inner_boundary",
        "fraction_of_input": (boundary_totals.get("Q_source_inferred_from_inner_boundary_W_per_m", np.nan) / args.q_input_per_length) if args.q_input_per_length else np.nan,
        "edge_count": np.nan,
        "integrated_length_m": np.nan,
        "flux_source": boundary_totals.get("flux_source", ""),
        "outer_boundary_escape_W_per_m": boundary_totals.get("Q_escape_outer_W_per_m", np.nan),
        "all_boundary_net_W_per_m": boundary_totals.get("Q_escape_all_boundaries_W_per_m", np.nan),
        "inferred_source_input_from_inner_boundary_W_per_m": boundary_totals.get("Q_source_inferred_from_inner_boundary_W_per_m", np.nan),
    }])

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

    # Momentum/buoyancy balance proxy.
    # This is not a closed Navier-Stokes momentum balance because pressure, viscous
    # traction, and lateral advective fluxes are not included.  It is a diagnostic:
    # compare how much upward buoyancy force is generated by the temperature field
    # against the observed change of vertical momentum flux through horizontal cuts.
    momentum_rows = []
    y_arr = np.array([r["y_m"] for r in balance_rows], dtype=float)
    h_arr = np.array([r["height_above_wire_m"] for r in balance_rows], dtype=float)
    M_arr = np.array([r["vertical_momentum_flux_N_per_m"] for r in balance_rows], dtype=float)
    Bp_arr = np.array([r.get("buoyancy_force_density_integral_N_per_m2", np.nan) for r in balance_rows], dtype=float)
    CB_arr = np.array([r.get("cumulative_buoyancy_N_per_m", np.nan) for r in balance_rows], dtype=float)
    dMdy_arr = np.full_like(M_arr, np.nan, dtype=float)
    if np.count_nonzero(np.isfinite(M_arr) & np.isfinite(y_arr)) >= 3:
        good = np.isfinite(M_arr) & np.isfinite(y_arr)
        # Gradient on the valid contiguous samples; this is a smooth finite-difference
        # estimate of d/dy int rho uy^2 dx.
        idx = np.where(good)[0]
        dvals = np.gradient(M_arr[idx], y_arr[idx])
        dMdy_arr[idx] = dvals
    delta_M_arr = M_arr - M_arr[0] if len(M_arr) else M_arr
    residual_local = Bp_arr - dMdy_arr
    residual_cumulative = CB_arr - delta_M_arr
    for yi, hi, Mi, dMi, Bpi, CBi, dMi_c, rli, rci in zip(y_arr, h_arr, M_arr, dMdy_arr, Bp_arr, CB_arr, delta_M_arr, residual_local, residual_cumulative):
        momentum_rows.append({
            "y_m": yi,
            "height_above_wire_m": hi,
            "vertical_momentum_flux_N_per_m": Mi,
            "d_vertical_momentum_flux_dy_N_per_m2": dMi,
            "buoyancy_force_per_height_N_per_m2": Bpi,
            "cumulative_buoyancy_N_per_m": CBi,
            "delta_vertical_momentum_flux_N_per_m": dMi_c,
            "local_unresolved_terms_proxy_N_per_m2": rli,
            "cumulative_unresolved_terms_proxy_N_per_m": rci,
        })
    write_csv(outdir / "momentum_balance_proxy.csv", momentum_rows)

    # Optional full rectangular-control-volume vertical momentum balance with pressure and viscous tractions.
    full_momentum_rows = []
    if pi is not None and args.mu is not None and args.beta is not None:
        if args.momentum_cv_half_width is not None:
            cv_hw = float(args.momentum_cv_half_width)
            x_left_cv = max(xmin, -cv_hw)
            x_right_cv = min(xmax, cv_hw)
        elif args.profile_half_width is not None:
            cv_hw = float(args.profile_half_width)
            x_left_cv = max(xmin, -cv_hw)
            x_right_cv = min(xmax, cv_hw)
        else:
            x_left_cv = xmin
            x_right_cv = xmax

        if args.momentum_cv_y0 is not None:
            y0_cv = wire_y_m + float(args.momentum_cv_y0)
        else:
            y0_cv = float(yy[0]) if len(yy) else wire_top_y_m
        y0_cv = max(y0_cv, ymin + 1e-10 * max(ymax - ymin, 1.0))

        # Use only top positions safely above the lower face.
        y_tops = [float(v) for v in yy if np.isfinite(v) and v > y0_cv + 1e-6 * max(ymax-ymin, 1.0)]
        for ytop in y_tops:
            row = integrate_control_volume_vertical_momentum(
                Ti, uxi, uyi, pi,
                x_left_cv, x_right_cv, y0_cv, ytop,
                rho=float(args.rho), mu=float(args.mu), beta=float(args.beta), g=float(args.g), T_inf=float(args.T_inf),
                n_side=max(201, min(args.nx, 801)),
                n_bottom_top=max(401, min(args.nx, 1201)),
                n_area_y=max(31, min(args.ny_balance, 151)),
            )
            if row:
                row["height_top_above_wire_m"] = row["y_top_m"] - wire_y_m
                row["height_bottom_above_wire_m"] = y0_cv - wire_y_m
                full_momentum_rows.append(row)
        write_csv(outdir / "momentum_balance_full.csv", full_momentum_rows)

    # Plots.
    plane_h = np.array([r["height_m"] for r in plane_rows], dtype=float)
    energy_series = [("plume vertical transport through plane", np.array([r["Q_total_W_per_m"] for r in plane_rows]))]
    if args.q_input_per_length:
        energy_series.append(("supplied heat", np.full_like(plane_h, args.q_input_per_length, dtype=float)))
    if "boundary_totals" in locals():
        energy_series.append(("outer boundary heat escape", np.full_like(plane_h, boundary_totals["Q_escape_total_W_per_m"], dtype=float)))
        energy_series.append(("inferred heat entering at cylinder", np.full_like(plane_h, boundary_totals.get("Q_source_inferred_from_inner_boundary_W_per_m", np.nan), dtype=float)))
    plot_xy(outdir / "energy_flux_vs_height.png", plane_h, energy_series,
            "height above wire [m]", "heat rate per unit depth [W/m]",
            "Supplied heat, vertical plume transport, and boundary heat escape")

    # Compact bar chart of global energy budget.
    if "boundary_totals" in locals():
        labels = ["supplied", "boundary escape", "max plane plume"]
        vals = [args.q_input_per_length if args.q_input_per_length else np.nan,
                boundary_totals["Q_escape_total_W_per_m"],
                float(np.nanmax([r["Q_total_W_per_m"] for r in plane_rows])) if plane_rows else np.nan]
        plt.figure(figsize=(7.2, 4.8))
        plt.bar(labels, vals)
        plt.ylabel("heat rate per unit depth [W/m]")
        plt.title("Global/diagnostic heat budget")
        plt.grid(True, axis="y", alpha=0.35)
        plt.tight_layout()
        plt.savefig(outdir / "energy_budget_summary.png", dpi=180)
        plt.close()
    plot_xy(outdir / "mass_momentum_vs_height.png", plane_h,
            [("mass flux", np.array([r["mass_flux_kg_per_s_per_m"] for r in plane_rows])),
             ("vertical momentum flux", np.array([r["vertical_momentum_flux_N_per_m"] for r in plane_rows]))],
            "height above wire [m]", "integral", "Mass and vertical-momentum flux")

    if args.beta is not None and momentum_rows:
        mh = np.array([r["height_above_wire_m"] for r in momentum_rows], dtype=float)
        plot_xy(outdir / "momentum_balance_proxy_local.png", mh,
                [("buoyancy source per height", np.array([r["buoyancy_force_per_height_N_per_m2"] for r in momentum_rows], dtype=float)),
                 ("d(momentum flux)/dy", np.array([r["d_vertical_momentum_flux_dy_N_per_m2"] for r in momentum_rows], dtype=float)),
                 ("unresolved = buoyancy - dM/dy", np.array([r["local_unresolved_terms_proxy_N_per_m2"] for r in momentum_rows], dtype=float))],
                "height above wire [m]", "force per height per depth [N/m²]",
                "Local vertical momentum-balance proxy")
        plot_xy(outdir / "momentum_balance_proxy_cumulative.png", mh,
                [("cumulative buoyancy", np.array([r["cumulative_buoyancy_N_per_m"] for r in momentum_rows], dtype=float)),
                 ("change in vertical momentum flux", np.array([r["delta_vertical_momentum_flux_N_per_m"] for r in momentum_rows], dtype=float)),
                 ("unresolved cumulative terms", np.array([r["cumulative_unresolved_terms_proxy_N_per_m"] for r in momentum_rows], dtype=float))],
                "height above wire [m]", "force per unit depth [N/m]",
                "Cumulative vertical momentum-balance proxy")

    if full_momentum_rows:
        mhf = np.array([r["height_top_above_wire_m"] for r in full_momentum_rows], dtype=float)
        plot_xy(outdir / "momentum_balance_full_terms.png", mhf,
                [("advective flux", np.array([r["advective_vertical_momentum_flux_N_per_m"] for r in full_momentum_rows], dtype=float)),
                 ("pressure force", np.array([r["pressure_vertical_force_N_per_m"] for r in full_momentum_rows], dtype=float)),
                 ("viscous force", np.array([r["viscous_vertical_force_N_per_m"] for r in full_momentum_rows], dtype=float)),
                 ("buoyancy force", np.array([r["buoyancy_vertical_force_N_per_m"] for r in full_momentum_rows], dtype=float)),
                 ("RHS total", np.array([r["rhs_pressure_plus_viscous_plus_buoyancy_N_per_m"] for r in full_momentum_rows], dtype=float))],
                "top height above wire [m]", "vertical force / momentum flux [N/m]",
                "Full vertical momentum-balance terms")
        plot_xy(outdir / "momentum_balance_full_residual.png", mhf,
                [("residual", np.array([r["momentum_balance_residual_N_per_m"] for r in full_momentum_rows], dtype=float)),
                 ("relative residual", np.array([r["momentum_balance_residual_relative"] for r in full_momentum_rows], dtype=float))],
                "top height above wire [m]", "residual [N/m] or relative residual [-]",
                "Full vertical momentum-balance residual")
        # Breakdown of side terms helps diagnose where confinement/entrainment enters.
        plot_xy(outdir / "momentum_balance_full_boundary_breakdown.png", mhf,
                [("advective top", np.array([r["advective_top_N_per_m"] for r in full_momentum_rows], dtype=float)),
                 ("advective bottom", np.array([r["advective_bottom_N_per_m"] for r in full_momentum_rows], dtype=float)),
                 ("advective sides", np.array([r["advective_left_N_per_m"] + r["advective_right_N_per_m"] for r in full_momentum_rows], dtype=float)),
                 ("pressure top+bottom", np.array([r["pressure_top_N_per_m"] + r["pressure_bottom_N_per_m"] for r in full_momentum_rows], dtype=float)),
                 ("pressure sides", np.array([r["pressure_left_N_per_m"] + r["pressure_right_N_per_m"] for r in full_momentum_rows], dtype=float)),
                 ("viscous all", np.array([r["viscous_vertical_force_N_per_m"] for r in full_momentum_rows], dtype=float))],
                "top height above wire [m]", "term [N/m]",
                "Full vertical momentum-balance boundary-term breakdown")

    # Centreline plots with fitted virtual-source curves.
    h_center = yy - wire_y_m
    dT_fit = np.full_like(yy, np.nan, dtype=float)
    uy_fit = np.full_like(yy, np.nan, dtype=float)
    if np.isfinite(fitT["C"]) and np.isfinite(fitT["y0"]):
        m = yy > fitT["y0"]
        dT_fit[m] = fitT["C"] * (yy[m] - fitT["y0"]) ** (-(3.0/5.0))
    if np.isfinite(fitU["C"]) and np.isfinite(fitU["y0"]):
        m = yy > fitU["y0"]
        uy_fit[m] = fitU["C"] * (yy[m] - fitU["y0"]) ** (+(1.0/5.0))

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
        z[mask] = amp_[mask] ** (1.0 / exponent)
        plt.plot(yy_[mask] - wire_y_m, z[mask], label="transformed centreline")
        if np.isfinite(fit.get("fit_y_min_m", np.nan)) and np.isfinite(fit.get("fit_y_max_m", np.nan)):
            plt.axvspan(fit["fit_y_min_m"] - wire_y_m, fit["fit_y_max_m"] - wire_y_m, alpha=0.15, label="fit window")
        if np.isfinite(fit["y0"]):
            A = fit["C"] ** (1.0 / exponent) if np.isfinite(fit["C"]) and fit["C"] > 0 else np.nan
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

    plot_linearized_virtual_origin(outdir / "virtual_origin_temperature_linearized.png", yy, dT_c, -(3.0/5.0), fitT,
                                   r"$\Delta T_c^{-5/3}$", "Linearized temperature virtual-origin fit")
    plot_linearized_virtual_origin(outdir / "virtual_origin_velocity_linearized.png", yy, uy_c, 1.0/5.0, fitU,
                                   r"$u_{y,c}^{5}$", "Linearized velocity virtual-origin fit")

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
            if quantity_key == "ux_m_per_s":
                qp = np.abs(qp) 
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

    # Optional thesis-style overlays: numerical solution as solid lines,
    # experimental data as symbols, boundary-layer/self-similar data as dashed lines.
    exp_overlay_rows: List[Dict[str, float]] = []
    for csv_name in args.experiment_profile_csv:
        csv_path = Path(csv_name)
        exp_overlay_rows.extend(read_profile_overlay_csv(csv_path, args.T_inf, default_label=csv_path.stem))

    theory_overlay_rows: List[Dict[str, float]] = []
    for csv_name in args.theory_profile_csv:
        csv_path = Path(csv_name)
        theory_overlay_rows.extend(read_profile_overlay_csv(csv_path, args.T_inf, default_label=csv_path.stem))

    comp_half_width = args.comparison_profile_half_width
    if comp_half_width is None:
        comp_half_width = args.profile_half_width

    def plot_profile_comparison(quantity_key: str, ylabel: str, filename_prefix: str, title_prefix: str) -> None:
        for h in args.planes:
            rows = [r for r in profile_rows if abs(r["height_m"] - h) < 1e-15]
            if not rows:
                continue
            xp = np.array([r["x_m"] for r in rows], dtype=float)
            qp = np.array([r[quantity_key] for r in rows], dtype=float)
            mask = np.isfinite(xp) & np.isfinite(qp)
            if comp_half_width is not None:
                mask &= np.abs(xp) <= comp_half_width
            order = np.argsort(xp[mask])
            plt.figure(figsize=(7.2, 4.8))
            plt.plot(args.comparison_x_scale * xp[mask][order], qp[mask][order], label="numerical", linewidth=2.0)

            exp_grouped = group_overlay_rows_by_label_and_height(exp_overlay_rows, h, args.comparison_height_tol)
            for label, group in exp_grouped.items():
                xe = np.array([r["x_m"] for r in group], dtype=float)
                qe = np.array([r[quantity_key] for r in group], dtype=float)
                m = np.isfinite(xe) & np.isfinite(qe)
                if comp_half_width is not None:
                    m &= np.abs(xe) <= comp_half_width
                if np.any(m):
                    o = np.argsort(xe[m])
                    plt.plot(args.comparison_x_scale * xe[m][o], qe[m][o], linestyle="None", marker="o", label=f"{label} exp.")

            th_grouped = group_overlay_rows_by_label_and_height(theory_overlay_rows, h, args.comparison_height_tol)
            for label, group in th_grouped.items():
                xt = np.array([r["x_m"] for r in group], dtype=float)
                qt = np.array([r[quantity_key] for r in group], dtype=float)
                m = np.isfinite(xt) & np.isfinite(qt)
                if comp_half_width is not None:
                    m &= np.abs(xt) <= comp_half_width
                if np.any(m):
                    o = np.argsort(xt[m])
                    plt.plot(args.comparison_x_scale * xt[m][o], qt[m][o], linestyle="--", label=f"{label} BL")

            xunit = "mm" if abs(args.comparison_x_scale - 1000.0) < 1e-12 else f"{args.comparison_x_scale:g} x m"
            plt.xlabel(f"x [{xunit}]")
            plt.ylabel(ylabel)
            plt.title(f"{title_prefix}, h={h:g} m")
            plt.grid(True, alpha=0.35)
            plt.legend()
            plt.tight_layout()
            safe_h = str(f"{h:.6g}").replace(".", "p").replace("-", "m")
            plt.savefig(outdir / f"{filename_prefix}_h_{safe_h}m_comparison.png", dpi=220)
            plt.close()

    plot_profile_comparison("DeltaT_K", r"$T-T_\infty$ [K]", "profile_temperature", "Temperature profile comparison")
    plot_profile_comparison("uy_m_per_s", r"$u_y$ [m/s]", "profile_uy", "Vertical-velocity profile comparison")

    # Log-log centreline power-law diagnostics requested for the thesis.
    h_center = yy - wire_y_m
    temp_power = fit_loglog_powerlaw(
        h_center, dT_c,
        fitT["fit_height_min_m"] if np.isfinite(fitT.get("fit_height_min_m", np.nan)) else np.nanmin(h_center[h_center > 0]),
        fitT["fit_height_max_m"] if np.isfinite(fitT.get("fit_height_max_m", np.nan)) else np.nanmax(h_center),
    )
    vel_power = fit_loglog_powerlaw(
        h_center, uy_c,
        fitU["fit_height_min_m"] if np.isfinite(fitU.get("fit_height_min_m", np.nan)) else np.nanmin(h_center[h_center > 0]),
        fitU["fit_height_max_m"] if np.isfinite(fitU.get("fit_height_max_m", np.nan)) else np.nanmax(h_center),
    )
    write_csv(outdir / "centerline_loglog_powerlaw_fits.csv", [
        {"field": "temperature_centerline", **temp_power},
        {"field": "velocity_centerline", **vel_power},
    ])

    def plot_centerline_loglog(path: Path, amp: np.ndarray, fit: Dict[str, float], ylabel: str, title: str) -> None:
        plt.figure(figsize=(7.2, 4.8))
        mask = np.isfinite(h_center) & np.isfinite(amp) & (h_center > 0.0) & (amp > 0.0)
        plt.loglog(h_center[mask], amp[mask], label="numerical centreline")
        if np.isfinite(fit.get("C", np.nan)) and np.isfinite(fit.get("exponent", np.nan)):
            hfit = h_center[mask]
            w = (hfit >= fit["fit_height_min_m"]) & (hfit <= fit["fit_height_max_m"])
            if np.any(w):
                plt.loglog(hfit[w], fit["C"] * hfit[w] ** fit["exponent"],
                           linestyle="--", label=f"fit: exponent={fit['exponent']:.3f}, R2={fit['r2']:.4f}")
                plt.axvspan(fit["fit_height_min_m"], fit["fit_height_max_m"], alpha=0.12, label="fit window")
        plt.xlabel("height above wire centre [m]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, which="both", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()

    plot_centerline_loglog(outdir / "centerline_temperature_loglog_powerlaw.png", dT_c, temp_power,
                           r"$\Delta T_c$ [K]", "Centreline temperature decay on log-log axes")
    plot_centerline_loglog(outdir / "centerline_velocity_loglog_powerlaw.png", uy_c, vel_power,
                           r"$u_{y,c}$ [m/s]", "Centreline vertical velocity on log-log axes")

    summary_path = outdir / "README_summary.txt"
    with summary_path.open("w") as f:
        f.write("Steady plume post-processing summary\n")
        f.write("====================================\n\n")
        f.write(f"Temperature file: {args.temperature_xdmf}\n")
        f.write(f"Velocity file:    {args.velocity_xdmf}\n")
        f.write(f"Heat-flux file:   {args.heatflux_xdmf}\n")
        f.write(f"Pressure file:    {args.pressure_xdmf}\n")
        if args.pressure_xdmf:
            f.write(f"Applied pressure scale: {args.pressure_scale:.16e}\n")
        if Qdata is not None:
            f.write(f"Applied heat-flux field scale: {heatflux_scale:.16e}\n")
        f.write(f"Coordinate scale to metres: {coordinate_scale:.16e}\n")
        f.write(f"Wire/source y-coordinate inferred as y_min + H/10 + 11*r: {wire_y_m:.16e} m\n")
        f.write(f"Wire radius r = lref: {wire_radius_m:.16e} m\n")
        f.write(f"Velocity scale factor applied to exported velocity field: {args.velocity_scale_factor:.16e}\n")
        f.write(f"Near-wire 1% thermal boundary-layer thickness, angular mean: {thermal_bl_mean_m:.8e} m ({thermal_bl_mean_m / wire_radius_m if np.isfinite(thermal_bl_mean_m) else np.nan:.6g} r)\n")
        f.write(f"Near-wire 1% thickness angular std/min/median/max: {thermal_bl_std_m:.8e}, {thermal_bl_min_m:.8e}, {thermal_bl_median_m:.8e}, {thermal_bl_max_m:.8e} m\n")
        f.write(f"Valid boundary-layer ray crossings: {delta_arr.size}/{args.bl_angles}\n")
        f.write(f"Physical mesh bounds: x=[{xmin:.6e}, {xmax:.6e}], y=[{ymin:.6e}, {ymax:.6e}] m\n")
        f.write(f"T_inf: {args.T_inf:.12g} K\n")
        if args.q_input_per_length is not None:
            f.write(f"Input heat per length: {args.q_input_per_length:.12g} W/m\n")
        f.write("\nVirtual-origin fits use line-plume exponents:\n")
        f.write("  DeltaT_c ~ C_T * (y - y0_T)^(-3/5)\n")
        f.write("  uy_c     ~ C_U * (y - y0_U)^(+1/5)\n")
        f.write(f"Temperature virtual origin y0 = {fitT['y0']:.8e} m, R2={fitT['r2']:.6f}, n={fitT['npoints']}\n")
        f.write(f"Velocity virtual origin    y0 = {fitU['y0']:.8e} m, R2={fitU['r2']:.6f}, n={fitU['npoints']}\n")
        f.write("\nBoundary heat escape is integrated over exterior mesh facets; no cell-centred boundary extrapolation is used.\n\nMain outputs:\n")
        f.write("  plane_integrals.csv\n")
        f.write("  plane_profiles.csv\n")
        f.write("  centerline.csv\n")
        f.write("  virtual_origin_fits.csv\n")
        f.write("  near_wire_boundary_layer.csv\n")
        f.write("  near_wire_boundary_layer_by_angle.csv\n")
        f.write("  balance_curves.csv\n")
        f.write("  momentum_balance_proxy.csv\n")
        if full_momentum_rows:
            f.write("  momentum_balance_full.csv\n")
            f.write("  momentum_balance_full_terms.png\n")
            f.write("  momentum_balance_full_residual.png\n")
            f.write("  momentum_balance_full_boundary_breakdown.png\n")
        f.write("  *.png diagnostic plots\n")

    print(f"Wrote outputs to: {outdir}")
    print(f"Temperature virtual origin y0 = {fitT['y0']:.6e} m, R2={fitT['r2']:.4f}")
    print(f"Velocity virtual origin    y0 = {fitU['y0']:.6e} m, R2={fitU['r2']:.4f}")
    print(f"Near-wire 1% thermal boundary-layer thickness angular mean = {thermal_bl_mean_m:.6e} m")
    print(f"Valid boundary-layer ray crossings = {delta_arr.size}/{args.bl_angles}")


if __name__ == "__main__":
    main()
