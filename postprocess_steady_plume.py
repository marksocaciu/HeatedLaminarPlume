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
  * black-body/gray-body wire radiation estimate and fraction of supplied heat
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


def configure_plot_style(font_size: float = 12.0) -> None:
    matplotlib.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def thesis_figsize(width_in: float, height_in: Optional[float] = None) -> Tuple[float, float]:
    width = float(width_in)
    if height_in is None:
        height = 0.72 * width
    else:
        height = float(height_in)
    return (width, height)


def maybe_set_title(title: str, show_titles: bool) -> None:
    if show_titles and title:
        plt.title(title)


def maybe_set_ax_title(ax, title: str, show_titles: bool) -> None:
    if show_titles and title:
        ax.set_title(title)


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





def sample_energy_flux_vector(Ti, uxi, uyi, xpts: np.ndarray, ypts: np.ndarray, rho: float, cp: float, T_inf: float, k: float, qx_i=None, qy_i=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return total, convective, conductive heat-flux vectors sampled at arbitrary points.

    F_total = rho*cp*(T-T_inf)*u + q_cond, where q_cond = -k*grad(T) unless
    a supplied cell-centred heat-flux interpolator is available. Units are W/m^2;
    line integration gives W/m per unit out-of-plane depth.
    """
    xpts = np.asarray(xpts, dtype=float)
    ypts = np.asarray(ypts, dtype=float)
    Tloc = finite_or_nan(Ti(xpts, ypts))
    ux = finite_or_nan(uxi(xpts, ypts))
    uy = finite_or_nan(uyi(xpts, ypts))
    theta = Tloc - float(T_inf)
    qconv_x = float(rho) * float(cp) * ux * theta
    qconv_y = float(rho) * float(cp) * uy * theta

    if qx_i is not None and qy_i is not None:
        qcond_x = finite_or_nan(qx_i(xpts, ypts))
        qcond_y = finite_or_nan(qy_i(xpts, ypts))
    else:
        dTdx, dTdy = Ti.gradient(xpts, ypts)
        qcond_x = -float(k) * finite_or_nan(dTdx)
        qcond_y = -float(k) * finite_or_nan(dTdy)

    qtot_x = qconv_x + qcond_x
    qtot_y = qconv_y + qcond_y
    return qtot_x, qtot_y, qconv_x, qconv_y, qcond_x, qcond_y


def eta_halfwidth_at_y(ypts: np.ndarray, eta_abs: float, eta_origin_y: float, q_input_per_length: float, rho: float, cp: float, mu: float, beta: float, g: float) -> np.ndarray:
    """Physical half-width x(y) corresponding to |eta| = eta_abs for the line-plume scaling."""
    ypts = np.asarray(ypts, dtype=float)
    nu = float(mu) / float(rho)
    theta_line = float(q_input_per_length) / (float(rho) * float(cp) * nu) if nu > 0 else np.nan
    h = ypts - float(eta_origin_y)
    Gr_h = np.where(h > 0.0, float(g) * float(beta) * theta_line * h**3 / nu**2, np.nan)
    return np.where(np.isfinite(Gr_h) & (Gr_h > 0.0) & (h > 0.0), float(eta_abs) * h / (Gr_h**0.2), np.nan)


def integrate_energy_control_volume(
    Ti, uxi, uyi,
    x_left: float, x_right: float, y_bottom: float, y_top: float,
    rho: float, cp: float, T_inf: float, k: float,
    qx_i=None, qy_i=None,
    n_x: int = 1201, n_y: int = 801,
    x_half_fun=None,
) -> Tuple[List[Dict[str, float | str]], Dict[str, float | str]]:
    """Integrate convective and conductive heat transport over a rectangular or eta-width CV.

    Positive values mean outward through the named control-surface boundary. For an
    eta-width CV, x_half_fun(y) supplies the curved side half-width; top and bottom
    use their local half-widths.
    """
    if y_top <= y_bottom:
        raise ValueError("Control-volume y_top must be larger than y_bottom.")

    rows: List[Dict[str, float | str]] = []

    def add_row(boundary: str, qtot: float, qconv: float, qcond: float, extra: Dict[str, float | str]):
        rows.append({
            "boundary": boundary,
            "Q_total_out_W_per_m": qtot,
            "Q_convection_out_W_per_m": qconv,
            "Q_conduction_out_W_per_m": qcond,
            "positive_means": "out_of_control_volume",
            **extra,
        })

    if x_half_fun is None:
        # Rectangular control volume with constant x_left/x_right.
        xs = np.linspace(float(x_left), float(x_right), int(n_x))
        for boundary, yy, ny in (("bottom", y_bottom, -1.0), ("top", y_top, 1.0)):
            xline = xs
            yline = np.full_like(xline, yy)
            qtx, qty, qcx, qcy, qkx, qky = sample_energy_flux_vector(Ti, uxi, uyi, xline, yline, rho, cp, T_inf, k, qx_i, qy_i)
            add_row(boundary, robust_trapz(qty * ny, xline), robust_trapz(qcy * ny, xline), robust_trapz(qky * ny, xline), {"x_left_m": x_left, "x_right_m": x_right, "y_m": yy})

        ys = np.linspace(float(y_bottom), float(y_top), int(n_y))
        for boundary, xx, nx in (("left", x_left, -1.0), ("right", x_right, 1.0)):
            xline = np.full_like(ys, xx)
            yline = ys
            qtx, qty, qcx, qcy, qkx, qky = sample_energy_flux_vector(Ti, uxi, uyi, xline, yline, rho, cp, T_inf, k, qx_i, qy_i)
            add_row(boundary, robust_trapz(qtx * nx, ys), robust_trapz(qcx * nx, ys), robust_trapz(qkx * nx, ys), {"x_m": xx, "y_bottom_m": y_bottom, "y_top_m": y_top})
    else:
        # Curvilinear eta control volume, x = +/- xh(y).
        ys = np.linspace(float(y_bottom), float(y_top), int(n_y))
        xh = np.asarray(x_half_fun(ys), dtype=float)
        valid = np.isfinite(xh) & (xh > 0.0)
        if np.count_nonzero(valid) < 4:
            raise ValueError("Eta-width control volume is not valid over the selected y-range.")
        # Restrict to valid segment if the very bottom is invalid.
        ys = ys[valid]
        xh = xh[valid]
        dxh_dy = np.gradient(xh, ys)

        for boundary, yy, ny in (("bottom", ys[0], -1.0), ("top", ys[-1], 1.0)):
            half = float(xh[0] if boundary == "bottom" else xh[-1])
            xline = np.linspace(-half, half, int(n_x))
            yline = np.full_like(xline, yy)
            qtx, qty, qcx, qcy, qkx, qky = sample_energy_flux_vector(Ti, uxi, uyi, xline, yline, rho, cp, T_inf, k, qx_i, qy_i)
            add_row(boundary, robust_trapz(qty * ny, xline), robust_trapz(qcy * ny, xline), robust_trapz(qky * ny, xline), {"x_left_m": -half, "x_right_m": half, "y_m": yy})

        for side, sign in (("left", -1.0), ("right", 1.0)):
            xline = sign * xh
            yline = ys
            # outward normal times ds: right=(1,-xh'), left=(-1,-xh')
            nx = sign
            ny_ds = -dxh_dy
            qtx, qty, qcx, qcy, qkx, qky = sample_energy_flux_vector(Ti, uxi, uyi, xline, yline, rho, cp, T_inf, k, qx_i, qy_i)
            add_row(side, robust_trapz(qtx * nx + qty * ny_ds, ys), robust_trapz(qcx * nx + qcy * ny_ds, ys), robust_trapz(qkx * nx + qky * ny_ds, ys), {"y_bottom_m": float(ys[0]), "y_top_m": float(ys[-1]), "x_half_min_m": float(np.nanmin(xh)), "x_half_max_m": float(np.nanmax(xh))})

    total = {
        "boundary": "total",
        "Q_total_out_W_per_m": float(sum(float(r["Q_total_out_W_per_m"]) for r in rows)),
        "Q_convection_out_W_per_m": float(sum(float(r["Q_convection_out_W_per_m"]) for r in rows)),
        "Q_conduction_out_W_per_m": float(sum(float(r["Q_conduction_out_W_per_m"]) for r in rows)),
        "positive_means": "out_of_control_volume",
    }
    return rows, total



def momentum_cv_geometry_from_energy_settings(
    args,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    wire_y_m: float,
    wire_radius_m: float,
    thermal_bl_mean_m: float,
    rho: float,
    cp: float,
    mu: float,
    beta: float,
    g: float,
) -> Dict[str, object]:
    """Build the thesis-facing selected momentum-CV geometry from the energy-CV settings.

    The returned geometry uses the same convention as the eta-width energy CV:
    x = +/- x_half(y).  In the default eta mode, x_half(y) follows the selected
    eta half-width only after the eta width has grown to the minimum half-width;
    below that height it is frozen at --energy-cv-min-half-width-m.  This is the
    requested finite-cylinder near-source CV shape.
    """
    if args.q_input_per_length is None:
        raise ValueError("Selected momentum CV requires --q-input-per-length for eta-width geometry.")
    if mu is None or beta is None:
        raise ValueError("Selected momentum CV requires --mu and --beta for eta-width geometry.")

    if getattr(args, "energy_cv_y_bottom", None) is not None:
        y_bottom = float(wire_y_m) + float(args.energy_cv_y_bottom)
        y_bottom_definition = "user_relative_to_wire_center"
    else:
        bl = float(thermal_bl_mean_m) if np.isfinite(thermal_bl_mean_m) else 0.0
        y_bottom = float(wire_y_m) - float(wire_radius_m) - bl
        y_bottom_definition = "lower_wire_surface_minus_angular_mean_thermal_bl"

    y_top = float(wire_y_m) + float(args.energy_cv_y_top)

    eps = 1e-10 * max(float(ymax) - float(ymin), float(xmax) - float(xmin), 1.0)
    y_bottom_requested = float(y_bottom)
    y_top_requested = float(y_top)
    y_bottom = max(float(y_bottom), float(ymin) + eps)
    y_top = min(float(y_top), float(ymax) - eps)
    if y_top <= y_bottom:
        raise ValueError(f"Invalid selected momentum CV: y_top={y_top:g} <= y_bottom={y_bottom:g} after clipping to mesh bounds.")

    mode_requested = str(getattr(args, "energy_cv_width_mode", "eta"))
    mode_effective = mode_requested
    qin = float(args.q_input_per_length)
    nu = float(mu) / float(rho) if float(rho) != 0.0 else np.nan
    theta_line_source = qin / (float(rho) * float(cp) * nu) if np.isfinite(nu) and nu > 0.0 else np.nan

    eta_origin_mode = str(getattr(args, "eta_origin", "wire"))
    # At this stage the virtual-origin fits are not passed into this geometry function;
    # keep the same safe default used by the energy CV unless an explicit origin height is supplied.
    if getattr(args, "eta_origin_height", None) is not None:
        eta_origin_y_m = float(wire_y_m) + float(args.eta_origin_height)
        eta_origin_mode = "explicit_height_relative_to_wire"
    else:
        eta_origin_y_m = float(wire_y_m)

    eta_enabled = (
        np.isfinite(theta_line_source) and theta_line_source > 0.0
        and np.isfinite(nu) and nu > 0.0
        and np.isfinite(float(beta)) and float(beta) > 0.0
        and np.isfinite(float(g)) and float(g) > 0.0
    )
    if mode_effective == "eta" and not eta_enabled:
        raise ValueError("--energy-cv-width-mode eta requires positive --q-input-per-length, --mu, --beta, --rho, --cp, and --g.")
    if mode_effective == "auto" and not eta_enabled:
        mode_effective = "fixed"

    min_hw = float(getattr(args, "energy_cv_min_half_width_m", 1.5913e-2))
    fixed_hw = float(getattr(args, "energy_cv_fixed_half_width", min_hw))
    eta_hw = float(getattr(args, "energy_cv_eta_half_width", 9.0))
    max_hw = 0.999 * max(abs(float(xmin)), abs(float(xmax)))

    def raw_eta_half_width(yvals: np.ndarray) -> np.ndarray:
        return eta_halfwidth_at_y(
            np.asarray(yvals, dtype=float),
            eta_abs=eta_hw,
            eta_origin_y=eta_origin_y_m,
            q_input_per_length=qin,
            rho=float(rho), cp=float(cp), mu=float(mu), beta=float(beta), g=float(g),
        )

    def x_half_fun(yvals: np.ndarray) -> np.ndarray:
        yarr = np.asarray(yvals, dtype=float)
        if mode_effective == "fixed":
            hw = np.full_like(yarr, fixed_hw, dtype=float)
        else:
            xeta = raw_eta_half_width(yarr)
            # Freeze to the minimum half-width while the eta plume is thinner/undefined.
            hw = np.where(np.isfinite(xeta) & (xeta >= min_hw), xeta, min_hw)
        return np.maximum(0.0, np.minimum(hw, max_hw))

    yprobe = np.linspace(y_bottom, y_top, max(101, int(getattr(args, "energy_cv_n_boundary", 1201))))
    xeta_probe = raw_eta_half_width(yprobe) if mode_effective != "fixed" else np.full_like(yprobe, np.nan)
    width_probe = x_half_fun(yprobe)
    transition_y = np.nan
    finite_xeta = np.isfinite(xeta_probe)
    ok = finite_xeta & (xeta_probe >= min_hw)
    if np.any(ok):
        transition_y = float(yprobe[np.where(ok)[0][0]])

    return {
        "kind": "eta_width_selected_cv",
        "x_half_fun": x_half_fun,
        "y_bottom_m": float(y_bottom),
        "y_top_m": float(y_top),
        "y_bottom_requested_m": float(y_bottom_requested),
        "y_top_requested_m": float(y_top_requested),
        "y_bottom_definition": y_bottom_definition,
        "wire_center_y_m": float(wire_y_m),
        "wire_radius_m": float(wire_radius_m),
        "thermal_bl_mean_m": float(thermal_bl_mean_m) if np.isfinite(thermal_bl_mean_m) else np.nan,
        "width_mode_requested": mode_requested,
        "width_mode_effective": mode_effective,
        "eta_half_width": float(eta_hw),
        "minimum_half_width_m": float(min_hw),
        "minimum_full_width_m": float(2.0 * min_hw),
        "fixed_half_width_m": float(fixed_hw),
        "bottom_half_width_m": float(width_probe[0]) if width_probe.size else np.nan,
        "top_half_width_m": float(width_probe[-1]) if width_probe.size else np.nan,
        "x_half_min_m": float(np.nanmin(width_probe)) if width_probe.size else np.nan,
        "x_half_max_m": float(np.nanmax(width_probe)) if width_probe.size else np.nan,
        "eta_origin_y_m": float(eta_origin_y_m),
        "eta_origin_mode": eta_origin_mode,
        "eta_to_min_width_transition_y_m": transition_y,
        "eta_to_min_width_transition_height_above_wire_m": float(transition_y - wire_y_m) if np.isfinite(transition_y) else np.nan,
    }


def integrate_selected_control_volume_vertical_momentum(
    Ti,
    uxi,
    uyi,
    pi,
    geom: Dict[str, object],
    rho: float,
    mu: float,
    beta: float,
    g: float,
    T_inf: float,
    n_side: int = 801,
    n_bottom_top: int = 1201,
    n_area_y: int = 101,
) -> Dict[str, float | str]:
    """Integrate vertical momentum over the selected eta/frozen-width CV.

    The lateral surfaces are x = +/- x_half(y).  For these curved sides the vertical
    momentum flux, pressure and viscous terms are integrated using the outward
    normal-times-arclength vector directly:
      right side: (n_x ds, n_y ds) = ( 1, -dxh/dy) dy
      left side:  (n_x ds, n_y ds) = (-1, -dxh/dy) dy
    Top and bottom are horizontal straight segments using their local half-widths.
    """
    if pi is None or geom is None:
        return {}
    x_half_fun = geom.get("x_half_fun")
    if x_half_fun is None:
        return {}
    y_bottom = float(geom["y_bottom_m"])
    y_top = float(geom["y_top_m"])
    if y_top <= y_bottom:
        return {}

    def line_terms_nds(xv, yv, nds_x, nds_y):
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)
        nds_x = np.asarray(nds_x, dtype=float) + np.zeros_like(xv, dtype=float)
        nds_y = np.asarray(nds_y, dtype=float) + np.zeros_like(xv, dtype=float)
        ux = finite_or_nan(uxi(xv, yv))
        uy = finite_or_nan(uyi(xv, yv))
        pp = finite_or_nan(pi(xv, yv))
        dux_dx, dux_dy = uxi.gradient(xv, yv)
        duy_dx, duy_dy = uyi.gradient(xv, yv)
        dux_dy = finite_or_nan(dux_dy)
        duy_dx = finite_or_nan(duy_dx)
        duy_dy = finite_or_nan(duy_dy)
        un_ds = ux * nds_x + uy * nds_y
        adv = float(rho) * uy * un_ds
        tau_yx = float(mu) * (duy_dx + dux_dy)
        tau_yy = 2.0 * float(mu) * duy_dy
        pressure = -pp * nds_y
        viscous = tau_yx * nds_x + tau_yy * nds_y
        return adv, pressure, viscous

    # Side geometry and derivative.
    ys = np.linspace(y_bottom, y_top, int(n_side))
    xh = np.asarray(x_half_fun(ys), dtype=float)
    valid = np.isfinite(xh) & (xh > 0.0)
    if np.count_nonzero(valid) < 4:
        return {}
    ys = ys[valid]
    xh = xh[valid]
    dxh_dy = np.gradient(xh, ys)
    y_bottom_eff = float(ys[0])
    y_top_eff = float(ys[-1])

    # Horizontal top/bottom boundaries.
    half_b = float(xh[0])
    half_t = float(xh[-1])
    xb = np.linspace(-half_b, half_b, int(n_bottom_top))
    xt = np.linspace(-half_t, half_t, int(n_bottom_top))
    adv_b, pres_b, visc_b = line_terms_nds(xb, np.full_like(xb, y_bottom_eff), 0.0, -1.0)
    adv_t, pres_t, visc_t = line_terms_nds(xt, np.full_like(xt, y_top_eff), 0.0, 1.0)
    adv_bottom = robust_trapz(adv_b, xb)
    adv_top = robust_trapz(adv_t, xt)
    pres_bottom = robust_trapz(pres_b, xb)
    pres_top = robust_trapz(pres_t, xt)
    visc_bottom = robust_trapz(visc_b, xb)
    visc_top = robust_trapz(visc_t, xt)

    # Curved left/right sides, parameterized by y.
    adv_l, pres_l, visc_l = line_terms_nds(-xh, ys, -1.0, -dxh_dy)
    adv_r, pres_r, visc_r = line_terms_nds( xh, ys,  1.0, -dxh_dy)
    adv_left = robust_trapz(adv_l, ys)
    adv_right = robust_trapz(adv_r, ys)
    pres_left = robust_trapz(pres_l, ys)
    pres_right = robust_trapz(pres_r, ys)
    visc_left = robust_trapz(visc_l, ys)
    visc_right = robust_trapz(visc_r, ys)

    adv_total = adv_top + adv_bottom + adv_left + adv_right
    pressure_total = pres_top + pres_bottom + pres_left + pres_right
    viscous_total = visc_top + visc_bottom + visc_left + visc_right

    # Area buoyancy integral over -x_half(y) <= x <= x_half(y).
    ya = np.linspace(y_bottom_eff, y_top_eff, int(n_area_y))
    bline = np.full_like(ya, np.nan, dtype=float)
    for i, yy in enumerate(ya):
        half = float(np.asarray(x_half_fun(np.asarray([yy], dtype=float)))[0])
        if not np.isfinite(half) or half <= 0.0:
            continue
        xa = np.linspace(-half, half, int(n_bottom_top))
        Tline = finite_or_nan(Ti(xa, np.full_like(xa, yy)))
        bline[i] = robust_trapz(float(rho) * float(g) * float(beta) * (Tline - float(T_inf)), xa)
    buoyancy_total = robust_trapz(bline, ya)

    rhs_total = pressure_total + viscous_total + buoyancy_total
    residual = adv_total - rhs_total
    scale = max(abs(adv_total), abs(pressure_total), abs(viscous_total), abs(buoyancy_total), abs(rhs_total), 1e-300)

    out: Dict[str, float | str] = {
        "enabled": "true",
        "cv_kind": str(geom.get("kind", "eta_width_selected_cv")),
        "cv_y_bottom_m": float(y_bottom_eff),
        "cv_y_top_m": float(y_top_eff),
        "cv_x_left_bottom_m": float(-half_b),
        "cv_x_right_bottom_m": float(half_b),
        "cv_x_left_top_m": float(-half_t),
        "cv_x_right_top_m": float(half_t),
        "cv_bottom_half_width_m": float(half_b),
        "cv_top_half_width_m": float(half_t),
        "cv_x_half_min_m": float(np.nanmin(xh)),
        "cv_x_half_max_m": float(np.nanmax(xh)),
        "advective_vertical_momentum_flux_N_per_m": float(adv_total),
        "advective_top_N_per_m": float(adv_top),
        "advective_bottom_N_per_m": float(adv_bottom),
        "advective_left_N_per_m": float(adv_left),
        "advective_right_N_per_m": float(adv_right),
        "advective_top_plus_bottom_N_per_m": float(adv_top + adv_bottom),
        "advective_side_entrainment_N_per_m": float(adv_left + adv_right),
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
        "n_side_samples": int(len(ys)),
        "n_bottom_top_samples": int(n_bottom_top),
        "n_area_y_samples": int(n_area_y),
    }
    # Copy scalar metadata from geometry into the CSV row; skip callables.
    for k, v in geom.items():
        if callable(v) or k in out:
            continue
        if isinstance(v, (str, int, float, np.floating)):
            out[k] = float(v) if isinstance(v, np.floating) else v
    return out


def integrate_cylinder_traction_from_facets(points: np.ndarray, cells: np.ndarray, pi, uxi, uyi, mu: float) -> Dict[str, float]:
    """Approximate the cylinder/inner-boundary vertical traction from non-outer boundary facets.

    Boundary facets not lying on the rectangular outer box are treated as the immersed-cylinder
    boundary. The outward normal is chosen away from the adjacent fluid cell centroid.
    """
    if pi is None:
        return {}
    pts = np.asarray(points, dtype=float)
    if pts.size == 0 or len(cells) == 0:
        return {}
    xmin, xmax = float(np.nanmin(pts[:, 0])), float(np.nanmax(pts[:, 0]))
    ymin, ymax = float(np.nanmin(pts[:, 1])), float(np.nanmax(pts[:, 1]))
    span = max(xmax - xmin, ymax - ymin, 1.0)
    tol = 1e-7 * span
    total_pressure = 0.0
    total_viscous = 0.0
    total = 0.0
    length = 0.0
    count = 0
    for i, j, ci in boundary_edges_with_cells(cells):
        p0 = pts[i]
        p1 = pts[j]
        mid = 0.5 * (p0 + p1)
        if (abs(mid[0] - xmin) <= tol or abs(mid[0] - xmax) <= tol or abs(mid[1] - ymin) <= tol or abs(mid[1] - ymax) <= tol):
            continue
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        nds = np.asarray([dy, -dx], dtype=float)
        centroid = pts[cells[ci]].mean(axis=0)
        if np.dot(nds, mid - centroid) < 0.0:
            nds *= -1.0
        xm = np.asarray([mid[0]], dtype=float)
        ym = np.asarray([mid[1]], dtype=float)
        pp = float(finite_or_nan(pi(xm, ym))[0])
        dux_dx, dux_dy = uxi.gradient(xm, ym)
        duy_dx, duy_dy = uyi.gradient(xm, ym)
        tau_yx = float(mu) * (float(finite_or_nan(duy_dx)[0]) + float(finite_or_nan(dux_dy)[0]))
        tau_yy = 2.0 * float(mu) * float(finite_or_nan(duy_dy)[0])
        pressure = -pp * float(nds[1])
        viscous = tau_yx * float(nds[0]) + tau_yy * float(nds[1])
        if np.isfinite(pressure):
            total_pressure += pressure
        if np.isfinite(viscous):
            total_viscous += viscous
        if np.isfinite(pressure + viscous):
            total += pressure + viscous
        length += math.hypot(dx, dy)
        count += 1
    return {
        "cylinder_traction_vertical_force_N_per_m": float(total),
        "cylinder_pressure_vertical_force_N_per_m": float(total_pressure),
        "cylinder_viscous_vertical_force_N_per_m": float(total_viscous),
        "cylinder_boundary_edge_count": int(count),
        "cylinder_boundary_integrated_length_m": float(length),
    }
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
    adv_increase = adv_top + adv_bottom
    adv_entr = adv_left + adv_right
    pressure_total = pres_top + pres_bottom + pres_left + pres_right
    pressure_streamwise = pres_top + pres_bottom
    pressure_edgewise = pres_left + pres_right
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


def classify_boundary_edge_name(mid: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float, tol: float) -> str:
    """Classify an exterior mesh edge by midpoint location."""
    if abs(float(mid[1]) - ymax) <= tol:
        return "top"
    if abs(float(mid[1]) - ymin) <= tol:
        return "bottom"
    if abs(float(mid[0]) - xmin) <= tol:
        return "left"
    if abs(float(mid[0]) - xmax) <= tol:
        return "right"
    return "other"



def compute_black_body_wire_radiation_estimate(
    Ti,
    T_inf: float,
    wire_center_x_m: float,
    wire_center_y_m: float,
    wire_radius_m: float,
    q_input_per_length: Optional[float] = None,
    wall_temperature_K: Optional[float] = None,
    emissivity: float = 1.0,
    n_angles: int = 721,
    surface_offset_m: Optional[float] = None,
) -> Tuple[Dict[str, float | str], List[Dict[str, float]]]:
    """
    Estimate radiative heat loss per unit length from the heated wire to the enclosure.

    The default is the black-body upper-bound estimate
        q'_rad = epsilon * 2*pi*r*sigma*(Tw^4 - Twall^4).

    Tw is estimated from an angular average of interpolated temperatures just outside
    the wire surface. The estimate is diagnostic only; it is not coupled back into the
    conduction/convection post-processing integrals.
    """
    if wire_radius_m <= 0.0:
        raise ValueError("Radiation estimate requires a positive wire radius.")
    if n_angles < 8:
        raise ValueError("Radiation estimate requires at least 8 angular samples.")

    sigma = 5.670374419e-8  # W/(m^2 K^4)
    wall_T = float(T_inf if wall_temperature_K is None else wall_temperature_K)
    eps = float(emissivity)
    if surface_offset_m is None:
        surface_offset_m = max(1e-9, 1e-4 * float(wire_radius_m))

    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    sample_radius = float(wire_radius_m) + float(surface_offset_m)
    xs = float(wire_center_x_m) + sample_radius * np.cos(angles)
    ys = float(wire_center_y_m) + sample_radius * np.sin(angles)
    Ts = finite_or_nan(Ti(xs, ys))
    valid = np.isfinite(Ts) & (Ts > 0.0)

    local_rows: List[Dict[str, float]] = []
    qpp = np.full_like(Ts, np.nan, dtype=float)
    qprime_density = np.full_like(Ts, np.nan, dtype=float)
    if np.any(valid):
        qpp[valid] = eps * sigma * (Ts[valid] ** 4 - wall_T ** 4)
        # Per-unit-angle contribution to q' if integrated over theta.
        qprime_density[valid] = float(wire_radius_m) * qpp[valid]

    for ang, Tx, qppx, qdx in zip(angles, Ts, qpp, qprime_density):
        local_rows.append({
            "angle_rad": float(ang),
            "angle_deg": float(np.degrees(ang)),
            "T_surface_sample_K": float(Tx) if np.isfinite(Tx) else np.nan,
            "T_wall_K": wall_T,
            "radiative_heat_flux_W_per_m2": float(qppx) if np.isfinite(qppx) else np.nan,
            "radiative_heat_per_length_density_W_per_m_per_rad": float(qdx) if np.isfinite(qdx) else np.nan,
        })

    if np.any(valid):
        T_avg = float(np.mean(Ts[valid]))
        T_min = float(np.min(Ts[valid]))
        T_max = float(np.max(Ts[valid]))
        T_std = float(np.std(Ts[valid], ddof=1)) if np.count_nonzero(valid) > 1 else 0.0
        # Primary estimate: use angular mean of T^4, not (mean T)^4, so nonuniformity is retained.
        q_rad_per_m = float(2.0 * np.pi * wire_radius_m * eps * sigma * (np.mean(Ts[valid] ** 4) - wall_T ** 4))
        q_rad_per_m_from_avg_T = float(2.0 * np.pi * wire_radius_m * eps * sigma * (T_avg ** 4 - wall_T ** 4))
        qpp_avg = float(eps * sigma * (np.mean(Ts[valid] ** 4) - wall_T ** 4))
    else:
        T_avg = T_min = T_max = T_std = np.nan
        q_rad_per_m = q_rad_per_m_from_avg_T = qpp_avg = np.nan

    q_input = float(q_input_per_length) if q_input_per_length is not None else np.nan
    summary: Dict[str, float | str] = {
        "model": "black_or_gray_diffuse_wire_to_large_isothermal_surroundings",
        "notes": "Black-body upper bound when emissivity=1. Uses angular mean of sampled T^4 just outside the wire surface.",
        "stefan_boltzmann_constant_W_per_m2_K4": sigma,
        "emissivity": eps,
        "wire_radius_m": float(wire_radius_m),
        "wire_diameter_m": 2.0 * float(wire_radius_m),
        "wire_center_x_m": float(wire_center_x_m),
        "wire_center_y_m": float(wire_center_y_m),
        "surface_sample_offset_m": float(surface_offset_m),
        "surface_sample_radius_m": sample_radius,
        "n_angles_requested": int(n_angles),
        "n_angles_valid": int(np.count_nonzero(valid)),
        "valid_angle_fraction": float(np.count_nonzero(valid) / int(n_angles)),
        "T_wall_K": wall_T,
        "T_inf_K": float(T_inf),
        "T_wire_surface_angular_mean_K": T_avg,
        "T_wire_surface_min_K": T_min,
        "T_wire_surface_max_K": T_max,
        "T_wire_surface_std_K": T_std,
        "DeltaT_wire_surface_mean_vs_wall_K": T_avg - wall_T if np.isfinite(T_avg) else np.nan,
        "radiative_heat_flux_area_average_W_per_m2": qpp_avg,
        "radiative_heat_per_length_W_per_m": q_rad_per_m,
        "radiative_heat_per_length_from_mean_T_W_per_m": q_rad_per_m_from_avg_T,
        "q_input_per_length_W_per_m": q_input,
        "radiative_fraction_of_input": q_rad_per_m / q_input if np.isfinite(q_rad_per_m) and np.isfinite(q_input) and q_input != 0.0 else np.nan,
        "convective_conductive_remainder_if_subtracted_W_per_m": q_input - q_rad_per_m if np.isfinite(q_rad_per_m) and np.isfinite(q_input) else np.nan,
    }
    return summary, local_rows

def compute_wire_nusselt_diagnostics(
    points: np.ndarray,
    cells: np.ndarray,
    T: np.ndarray,
    Ti,
    *,
    T_inf: float,
    k: float,
    q_input_per_length: float,
    wire_center_x_m: float,
    wire_center_y_m: float,
    wire_radius_m: float,
    rho: Optional[float] = None,
    mu: Optional[float] = None,
    beta: Optional[float] = None,
    g: float = 9.81,
    solid_k: Optional[float] = None,
    n_angles: int = 361,
    surface_offset_m: Optional[float] = None,
) -> Tuple[List[Dict[str, float | str]], List[Dict[str, float | str]]]:
    """
    Compute local and overall wire/cylinder Nusselt diagnostics.

    Base definition:
        Nu_D = h D / k, with D = 2R.

    Overall value:
        h = Q_L / (S_w*(Tbar_s - T_inf)),
        Nu_D = Q_L*D/(S_w*k*(Tbar_s - T_inf)).

    Optional extended values:
        Gr_D = g*beta*(T_s - T_inf)*D^3/nu^2, with nu=mu/rho,
        Nu_delta = Nu_D/Gr_D^(1/5),
        Bi = Nu_D*k/k_s.
    """
    if q_input_per_length is None or not np.isfinite(float(q_input_per_length)):
        raise ValueError("Nusselt diagnostics require a finite --q-input-per-length.")
    if wire_radius_m <= 0.0:
        raise ValueError("Nusselt diagnostics require a positive wire radius.")
    if k <= 0.0:
        raise ValueError("Nusselt diagnostics require a positive fluid thermal conductivity --k.")

    D = 2.0 * float(wire_radius_m)
    nominal_perimeter = math.pi * D
    QL = float(q_input_per_length)

    have_gr = (
        rho is not None and mu is not None and beta is not None
        and np.isfinite(float(rho)) and np.isfinite(float(mu)) and np.isfinite(float(beta))
        and float(rho) > 0.0 and float(mu) > 0.0 and float(beta) > 0.0
    )
    nu = float(mu) / float(rho) if have_gr else np.nan

    have_bi = solid_k is not None and np.isfinite(float(solid_k)) and float(solid_k) > 0.0
    solid_k_value = float(solid_k) if have_bi else np.nan

    def _gr_d(delta_t: float) -> float:
        if not have_gr or not np.isfinite(delta_t) or delta_t <= 0.0:
            return np.nan
        return float(g) * float(beta) * float(delta_t) * D**3 / (nu**2)

    def _nu_delta(nu_d: float, gr_d: float) -> float:
        if not np.isfinite(nu_d) or not np.isfinite(gr_d) or gr_d <= 0.0:
            return np.nan
        return float(nu_d) / (float(gr_d) ** 0.2)

    def _bi(nu_d: float) -> float:
        if not have_bi or not np.isfinite(nu_d):
            return np.nan
        return float(nu_d) * float(k) / solid_k_value

    xmin, xmax = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
    ymin, ymax = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
    tol = 1e-8 * max(xmax - xmin, ymax - ymin, 1.0)

    inner_length = 0.0
    inner_T_int = 0.0
    inner_dT_int = 0.0
    inner_Nu_int = 0.0
    inner_count = 0

    for a, b, ci in boundary_edges_with_cells(cells):
        p0 = points[a]
        p1 = points[b]
        mid = 0.5 * (p0 + p1)
        if classify_boundary_edge_name(mid, xmin, xmax, ymin, ymax, tol) != "other":
            continue
        L = float(np.linalg.norm(p1 - p0))
        if L <= 0.0:
            continue
        Tmid = 0.5 * (float(T[a]) + float(T[b]))
        dTmid = Tmid - float(T_inf)
        inner_length += L
        inner_T_int += Tmid * L
        inner_dT_int += dTmid * L
        inner_count += 1

    if inner_length <= 0.0 or inner_count == 0:
        raise ValueError("Could not identify the inner wire boundary. Expected non-outer boundary facets classified as 'other'.")

    Tbar_edge = inner_T_int / inner_length
    dTbar_edge = Tbar_edge - float(T_inf)
    qpp_actual = QL / inner_length
    qpp_nominal = QL / nominal_perimeter

    Nu_overall_actual = qpp_actual * D / (float(k) * dTbar_edge) if dTbar_edge > 0.0 else np.nan
    Nu_overall_nominal = qpp_nominal * D / (float(k) * dTbar_edge) if dTbar_edge > 0.0 else np.nan
    Nu_overall_formula = QL / (math.pi * float(k) * dTbar_edge) if dTbar_edge > 0.0 else np.nan

    GrD_overall = _gr_d(dTbar_edge)
    Nu_delta_overall_actual = _nu_delta(Nu_overall_actual, GrD_overall)
    Nu_delta_overall_nominal = _nu_delta(Nu_overall_nominal, GrD_overall)
    Bi_overall_actual = _bi(Nu_overall_actual)
    Bi_overall_nominal = _bi(Nu_overall_nominal)

    if surface_offset_m is None:
        surface_offset_m = max(1e-9, 1e-4 * wire_radius_m)
    surface_offset_m = float(surface_offset_m)

    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    local_rows: List[Dict[str, float | str]] = []
    Nu_loc_values = []
    Ts_values = []
    GrD_values = []
    Nu_delta_values = []
    Bi_values = []

    for phi in angles:
        c = math.cos(float(phi))
        s = math.sin(float(phi))
        xs = float(wire_center_x_m) + (wire_radius_m + surface_offset_m) * c
        ys = float(wire_center_y_m) + (wire_radius_m + surface_offset_m) * s
        Ts = float(finite_or_nan(Ti(np.array([xs]), np.array([ys])))[0])
        dTs = Ts - float(T_inf) if np.isfinite(Ts) else np.nan
        Nu_loc_actual = qpp_actual * D / (float(k) * dTs) if np.isfinite(dTs) and dTs > 0.0 else np.nan
        Nu_loc_nominal = qpp_nominal * D / (float(k) * dTs) if np.isfinite(dTs) and dTs > 0.0 else np.nan
        GrD_loc = _gr_d(dTs)
        Nu_delta_loc_actual = _nu_delta(Nu_loc_actual, GrD_loc)
        Nu_delta_loc_nominal = _nu_delta(Nu_loc_nominal, GrD_loc)
        Bi_loc_actual = _bi(Nu_loc_actual)
        Bi_loc_nominal = _bi(Nu_loc_nominal)

        local_rows.append({
            "angle_rad": float(phi),
            "angle_deg": float(np.degrees(phi)),
            "x_m": xs,
            "y_m": ys,
            "T_surface_sample_K": Ts,
            "DeltaT_surface_sample_K": dTs,
            "Nu_local_perimeter_corrected": Nu_loc_actual,
            "Nu_local_nominal_circle": Nu_loc_nominal,
            "Gr_D_local_from_surface_sample": GrD_loc,
            "Nu_delta_local_perimeter_corrected": Nu_delta_loc_actual,
            "Nu_delta_local_nominal_circle": Nu_delta_loc_nominal,
            "Bi_local_perimeter_corrected": Bi_loc_actual,
            "Bi_local_nominal_circle": Bi_loc_nominal,
            "qpp_perimeter_corrected_W_per_m2": qpp_actual,
            "qpp_nominal_circle_W_per_m2": qpp_nominal,
            "sampling_offset_from_surface_m": surface_offset_m,
        })
        if np.isfinite(Nu_loc_actual):
            Nu_loc_values.append(Nu_loc_actual)
        if np.isfinite(Ts):
            Ts_values.append(Ts)
        if np.isfinite(GrD_loc):
            GrD_values.append(GrD_loc)
        if np.isfinite(Nu_delta_loc_actual):
            Nu_delta_values.append(Nu_delta_loc_actual)
        if np.isfinite(Bi_loc_actual):
            Bi_values.append(Bi_loc_actual)

    for a, b, ci in boundary_edges_with_cells(cells):
        p0 = points[a]
        p1 = points[b]
        mid = 0.5 * (p0 + p1)
        if classify_boundary_edge_name(mid, xmin, xmax, ymin, ymax, tol) != "other":
            continue
        L = float(np.linalg.norm(p1 - p0))
        if L <= 0.0:
            continue
        Tmid = 0.5 * (float(T[a]) + float(T[b]))
        dTmid = Tmid - float(T_inf)
        Nu_mid = qpp_actual * D / (float(k) * dTmid) if dTmid > 0.0 else np.nan
        if np.isfinite(Nu_mid):
            inner_Nu_int += Nu_mid * L

    Nu_arr = np.asarray(Nu_loc_values, dtype=float)
    Ts_arr = np.asarray(Ts_values, dtype=float)
    GrD_arr = np.asarray(GrD_values, dtype=float)
    Nu_delta_arr = np.asarray(Nu_delta_values, dtype=float)
    Bi_arr = np.asarray(Bi_values, dtype=float)

    summary_rows: List[Dict[str, float | str]] = [{
        "definition": "Nu_D=hD/k, Gr_D=g*beta*(T_s-T_inf)*D^3/nu^2, Nu_delta=Nu_D/Gr_D^(1/5), Bi=Nu_D*k/k_s",
        "wire_center_x_m": float(wire_center_x_m),
        "wire_center_y_m": float(wire_center_y_m),
        "wire_radius_m": float(wire_radius_m),
        "wire_diameter_m": D,
        "wire_perimeter_integrated_from_mesh_m": inner_length,
        "wire_perimeter_nominal_circle_m": nominal_perimeter,
        "wire_perimeter_relative_error_vs_nominal": (inner_length - nominal_perimeter) / nominal_perimeter if nominal_perimeter > 0.0 else np.nan,
        "wire_boundary_edge_count": int(inner_count),
        "T_surface_mean_edge_length_weighted_K": Tbar_edge,
        "DeltaT_surface_mean_edge_length_weighted_K": dTbar_edge,
        "q_input_per_length_W_per_m": QL,
        "qpp_perimeter_corrected_W_per_m2": qpp_actual,
        "qpp_nominal_circle_W_per_m2": qpp_nominal,
        "Nu_overall_perimeter_corrected": Nu_overall_actual,
        "Nu_overall_nominal_circle": Nu_overall_nominal,
        "Nu_overall_Q_over_pi_k_DeltaT": Nu_overall_formula,
        "rho_kg_per_m3_for_Gr_D": float(rho) if rho is not None else np.nan,
        "mu_Pa_s_for_Gr_D": float(mu) if mu is not None else np.nan,
        "nu_m2_per_s_for_Gr_D": nu,
        "beta_1_per_K_for_Gr_D": float(beta) if beta is not None else np.nan,
        "g_m_per_s2_for_Gr_D": float(g),
        "Gr_D_overall_from_mean_surface_DeltaT": GrD_overall,
        "Nu_delta_overall_perimeter_corrected": Nu_delta_overall_actual,
        "Nu_delta_overall_nominal_circle": Nu_delta_overall_nominal,
        "solid_k_W_per_mK_for_Bi": solid_k_value,
        "Bi_overall_perimeter_corrected": Bi_overall_actual,
        "Bi_overall_nominal_circle": Bi_overall_nominal,
        "Nu_local_surface_average_edge_length_weighted": inner_Nu_int / inner_length if inner_length > 0.0 else np.nan,
        "Nu_local_sample_mean_perimeter_corrected": float(np.mean(Nu_arr)) if Nu_arr.size else np.nan,
        "Nu_local_sample_min_perimeter_corrected": float(np.min(Nu_arr)) if Nu_arr.size else np.nan,
        "Nu_local_sample_max_perimeter_corrected": float(np.max(Nu_arr)) if Nu_arr.size else np.nan,
        "Gr_D_local_sample_mean": float(np.mean(GrD_arr)) if GrD_arr.size else np.nan,
        "Gr_D_local_sample_min": float(np.min(GrD_arr)) if GrD_arr.size else np.nan,
        "Gr_D_local_sample_max": float(np.max(GrD_arr)) if GrD_arr.size else np.nan,
        "Nu_delta_local_sample_mean_perimeter_corrected": float(np.mean(Nu_delta_arr)) if Nu_delta_arr.size else np.nan,
        "Nu_delta_local_sample_min_perimeter_corrected": float(np.min(Nu_delta_arr)) if Nu_delta_arr.size else np.nan,
        "Nu_delta_local_sample_max_perimeter_corrected": float(np.max(Nu_delta_arr)) if Nu_delta_arr.size else np.nan,
        "Bi_local_sample_mean_perimeter_corrected": float(np.mean(Bi_arr)) if Bi_arr.size else np.nan,
        "Bi_local_sample_min_perimeter_corrected": float(np.min(Bi_arr)) if Bi_arr.size else np.nan,
        "Bi_local_sample_max_perimeter_corrected": float(np.max(Bi_arr)) if Bi_arr.size else np.nan,
        "T_surface_sample_mean_K": float(np.mean(Ts_arr)) if Ts_arr.size else np.nan,
        "T_surface_sample_min_K": float(np.min(Ts_arr)) if Ts_arr.size else np.nan,
        "T_surface_sample_max_K": float(np.max(Ts_arr)) if Ts_arr.size else np.nan,
        "n_local_angle_samples_requested": int(n_angles),
        "n_local_angle_samples_valid": int(Nu_arr.size),
        "local_sampling_offset_from_surface_m": surface_offset_m,
        "notes": "Use Nu_overall_perimeter_corrected as the primary total wire Nusselt number. Nu_delta requires rho, mu, beta. Bi requires --solid-k.",
    }]
    return summary_rows, local_rows


def plot_local_nusselt(path: Path, rows: List[Dict[str, float | str]], *, figsize: Tuple[float, float], show_titles: bool = False) -> None:
    if not rows:
        return
    phi = np.asarray([r["angle_deg"] for r in rows], dtype=float)
    Nu = np.asarray([r["Nu_local_perimeter_corrected"] for r in rows], dtype=float)
    Nu_delta = np.asarray([r["Nu_delta_local_perimeter_corrected"] for r in rows], dtype=float)
    mask = np.isfinite(phi) & np.isfinite(Nu)
    if np.count_nonzero(mask) < 2:
        return
    order = np.argsort(phi[mask])
    plt.figure(figsize=figsize)
    plt.plot(phi[mask][order], Nu[mask][order], linewidth=1.8, label=r"$Nu_D$")
    if np.any(np.isfinite(Nu_delta)):
        maskd = np.isfinite(phi) & np.isfinite(Nu_delta)
        orderd = np.argsort(phi[maskd])
        plt.plot(phi[maskd][orderd], Nu_delta[maskd][orderd], linewidth=1.8, label=r"$Nu_\delta$")
        plt.legend()
    plt.xlabel(r"surface angle $\phi$ [deg]")
    plt.ylabel(r"Nusselt number [-]")
    maybe_set_title("Local wire Nusselt diagnostics", show_titles)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()



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



def plot_xy(path: Path, x, ys: Sequence[Tuple[str, np.ndarray]], xlabel: str, ylabel: str, title: str, semilogy: bool = False, *, figsize: Tuple[float, float] = (3.0, 2.16), show_titles: bool = False, dpi: int = 220) -> None:
    plt.figure(figsize=figsize)
    for label, y in ys:
        if semilogy:
            plt.semilogy(x, y, label=label)
        else:
            plt.plot(x, y, label=label)
    plt.xlabel(xlabel, size=22)
    plt.ylabel(ylabel, size=22)
    maybe_set_title(title, show_titles)
    plt.grid(True, which="both", alpha=0.35)
    if len(ys) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
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


def fit_fixed_exponent_powerlaw(h: np.ndarray, a: np.ndarray, exponent: float, hmin: float, hmax: float) -> Dict[str, float]:
    """Fit only C in a = C h^exponent over a selected positive h-window."""
    h = np.asarray(h, dtype=float)
    a = np.asarray(a, dtype=float)
    mask = np.isfinite(h) & np.isfinite(a) & (h > 0.0) & (a > 0.0)
    mask &= h >= hmin
    mask &= h <= hmax
    if np.count_nonzero(mask) < 4:
        return {"C": np.nan, "exponent": float(exponent), "r2": np.nan, "npoints": int(np.count_nonzero(mask)),
                "fit_height_min_m": hmin, "fit_height_max_m": hmax}
    x = np.log(h[mask])
    y = np.log(a[mask])
    logC = float(np.mean(y - exponent * x))
    yhat = exponent * x + logC
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"C": float(np.exp(logC)), "exponent": float(exponent), "r2": float(r2),
            "npoints": int(np.count_nonzero(mask)), "fit_height_min_m": float(hmin), "fit_height_max_m": float(hmax)}


def fit_loglog_powerlaw_about_origin(y: np.ndarray, a: np.ndarray, y0: float, y_min: float, y_max: float) -> Dict[str, float]:
    """Fit a = C (y-y0)^n on y_min <= y <= y_max."""
    h_shifted = np.asarray(y, dtype=float) - float(y0)
    return fit_loglog_powerlaw(h_shifted, a, float(y_min) - float(y0), float(y_max) - float(y0))


def fit_fixed_exponent_powerlaw_about_origin(y: np.ndarray, a: np.ndarray, y0: float, exponent: float, y_min: float, y_max: float) -> Dict[str, float]:
    """Fit only C in a = C (y-y0)^exponent on y_min <= y <= y_max."""
    h_shifted = np.asarray(y, dtype=float) - float(y0)
    return fit_fixed_exponent_powerlaw(h_shifted, a, exponent, float(y_min) - float(y0), float(y_max) - float(y0))

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
    ap.add_argument("--k", type=float, required=True, help="Thermal conductivity of the fluid [W/(m K)]")
    ap.add_argument("--solid-k", type=float, default=None,
                    help="Thermal conductivity of the wire/cylinder solid [W/(m K)] for Bi = Nu_D*k/solid_k.")
    ap.add_argument("--mu", type=float, default=None, help="Dynamic viscosity [Pa s]; used for Re-like diagnostics and Gr_D if --nusselt is active")
    ap.add_argument("--beta", type=float, default=None, help="Thermal expansion coefficient [1/K]; used for buoyancy diagnostics and Gr_D if --nusselt is active")
    ap.add_argument("--g", type=float, default=9.81, help="Gravity magnitude [m/s^2]")
    ap.add_argument("--q-input-per-length", type=float, default=None, help="Known heat input per unit length [W/m] for energy-balance error and Nusselt diagnostics")
    ap.add_argument("--radiation-emissivity", type=float, default=1.0,
                    help="Emissivity used in the wire-to-wall radiation estimate. Default 1.0 gives the black-body upper bound.")
    ap.add_argument("--radiation-wall-temperature", type=float, default=None,
                    help="Wall/enclosure temperature [K] for the radiation estimate. Default: --T-inf.")
    ap.add_argument("--radiation-angles", type=int, default=721,
                    help="Number of angular samples used to estimate the mean wire surface temperature for radiation.")
    ap.add_argument("--radiation-surface-offset", type=float, default=None,
                    help="Distance [m] outside the wire surface used to sample Tw for radiation. Default: max(1e-9, 1e-4*r).")
    ap.add_argument("--nusselt", action="store_true",
                    help="Compute local and overall wire/cylinder Nusselt numbers, Gr_D-scaled Nu_delta, and optional Biot number.")
    ap.add_argument("--nusselt-angles", type=int, default=361,
                    help="Number of angular samples for the local surface Nusselt-number distribution.")
    ap.add_argument("--nusselt-surface-offset", type=float, default=None,
                    help="Distance [m] outside the wire surface used for local temperature sampling. Default: max(1e-9, 1e-4*r).")
    ap.add_argument("--wire-center-x", type=float, default=0.0,
                    help="Wire centre x-coordinate [m] used for local Nusselt sampling. Default: 0, matching the symmetric plume setup.")
    ap.add_argument("--velocity-scale-factor", type=float, default=1.0,
                    help="Multiplicative factor applied to the exported velocity field before all diagnostics. Use this when the saved dimensional velocity was reconstructed with Uref but should be reported with Uplume, or vice versa.")
    ap.add_argument("--energy-eta-half-width", type=float, default=8.0,
                    help="Eta half-width used for the plume enthalpy-flow balance, i.e. integrate only samples with |eta| <= this value. Use 9 for the wider measured plume window.")
    ap.add_argument("--energy-cv", action="store_true",
                    help="Compute a closed control-volume energy budget and write energy_control_volume_budget.csv.")
    ap.add_argument("--energy-cv-width-mode", choices=["fixed", "eta", "auto"], default="eta",
                    help="Shape of the energy control volume. fixed: vertical sides at fixed half-width; eta: plume-following |eta| sidewalls with a minimum half-width; auto: eta if possible, otherwise fixed.")
    ap.add_argument("--energy-cv-fixed-half-width", type=float, default=0.015,
                    help="Fixed Cartesian half-width [m] used for --energy-cv-width-mode fixed, or as fallback for auto. Default: 0.015 m.")
    ap.add_argument("--energy-cv-eta-half-width", type=float, default=9.0,
                    help="Eta half-width used for plume-following energy control-volume sidewalls. Default: 9.")
    ap.add_argument("--energy-cv-min-half-width-m", type=float, default=1.5913e-2,
                    help="Minimum physical half-width [m] for eta-based energy CV. The sidewalls follow |eta| only after x_eta(y) reaches this value; below that, they remain vertical at x=±this value. Default: 1.5913e-2 m.")
    ap.add_argument("--energy-cv-y-bottom", type=float, default=None,
                    help="Optional lower face height relative to wire centre [m]. Default: lower wire surface minus angular mean thermal boundary-layer thickness.")
    ap.add_argument("--energy-cv-y-top", type=float, default=0.035,
                    help="Upper face height relative to wire centre [m]. Default: 0.035 m.")
    ap.add_argument("--energy-cv-n-boundary", type=int, default=1201,
                    help="Number of quadrature samples on each energy-CV boundary. Default: 1201.")
    ap.add_argument("--plot-width-inch", type=float, default=3.0,
                    help="Width of saved plot figures in inches. Default is thesis-friendly for two plots per row.")
    ap.add_argument("--plot-height-inch", type=float, default=None,
                    help="Optional height of saved plot figures in inches. Default is 0.72 * plot width.")
    ap.add_argument("--plot-font-size", type=float, default=12.0,
                    help="Base font size for all plots.")
    ap.add_argument("--plot-titles", action="store_true",
                    help="Enable titles inside plot images. Default: no plot titles, because captions provide them in the thesis.")

    ap.add_argument("--planes", type=float, nargs="+", default=[0.01, 0.02, 0.04, 0.08], help="Physical heights above wire [m]")
    ap.add_argument("--eta-origin", choices=["wire", "temperature-virtual-origin", "velocity-virtual-origin"], default="wire",
                    help="Origin used for eta-profile scaling. 'wire' uses the inferred wire centre; the virtual-origin options use the fitted y0 after the fits are available.")
    ap.add_argument("--eta-origin-height", type=float, default=None,
                    help="Optional explicit eta source height above wire centre [m]. Overrides --eta-origin when supplied. For example, use -2*d if you want a virtual line source below the wire.")
    ap.add_argument("--eta-plot-half-width", type=float, default=8.0,
                    help="Eta half-width used only for eta-coordinate plots. Default: 8, so eta plots show |eta| <= 8. Use a value <= 0 to plot the full sampled eta range.")
    ap.add_argument("--fit-y-min", type=float, default=None, help="Minimum physical height above wire [m] used for virtual-origin fits. If omitted, an automatic window is selected.")
    ap.add_argument("--fit-y-max", type=float, default=None, help="Maximum physical height above wire [m] used for virtual-origin fits. If omitted, an automatic window is selected.")
    ap.add_argument("--auto-fit-min-points", type=int, default=30, help="Minimum number of centreline samples in automatic virtual-origin fit window")
    ap.add_argument("--auto-fit-min-span", type=float, default=None, help="Minimum physical height span [m] of automatic virtual-origin fit window. Default: max(0.025 m, 25*r)")
    ap.add_argument("--auto-fit-lower-cutoff", type=float, default=None, help="Minimum height above wire [m] for automatic fits. Default: max(12*r, 0.010 m)")
    ap.add_argument("--auto-fit-upper-cutoff", type=float, default=None,
                    help="Maximum height above wire [m] allowed for automatic virtual-origin fits. Default: stop at 90%% of the distance from wire centre to the top boundary, to avoid fitting the cooled top-wall region.")
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
                    help="Maximum height above wire [m] for continuous centreline and balance diagnostics. Default: 90%% of available height from wire to top boundary.")
    ap.add_argument("--balance-y-min", type=float, default=None,
                    help="Minimum height above wire [m] for continuous centreline and balance diagnostics. Default: just above the wire/source region.")
    ap.add_argument("--threshold", type=float, default=0.01, help="Boundary-layer threshold fraction of local near-surface temperature excess")
    ap.add_argument("--bl-angles", type=int, default=181, help="Number of radial directions used for angular-average near-wire boundary-layer thickness")
    ap.add_argument("--bl-r-max", type=float, default=None, help="Maximum radial distance from cylinder surface [m] for boundary-layer search. Default: largest distance fitting in domain")
    ap.add_argument("--bl-nr", type=int, default=600, help="Number of radial samples per angular direction for boundary-layer search")
    args = ap.parse_args()
    configure_plot_style(args.plot_font_size)
    plot_figsize = thesis_figsize(args.plot_width_inch, args.plot_height_inch)

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

    radiation_summary, radiation_local_rows = compute_black_body_wire_radiation_estimate(
        Ti,
        T_inf=args.T_inf,
        wire_center_x_m=args.wire_center_x,
        wire_center_y_m=wire_y_m,
        wire_radius_m=wire_radius_m,
        q_input_per_length=args.q_input_per_length,
        wall_temperature_K=args.radiation_wall_temperature,
        emissivity=args.radiation_emissivity,
        n_angles=args.radiation_angles,
        surface_offset_m=args.radiation_surface_offset,
    )
    write_csv(outdir / "black_body_radiation_estimate.csv", [radiation_summary])
    write_csv(outdir / "black_body_radiation_by_angle.csv", radiation_local_rows)

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
                "qconv_y_W_per_m2": args.rho * args.cp * uyi_ * (Ti_ - args.T_inf),
                "qcond_y_W_per_m2": qt_ - args.rho * args.cp * uyi_ * (Ti_ - args.T_inf),
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
        # as the temperature fit; if a clear maximum exists, cap the fit below it.
        #
        # IMPORTANT: for the velocity branch the fitted empirical origin may lie
        # above the physical wire centre in a confined finite-cylinder calculation.
        # This is not the same object as the temperature virtual origin. Therefore
        # the velocity selector below must not enforce y0 <= wire_y_m.
        vel_lower = lower
        vel_upper = upper
        vel_min_span = min_span
        if np.any(np.isfinite(uy_c)):
            imax_u = int(np.nanargmax(uy_c))
            h_umax = float(yy[imax_u] - wire_y_m)
            candidate_upper = h_umax - 3.0 * wire_radius_m
            if candidate_upper > vel_lower:
                vel_upper = min(vel_upper, candidate_upper)

            # The usable increasing branch can be shorter than the conservative
            # temperature-fit span, especially in a closed cavity where uy_c reaches
            # a maximum and then decelerates before the top wall. Keep the global
            # default for temperature, but relax the velocity span if necessary.
            available_vel_span = vel_upper - vel_lower if np.isfinite(vel_upper) else np.nan
            if np.isfinite(available_vel_span) and available_vel_span > 0.0:
                vel_min_span = min(min_span, max(0.010, 0.60 * available_vel_span))
            else:
                vel_min_span = min(min_span, 0.015)

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
            min_span=vel_min_span,
            require_monotone=True,
            require_y0_below_wire=False,
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

    if args.nusselt:
        if args.q_input_per_length is None:
            raise SystemExit("--nusselt requires --q-input-per-length because the imposed heat per unit length defines the heat-transfer coefficient.")
        nusselt_summary_rows, nusselt_local_rows = compute_wire_nusselt_diagnostics(
            Tdata.points_m, Tdata.cells, T, Ti,
            T_inf=args.T_inf,
            k=args.k,
            q_input_per_length=args.q_input_per_length,
            wire_center_x_m=args.wire_center_x,
            wire_center_y_m=wire_y_m,
            wire_radius_m=wire_radius_m,
            rho=args.rho,
            mu=args.mu,
            beta=args.beta,
            g=args.g,
            solid_k=args.solid_k,
            n_angles=args.nusselt_angles,
            surface_offset_m=args.nusselt_surface_offset,
        )
        write_csv(outdir / "wire_nusselt_summary.csv", nusselt_summary_rows)
        write_csv(outdir / "wire_nusselt_local.csv", nusselt_local_rows)
        plot_local_nusselt(outdir / "wire_nusselt_local.png", nusselt_local_rows, figsize=plot_figsize, show_titles=args.plot_titles)

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

    # Selected control-volume vertical momentum budget.
    # This thesis-facing output deliberately does not modify the older momentum proxy/full plots.
    selected_momentum_rows: List[Dict[str, float | str]] = []
    if pi is not None and args.mu is not None and args.beta is not None:
        try:
            selected_geom = momentum_cv_geometry_from_energy_settings(
                args, xmin, xmax, ymin, ymax, wire_y_m, wire_radius_m, thermal_bl_mean_m,
                rho=float(args.rho), cp=float(args.cp), mu=float(args.mu), beta=float(args.beta), g=float(args.g),
            )
            selected_momentum = integrate_selected_control_volume_vertical_momentum(
                Ti, uxi, uyi, pi, selected_geom,
                rho=float(args.rho), mu=float(args.mu), beta=float(args.beta), g=float(args.g), T_inf=float(args.T_inf),
                n_side=max(201, int(getattr(args, "energy_cv_ny", getattr(args, "energy_cv_n_boundary", 1201)))),
                n_bottom_top=max(401, int(getattr(args, "energy_cv_nx", getattr(args, "energy_cv_n_boundary", 1201)))),
                n_area_y=max(51, min(int(getattr(args, "energy_cv_ny", getattr(args, "energy_cv_n_boundary", 1201))), 401)),
            )
            if selected_momentum:
                selected_momentum["cv_y_bottom_height_above_wire_m"] = float(selected_momentum.get("cv_y_bottom_m", np.nan)) - wire_y_m
                selected_momentum["cv_y_top_height_above_wire_m"] = float(selected_momentum.get("cv_y_top_m", np.nan)) - wire_y_m
                selected_momentum["description"] = (
                    "Single selected CV momentum budget: buoyancy area integral, cylinder drag, "
                    "side entrainment terms, and top/bottom loading terms. Geometry follows the "
                    "energy CV eta/frozen-minimum-width definition."
                )
                drag = integrate_cylinder_traction_from_facets(Tdata.points_m, Tdata.cells, pi, uxi, uyi, float(args.mu))
                selected_momentum.update(drag)
                selected_momentum_rows.append(selected_momentum)
            else:
                selected_momentum_rows.append({
                    "enabled": "false",
                    "reason": "selected momentum integration returned no data",
                    "description": "Selected CV momentum budget was not computed because the selected geometry was invalid over the mesh.",
                })
        except Exception as exc:
            selected_momentum_rows.append({
                "enabled": "false",
                "reason": str(exc),
                "description": "Selected CV momentum budget failed during geometry construction or integration.",
            })
        write_csv(outdir / "momentum_control_volume_budget.csv", selected_momentum_rows)
    else:
        missing = []
        if pi is None:
            missing.append("--pressure-xdmf")
        if args.mu is None:
            missing.append("--mu")
        if args.beta is None:
            missing.append("--beta")
        selected_momentum_rows.append({
            "enabled": "false",
            "reason": "requires " + ", ".join(missing),
            "description": "Selected CV momentum budget was not computed because pressure/viscosity/buoyancy inputs are incomplete.",
        })
        write_csv(outdir / "momentum_control_volume_budget.csv", selected_momentum_rows)

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
        plt.figure(figsize=plot_figsize)
        plt.bar(labels, vals)
        plt.ylabel("heat rate per unit depth [W/m]")
        maybe_set_title("Global/diagnostic heat budget", args.plot_titles)
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
        plt.figure(figsize=plot_figsize)
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
        maybe_set_title(title, args.plot_titles)
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
    plt.figure(figsize=plot_figsize)
    plt.plot(angle_deg, delta_ang / wire_radius_m)
    if np.isfinite(thermal_bl_mean_m):
        plt.axhline(thermal_bl_mean_m / wire_radius_m, linestyle="--", label=f"mean={thermal_bl_mean_m / wire_radius_m:.4g} r")
    plt.xlabel("angle around cylinder [deg]; 0=right, 90=up")
    plt.ylabel("1% thermal thickness / r")
    maybe_set_title("Near-wire angular thermal boundary-layer thickness", args.plot_titles)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "near_wire_boundary_layer_by_angle.png", dpi=180)
    plt.close()

    # Polar-style visualization of the same radial thickness values.
    plt.figure(figsize=(args.plot_width_inch, args.plot_width_inch))
    ax = plt.subplot(111, projection="polar")
    ax.plot(angles, delta_ang / wire_radius_m)
    if np.isfinite(thermal_bl_mean_m):
        ax.plot(angles, np.full_like(angles, thermal_bl_mean_m / wire_radius_m), linestyle="--", label="mean")
    maybe_set_ax_title(ax, "Angular 1% thermal thickness / r", args.plot_titles)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outdir / "near_wire_boundary_layer_polar.png", dpi=180)
    plt.close()

    # Combined profile plots: one figure per quantity, with all requested heights overlaid.
    def combined_profile_plot(filename: str, quantity_key: str, ylabel: str, title: str) -> None:
        plt.figure(figsize=plot_figsize)
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
        maybe_set_title(title, args.plot_titles)
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


    # Gebhart/Fujii-style eta profiles.  In the paper notation the vertical
    # coordinate is x and the transverse coordinate is y; in this script those
    # correspond to height h above the line source and horizontal coordinate xs.
    # eta = (x_transverse / h) * Gr_h^(1/4),
    # Gr_h = g beta Theta h^3 / nu^2, with Theta = q/(rho cp nu) = q/(mu cp).
    # The same rows also contain the usual similarity-scaled ordinates:
    #   h(eta)  = DeltaT / (Theta Gr_h^(-1/2))
    #   f'(eta) = uy / ((nu/h) Gr_h^(1/2))
    eta_rows = []
    eta_origin_y_m = wire_y_m
    eta_origin_mode = args.eta_origin
    if args.eta_origin_height is not None:
        eta_origin_y_m = wire_y_m + float(args.eta_origin_height)
        eta_origin_mode = "explicit-height"
    elif args.eta_origin == "temperature-virtual-origin" and np.isfinite(fitT.get("y0", np.nan)):
        eta_origin_y_m = float(fitT["y0"])
    elif args.eta_origin == "velocity-virtual-origin" and np.isfinite(fitU.get("y0", np.nan)):
        eta_origin_y_m = float(fitU["y0"])

    eta_enabled = (args.mu is not None and args.beta is not None and
                   args.q_input_per_length is not None and args.q_input_per_length > 0.0)
    nu = float(args.mu) / float(args.rho) if args.mu is not None else np.nan
    theta_line_source = (float(args.q_input_per_length) / (float(args.rho) * float(args.cp) * nu)
                         if eta_enabled and nu > 0.0 else np.nan)

    if eta_enabled and nu > 0.0 and theta_line_source > 0.0:
        for r in profile_rows:
            h_eta = float(r["y_m"]) - eta_origin_y_m
            Gr_h = (float(args.g) * float(args.beta) * theta_line_source * h_eta**3 / nu**2) if h_eta > 0.0 else np.nan
            eta = (float(r["x_m"]) / h_eta) * Gr_h**0.2 if h_eta > 0.0 and np.isfinite(Gr_h) and Gr_h > 0.0 else np.nan
            temp_scale = theta_line_source * Gr_h**(-0.2) if np.isfinite(Gr_h) and Gr_h > 0.0 else np.nan
            vel_scale = (nu / h_eta) * Gr_h**0.4 if h_eta > 0.0 and np.isfinite(Gr_h) and Gr_h > 0.0 else np.nan
            eta_rows.append({
                **r,
                "eta": eta,
                "eta_origin_mode": eta_origin_mode,
                "eta_origin_y_m": eta_origin_y_m,
                "height_above_eta_origin_m": h_eta,
                "Theta_line_source_K": theta_line_source,
                "nu_m2_per_s": nu,
                "Gr_x": Gr_h,
                "DeltaT_similarity_h": float(r["DeltaT_K"]) / temp_scale if np.isfinite(temp_scale) and temp_scale != 0.0 else np.nan,
                "uy_similarity_fprime": float(r["uy_m_per_s"]) / vel_scale if np.isfinite(vel_scale) and vel_scale != 0.0 else np.nan,
            })
        write_csv(outdir / "plane_profiles_eta.csv", eta_rows)

        def combined_eta_plot(filename: str, quantity_key: str, ylabel: str, title: str, xlim_percentile: float = 99.0) -> None:
            plt.figure(figsize=plot_figsize)
            all_eta = []
            eta_plot_half_width = float(args.eta_plot_half_width) if args.eta_plot_half_width is not None else 0.0
            use_eta_window = np.isfinite(eta_plot_half_width) and eta_plot_half_width > 0.0
            for h in args.planes:
                rows = [rr for rr in eta_rows if abs(rr["height_m"] - h) < 1e-15]
                if not rows:
                    continue
                et = np.array([rr["eta"] for rr in rows], dtype=float)
                qp = np.array([rr[quantity_key] for rr in rows], dtype=float)
                m = np.isfinite(et) & np.isfinite(qp)
                if use_eta_window:
                    m &= np.abs(et) <= eta_plot_half_width
                if np.any(m):
                    all_eta.extend(np.abs(et[m]).tolist())
                    o = np.argsort(et[m])
                    plt.plot(et[m][o], qp[m][o], label=f"h={h:g} m")
            if use_eta_window:
                plt.xlim(-eta_plot_half_width, eta_plot_half_width)
            elif all_eta:
                lim = np.nanpercentile(np.array(all_eta), xlim_percentile)
                if np.isfinite(lim) and lim > 0:
                    plt.xlim(-lim, lim)
            plt.xlabel(r"$\eta=(x/h)Gr_h^{1/5}$ [-]", size=22)
            plt.ylabel(ylabel, size=22)
            maybe_set_title(title, args.plot_titles)
            plt.grid(True, alpha=0.35)
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / filename, dpi=220)
            plt.close()

        combined_eta_plot(
            "profiles_temperature_eta_physical_all_heights.png",
            "DeltaT_K",
            r"$T-T_\infty$ [K]",
            r"Temperature profiles in $\eta$ coordinates",
        )
        combined_eta_plot(
            "profiles_uy_eta_physical_all_heights.png",
            "uy_m_per_s",
            r"$u_y$ [m/s]",
            r"Vertical-velocity profiles in $\eta$ coordinates",
        )
        combined_eta_plot(
            "profiles_temperature_eta_similarity_all_heights.png",
            "DeltaT_similarity_h",
            r"$h(\eta)=\Delta T/(\Theta Gr_h^{-1/5})$ [-]",
            r"Similarity-scaled temperature profiles",
        )
        combined_eta_plot(
            "profiles_uy_eta_similarity_all_heights.png",
            "uy_similarity_fprime",
            r"$f'(\eta)=u_y/[(\nu/h)Gr_h^{2/5}]$ [-]",
            r"Similarity-scaled vertical-velocity profiles",
        )
    else:
        write_csv(outdir / "plane_profiles_eta.csv", [{
            "eta_profiles_enabled": False,
            "reason": "requires --mu, --beta, and positive --q-input-per-length",
            "mu_supplied": args.mu is not None,
            "beta_supplied": args.beta is not None,
            "q_input_per_length_supplied": args.q_input_per_length is not None,
        }])

    # Plume enthalpy-flow balance over a constant eta width.
    # This is intentionally separate from Q_total_W_per_m: here we integrate only
    # the convective enthalpy flux rho*cp*uy*(T-T_inf) inside |eta| <= eta_half_width.
    enthalpy_balance_rows = []
    if eta_enabled and eta_rows:
        eta_hw = float(args.energy_eta_half_width)
        for h in args.planes:
            rows_h = [rr for rr in eta_rows if abs(rr["height_m"] - h) < 1e-15]
            if not rows_h:
                continue
            xh = np.array([rr["x_m"] for rr in rows_h], dtype=float)
            etah = np.array([rr.get("eta", np.nan) for rr in rows_h], dtype=float)
            qh = np.array([rr.get("qconv_y_W_per_m2", np.nan) for rr in rows_h], dtype=float)
            uyh = np.array([rr.get("uy_m_per_s", np.nan) for rr in rows_h], dtype=float)
            dth = np.array([rr.get("DeltaT_K", np.nan) for rr in rows_h], dtype=float)
            mask = np.isfinite(xh) & np.isfinite(etah) & np.isfinite(qh) & (np.abs(etah) <= eta_hw)
            if np.count_nonzero(mask) >= 2:
                order = np.argsort(xh[mask])
                xx = xh[mask][order]
                qq = qh[mask][order]
                uu = uyh[mask][order]
                dd = dth[mask][order]
                H_signed = robust_trapz(qq, xx)
                H_upward = robust_trapz(np.where(uu > 0.0, qq, 0.0), xx)
                H_downward = robust_trapz(np.where(uu < 0.0, qq, 0.0), xx)
                H_positive_heat_upward = robust_trapz(np.where((uu > 0.0) & (dd > 0.0), qq, 0.0), xx)
                x_min_eta = float(np.nanmin(xx))
                x_max_eta = float(np.nanmax(xx))
            else:
                H_signed = H_upward = H_downward = H_positive_heat_upward = np.nan
                x_min_eta = x_max_eta = np.nan
            qin = float(args.q_input_per_length) if args.q_input_per_length is not None else np.nan
            if np.isfinite(qin) and abs(qin) > 0.0:
                signed_diff = H_signed - qin
                upward_diff = H_upward - qin
                signed_pct = 100.0 * signed_diff / qin
                upward_pct = 100.0 * upward_diff / qin
            else:
                signed_diff = upward_diff = signed_pct = upward_pct = np.nan
            enthalpy_balance_rows.append({
                "height_m": float(h),
                "eta_half_width": eta_hw,
                "x_min_eta_window_m": x_min_eta,
                "x_max_eta_window_m": x_max_eta,
                "physical_width_eta_window_m": x_max_eta - x_min_eta if np.isfinite(x_min_eta) and np.isfinite(x_max_eta) else np.nan,
                "n_samples_in_eta_window": int(np.count_nonzero(mask)),
                "enthalpy_signed_W_per_m": H_signed,
                "enthalpy_upward_only_W_per_m": H_upward,
                "enthalpy_downward_only_W_per_m": H_downward,
                "enthalpy_upward_positive_temperature_W_per_m": H_positive_heat_upward,
                "supplied_heat_W_per_m": qin,
                "signed_minus_supplied_W_per_m": signed_diff,
                "signed_deviation_percent": signed_pct,
                "upward_minus_supplied_W_per_m": upward_diff,
                "upward_deviation_percent": upward_pct,
            })
        write_csv(outdir / "plume_enthalpy_balance.csv", enthalpy_balance_rows)
    else:
        enthalpy_balance_rows.append({
            "enthalpy_balance_enabled": False,
            "reason": "requires eta profiles, hence --mu, --beta, and positive --q-input-per-length",
            "eta_half_width": float(args.energy_eta_half_width),
        })
        write_csv(outdir / "plume_enthalpy_balance.csv", enthalpy_balance_rows)



    # Fixed-eta plume mass and vertical-momentum fluxes at the requested profile planes.
    # These use exactly the same eta window as the enthalpy-balance integration.
    fixed_eta_mass_momentum_rows: List[Dict[str, float | int]] = []
    if eta_enabled and eta_rows:
        eta_hw_flux = float(args.energy_eta_half_width)
        for h in args.planes:
            rows_h = [rr for rr in eta_rows if abs(rr["height_m"] - h) < 1e-15]
            if not rows_h:
                continue
            xh = np.array([rr["x_m"] for rr in rows_h], dtype=float)
            etah = np.array([rr.get("eta", np.nan) for rr in rows_h], dtype=float)
            uyh = np.array([rr.get("uy_m_per_s", np.nan) for rr in rows_h], dtype=float)
            mask = np.isfinite(xh) & np.isfinite(etah) & np.isfinite(uyh) & (np.abs(etah) <= eta_hw_flux)
            if np.count_nonzero(mask) >= 2:
                order = np.argsort(xh[mask])
                xx = xh[mask][order]
                uu = uyh[mask][order]
                mass_signed = robust_trapz(float(args.rho) * uu, xx)
                mass_upward = robust_trapz(np.where(uu > 0.0, float(args.rho) * uu, 0.0), xx)
                mass_downward = robust_trapz(np.where(uu < 0.0, float(args.rho) * uu, 0.0), xx)
                mom_positive = robust_trapz(float(args.rho) * uu * uu, xx)
                mom_signed = robust_trapz(float(args.rho) * uu * np.abs(uu), xx)
                x_min_eta = float(np.nanmin(xx))
                x_max_eta = float(np.nanmax(xx))
            else:
                mass_signed = mass_upward = mass_downward = mom_positive = mom_signed = np.nan
                x_min_eta = x_max_eta = np.nan
            fixed_eta_mass_momentum_rows.append({
                "height_m": float(h),
                "eta_half_width": eta_hw_flux,
                "x_min_eta_window_m": x_min_eta,
                "x_max_eta_window_m": x_max_eta,
                "physical_width_eta_window_m": x_max_eta - x_min_eta if np.isfinite(x_min_eta) and np.isfinite(x_max_eta) else np.nan,
                "n_samples_in_eta_window": int(np.count_nonzero(mask)),
                "mass_flux_signed_kg_per_s_per_m": mass_signed,
                "mass_flux_upward_kg_per_s_per_m": mass_upward,
                "mass_flux_downward_kg_per_s_per_m": mass_downward,
                "vertical_momentum_flux_positive_N_per_m": mom_positive,
                "vertical_momentum_flux_signed_N_per_m": mom_signed,
            })
        write_csv(outdir / "fixed_eta_mass_momentum_fluxes.csv", fixed_eta_mass_momentum_rows)

        if fixed_eta_mass_momentum_rows:
            hh = np.array([r["height_m"] for r in fixed_eta_mass_momentum_rows], dtype=float)
            mm = np.array([r["mass_flux_signed_kg_per_s_per_m"] for r in fixed_eta_mass_momentum_rows], dtype=float)
            mom = np.array([r["vertical_momentum_flux_positive_N_per_m"] for r in fixed_eta_mass_momentum_rows], dtype=float)
            fig, ax1 = plt.subplots(figsize=plot_figsize)
            ax2 = ax1.twinx()
            l1, = ax1.plot(hh, mom, marker="o", label="vertical momentum flux")
            l2, = ax2.plot(hh, mm, marker="s", linestyle="--", label="mass flux")
            ax1.set_xlabel("height above wire [m]")
            ax1.set_ylabel("vertical momentum flux [N/m]")
            ax2.set_ylabel("mass flux [kg/(s m)]")
            maybe_set_ax_title(ax1, "Fixed-eta mass and vertical momentum flux", args.plot_titles)
            ax1.grid(True, alpha=0.35)
            ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="best")
            fig.tight_layout()
            fig.savefig(outdir / "fixed_eta_mass_momentum_fluxes.png", dpi=220)
            plt.close(fig)
    else:
        fixed_eta_mass_momentum_rows.append({
            "enabled": False,
            "reason": "requires eta profiles, hence --mu, --beta, and positive --q-input-per-length",
            "eta_half_width": float(args.energy_eta_half_width),
        })
        write_csv(outdir / "fixed_eta_mass_momentum_fluxes.csv", fixed_eta_mass_momentum_rows)

    # Cumulative selected-CV vertical momentum budget, accumulated from the top of the wire.
    # The terms are evaluated with the same eta/frozen-width sidewalls as the selected CV,
    # but the lower horizontal face is fixed at the upper cylinder surface and the upper face
    # is swept from there up to 0.09 m above the wire centre.
    cumulative_selected_momentum_rows: List[Dict[str, float | str | int | bool]] = []
    if (pi is not None and args.mu is not None and args.beta is not None and args.q_input_per_length is not None):
        try:
            cumulative_geom_base = momentum_cv_geometry_from_energy_settings(
                args, xmin, xmax, ymin, ymax, wire_y_m, wire_radius_m, thermal_bl_mean_m,
                rho=float(args.rho), cp=float(args.cp), mu=float(args.mu), beta=float(args.beta), g=float(args.g),
            )
            y_bottom_cum = max(float(wire_y_m + wire_radius_m), float(ymin) + 1e-10 * max(ymax - ymin, xmax - xmin, 1.0))
            y_top_limit = min(float(wire_y_m + args.energy_cv_y_top), float(ymax) - 1e-10 * max(ymax - ymin, xmax - xmin, 1.0))
            if y_top_limit <= y_bottom_cum:
                raise ValueError("0.09 m cumulative top is outside the available mesh above the wire top.")
            top_heights = np.linspace(y_bottom_cum - wire_y_m, y_top_limit - wire_y_m, 80)
            # Also include the user-requested planes that lie inside the cumulative interval.
            extra_heights = [float(h) for h in args.planes if (y_bottom_cum - wire_y_m) <= float(h) <= (y_top_limit - wire_y_m)]
            top_heights = np.unique(np.concatenate([top_heights, np.asarray(extra_heights, dtype=float)]))
            for ht in top_heights:
                geom_i = dict(cumulative_geom_base)
                geom_i["y_bottom_m"] = float(y_bottom_cum)
                geom_i["y_top_m"] = float(wire_y_m + ht)
                geom_i["y_bottom_requested_m"] = float(y_bottom_cum)
                geom_i["y_top_requested_m"] = float(wire_y_m + ht)
                if geom_i["y_top_m"] <= geom_i["y_bottom_m"]:
                    continue
                row_i = integrate_selected_control_volume_vertical_momentum(
                    Ti, uxi, uyi, pi, geom_i,
                    rho=float(args.rho), mu=float(args.mu), beta=float(args.beta), g=float(args.g), T_inf=float(args.T_inf),
                    n_side=max(101, min(int(args.energy_cv_n_boundary), 401)),
                    n_bottom_top=max(201, min(int(args.energy_cv_n_boundary), 601)),
                    n_area_y=81,
                )
                if not row_i:
                    continue
                adv_tb = float(row_i.get("advective_top_plus_bottom_N_per_m", np.nan))
                entrainment = float(row_i.get("advective_side_entrainment_N_per_m", np.nan))
                buoy = float(row_i.get("buoyancy_vertical_force_N_per_m", np.nan))
                pres = float(row_i.get("pressure_vertical_force_N_per_m", np.nan))
                visc = float(row_i.get("viscous_vertical_force_N_per_m", np.nan))
                cumulative_selected_momentum_rows.append({
                    "top_height_above_wire_m": float(ht),
                    "bottom_height_above_wire_m": float(y_bottom_cum - wire_y_m),
                    "increase_of_momentum_top_minus_bottom_N_per_m": adv_tb,
                    "buoyancy_N_per_m": buoy,
                    "entrainment_advective_side_flux_N_per_m": entrainment,
                    "pressure_force_N_per_m": pres,
                    "viscous_stress_force_N_per_m": visc,
                    "rhs_pressure_plus_viscous_plus_buoyancy_N_per_m": float(row_i.get("rhs_pressure_plus_viscous_plus_buoyancy_N_per_m", np.nan)),
                    "total_advective_boundary_flux_N_per_m": float(row_i.get("advective_vertical_momentum_flux_N_per_m", np.nan)),
                    "residual_N_per_m": float(row_i.get("momentum_balance_residual_N_per_m", np.nan)),
                    "eta_half_width": float(cumulative_geom_base.get("eta_half_width", np.nan)),
                    "minimum_half_width_m": float(cumulative_geom_base.get("minimum_half_width_m", np.nan)),
                    "width_mode_effective": str(cumulative_geom_base.get("width_mode_effective", "")),
                })
            write_csv(outdir / "selected_cv_momentum_cumulative.csv", cumulative_selected_momentum_rows)
            if cumulative_selected_momentum_rows:
                hc = np.array([r["top_height_above_wire_m"] for r in cumulative_selected_momentum_rows], dtype=float)
                plot_xy(outdir / "selected_cv_momentum_cumulative_terms.png", hc,
                        [("momentum gain", np.array([r["increase_of_momentum_top_minus_bottom_N_per_m"] for r in cumulative_selected_momentum_rows], dtype=float)),
                         ("buoyancy force", np.array([r["buoyancy_N_per_m"] for r in cumulative_selected_momentum_rows], dtype=float)),
                         ("entrainment", np.array([r["entrainment_advective_side_flux_N_per_m"] for r in cumulative_selected_momentum_rows], dtype=float)),
                         ("pressure force", np.array([r["pressure_force_N_per_m"] for r in cumulative_selected_momentum_rows], dtype=float)),
                         ("viscous stress force", np.array([r["viscous_stress_force_N_per_m"] for r in cumulative_selected_momentum_rows], dtype=float))],
                        "top height above wire [m]", "vertical momentum term [N/m]",
                        "Selected-CV cumulative vertical momentum terms",
                        figsize=plot_figsize, show_titles=args.plot_titles, dpi=220)
        except Exception as exc:
            cumulative_selected_momentum_rows.append({
                "enabled": False,
                "reason": str(exc),
                "description": "Cumulative selected-CV momentum plot was not computed.",
            })
            write_csv(outdir / "selected_cv_momentum_cumulative.csv", cumulative_selected_momentum_rows)
    else:
        missing = []
        if pi is None:
            missing.append("--pressure-xdmf")
        if args.mu is None:
            missing.append("--mu")
        if args.beta is None:
            missing.append("--beta")
        if args.q_input_per_length is None:
            missing.append("--q-input-per-length")
        cumulative_selected_momentum_rows.append({
            "enabled": False,
            "reason": "missing " + ", ".join(missing),
            "description": "Cumulative selected-CV momentum plot requires pressure, viscosity, buoyancy, and eta-width inputs.",
        })
        write_csv(outdir / "selected_cv_momentum_cumulative.csv", cumulative_selected_momentum_rows)

    def format_enthalpy_balance_summary(rows: List[Dict[str, float | str]]) -> str:
        if not rows or rows[0].get("enthalpy_balance_enabled", True) is False:
            return ("\nPlume enthalpy-flow balance\n"
                    "============================\n"
                    f"Disabled: {rows[0].get('reason', 'missing data') if rows else 'missing data'}\n")
        lines = []
        lines.append("\nPlume enthalpy-flow balance")
        lines.append("============================")
        lines.append("Definition: integral rho cp uy (T - T_inf) dx over |eta| <= eta_half_width. Units are W/m.")
        lines.append("")
        header = ("height[m]   eta_hw   width[m]   signed[W/m]   upward[W/m]   "
                  "signed-Qin[W/m]   signed dev[%]   upward dev[%]")
        lines.append(header)
        for rr in rows:
            lines.append(
                f"{rr['height_m']:9.4e}  {rr['eta_half_width']:6.2f}  "
                f"{rr['physical_width_eta_window_m']:9.3e}  "
                f"{rr['enthalpy_signed_W_per_m']:12.6e}  "
                f"{rr['enthalpy_upward_only_W_per_m']:12.6e}  "
                f"{rr['signed_minus_supplied_W_per_m']:14.6e}  "
                f"{rr['signed_deviation_percent']:12.3f}  "
                f"{rr['upward_deviation_percent']:12.3f}"
            )
        signed_err = np.array([rr.get("signed_deviation_percent", np.nan) for rr in rows], dtype=float)
        upward_err = np.array([rr.get("upward_deviation_percent", np.nan) for rr in rows], dtype=float)
        signed_err = signed_err[np.isfinite(signed_err)]
        upward_err = upward_err[np.isfinite(upward_err)]
        if signed_err.size:
            lines.append("")
            lines.append(f"Signed deviation: mean={np.mean(signed_err):.3f} %, min={np.min(signed_err):.3f} %, max={np.max(signed_err):.3f} %.")
        if upward_err.size:
            lines.append(f"Upward-only deviation: mean={np.mean(upward_err):.3f} %, min={np.min(upward_err):.3f} %, max={np.max(upward_err):.3f} %.")
        return "\n".join(lines) + "\n"

    enthalpy_balance_summary = format_enthalpy_balance_summary(enthalpy_balance_rows)
    print(enthalpy_balance_summary)

    # Closed energy control-volume budget.
    # Positive flux means outward through the selected control-volume boundary.
    def _sample_energy_flux_vector(xp: np.ndarray, yp: np.ndarray):
        TT = finite_or_nan(Ti(xp, yp))
        uu = finite_or_nan(uxi(xp, yp))
        vv = finite_or_nan(uyi(xp, yp))
        theta = TT - float(args.T_inf)

        if qx_i is not None and qy_i is not None:
            qcx = finite_or_nan(qx_i(xp, yp))
            qcy = finite_or_nan(qy_i(xp, yp))
        else:
            dTx, dTy = Ti.gradient(xp, yp)
            qcx = -float(args.k) * finite_or_nan(dTx)
            qcy = -float(args.k) * finite_or_nan(dTy)

        qvx = float(args.rho) * float(args.cp) * theta * uu
        qvy = float(args.rho) * float(args.cp) * theta * vv
        return qvx, qvy, qcx, qcy, TT, uu, vv

    def _eta_cv_half_width(yval: float, eta_hw: float, min_hw: float, fixed_hw: float, mode: str) -> Tuple[float, str, float]:
        if mode == "fixed":
            return float(fixed_hw), "fixed", np.nan
        if mode == "auto" and not (eta_enabled and np.isfinite(theta_line_source) and theta_line_source > 0.0 and np.isfinite(nu) and nu > 0.0):
            return float(fixed_hw), "auto_fallback_fixed", np.nan
        if mode in ("eta", "auto"):
            h_eta = float(yval) - float(eta_origin_y_m)
            if h_eta <= 0.0:
                return float(min_hw), "minimum_width_eta_undefined", np.nan
            Gr_h = float(args.g) * float(args.beta) * float(theta_line_source) * h_eta**3 / float(nu)**2
            if not np.isfinite(Gr_h) or Gr_h <= 0.0:
                return float(min_hw), "minimum_width_eta_invalid", np.nan
            x_eta = abs(float(eta_hw)) * h_eta / (Gr_h**0.2)
            if not np.isfinite(x_eta):
                return float(min_hw), "minimum_width_eta_invalid", np.nan
            if x_eta < float(min_hw):
                return float(min_hw), "minimum_width", float(x_eta)
            return float(x_eta), "eta", float(x_eta)
        return float(fixed_hw), "fixed", np.nan

    def _energy_cv_width_arrays(yvals: np.ndarray, mode: str):
        widths = np.empty_like(yvals, dtype=float)
        source = []
        xeta = np.empty_like(yvals, dtype=float)
        for ii, yyv in enumerate(yvals):
            widths[ii], s, xei = _eta_cv_half_width(
                float(yyv),
                eta_hw=float(args.energy_cv_eta_half_width),
                min_hw=float(args.energy_cv_min_half_width_m),
                fixed_hw=float(args.energy_cv_fixed_half_width),
                mode=str(args.energy_cv_width_mode),
            )
            source.append(s)
            xeta[ii] = xei
        return widths, source, xeta

    def _integrate_polyline_energy_boundary(name: str, xb: np.ndarray, yb: np.ndarray) -> Dict[str, float | str]:
        xb = np.asarray(xb, dtype=float)
        yb = np.asarray(yb, dtype=float)
        if xb.size < 2 or yb.size < 2:
            return {"boundary": name, "Q_total_W_per_m": np.nan, "Q_convective_W_per_m": np.nan, "Q_conductive_W_per_m": np.nan}

        # Segment-midpoint quadrature gives robust geometry normals for both straight and curved sides.
        x0, x1 = xb[:-1], xb[1:]
        y0, y1 = yb[:-1], yb[1:]
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)
        dxs = x1 - x0
        dys = y1 - y0

        # The boundary points are ordered counter-clockwise around the CV.
        # For a CCW contour the outward normal times segment length is (dy, -dx).
        nds_x = dys
        nds_y = -dxs

        qvx, qvy, qcx, qcy, TT, uu, vv = _sample_energy_flux_vector(xm, ym)
        conv_seg = qvx * nds_x + qvy * nds_y
        cond_seg = qcx * nds_x + qcy * nds_y
        valid_conv = np.isfinite(conv_seg)
        valid_cond = np.isfinite(cond_seg)
        valid_tot = valid_conv & valid_cond
        length = np.sqrt(dxs**2 + dys**2)

        return {
            "boundary": name,
            "Q_total_W_per_m": float(np.nansum(conv_seg[valid_tot] + cond_seg[valid_tot])) if np.any(valid_tot) else np.nan,
            "Q_convective_W_per_m": float(np.nansum(conv_seg[valid_conv])) if np.any(valid_conv) else np.nan,
            "Q_conductive_W_per_m": float(np.nansum(cond_seg[valid_cond])) if np.any(valid_cond) else np.nan,
            "positive_means": "out_of_control_volume",
            "n_segments": int(len(conv_seg)),
            "n_valid_total": int(np.count_nonzero(valid_tot)),
            "n_valid_convection": int(np.count_nonzero(valid_conv)),
            "n_valid_conduction": int(np.count_nonzero(valid_cond)),
            "valid_fraction_total": float(np.count_nonzero(valid_tot) / len(conv_seg)) if len(conv_seg) else np.nan,
            "valid_fraction_convection": float(np.count_nonzero(valid_conv) / len(conv_seg)) if len(conv_seg) else np.nan,
            "valid_fraction_conduction": float(np.count_nonzero(valid_cond) / len(conv_seg)) if len(cond_seg) else np.nan,
            "integrated_boundary_length_m": float(np.nansum(length)),
        }

    def _compute_energy_control_volume_budget() -> List[Dict[str, float | str]]:
        if not args.energy_cv:
            return [{
                "boundary": "disabled",
                "reason": "enable with --energy-cv",
            }]

        if args.energy_cv_y_bottom is not None:
            y_bottom_cv = wire_y_m + float(args.energy_cv_y_bottom)
            y_bottom_definition = "user_relative_to_wire_center"
        else:
            # Lower wire surface minus the angular mean thermal boundary-layer thickness.
            # This follows the user's intended near-source CV definition.
            bl = thermal_bl_mean_m if np.isfinite(thermal_bl_mean_m) else 0.0
            y_bottom_cv = wire_y_m - wire_radius_m - float(bl)
            y_bottom_definition = "lower_wire_surface_minus_angular_mean_thermal_bl"

        y_top_cv = wire_y_m + float(args.energy_cv_y_top)

        eps = 1e-10 * max(ymax - ymin, xmax - xmin, 1.0)
        y_bottom_req = float(y_bottom_cv)
        y_top_req = float(y_top_cv)
        y_bottom_cv = max(float(y_bottom_cv), ymin + eps)
        y_top_cv = min(float(y_top_cv), ymax - eps)
        if y_top_cv <= y_bottom_cv:
            raise SystemExit(f"Invalid energy CV: y_top={y_top_cv:g} <= y_bottom={y_bottom_cv:g} after clipping to mesh bounds.")

        mode_eff = str(args.energy_cv_width_mode)
        if mode_eff == "eta" and not eta_enabled:
            raise SystemExit("--energy-cv-width-mode eta requires eta scaling inputs: --mu, --beta, and positive --q-input-per-length.")
        if mode_eff == "auto" and not eta_enabled:
            mode_eff = "fixed"

        n = max(101, int(args.energy_cv_n_boundary))
        yside = np.linspace(y_bottom_cv, y_top_cv, n)
        widths, sources, xeta_vals = _energy_cv_width_arrays(yside, mode_eff)
        widths = np.minimum(widths, 0.999 * max(abs(xmin), abs(xmax)))
        widths = np.maximum(widths, 0.0)

        # Transition height: first y where the raw eta half-width reaches the minimum width.
        finite_xeta = np.isfinite(xeta_vals)
        transition_y = np.nan
        if np.any(finite_xeta & (xeta_vals >= float(args.energy_cv_min_half_width_m))):
            transition_y = float(yside[np.where(finite_xeta & (xeta_vals >= float(args.energy_cv_min_half_width_m)))[0][0]])

        w_bottom = float(widths[0])
        w_top = float(widths[-1])

        # Counter-clockwise boundary orientation:
        # bottom: right -> left; left: bottom -> top; top: left -> right; right: top -> bottom.
        xb_bottom = np.linspace(w_bottom, -w_bottom, n)
        yb_bottom = np.full_like(xb_bottom, y_bottom_cv)

        xb_left = -widths
        yb_left = yside

        xb_top = np.linspace(-w_top, w_top, n)
        yb_top = np.full_like(xb_top, y_top_cv)

        xb_right = widths[::-1]
        yb_right = yside[::-1]

        rows = [
            _integrate_polyline_energy_boundary("bottom", xb_bottom, yb_bottom),
            _integrate_polyline_energy_boundary("left", xb_left, yb_left),
            _integrate_polyline_energy_boundary("top", xb_top, yb_top),
            _integrate_polyline_energy_boundary("right", xb_right, yb_right),
        ]

        total_conv = np.nansum([float(r.get("Q_convective_W_per_m", np.nan)) for r in rows])
        total_cond = np.nansum([float(r.get("Q_conductive_W_per_m", np.nan)) for r in rows])
        total = total_conv + total_cond
        qin = float(args.q_input_per_length) if args.q_input_per_length is not None else np.nan

        meta = {
            "boundary": "total",
            "Q_total_W_per_m": float(total),
            "Q_convective_W_per_m": float(total_conv),
            "Q_conductive_W_per_m": float(total_cond),
            "positive_means": "out_of_control_volume",
            "supplied_heat_W_per_m": qin,
            "fraction_of_input_total": float(total / qin) if np.isfinite(qin) and qin != 0.0 else np.nan,
            "total_minus_input_W_per_m": float(total - qin) if np.isfinite(qin) else np.nan,
            "width_mode_requested": str(args.energy_cv_width_mode),
            "width_mode_effective": mode_eff,
            "eta_half_width": float(args.energy_cv_eta_half_width),
            "minimum_half_width_m": float(args.energy_cv_min_half_width_m),
            "minimum_full_width_m": 2.0 * float(args.energy_cv_min_half_width_m),
            "fixed_half_width_m": float(args.energy_cv_fixed_half_width),
            "y_bottom_requested_m": y_bottom_req,
            "y_bottom_effective_m": float(y_bottom_cv),
            "y_bottom_definition": y_bottom_definition,
            "y_bottom_height_above_wire_m": float(y_bottom_cv - wire_y_m),
            "y_top_requested_m": y_top_req,
            "y_top_effective_m": float(y_top_cv),
            "y_top_height_above_wire_m": float(y_top_cv - wire_y_m),
            "wire_center_y_m": float(wire_y_m),
            "wire_radius_m": float(wire_radius_m),
            "thermal_bl_mean_m": float(thermal_bl_mean_m) if np.isfinite(thermal_bl_mean_m) else np.nan,
            "eta_origin_y_m": float(eta_origin_y_m) if "eta_origin_y_m" in locals() else np.nan,
            "eta_origin_mode": str(eta_origin_mode) if "eta_origin_mode" in locals() else "",
            "eta_to_min_width_transition_y_m": transition_y,
            "eta_to_min_width_transition_height_above_wire_m": float(transition_y - wire_y_m) if np.isfinite(transition_y) else np.nan,
            "bottom_half_width_m": w_bottom,
            "top_half_width_m": w_top,
            "n_side_samples": int(n),
            "n_minimum_width_samples": int(sum(1 for s in sources if str(s).startswith("minimum"))),
            "n_eta_width_samples": int(sum(1 for s in sources if s == "eta")),
            "conductive_flux_source": "cell_centered_q_heat_field" if (qx_i is not None and qy_i is not None) else "temperature_gradient",
        }
        rows.append(meta)
        return rows

    energy_cv_rows = _compute_energy_control_volume_budget()
    write_csv(outdir / "energy_control_volume_budget.csv", energy_cv_rows)

    def format_energy_cv_summary(rows: List[Dict[str, float | str]]) -> str:
        if not rows or rows[0].get("boundary") == "disabled":
            return "\nEnergy control-volume budget\n============================\nDisabled; enable with --energy-cv.\n"
        out = ["\nEnergy control-volume budget",
               "============================",
               "Positive flux is outward through the selected CV boundary.",
               "boundary        total[W/m]      conv[W/m]       cond[W/m]      valid_total"]
        for rr in rows:
            if rr.get("boundary") == "total":
                continue
            out.append(f"{str(rr.get('boundary','')):10s}  {float(rr.get('Q_total_W_per_m', np.nan)):13.6e}  "
                       f"{float(rr.get('Q_convective_W_per_m', np.nan)):13.6e}  "
                       f"{float(rr.get('Q_conductive_W_per_m', np.nan)):13.6e}  "
                       f"{float(rr.get('valid_fraction_total', np.nan)):10.3f}")
        total_row = next((r for r in rows if r.get("boundary") == "total"), None)
        if total_row:
            out.append("")
            out.append(f"total outward heat flux = {float(total_row.get('Q_total_W_per_m', np.nan)):.6e} W/m")
            if np.isfinite(float(total_row.get("fraction_of_input_total", np.nan))):
                out.append(f"fraction of supplied heat = {float(total_row.get('fraction_of_input_total', np.nan)):.6f}")
            out.append(f"CV width mode = {total_row.get('width_mode_effective')}; minimum half-width = {float(total_row.get('minimum_half_width_m', np.nan)):.6e} m")
            out.append(f"eta/min-width transition height above wire = {float(total_row.get('eta_to_min_width_transition_height_above_wire_m', np.nan)):.6e} m")
        return "\\n".join(out) + "\\n"

    print(format_energy_cv_summary(energy_cv_rows))

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
            plt.figure(figsize=plot_figsize)
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
            maybe_set_title(f"{title_prefix}, h={h:g} m", args.plot_titles)
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
    temp_bl_power = fit_fixed_exponent_powerlaw(
        h_center, dT_c, -(3.0 / 5.0),
        temp_power["fit_height_min_m"], temp_power["fit_height_max_m"],
    )
    vel_bl_power = fit_fixed_exponent_powerlaw(
        h_center, uy_c, +(1.0 / 5.0),
        vel_power["fit_height_min_m"], vel_power["fit_height_max_m"],
    )

    temp_vo_power = {"C": np.nan, "exponent": np.nan, "r2": np.nan, "npoints": 0,
                     "fit_height_min_m": np.nan, "fit_height_max_m": np.nan}
    temp_vo_bl_power = dict(temp_vo_power)
    vel_vo_power = dict(temp_vo_power)
    vel_vo_bl_power = dict(temp_vo_power)
    if np.isfinite(fitT.get("y0", np.nan)) and np.isfinite(fitT.get("fit_y_min_m", np.nan)) and np.isfinite(fitT.get("fit_y_max_m", np.nan)):
        temp_vo_power = fit_loglog_powerlaw_about_origin(yy, dT_c, fitT["y0"], fitT["fit_y_min_m"], fitT["fit_y_max_m"])
        temp_vo_bl_power = fit_fixed_exponent_powerlaw_about_origin(yy, dT_c, fitT["y0"], -(3.0 / 5.0), fitT["fit_y_min_m"], fitT["fit_y_max_m"])
    if np.isfinite(fitU.get("y0", np.nan)) and np.isfinite(fitU.get("fit_y_min_m", np.nan)) and np.isfinite(fitU.get("fit_y_max_m", np.nan)):
        vel_vo_power = fit_loglog_powerlaw_about_origin(yy, uy_c, fitU["y0"], fitU["fit_y_min_m"], fitU["fit_y_max_m"])
        vel_vo_bl_power = fit_fixed_exponent_powerlaw_about_origin(yy, uy_c, fitU["y0"], +(1.0 / 5.0), fitU["fit_y_min_m"], fitU["fit_y_max_m"])

    write_csv(outdir / "centerline_loglog_powerlaw_fits.csv", [
        {"field": "temperature_centerline_height_from_wire_free_exponent", **temp_power},
        {"field": "temperature_centerline_height_from_wire_boundary_layer_exponent", **temp_bl_power},
        {"field": "velocity_centerline_height_from_wire_free_exponent", **vel_power},
        {"field": "velocity_centerline_height_from_wire_boundary_layer_exponent", **vel_bl_power},
        {"field": "temperature_centerline_virtual_origin_free_exponent", **temp_vo_power},
        {"field": "temperature_centerline_virtual_origin_boundary_layer_exponent", **temp_vo_bl_power},
        {"field": "velocity_centerline_virtual_origin_free_exponent", **vel_vo_power},
        {"field": "velocity_centerline_virtual_origin_boundary_layer_exponent", **vel_vo_bl_power},
    ])

    def plot_centerline_loglog(path: Path, h_axis: np.ndarray, amp: np.ndarray, free_fit: Dict[str, float],
                               bl_fit: Dict[str, float], ylabel: str, xlabel: str, title: str,
                               theory_exponent: float) -> None:
        plt.figure(figsize=plot_figsize)
        mask = np.isfinite(h_axis) & np.isfinite(amp) & (h_axis > 0.0) & (amp > 0.0)
        plt.loglog(h_axis[mask], amp[mask], label="numerical")
        hfit = h_axis[mask]
        if np.isfinite(free_fit.get("C", np.nan)) and np.isfinite(free_fit.get("exponent", np.nan)):
            w = (hfit >= free_fit["fit_height_min_m"]) & (hfit <= free_fit["fit_height_max_m"])
            if np.any(w):
                plt.loglog(hfit[w], free_fit["C"] * hfit[w] ** free_fit["exponent"],
                           linestyle="--", label=f"free fit: n={free_fit['exponent']:.3f}, R2={free_fit['r2']:.4f}")
                plt.axvspan(free_fit["fit_height_min_m"], free_fit["fit_height_max_m"], alpha=0.12, label="fit window")
        if np.isfinite(bl_fit.get("C", np.nan)):
            w = (hfit >= bl_fit["fit_height_min_m"]) & (hfit <= bl_fit["fit_height_max_m"])
            if np.any(w):
                plt.loglog(hfit[w], bl_fit["C"] * hfit[w] ** theory_exponent,
                           linestyle=":", linewidth=2.0, label=f"BL theory: n={theory_exponent:.3f}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(left=4e-3, right=5e-2)
        plt.ylim(bottom=2e-2)
        maybe_set_title(title, args.plot_titles)
        plt.grid(True, which="both", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()

    plot_centerline_loglog(outdir / "centerline_temperature_loglog_powerlaw.png", h_center, dT_c, temp_power, temp_bl_power,
                           r"$\Delta T_c$ [K]", "height above wire centre [m]",
                           "Centreline temperature decay on log-log axes", -(3.0 / 5.0))
    plot_centerline_loglog(outdir / "centerline_velocity_loglog_powerlaw.png", h_center, uy_c, vel_power, vel_bl_power,
                           r"$u_{y,c}$ [m/s]", "height above wire centre [m]",
                           "Centreline vertical velocity on log-log axes", +(1.0 / 5.0))

    if np.isfinite(fitT.get("y0", np.nan)):
        hT_vo = yy - fitT["y0"]
        plot_centerline_loglog(outdir / "virtual_origin_temperature_loglog.png", hT_vo, dT_c, temp_vo_power, temp_vo_bl_power,
                               r"$\Delta T_c$ [K]", r"$y-y_{0,T}$ [m]",
                               "Centreline temperature using fitted virtual origin", -(3.0 / 5.0))
    if np.isfinite(fitU.get("y0", np.nan)):
        hU_vo = yy - fitU["y0"]
        plot_centerline_loglog(outdir / "virtual_origin_velocity_loglog.png", hU_vo, uy_c, vel_vo_power, vel_vo_bl_power,
                               r"$u_{y,c}$ [m/s]", r"$y-y_{0,U}$ [m]",
                               "Centreline vertical velocity using fitted virtual origin", +(1.0 / 5.0))

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
        f.write("\nBoundary heat escape is integrated over exterior mesh facets; no cell-centred boundary extrapolation is used.\n")
        f.write(enthalpy_balance_summary)
        if args.energy_cv:
            f.write("\nEnergy control-volume budget is written to energy_control_volume_budget.csv.\n")
        f.write("\nMain outputs:\n")
        f.write("  plane_integrals.csv\n")
        f.write("  plane_profiles.csv\n")
        f.write("  plume_enthalpy_balance.csv\n")
        f.write("  fixed_eta_mass_momentum_fluxes.csv\n")
        f.write("  fixed_eta_mass_momentum_fluxes.png\n")
        f.write("  selected_cv_momentum_cumulative.csv\n")
        f.write("  selected_cv_momentum_cumulative_terms.png\n")
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
    print("Black-body / gray-body radiation estimate:")
    print(f"  emissivity = {float(radiation_summary.get('emissivity', np.nan)):.6g}")
    print(f"  mean sampled wire surface temperature = {float(radiation_summary.get('T_wire_surface_angular_mean_K', np.nan)):.6f} K")
    print(f"  wall/enclosure temperature = {float(radiation_summary.get('T_wall_K', np.nan)):.6f} K")
    print(f"  radiative heat loss = {float(radiation_summary.get('radiative_heat_per_length_W_per_m', np.nan)):.6e} W/m")
    print(f"  radiative fraction of input = {float(radiation_summary.get('radiative_fraction_of_input', np.nan)):.6e}")
    print(f"  convective/conductive remainder if subtracted = {float(radiation_summary.get('convective_conductive_remainder_if_subtracted_W_per_m', np.nan)):.6e} W/m")
    print("Black-body radiation estimate written to black_body_radiation_estimate.csv")
    print("Plume enthalpy-flow balance written to plume_enthalpy_balance.csv")
    print("Fixed-eta mass/momentum fluxes written to fixed_eta_mass_momentum_fluxes.csv")
    print("Cumulative selected-CV momentum terms written to selected_cv_momentum_cumulative.csv")
    if args.nusselt:
        print("Wire Nusselt/Biot diagnostics written to wire_nusselt_summary.csv and wire_nusselt_local.csv")
    if args.energy_cv:
        _energy_total_row = next((r for r in energy_cv_rows if r.get("boundary") == "total"), {})
        print(f"Energy control-volume budget written to energy_control_volume_budget.csv; total outward heat = {float(_energy_total_row.get('Q_total_W_per_m', np.nan)):.6e} W/m")


if __name__ == "__main__":
    main()
