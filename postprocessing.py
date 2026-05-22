#!/usr/bin/env python3
"""
Post-process saved plume temperature fields and convective enthalpy flux.

Implemented diagnostics:
  1. Q_conv_net / Q_conv_up / Q_conv_down
  2. uy_min / uy_mean / uy_max / ∫uy dx / ∫|uy| dx
  3. Multiple integration half-widths
  4. Final uy(x) and 10*ux(x) profile plots over the full requested line width
  5. Final CSV with T, theta=T-T_inf, ux, 10ux, uy, and h_flux density
  6. Temperature evolution plots use temperature excess T-T_inf
  7. Outward enthalpy-flow integrals on the full outer boundary

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python postprocess_temperature_convergence.py /path/to/results/folder \
        --workers 8 \
        --velocity-scale-factor 0.153657 \
        --T-inf 292.96

Expected temperature files:
    air_temperature_transient_*.h5

Expected velocity files:
    air_velocity_transient_*.h5

Expected HDF5 layout:
    Mesh/mesh/geometry
    Mesh/mesh/topology
    VisualisationVector/0

Temperature is expected to be dimensional [K].
Velocity is expected to be dimensional [m/s], after applying --velocity-scale-factor.

Mesh coordinates are expected to be nondimensional by Lref unless
the coordinate range already looks physical.
"""

from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def parse_step(path: Path, prefix: str) -> int:
    pattern = rf"{re.escape(prefix)}_(\d+)\.h5$"
    match = re.search(pattern, path.name)
    if not match:
        raise ValueError(f"Could not parse timestep from {path.name}")
    return int(match.group(1))


def read_xdmf_h5_function(path: Path):
    with h5py.File(path, "r") as h5:
        coords = np.asarray(h5["Mesh/mesh/geometry"], dtype=float)
        topology = np.asarray(h5["Mesh/mesh/topology"], dtype=np.int64)
        values = np.asarray(h5["VisualisationVector/0"], dtype=float)

    if coords.shape[1] == 3:
        coords = coords[:, :2]

    return coords, topology, values


def read_temperature_h5(path: Path):
    coords, topology, values = read_xdmf_h5_function(path)
    temperature = np.asarray(values, dtype=float).reshape(-1)
    return coords, topology, temperature


def read_velocity_h5(path: Path):
    coords, topology, values = read_xdmf_h5_function(path)

    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        if values.size % 2 != 0:
            raise RuntimeError(f"Velocity array in {path} is 1D but not divisible by 2.")
        velocity = values.reshape((-1, 2))
    elif values.ndim == 2:
        if values.shape[1] >= 2:
            velocity = values[:, :2]
        else:
            raise RuntimeError(f"Velocity array in {path} has shape {values.shape}.")
    else:
        raise RuntimeError(f"Velocity array in {path} has unsupported shape {values.shape}.")

    return coords, topology, velocity


def physical_to_mesh_coords(
    x_phys,
    y_phys,
    coordinates_are_nondim: bool,
    lref: float,
):
    if coordinates_are_nondim:
        return np.asarray(x_phys) / lref, np.asarray(y_phys) / lref

    return np.asarray(x_phys), np.asarray(y_phys)


def interpolate_scalar_on_line(
    coords,
    topology,
    scalar,
    x_phys,
    y_phys,
    coordinates_are_nondim: bool,
    lref: float,
):
    x_mesh, y_mesh = physical_to_mesh_coords(
        x_phys,
        y_phys,
        coordinates_are_nondim=coordinates_are_nondim,
        lref=lref,
    )

    triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1], topology)
    interpolator = mtri.LinearTriInterpolator(triangulation, scalar)

    sampled = interpolator(x_mesh, y_mesh)
    sampled = np.asarray(sampled.filled(np.nan), dtype=float)

    return sampled


def interpolate_vector_component_on_line(
    coords,
    topology,
    vector,
    component: int,
    x_phys,
    y_phys,
    coordinates_are_nondim: bool,
    lref: float,
):
    return interpolate_scalar_on_line(
        coords=coords,
        topology=topology,
        scalar=vector[:, component],
        x_phys=x_phys,
        y_phys=y_phys,
        coordinates_are_nondim=coordinates_are_nondim,
        lref=lref,
    )


def integrate_trapezoid_ignore_nan(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    valid = np.isfinite(y) & np.isfinite(x)
    if np.count_nonzero(valid) < 2:
        return np.nan

    return float(np.trapezoid(y[valid], x[valid]))


def add_plane_window_diagnostics(
    flux_row: dict,
    *,
    offset: float,
    half_width: float,
    x_line: np.ndarray,
    T_profile: np.ndarray,
    ux_profile: np.ndarray,
    uy_profile: np.ndarray,
    rho: float,
    cp: float,
    T_inf: float,
):
    window = np.abs(x_line) <= half_width

    xw = x_line[window]
    Tw = T_profile[window]
    uxw = ux_profile[window]
    uyw = uy_profile[window]

    h_flux_w = rho * cp * (Tw - T_inf) * uyw

    Q_conv_net = integrate_trapezoid_ignore_nan(h_flux_w, xw)
    Q_conv_up = integrate_trapezoid_ignore_nan(np.maximum(h_flux_w, 0.0), xw)
    Q_conv_down = integrate_trapezoid_ignore_nan(np.minimum(h_flux_w, 0.0), xw)

    int_uy = integrate_trapezoid_ignore_nan(uyw, xw)
    int_abs_uy = integrate_trapezoid_ignore_nan(np.abs(uyw), xw)

    int_ux = integrate_trapezoid_ignore_nan(uxw, xw)
    int_abs_ux = integrate_trapezoid_ignore_nan(np.abs(uxw), xw)

    key = f"y_plus_{offset:.3f}_m_halfwidth_{half_width:.3f}_m"

    flux_row[f"Q_conv_net_{key}_W_per_m"] = Q_conv_net
    flux_row[f"Q_conv_up_{key}_W_per_m"] = Q_conv_up
    flux_row[f"Q_conv_down_{key}_W_per_m"] = Q_conv_down

    flux_row[f"ux_min_{key}_m_per_s"] = float(np.nanmin(uxw))
    flux_row[f"ux_mean_{key}_m_per_s"] = float(np.nanmean(uxw))
    flux_row[f"ux_max_{key}_m_per_s"] = float(np.nanmax(uxw))
    flux_row[f"int_ux_{key}_m2_per_s"] = int_ux
    flux_row[f"int_abs_ux_{key}_m2_per_s"] = int_abs_ux
    flux_row[f"ux_net_fraction_{key}"] = abs(int_ux) / max(int_abs_ux, 1.0e-300)

    flux_row[f"uy_min_{key}_m_per_s"] = float(np.nanmin(uyw))
    flux_row[f"uy_mean_{key}_m_per_s"] = float(np.nanmean(uyw))
    flux_row[f"uy_max_{key}_m_per_s"] = float(np.nanmax(uyw))
    flux_row[f"int_uy_{key}_m2_per_s"] = int_uy
    flux_row[f"int_abs_uy_{key}_m2_per_s"] = int_abs_uy
    flux_row[f"uy_net_fraction_{key}"] = abs(int_uy) / max(int_abs_uy, 1.0e-300)



def _sample_boundary_side(
    *,
    coords,
    topology,
    temperature,
    velocity,
    side: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n_points: int,
    inset: float,
    coordinates_are_nondim: bool,
    lref: float,
):
    """
    Sample T and u just inside one physical outer boundary side.

    The returned normal is the outward normal of the physical rectangular domain.
    Sampling is inset by a small positive distance to avoid interpolation failures
    exactly on triangulation edges.
    """
    if side == "top":
        s_coord = np.linspace(x_min, x_max, n_points)
        x = s_coord
        y = np.full_like(x, y_max - inset)
        normal = np.array([0.0, 1.0])
    elif side == "bottom":
        s_coord = np.linspace(x_min, x_max, n_points)
        x = s_coord
        y = np.full_like(x, y_min + inset)
        normal = np.array([0.0, -1.0])
    elif side == "right":
        s_coord = np.linspace(y_min, y_max, n_points)
        x = np.full_like(s_coord, x_max - inset)
        y = s_coord
        normal = np.array([1.0, 0.0])
    elif side == "left":
        s_coord = np.linspace(y_min, y_max, n_points)
        x = np.full_like(s_coord, x_min + inset)
        y = s_coord
        normal = np.array([-1.0, 0.0])
    else:
        raise ValueError(f"Unknown boundary side {side!r}")

    T = interpolate_scalar_on_line(
        coords=coords,
        topology=topology,
        scalar=temperature,
        x_phys=x,
        y_phys=y,
        coordinates_are_nondim=coordinates_are_nondim,
        lref=lref,
    )
    ux = interpolate_vector_component_on_line(
        coords=coords,
        topology=topology,
        vector=velocity,
        component=0,
        x_phys=x,
        y_phys=y,
        coordinates_are_nondim=coordinates_are_nondim,
        lref=lref,
    )
    uy = interpolate_vector_component_on_line(
        coords=coords,
        topology=topology,
        vector=velocity,
        component=1,
        x_phys=x,
        y_phys=y,
        coordinates_are_nondim=coordinates_are_nondim,
        lref=lref,
    )
    un = normal[0] * ux + normal[1] * uy
    return s_coord, T, ux, uy, un


def add_outer_boundary_enthalpy_diagnostics(
    boundary_row: dict,
    *,
    coords,
    topology,
    temperature,
    velocity,
    coordinates_are_nondim: bool,
    lref: float,
    rho: float,
    cp: float,
    T_inf: float,
    boundary_n_points: int,
    boundary_inset: float,
):
    """
    Integrate outward convective enthalpy flux over the rectangular outer boundary.

        Q_h,out = ∮ rho*cp*(T - T_inf)*(u · n) ds

    Units are W/m, i.e. per unit out-of-plane wire length. Positive values mean
    heat leaves the computational room; negative values mean thermal enthalpy enters.
    """
    coords_phys = coords * lref if coordinates_are_nondim else coords
    x_min = float(np.nanmin(coords_phys[:, 0]))
    x_max = float(np.nanmax(coords_phys[:, 0]))
    y_min = float(np.nanmin(coords_phys[:, 1]))
    y_max = float(np.nanmax(coords_phys[:, 1]))

    if boundary_inset <= 0.0:
        # Keep the samples just inside the triangulation to avoid NaNs on edges.
        # This is intentionally small relative to the room size.
        room_size = min(x_max - x_min, y_max - y_min)
        boundary_inset = max(1.0e-6 * room_size, 1.0e-9)

    total_net = 0.0
    total_out_positive = 0.0
    total_in_negative = 0.0

    for side in ("top", "bottom", "left", "right"):
        s_coord, T, ux, uy, un = _sample_boundary_side(
            coords=coords,
            topology=topology,
            temperature=temperature,
            velocity=velocity,
            side=side,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            n_points=boundary_n_points,
            inset=boundary_inset,
            coordinates_are_nondim=coordinates_are_nondim,
            lref=lref,
        )

        hflux_out = rho * cp * (T - T_inf) * un
        q_net = integrate_trapezoid_ignore_nan(hflux_out, s_coord)
        q_out = integrate_trapezoid_ignore_nan(np.maximum(hflux_out, 0.0), s_coord)
        q_in = integrate_trapezoid_ignore_nan(np.minimum(hflux_out, 0.0), s_coord)

        boundary_row[f"Qh_out_{side}_net_W_per_m"] = q_net
        boundary_row[f"Qh_out_{side}_positive_W_per_m"] = q_out
        boundary_row[f"Qh_out_{side}_negative_W_per_m"] = q_in
        boundary_row[f"un_{side}_min_m_per_s"] = float(np.nanmin(un))
        boundary_row[f"un_{side}_mean_m_per_s"] = float(np.nanmean(un))
        boundary_row[f"un_{side}_max_m_per_s"] = float(np.nanmax(un))

        if np.isfinite(q_net):
            total_net += q_net
        if np.isfinite(q_out):
            total_out_positive += q_out
        if np.isfinite(q_in):
            total_in_negative += q_in

    boundary_row["Qh_out_boundary_total_net_W_per_m"] = total_net
    boundary_row["Qh_out_boundary_total_positive_W_per_m"] = total_out_positive
    boundary_row["Qh_out_boundary_total_negative_W_per_m"] = total_in_negative

def process_saved_step_worker(payload):
    """
    Worker for independent per-step post-processing.
    """
    (
        temperature_file,
        velocity_file,
        box_bounds,
        plane_offsets,
        x_line,
        wire_y,
        coordinates_are_nondim,
        lref,
        rho,
        cp,
        T_inf,
        flux_half_widths,
        velocity_scale_factor,
        boundary_n_points,
        boundary_inset,
    ) = payload

    temperature_file = Path(temperature_file)
    velocity_file = Path(velocity_file)

    step = parse_step(temperature_file, "air_temperature_transient")

    coords_T, topology_T, temperature = read_temperature_h5(temperature_file)
    coords_u, topology_u, velocity = read_velocity_h5(velocity_file)

    velocity = velocity_scale_factor * velocity

    if coords_T.shape != coords_u.shape:
        raise RuntimeError(
            f"Temperature/velocity coordinate shape mismatch at step {step}: "
            f"{coords_T.shape} vs {coords_u.shape}"
        )

    if topology_T.shape != topology_u.shape:
        raise RuntimeError(
            f"Temperature/velocity topology shape mismatch at step {step}: "
            f"{topology_T.shape} vs {topology_u.shape}"
        )

    # Build the convergence-box mask from this step's own mesh.
    # This is required when a results directory contains files from different meshes
    # (for example, pre-MPI and MPI runs written into the same folder).
    box_x_min, box_x_max, box_y_min, box_y_max = box_bounds
    if coordinates_are_nondim:
        coords_phys_for_box = coords_T * lref
    else:
        coords_phys_for_box = coords_T

    box_mask = (
        (coords_phys_for_box[:, 0] >= box_x_min)
        & (coords_phys_for_box[:, 0] <= box_x_max)
        & (coords_phys_for_box[:, 1] >= box_y_min)
        & (coords_phys_for_box[:, 1] <= box_y_max)
    )

    if int(np.count_nonzero(box_mask)) == 0:
        raise RuntimeError(f"No mesh nodes found inside convergence box at step {step}.")

    temp_box = temperature[box_mask]
    theta_box = temp_box - T_inf

    result = {
        "step": step,
        "temperature_file": str(temperature_file),
        "velocity_file": str(velocity_file),
        "box_T_min_K": float(np.nanmin(temp_box)),
        "box_T_mean_K": float(np.nanmean(temp_box)),
        "box_T_max_K": float(np.nanmax(temp_box)),
        "box_theta_min_K": float(np.nanmin(theta_box)),
        "box_theta_mean_K": float(np.nanmean(theta_box)),
        "box_theta_max_K": float(np.nanmax(theta_box)),
        "T_box": temp_box,
    }

    peak_row = {"step": step}
    flux_row = {"step": step}
    boundary_row = {"step": step}

    add_outer_boundary_enthalpy_diagnostics(
        boundary_row,
        coords=coords_T,
        topology=topology_T,
        temperature=temperature,
        velocity=velocity,
        coordinates_are_nondim=coordinates_are_nondim,
        lref=lref,
        rho=rho,
        cp=cp,
        T_inf=T_inf,
        boundary_n_points=boundary_n_points,
        boundary_inset=boundary_inset,
    )

    for offset in plane_offsets:
        y_line = np.full_like(x_line, wire_y + offset)

        T_profile = interpolate_scalar_on_line(
            coords=coords_T,
            topology=topology_T,
            scalar=temperature,
            x_phys=x_line,
            y_phys=y_line,
            coordinates_are_nondim=coordinates_are_nondim,
            lref=lref,
        )

        ux_profile = interpolate_vector_component_on_line(
            coords=coords_u,
            topology=topology_u,
            vector=velocity,
            component=0,
            x_phys=x_line,
            y_phys=y_line,
            coordinates_are_nondim=coordinates_are_nondim,
            lref=lref,
        )

        uy_profile = interpolate_vector_component_on_line(
            coords=coords_u,
            topology=topology_u,
            vector=velocity,
            component=1,
            x_phys=x_line,
            y_phys=y_line,
            coordinates_are_nondim=coordinates_are_nondim,
            lref=lref,
        )

        theta_profile = T_profile - T_inf

        peak_row[f"theta_peak_y_plus_{offset:.3f}_m_K"] = float(np.nanmax(theta_profile))
        peak_row[f"theta_min_y_plus_{offset:.3f}_m_K"] = float(np.nanmin(theta_profile))
        peak_row[f"theta_mean_y_plus_{offset:.3f}_m_K"] = float(np.nanmean(theta_profile))

        peak_row[f"T_peak_y_plus_{offset:.3f}_m_K"] = float(np.nanmax(T_profile))
        peak_row[f"T_min_y_plus_{offset:.3f}_m_K"] = float(np.nanmin(T_profile))
        peak_row[f"T_mean_y_plus_{offset:.3f}_m_K"] = float(np.nanmean(T_profile))

        for half_width in flux_half_widths:
            add_plane_window_diagnostics(
                flux_row,
                offset=offset,
                half_width=half_width,
                x_line=x_line,
                T_profile=T_profile,
                ux_profile=ux_profile,
                uy_profile=uy_profile,
                rho=rho,
                cp=cp,
                T_inf=T_inf,
            )

    result["peak_row"] = peak_row
    result["flux_row"] = flux_row
    result["boundary_row"] = boundary_row

    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pair_temperature_velocity_files(results_dir: Path):
    temperature_files = sorted(
        results_dir.glob("air_temperature_transient_*.h5"),
        key=lambda p: parse_step(p, "air_temperature_transient"),
    )

    velocity_files = sorted(
        results_dir.glob("air_velocity_transient_*.h5"),
        key=lambda p: parse_step(p, "air_velocity_transient"),
    )

    if not temperature_files:
        raise FileNotFoundError(
            f"No air_temperature_transient_*.h5 files found in {results_dir}"
        )

    if not velocity_files:
        raise FileNotFoundError(
            f"No air_velocity_transient_*.h5 files found in {results_dir}"
        )

    temperature_by_step = {
        parse_step(path, "air_temperature_transient"): path for path in temperature_files
    }

    velocity_by_step = {
        parse_step(path, "air_velocity_transient"): path for path in velocity_files
    }

    common_steps = sorted(set(temperature_by_step) & set(velocity_by_step))

    missing_velocity = sorted(set(temperature_by_step) - set(velocity_by_step))
    missing_temperature = sorted(set(velocity_by_step) - set(temperature_by_step))

    if missing_velocity:
        print(
            f"Warning: {len(missing_velocity)} temperature steps have no matching velocity file. "
            f"First few: {missing_velocity[:10]}"
        )

    if missing_temperature:
        print(
            f"Warning: {len(missing_temperature)} velocity steps have no matching temperature file. "
            f"First few: {missing_temperature[:10]}"
        )

    if not common_steps:
        raise RuntimeError("No matching temperature/velocity timesteps found.")

    return [
        (step, temperature_by_step[step], velocity_by_step[step])
        for step in common_steps
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process plume temperature convergence and enthalpy flux."
    )

    parser.add_argument(
        "results_dir",
        type=Path,
        help="Folder containing air_temperature_transient_*.h5 and air_velocity_transient_*.h5 files.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for reading/interpolating saved fields.",
    )

    parser.add_argument(
        "--lref",
        type=float,
        default=3.75e-5,
        help="Length scale [m]. Default is Brodowicz wire radius.",
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=3.75e-5,
        help="Wire radius [m].",
    )

    parser.add_argument(
        "--domain-height",
        type=float,
        default=1.0,
        help="Domain height h [m], used only if --wire-y is not provided.",
    )

    parser.add_argument(
        "--wire-y",
        type=float,
        default=None,
        help="Wire center height [m]. If omitted, uses h/10 + 11R.",
    )

    parser.add_argument(
        "--box-half-width",
        type=float,
        default=0.20,
        help="Convergence box half-width around wire centerline [m].",
    )

    parser.add_argument(
        "--box-below-wire",
        type=float,
        default=0.02,
        help="Convergence box starts this distance below wire center [m].",
    )

    parser.add_argument(
        "--box-height",
        type=float,
        default=0.20,
        help="Convergence box height [m].",
    )

    parser.add_argument(
        "--line-half-width",
        type=float,
        default=None,
        help="Half-width of extracted horizontal profiles [m]. Defaults to --box-half-width.",
    )

    parser.add_argument(
        "--num-line-points",
        type=int,
        default=1000,
        help="Number of sample points per horizontal profile.",
    )

    parser.add_argument(
        "--plane-offsets",
        type=float,
        nargs="+",
        default=[0.01, 0.04, 0.08],
        help="Horizontal plane offsets above wire center [m].",
    )

    parser.add_argument(
        "--flux-half-widths",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.04, 0.08, 0.20],
        help="Half-widths [m] used for convective enthalpy flux integrals.",
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=1.1614,
        help="Density [kg/m^3].",
    )

    parser.add_argument(
        "--cp",
        type=float,
        default=1007.0,
        help="Specific heat capacity [J/(kg K)].",
    )

    parser.add_argument(
        "--T-inf",
        type=float,
        default=293.15,
        help="Ambient temperature [K].",
    )

    parser.add_argument(
        "--velocity-scale-factor",
        type=float,
        default=1.0,
        help="Optional multiplier applied to saved velocity before diagnostics.",
    )

    parser.add_argument(
        "--input-line-power",
        type=float,
        default=9.75,
        help="Input line power [W/m] used as horizontal reference in flux plots.",
    )

    parser.add_argument(
        "--boundary-n-points",
        type=int,
        default=1200,
        help="Number of samples per outer-boundary side for enthalpy-flow integrals.",
    )

    parser.add_argument(
        "--boundary-inset",
        type=float,
        default=0.0,
        help=(
            "Physical inset [m] used when sampling the outer boundary. "
            "Use 0 to choose a small automatic inset."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for CSV and PNG output. Defaults to results_dir/postprocess_temperature.",
    )

    parser.add_argument(
        "--plot-label-font-size",
        type=float,
        default=12.0,
        help="Font size for x/y axis labels in generated plots.",
    )

    parser.add_argument(
        "--plot-tick-font-size",
        type=float,
        default=11.0,
        help="Font size for tick labels in generated plots.",
    )

    parser.add_argument(
        "--plot-legend-font-size",
        type=float,
        default=8.5,
        help="Font size for plot legends.",
    )

    parser.add_argument(
        "--plot-figure-width",
        type=float,
        default=3.3,
        help="Figure width in inches. 3.0--3.4 in is suitable for two figures side by side on A4.",
    )

    parser.add_argument(
        "--plot-figure-height",
        type=float,
        default=2.45,
        help="Figure height in inches.",
    )

    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=300,
        help="Resolution used when saving PNG figures.",
    )

    args = parser.parse_args()

    plt.rcParams.update({
        "figure.figsize": (args.plot_figure_width, args.plot_figure_height),
        "figure.dpi": args.plot_dpi,
        "savefig.dpi": args.plot_dpi,
        "axes.labelsize": args.plot_label_font_size,
        "xtick.labelsize": args.plot_tick_font_size,
        "ytick.labelsize": args.plot_tick_font_size,
        "legend.fontsize": args.plot_legend_font_size,
        "font.size": args.plot_label_font_size,
    })

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    if args.line_half_width is None:
        args.line_half_width = args.box_half_width

    if max(args.flux_half_widths) > args.line_half_width:
        raise ValueError(
            "max(--flux-half-widths) must be <= --line-half-width. "
            f"Got max flux half-width {max(args.flux_half_widths)} and "
            f"line half-width {args.line_half_width}."
        )

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "postprocess_temperature"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_temperature_velocity_files(results_dir)

    wire_y = args.wire_y
    if wire_y is None:
        wire_y = args.domain_height / 10.0 + 11.0 * args.radius

    box_x_min = -args.box_half_width
    box_x_max = args.box_half_width
    box_y_min = wire_y - args.box_below_wire
    box_y_max = box_y_min + args.box_height

    print("Post-processing temperature fields and convective enthalpy flux")
    print(f"  results_dir = {results_dir}")
    print(f"  output_dir  = {output_dir}")
    print(f"  paired files = {len(pairs)}")
    print(f"  workers     = {args.workers}")
    print(f"  Lref        = {args.lref:.8e} m")
    print(f"  wire_y      = {wire_y:.8e} m")
    print(f"  rho         = {args.rho:.8e} kg/m^3")
    print(f"  cp          = {args.cp:.8e} J/(kg K)")
    print(f"  T_inf       = {args.T_inf:.8e} K")
    print(f"  velocity_scale_factor = {args.velocity_scale_factor:.8e}")
    print(f"  line_half_width = {args.line_half_width:.8e} m")
    print(f"  flux_half_widths = {args.flux_half_widths}")
    print(f"  boundary_n_points = {args.boundary_n_points}")
    print(f"  boundary_inset = {args.boundary_inset:.8e} m")
    print("  convergence box [physical]:")
    print(f"    x = [{box_x_min:.6e}, {box_x_max:.6e}] m")
    print(f"    y = [{box_y_min:.6e}, {box_y_max:.6e}] m")

    _, first_temperature_file, _ = pairs[0]
    first_coords, first_topology, first_temperature = read_temperature_h5(first_temperature_file)

    coordinates_are_nondim = np.nanmax(np.abs(first_coords)) > 10.0
    print(f"  coordinates_are_nondim = {coordinates_are_nondim}")

    if coordinates_are_nondim:
        coords_phys = first_coords * args.lref
    else:
        coords_phys = first_coords.copy()

    first_box_mask = (
        (coords_phys[:, 0] >= box_x_min)
        & (coords_phys[:, 0] <= box_x_max)
        & (coords_phys[:, 1] >= box_y_min)
        & (coords_phys[:, 1] <= box_y_max)
    )

    n_box = int(np.count_nonzero(first_box_mask))
    if n_box == 0:
        raise RuntimeError("No mesh nodes found inside the requested convergence box in the first file.")

    print(f"  nodes in convergence box in first file = {n_box}")
    print("  mixed-mesh safe mode: convergence-box mask is rebuilt per file")

    box_bounds = (box_x_min, box_x_max, box_y_min, box_y_max)

    x_line = np.linspace(
        -args.line_half_width,
        args.line_half_width,
        args.num_line_points,
    )

    payloads = [
        (
            str(temperature_file),
            str(velocity_file),
            box_bounds,
            args.plane_offsets,
            x_line,
            wire_y,
            coordinates_are_nondim,
            args.lref,
            args.rho,
            args.cp,
            args.T_inf,
            args.flux_half_widths,
            args.velocity_scale_factor,
            args.boundary_n_points,
            args.boundary_inset,
        )
        for _, temperature_file, velocity_file in pairs
    ]

    if args.workers > 1:
        print("  processing files in parallel...")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            processed = list(executor.map(process_saved_step_worker, payloads))
    else:
        print("  processing files serially...")
        processed = [process_saved_step_worker(payload) for payload in payloads]

    processed = sorted(processed, key=lambda row: row["step"])

    convergence_rows = []
    plane_peak_rows = []
    enthalpy_flux_rows = []
    boundary_enthalpy_rows = []

    previous_T_box = None
    previous_step = None

    for row in processed:
        step = row["step"]
        T_box = row["T_box"]

        if previous_T_box is None:
            l2_update = np.nan
            rel_l2_update = np.nan
            linf_update = np.nan
            previous_step_out = ""
        elif T_box.shape != previous_T_box.shape:
            # Mixed meshes can have different numbers of nodes in the convergence box.
            # A nodewise update norm is not meaningful unless both vectors live on the same mesh.
            l2_update = np.nan
            rel_l2_update = np.nan
            linf_update = np.nan
            previous_step_out = previous_step
        else:
            delta_box = T_box - previous_T_box

            l2_update = float(np.linalg.norm(delta_box))
            rel_l2_update = float(
                np.linalg.norm(delta_box) / max(np.linalg.norm(T_box), 1.0e-300)
            )
            linf_update = float(np.nanmax(np.abs(delta_box)))
            previous_step_out = previous_step

        convergence_rows.append(
            {
                "step": step,
                "previous_step": previous_step_out,
                "box_T_min_K": row["box_T_min_K"],
                "box_T_mean_K": row["box_T_mean_K"],
                "box_T_max_K": row["box_T_max_K"],
                "box_theta_min_K": row["box_theta_min_K"],
                "box_theta_mean_K": row["box_theta_mean_K"],
                "box_theta_max_K": row["box_theta_max_K"],
                "box_l2_update_K": l2_update,
                "box_rel_l2_update": rel_l2_update,
                "box_linf_update_K": linf_update,
            }
        )

        plane_peak_rows.append(row["peak_row"])
        enthalpy_flux_rows.append(row["flux_row"])
        boundary_enthalpy_rows.append(row["boundary_row"])

        previous_T_box = T_box
        previous_step = step

    convergence_csv = output_dir / "temperature_convergence_box.csv"
    peaks_csv = output_dir / "temperature_plane_peaks.csv"
    enthalpy_flux_csv = output_dir / "enthalpy_flux_planes.csv"
    boundary_enthalpy_csv = output_dir / "enthalpy_flux_outer_boundary.csv"

    write_csv(convergence_csv, convergence_rows)
    write_csv(peaks_csv, plane_peak_rows)
    write_csv(enthalpy_flux_csv, enthalpy_flux_rows)
    write_csv(boundary_enthalpy_csv, boundary_enthalpy_rows)

    print(f"  wrote {convergence_csv}")
    print(f"  wrote {peaks_csv}")
    print(f"  wrote {enthalpy_flux_csv}")
    print(f"  wrote {boundary_enthalpy_csv}")

    steps = np.asarray([row["step"] for row in convergence_rows], dtype=float)

    rel_updates = np.asarray(
        [row["box_rel_l2_update"] for row in convergence_rows],
        dtype=float,
    )

    linf_updates = np.asarray(
        [row["box_linf_update_K"] for row in convergence_rows],
        dtype=float,
    )

    box_theta_max = np.asarray(
        [row["box_theta_max_K"] for row in convergence_rows],
        dtype=float,
    )

    plt.figure()
    plt.semilogy(steps, rel_updates)
    plt.xlabel("Saved step")
    plt.ylabel("Relative L2 update in box")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_box_relative_l2_update.png", dpi=300)
    plt.close()

    plt.figure()
    plt.semilogy(steps, linf_updates)
    plt.xlabel("Saved step")
    plt.ylabel("L∞ update in box [K]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_box_linf_update.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(steps, box_theta_max)
    plt.xlabel("Saved step")
    plt.ylabel(r"Maximum temperature excess in box $\max(T-T_\infty)$ [K]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_excess_box_peak.png", dpi=300)
    plt.close()

    peak_steps = np.asarray([row["step"] for row in plane_peak_rows], dtype=float)

    plt.figure()
    for offset in args.plane_offsets:
        key = f"theta_peak_y_plus_{offset:.3f}_m_K"
        values = np.asarray([row[key] for row in plane_peak_rows], dtype=float)
        plt.plot(
            peak_steps,
            values,
            label=f"y = {offset * 100:.0f} cm above wire",
        )

    plt.xlabel("Saved step")
    plt.ylabel(r"Peak temperature excess $\max(T-T_\infty)$ [K]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_excess_plane_peak_evolution.png", dpi=300)
    plt.close()

    flux_steps = np.asarray([row["step"] for row in enthalpy_flux_rows], dtype=float)

    for offset in args.plane_offsets:
        plt.figure()

        for half_width in args.flux_half_widths:
            key = (
                f"Q_conv_net_y_plus_{offset:.3f}_m_"
                f"halfwidth_{half_width:.3f}_m_W_per_m"
            )
            values = np.asarray([row[key] for row in enthalpy_flux_rows], dtype=float)
            plt.plot(
                flux_steps,
                values,
                label=f"net, ±{half_width * 100:.0f} cm",
            )

        plt.axhline(
            args.input_line_power,
            linestyle="--",
            linewidth=1.0,
            label=f"input QL = {args.input_line_power:g} W/m",
        )
        plt.xlabel("Saved step")
        plt.ylabel(r"$Q_{conv,net}$ [W/m]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"enthalpy_flux_net_window_evolution_y_plus_{offset:.3f}_m.png",
            dpi=300,
        )
        plt.close()

        plt.figure()

        for half_width in args.flux_half_widths:
            key_up = (
                f"Q_conv_up_y_plus_{offset:.3f}_m_"
                f"halfwidth_{half_width:.3f}_m_W_per_m"
            )
            key_down = (
                f"Q_conv_down_y_plus_{offset:.3f}_m_"
                f"halfwidth_{half_width:.3f}_m_W_per_m"
            )

            values_up = np.asarray(
                [row[key_up] for row in enthalpy_flux_rows],
                dtype=float,
            )
            values_down = np.asarray(
                [row[key_down] for row in enthalpy_flux_rows],
                dtype=float,
            )

            plt.plot(
                flux_steps,
                values_up,
                label=f"up, ±{half_width * 100:.0f} cm",
            )
            plt.plot(
                flux_steps,
                values_down,
                linestyle="--",
                label=f"down, ±{half_width * 100:.0f} cm",
            )

        plt.axhline(
            args.input_line_power,
            linestyle=":",
            linewidth=1.0,
            label=f"input QL = {args.input_line_power:g} W/m",
        )
        plt.xlabel("Saved step")
        plt.ylabel(r"$Q_{conv,up/down}$ [W/m]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"enthalpy_flux_up_down_window_evolution_y_plus_{offset:.3f}_m.png",
            dpi=300,
        )
        plt.close()

        plt.figure()

        for half_width in args.flux_half_widths:
            key = (
                f"uy_net_fraction_y_plus_{offset:.3f}_m_"
                f"halfwidth_{half_width:.3f}_m"
            )
            values = np.asarray([row[key] for row in enthalpy_flux_rows], dtype=float)
            plt.plot(
                flux_steps,
                values,
                label=f"±{half_width * 100:.0f} cm",
            )

        plt.xlabel("Saved step")
        plt.ylabel(r"$|\int u_y dx| / \int |u_y| dx$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"uy_net_fraction_window_evolution_y_plus_{offset:.3f}_m.png",
            dpi=300,
        )
        plt.close()


    boundary_steps = np.asarray([row["step"] for row in boundary_enthalpy_rows], dtype=float)

    plt.figure()
    for side in ("top", "bottom", "left", "right"):
        values = np.asarray(
            [row[f"Qh_out_{side}_net_W_per_m"] for row in boundary_enthalpy_rows],
            dtype=float,
        )
        plt.plot(boundary_steps, values, label=side)

    total_values = np.asarray(
        [row["Qh_out_boundary_total_net_W_per_m"] for row in boundary_enthalpy_rows],
        dtype=float,
    )
    plt.plot(boundary_steps, total_values, linewidth=2.0, label="total boundary")
    plt.axhline(args.input_line_power, linestyle="--", linewidth=1.0, label=f"input QL = {args.input_line_power:g} W/m")
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("Saved step")
    plt.ylabel(r"$\oint \rho c_p (T-T_\infty)(\mathbf{u}\cdot\mathbf{n})\,ds$ [W/m]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "enthalpy_flux_outer_boundary_evolution.png", dpi=300)
    plt.close()

    plt.figure()
    positive_values = np.asarray(
        [row["Qh_out_boundary_total_positive_W_per_m"] for row in boundary_enthalpy_rows],
        dtype=float,
    )
    negative_values = np.asarray(
        [row["Qh_out_boundary_total_negative_W_per_m"] for row in boundary_enthalpy_rows],
        dtype=float,
    )
    plt.plot(boundary_steps, positive_values, label="outward positive parts")
    plt.plot(boundary_steps, negative_values, linestyle="--", label="inward negative parts")
    plt.plot(boundary_steps, total_values, linewidth=2.0, label="net")
    plt.axhline(args.input_line_power, linestyle=":", linewidth=1.0, label=f"input QL = {args.input_line_power:g} W/m")
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("Saved step")
    plt.ylabel(r"Boundary enthalpy flow [W/m]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "enthalpy_flux_outer_boundary_positive_negative_evolution.png", dpi=300)
    plt.close()

    final_step = processed[-1]["step"]
    final_temperature_file = Path(processed[-1]["temperature_file"])
    final_velocity_file = Path(processed[-1]["velocity_file"])

    final_coords_T, final_topology_T, final_temperature = read_temperature_h5(
        final_temperature_file
    )

    final_coords_u, final_topology_u, final_velocity = read_velocity_h5(
        final_velocity_file
    )
    final_velocity = args.velocity_scale_factor * final_velocity

    final_temperature_profiles_csv = output_dir / f"temperature_profiles_step_{final_step}.csv"
    final_combined_profiles_csv = output_dir / f"final_profiles_T_theta_ux_uy_hflux_step_{final_step}.csv"

    temperature_profile_columns = {"x_m": x_line}
    combined_profile_columns = {"x_m": x_line}

    plt.figure()
    for offset in args.plane_offsets:
        y_line = np.full_like(x_line, wire_y + offset)

        T_profile = interpolate_scalar_on_line(
            coords=final_coords_T,
            topology=final_topology_T,
            scalar=final_temperature,
            x_phys=x_line,
            y_phys=y_line,
            coordinates_are_nondim=coordinates_are_nondim,
            lref=args.lref,
        )

        theta_profile = T_profile - args.T_inf

        temperature_profile_columns[f"T_y_plus_{offset:.3f}_m_K"] = T_profile
        temperature_profile_columns[f"theta_y_plus_{offset:.3f}_m_K"] = theta_profile

        combined_profile_columns[f"T_y_plus_{offset:.3f}_m_K"] = T_profile
        combined_profile_columns[f"theta_y_plus_{offset:.3f}_m_K"] = theta_profile

        plt.plot(
            x_line,
            theta_profile,
            label=f"y = {offset * 100:.0f} cm above wire",
        )

    plt.xlabel("x [m]")
    plt.ylabel(r"Temperature excess $T-T_\infty$ [K]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"temperature_excess_profiles_step_{final_step}.png", dpi=300)
    plt.close()

    plt.figure()
    for offset in args.plane_offsets:
        y_line = np.full_like(x_line, wire_y + offset)

        ux_profile = interpolate_vector_component_on_line(
            coords=final_coords_u,
            topology=final_topology_u,
            vector=final_velocity,
            component=0,
            x_phys=x_line,
            y_phys=y_line,
            coordinates_are_nondim=coordinates_are_nondim,
            lref=args.lref,
        )

        uy_profile = interpolate_vector_component_on_line(
            coords=final_coords_u,
            topology=final_topology_u,
            vector=final_velocity,
            component=1,
            x_phys=x_line,
            y_phys=y_line,
            coordinates_are_nondim=coordinates_are_nondim,
            lref=args.lref,
        )

        combined_profile_columns[f"ux_y_plus_{offset:.3f}_m_m_per_s"] = ux_profile
        combined_profile_columns[f"10ux_y_plus_{offset:.3f}_m_m_per_s"] = 10.0 * np.abs(ux_profile)
        combined_profile_columns[f"uy_y_plus_{offset:.3f}_m_m_per_s"] = uy_profile

        plt.plot(
            x_line,
            uy_profile,
            label=f"$u_y$, y = {offset * 100:.0f} cm",
        )

        plt.plot(
            x_line,
            10.0 * np.abs(ux_profile),
            linestyle="--",
            label=f"$10u_x$, y = {offset * 100:.0f} cm",
        )

    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("x [m]")
    plt.ylabel(r"Velocity [m/s], with $10u_x$ scaled")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"velocity_profiles_uy_and_10ux_step_{final_step}.png", dpi=300)
    plt.close()

    plt.figure()
    for offset in args.plane_offsets:
        y_line = np.full_like(x_line, wire_y + offset)

        T_profile = interpolate_scalar_on_line(
            coords=final_coords_T,
            topology=final_topology_T,
            scalar=final_temperature,
            x_phys=x_line,
            y_phys=y_line,
            coordinates_are_nondim=coordinates_are_nondim,
            lref=args.lref,
        )

        ux_profile = interpolate_vector_component_on_line(
            coords=final_coords_u,
            topology=final_topology_u,
            vector=final_velocity,
            component=0,
            x_phys=x_line,
            y_phys=y_line,
            coordinates_are_nondim=coordinates_are_nondim,
            lref=args.lref,
        )

        uy_profile = interpolate_vector_component_on_line(
            coords=final_coords_u,
            topology=final_topology_u,
            vector=final_velocity,
            component=1,
            x_phys=x_line,
            y_phys=y_line,
            coordinates_are_nondim=coordinates_are_nondim,
            lref=args.lref,
        )

        theta_profile = T_profile - args.T_inf
        h_flux_density = args.rho * args.cp * theta_profile * uy_profile

        combined_profile_columns[
            f"hflux_y_plus_{offset:.3f}_m_W_per_m2"
        ] = h_flux_density

        plt.plot(
            x_line,
            h_flux_density,
            label=f"y = {offset * 100:.0f} cm above wire",
        )

    plt.xlabel("x [m]")
    plt.ylabel(r"$\rho c_p (T - T_\infty) u_y$ [W/m²]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"enthalpy_flux_density_profiles_step_{final_step}.png",
        dpi=300,
    )
    plt.close()

    with final_temperature_profiles_csv.open("w", newline="") as f:
        fieldnames = list(temperature_profile_columns.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(x_line)):
            writer.writerow(
                {key: temperature_profile_columns[key][i] for key in fieldnames}
            )

    with final_combined_profiles_csv.open("w", newline="") as f:
        fieldnames = list(combined_profile_columns.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(x_line)):
            writer.writerow(
                {key: combined_profile_columns[key][i] for key in fieldnames}
            )

    print(f"  wrote {final_temperature_profiles_csv}")
    print(f"  wrote {final_combined_profiles_csv}")
    print("Done.")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_enthalpy_flux_1_4_8cm(
    csv_path,
    halfwidth=0.200,
    flux_kind="net",   # "net", "up", or "down"
    output_png=None,
):
    """
    Plot convective enthalpy flux at horizontal planes
    1 cm, 4 cm, and 8 cm above the wire.

    Parameters
    ----------
    csv_path : str or Path
        Path to enthalpy_flux_planes.csv.
    halfwidth : float
        Integration half-width in metres. Example: 0.200.
    flux_kind : str
        "net", "up", or "down".
        net  = upward minus downward contribution.
        up   = positive/upward enthalpy transport only.
        down = downward enthalpy transport only.
    output_png : str or Path or None
        If given, save the figure to this path.
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    y_planes = {
        0.010: "1 cm above wire",
        0.040: "4 cm above wire",
        0.080: "8 cm above wire",
    }

    if flux_kind not in {"net", "up", "down"}:
        raise ValueError("flux_kind must be one of: 'net', 'up', 'down'")

    col_prefix = {
        "net": "Q_conv_net",
        "up": "Q_conv_up",
        "down": "Q_conv_down",
    }[flux_kind]

    plt.figure(figsize=(3.3, 2.45))

    for y_plus, label in y_planes.items():
        col = (
            f"{col_prefix}_y_plus_{y_plus:.3f}_m_"
            f"halfwidth_{halfwidth:.3f}_m_W_per_m"
        )

        if col not in df.columns:
            raise KeyError(
                f"Column not found:\n  {col}\n\n"
                f"Check that halfwidth={halfwidth} exists in the CSV."
            )

        plt.plot(df["step"], df[col], label=label)

    plt.xlabel("Saved step")
    plt.ylabel("Convective enthalpy flux [W/m]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_png is not None:
        plt.savefig(output_png, dpi=300)
        print(f"Saved: {output_png}")

    plt.show()

if __name__ == "__main__":
    main()
    # plot_enthalpy_flux_1_4_8cm(
    #     "PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866/base/postprocess_temperature/enthalpy_flux_planes.csv",
    #     halfwidth=0.200,
    #     flux_kind="net",
    #     output_png="enthalpy_flux_1_4_8cm.png",
    # )
    
    # python postprocessing.py PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866/base/ --workers 46 --flux-half-widths 0.005 0.01 0.02 0.04 0.08 0.20 --line-half-width 0.20 --T-inf 292.96 --velocity-scale-factor 0.153657
