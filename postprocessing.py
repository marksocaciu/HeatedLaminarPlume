#!/usr/bin/env python3
"""
Post-process saved plume temperature fields and convective enthalpy flux.

Implemented diagnostics:
  1. Q_conv_net / Q_conv_up / Q_conv_down
  2. uy_min / uy_mean / uy_max / ∫uy dx / ∫|uy| dx
  3. Multiple integration half-widths
  4. Final uy(x) profile plots
  5. Final CSV with T, uy, and h_flux density

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python postprocess_temperature_convergence.py /path/to/results/folder --workers 8

Expected temperature files:
    air_temperature_transient_*.h5

Expected velocity files:
    air_velocity_transient_*.h5

Expected HDF5 layout:
    Mesh/mesh/geometry
    Mesh/mesh/topology
    VisualisationVector/0

Temperature is expected to be dimensional [K].
Velocity is expected to be dimensional [m/s].

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
    uy_profile: np.ndarray,
    rho: float,
    cp: float,
    T_inf: float,
):
    window = np.abs(x_line) <= half_width

    xw = x_line[window]
    Tw = T_profile[window]
    uyw = uy_profile[window]

    h_flux_w = rho * cp * (Tw - T_inf) * uyw

    Q_conv_net = integrate_trapezoid_ignore_nan(h_flux_w, xw)
    Q_conv_up = integrate_trapezoid_ignore_nan(np.maximum(h_flux_w, 0.0), xw)
    Q_conv_down = integrate_trapezoid_ignore_nan(np.minimum(h_flux_w, 0.0), xw)

    int_uy = integrate_trapezoid_ignore_nan(uyw, xw)
    int_abs_uy = integrate_trapezoid_ignore_nan(np.abs(uyw), xw)

    key = f"y_plus_{offset:.3f}_m_halfwidth_{half_width:.3f}_m"

    flux_row[f"Q_conv_net_{key}_W_per_m"] = Q_conv_net
    flux_row[f"Q_conv_up_{key}_W_per_m"] = Q_conv_up
    flux_row[f"Q_conv_down_{key}_W_per_m"] = Q_conv_down

    flux_row[f"uy_min_{key}_m_per_s"] = float(np.nanmin(uyw))
    flux_row[f"uy_mean_{key}_m_per_s"] = float(np.nanmean(uyw))
    flux_row[f"uy_max_{key}_m_per_s"] = float(np.nanmax(uyw))

    flux_row[f"int_uy_{key}_m2_per_s"] = int_uy
    flux_row[f"int_abs_uy_{key}_m2_per_s"] = int_abs_uy
    flux_row[f"uy_net_fraction_{key}"] = abs(int_uy) / max(int_abs_uy, 1.0e-300)


def process_saved_step_worker(payload):
    """
    Worker for independent per-step post-processing.
    """
    (
        temperature_file,
        velocity_file,
        box_mask,
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

    temp_box = temperature[box_mask]

    result = {
        "step": step,
        "temperature_file": str(temperature_file),
        "velocity_file": str(velocity_file),
        "box_T_min_K": float(np.nanmin(temp_box)),
        "box_T_mean_K": float(np.nanmean(temp_box)),
        "box_T_max_K": float(np.nanmax(temp_box)),
        "T_box": temp_box,
    }

    peak_row = {"step": step}
    flux_row = {"step": step}

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

        peak_row[f"T_peak_y_plus_{offset:.3f}_m_K"] = float(np.nanmax(T_profile))

        for half_width in flux_half_widths:
            add_plane_window_diagnostics(
                flux_row,
                offset=offset,
                half_width=half_width,
                x_line=x_line,
                T_profile=T_profile,
                uy_profile=uy_profile,
                rho=rho,
                cp=cp,
                T_inf=T_inf,
            )

    result["peak_row"] = peak_row
    result["flux_row"] = flux_row

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
        default=0.20,
        help="Half-width of extracted horizontal profiles [m]. Must be >= max(--flux-half-widths).",
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
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for CSV and PNG output. Defaults to results_dir/postprocess_temperature.",
    )

    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    if max(args.flux_half_widths) > args.line_half_width:
        raise ValueError(
            "max(--flux-half-widths) must be <= --line-half-width. "
            f"Got max flux half-width {max(args.flux_half_widths)} and line half-width {args.line_half_width}."
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
    print(f"  flux_half_widths = {args.flux_half_widths}")
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

    box_mask = (
        (coords_phys[:, 0] >= box_x_min)
        & (coords_phys[:, 0] <= box_x_max)
        & (coords_phys[:, 1] >= box_y_min)
        & (coords_phys[:, 1] <= box_y_max)
    )

    n_box = int(np.count_nonzero(box_mask))
    if n_box == 0:
        raise RuntimeError("No mesh nodes found inside the requested convergence box.")

    print(f"  nodes in convergence box = {n_box}")

    x_line = np.linspace(
        -args.line_half_width,
        args.line_half_width,
        args.num_line_points,
    )

    payloads = [
        (
            str(temperature_file),
            str(velocity_file),
            box_mask,
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
                "box_l2_update_K": l2_update,
                "box_rel_l2_update": rel_l2_update,
                "box_linf_update_K": linf_update,
            }
        )

        plane_peak_rows.append(row["peak_row"])
        enthalpy_flux_rows.append(row["flux_row"])

        previous_T_box = T_box
        previous_step = step

    convergence_csv = output_dir / "temperature_convergence_box.csv"
    peaks_csv = output_dir / "temperature_plane_peaks.csv"
    enthalpy_flux_csv = output_dir / "enthalpy_flux_planes.csv"

    write_csv(convergence_csv, convergence_rows)
    write_csv(peaks_csv, plane_peak_rows)
    write_csv(enthalpy_flux_csv, enthalpy_flux_rows)

    print(f"  wrote {convergence_csv}")
    print(f"  wrote {peaks_csv}")
    print(f"  wrote {enthalpy_flux_csv}")

    steps = np.asarray([row["step"] for row in convergence_rows], dtype=float)

    rel_updates = np.asarray(
        [row["box_rel_l2_update"] for row in convergence_rows],
        dtype=float,
    )

    linf_updates = np.asarray(
        [row["box_linf_update_K"] for row in convergence_rows],
        dtype=float,
    )

    box_max = np.asarray(
        [row["box_T_max_K"] for row in convergence_rows],
        dtype=float,
    )

    plt.figure()
    plt.semilogy(steps, rel_updates)
    plt.xlabel("Saved step")
    plt.ylabel("Relative L2 update in box")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_box_relative_l2_update.png", dpi=200)
    plt.close()

    plt.figure()
    plt.semilogy(steps, linf_updates)
    plt.xlabel("Saved step")
    plt.ylabel("L∞ update in box [K]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_box_linf_update.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(steps, box_max)
    plt.xlabel("Saved step")
    plt.ylabel("Maximum temperature in box [K]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_box_peak_temperature.png", dpi=200)
    plt.close()

    peak_steps = np.asarray([row["step"] for row in plane_peak_rows], dtype=float)

    plt.figure()
    for offset in args.plane_offsets:
        key = f"T_peak_y_plus_{offset:.3f}_m_K"
        values = np.asarray([row[key] for row in plane_peak_rows], dtype=float)
        plt.plot(
            peak_steps,
            values,
            label=f"y = {offset * 100:.0f} cm above wire",
        )

    plt.xlabel("Saved step")
    plt.ylabel("Peak temperature on horizontal plane [K]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_plane_peak_evolution.png", dpi=200)
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
        plt.title(f"Convective enthalpy flux, y = {offset * 100:.0f} cm above wire")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"enthalpy_flux_net_window_evolution_y_plus_{offset:.3f}_m.png",
            dpi=200,
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
        plt.title(f"Up/down convective enthalpy flux, y = {offset * 100:.0f} cm above wire")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"enthalpy_flux_up_down_window_evolution_y_plus_{offset:.3f}_m.png",
            dpi=200,
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
        plt.title(f"Vertical velocity net fraction, y = {offset * 100:.0f} cm above wire")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"uy_net_fraction_window_evolution_y_plus_{offset:.3f}_m.png",
            dpi=200,
        )
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

    final_profiles_csv = output_dir / f"temperature_profiles_step_{final_step}.csv"
    final_combined_profiles_csv = output_dir / f"final_profiles_T_uy_hflux_step_{final_step}.csv"

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

        temperature_profile_columns[f"T_y_plus_{offset:.3f}_m_K"] = T_profile
        combined_profile_columns[f"T_y_plus_{offset:.3f}_m_K"] = T_profile

        plt.plot(
            x_line,
            T_profile,
            label=f"y = {offset * 100:.0f} cm above wire",
        )

    plt.xlabel("x [m]")
    plt.ylabel("Temperature [K]")
    plt.title(f"Temperature profiles at saved step {final_step}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"temperature_profiles_step_{final_step}.png", dpi=200)
    plt.close()

    plt.figure()
    for offset in args.plane_offsets:
        y_line = np.full_like(x_line, wire_y + offset)

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

        combined_profile_columns[f"uy_y_plus_{offset:.3f}_m_m_per_s"] = uy_profile

        plt.plot(
            x_line,
            uy_profile,
            label=f"y = {offset * 100:.0f} cm above wire",
        )

    plt.xlabel("x [m]")
    plt.ylabel(r"$u_y$ [m/s]")
    plt.title(f"Vertical velocity profiles at saved step {final_step}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"vertical_velocity_profiles_step_{final_step}.png", dpi=200)
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

        h_flux_density = args.rho * args.cp * (T_profile - args.T_inf) * uy_profile

        combined_profile_columns[
            f"TminusTinf_y_plus_{offset:.3f}_m_K"
        ] = T_profile - args.T_inf
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
    plt.title(f"Convective enthalpy flux density at saved step {final_step}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"enthalpy_flux_density_profiles_step_{final_step}.png",
        dpi=200,
    )
    plt.close()

    with final_profiles_csv.open("w", newline="") as f:
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

    print(f"  wrote {final_profiles_csv}")
    print(f"  wrote {final_combined_profiles_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
