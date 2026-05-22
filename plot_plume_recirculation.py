#!/usr/bin/env python3
"""
plot_plume_recirculation.py

Post-process legacy FEniCS/DOLFIN XDMF+HDF5 transient plume snapshots and generate
streamline, temperature, speed, and vorticity figures.

Expected file naming pattern in --input-dir:
    air_velocity_transient_05000.h5
    air_velocity_transient_05000.xdmf
    air_temperature_transient_05000.h5
    air_temperature_transient_05000.xdmf

The script reads the HDF5 files directly. It assumes the standard DOLFIN layout:
    Mesh/mesh/geometry        shape (n_nodes, 2)
    Mesh/mesh/topology        shape (n_cells, 3)
    VisualisationVector/0     scalar temperature (n_nodes, 1) or velocity (n_nodes, 3)

Examples
--------
Reduced-domain plots:
    python plot_plume_recirculation.py \
        --input-dir . \
        --steps 05000 10000 14500 \
        --out-dir recirculation_figures \
        --make-sequence

Large-domain plots:
    python plot_plume_recirculation.py \
        --input-dir . \
        --steps 85000 105000 \
        --out-dir large_domain_figures \
        --no-speed

Notes
-----
The interpolation is done on a regular grid using matplotlib.tri.LinearTriInterpolator,
which is fast and avoids requiring scipy. Vorticity is then computed on the regular grid
as d(uy)/dx - d(ux)/dy using numpy gradients.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


@dataclass
class Snapshot:
    step: str
    xy: np.ndarray
    cells: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    temp: Optional[np.ndarray]


def _read_h5_dataset(path: Path, dataset: str = "VisualisationVector/0") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coordinates, triangle connectivity, and field values from a DOLFIN HDF5 output."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with h5py.File(path, "r") as h5:
        xy = np.asarray(h5["Mesh/mesh/geometry"])
        cells = np.asarray(h5["Mesh/mesh/topology"], dtype=np.int64)
        values = np.asarray(h5[dataset])

    if xy.shape[1] > 2:
        xy = xy[:, :2]
    return xy, cells, values


def read_snapshot(input_dir: Path, step: str) -> Snapshot:
    """Read one velocity/temperature snapshot for a given string step, e.g. '05000'."""
    step = str(step).zfill(5)
    v_path = input_dir / f"air_velocity_transient_{step}.h5"
    t_path = input_dir / f"air_temperature_transient_{step}.h5"

    xy, cells, vel = _read_h5_dataset(v_path)
    vel = np.asarray(vel)
    if vel.ndim != 2 or vel.shape[1] < 2:
        raise ValueError(f"Velocity field in {v_path} has unexpected shape {vel.shape}")

    ux = vel[:, 0]
    uy = vel[:, 1]

    temp = None
    if t_path.exists():
        xy_t, cells_t, temp_raw = _read_h5_dataset(t_path)
        # Most files here share the same mesh for velocity and temperature.
        # If not, the temperature is still interpolated using its own triangulation later.
        temp = np.asarray(temp_raw).reshape(-1)
        if len(temp) != len(xy):
            # Reinterpolate temperature from its own mesh to velocity nodes.
            tri_t = mtri.Triangulation(xy_t[:, 0], xy_t[:, 1], cells_t)
            interp_t = mtri.LinearTriInterpolator(tri_t, temp)
            temp = np.asarray(interp_t(xy[:, 0], xy[:, 1])).astype(float)
            temp = np.nan_to_num(temp, nan=np.nanmin(temp))

    return Snapshot(step=step, xy=xy, cells=cells, ux=ux, uy=uy, temp=temp)


def make_grid(xy: np.ndarray, nx: int = 360, ny: int = 360, margin: float = 0.0):
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= margin * dx
    xmax += margin * dx
    ymin -= margin * dy
    ymax += margin * dy
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def interpolate_to_grid(snapshot: Snapshot, nx: int = 360, ny: int = 360):
    tri = mtri.Triangulation(snapshot.xy[:, 0], snapshot.xy[:, 1], snapshot.cells)
    X, Y = make_grid(snapshot.xy, nx=nx, ny=ny)

    def interp(values: np.ndarray) -> np.ndarray:
        interpolator = mtri.LinearTriInterpolator(tri, values)
        Z = np.asarray(interpolator(X, Y), dtype=float)
        return Z

    U = interp(snapshot.ux)
    V = interp(snapshot.uy)
    T = interp(snapshot.temp) if snapshot.temp is not None else None

    # Convert masked/NaN values outside the triangulation to NaN consistently.
    U = np.where(np.isfinite(U), U, np.nan)
    V = np.where(np.isfinite(V), V, np.nan)
    if T is not None:
        T = np.where(np.isfinite(T), T, np.nan)

    speed = np.sqrt(U**2 + V**2)

    # vorticity_z = d(uy)/dx - d(ux)/dy. np.gradient returns derivatives along y, x.
    x = X[0, :]
    y = Y[:, 0]
    dV_dy, dV_dx = np.gradient(V, y, x)
    dU_dy, dU_dx = np.gradient(U, y, x)
    omega = dV_dx - dU_dy

    return X, Y, U, V, T, speed, omega


def _streamplot(ax, X, Y, U, V, density: float = 1.4, linewidth: float = 0.55):
    U_plot = np.nan_to_num(U, nan=0.0)
    V_plot = np.nan_to_num(V, nan=0.0)
    speed = np.sqrt(U_plot**2 + V_plot**2)
    # Avoid plotting streamlines in zero/invalid regions too aggressively.
    lw = linewidth * (0.4 + 0.6 * speed / np.nanmax(speed) if np.nanmax(speed) > 0 else 1.0)
    ax.streamplot(
        X[0, :],
        Y[:, 0],
        U_plot,
        V_plot,
        density=density,
        color="k",
        linewidth=lw,
        arrowsize=0.7,
    )


def plot_field_with_streamlines(
    snapshot: Snapshot,
    out_dir: Path,
    field: str,
    nx: int,
    ny: int,
    dpi: int,
    show_title: bool = False,
):
    X, Y, U, V, T, speed, omega = interpolate_to_grid(snapshot, nx=nx, ny=ny)

    if field == "temperature":
        if T is None:
            return None
        Z = T
        label = "Temperature"
        fname = f"small_domain_streamlines_{snapshot.step}.png"
        cmap = "coolwarm"
    elif field == "speed":
        Z = speed
        label = "Speed magnitude"
        fname = f"small_domain_speed_streamlines_{snapshot.step}.png"
        cmap = "viridis"
    elif field == "vorticity":
        Z = omega
        label = r"Vorticity $\omega_z = \partial u_y/\partial x - \partial u_x/\partial y$"
        fname = f"reduced_domain_vorticity_{snapshot.step}.png"
        cmap = "coolwarm"
    else:
        raise ValueError(f"Unknown field: {field}")

    fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)

    if field == "vorticity":
        vmax = np.nanpercentile(np.abs(Z), 99.0)
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else np.nanmax(np.abs(Z))
        levels = np.linspace(-vmax, vmax, 41)
        cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend="both")
    else:
        levels = 40
        cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)

    _streamplot(ax, X, Y, U, V)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x",size=15)
    ax.set_ylabel("y",size=15)
    ax.tick_params(axis='both', which='major', labelsize=14)
    if show_title:
        ax.set_title(f"{label}, transient {snapshot.step}")
    cbar = fig.colorbar(cf, ax=ax, shrink=0.86)
    cbar.set_label(label)

    out_path = out_dir / fname
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_temperature_sequence(
    snapshots: Iterable[Snapshot],
    out_dir: Path,
    nx: int,
    ny: int,
    dpi: int,
    show_title: bool = False,
):
    snapshots = list(snapshots)
    if not snapshots:
        return None

    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.4), constrained_layout=True, squeeze=False)
    axes = axes[0]

    last_cf = None
    for ax, snap in zip(axes, snapshots):
        X, Y, U, V, T, speed, omega = interpolate_to_grid(snap, nx=nx, ny=ny)
        if T is None:
            Z = speed
            label = "Speed magnitude"
            cmap = "viridis"
        else:
            Z = T
            label = "Temperature"
            cmap = "inferno"
        last_cf = ax.contourf(X, Y, Z, levels=40, cmap=cmap)
        _streamplot(ax, X, Y, U, V, density=1.25, linewidth=0.45)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if show_title:
            ax.set_title(f"transient {snap.step}")
        else:
            ax.text(0.02, 0.98, f"{snap.step}", transform=ax.transAxes, va="top", ha="left",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75})

    if last_cf is not None:
        cbar = fig.colorbar(last_cf, ax=axes.ravel().tolist(), shrink=0.82)
        cbar.set_label(label)

    out_path = out_dir / "reduced_domain_recirculation_sequence.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def write_summary_csv(snapshots: Iterable[Snapshot], out_dir: Path, nx: int, ny: int):
    """Write a small diagnostic CSV with grid/domain extrema and circulation proxies."""
    out_path = out_dir / "recirculation_summary.csv"
    rows = [
        "step,xmin,xmax,ymin,ymax,max_speed,omega_min,omega_max,omega_abs_integral_proxy,temp_min,temp_max"
    ]
    for snap in snapshots:
        X, Y, U, V, T, speed, omega = interpolate_to_grid(snap, nx=nx, ny=ny)
        dx = float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 1.0
        dy = float(Y[1, 0] - Y[0, 0]) if Y.shape[0] > 1 else 1.0
        omega_abs_int = np.nansum(np.abs(omega)) * dx * dy
        temp_min = np.nanmin(T) if T is not None else np.nan
        temp_max = np.nanmax(T) if T is not None else np.nan
        rows.append(
            f"{snap.step},{np.nanmin(X):.12g},{np.nanmax(X):.12g},{np.nanmin(Y):.12g},{np.nanmax(Y):.12g},"
            f"{np.nanmax(speed):.12g},{np.nanmin(omega):.12g},{np.nanmax(omega):.12g},{omega_abs_int:.12g},"
            f"{temp_min:.12g},{temp_max:.12g}"
        )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot transient plume recirculation diagnostics from DOLFIN HDF5 files.")
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing the .h5 files.")
    parser.add_argument("--out-dir", type=Path, default=Path("recirculation_figures"), help="Output directory for figures.")
    parser.add_argument("--steps", nargs="+", required=True, help="Transient step IDs, e.g. 05000 10000 14500.")
    parser.add_argument("--nx", type=int, default=360, help="Regular interpolation grid points in x.")
    parser.add_argument("--ny", type=int, default=360, help="Regular interpolation grid points in y.")
    parser.add_argument("--dpi", type=int, default=220, help="Figure resolution.")
    parser.add_argument("--make-sequence", action="store_true", help="Create a multi-panel temperature/speed streamline sequence.")
    parser.add_argument("--no-temperature", action="store_true", help="Do not create temperature + streamline plots.")
    parser.add_argument("--no-speed", action="store_true", help="Do not create speed + streamline plots.")
    parser.add_argument("--no-vorticity", action="store_true", help="Do not create vorticity + streamline plots.")
    parser.add_argument("--titles", action="store_true", help="Add titles to figures. Default is no titles, suitable for thesis captions.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    snapshots = []
    for step in args.steps:
        snap = read_snapshot(args.input_dir, step)
        snapshots.append(snap)
        print(
            f"Loaded {snap.step}: nodes={len(snap.xy)}, cells={len(snap.cells)}, "
            f"domain x=[{snap.xy[:,0].min():.6g},{snap.xy[:,0].max():.6g}], "
            f"y=[{snap.xy[:,1].min():.6g},{snap.xy[:,1].max():.6g}]"
        )

    written = []
    for snap in snapshots:
        if not args.no_temperature:
            p = plot_field_with_streamlines(snap, args.out_dir, "temperature", args.nx, args.ny, args.dpi, args.titles)
            if p:
                written.append(p)
        if not args.no_speed:
            p = plot_field_with_streamlines(snap, args.out_dir, "speed", args.nx, args.ny, args.dpi, args.titles)
            if p:
                written.append(p)
        if not args.no_vorticity:
            p = plot_field_with_streamlines(snap, args.out_dir, "vorticity", args.nx, args.ny, args.dpi, args.titles)
            if p:
                written.append(p)

    if args.make_sequence:
        p = plot_temperature_sequence(snapshots, args.out_dir, args.nx, args.ny, args.dpi, args.titles)
        if p:
            written.append(p)

    written.append(write_summary_csv(snapshots, args.out_dir, args.nx, args.ny))

    print("\nWrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
