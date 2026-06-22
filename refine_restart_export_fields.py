#!/usr/bin/env python3
"""
Offline refinement/export utility for legacy-FEniCS plume restart checkpoints.

Reads a solver restart checkpoint directory containing
    state.h5   with /mesh, /p_star, /u_star, /theta_star
    state.json with step/time/dt metadata
then refines the checkpoint mesh, transfers the nondimensional solution, and writes:

  1) an optional refined restart checkpoint, still in solver variables P1/P2/P1;
  2) postprocess-ready XDMF/HDF5 files for dimensional T, u, p, and q=-k grad(T).

The exported mesh coordinates are kept in the same coordinates as the checkpoint
mesh. In this project that normally means nondimensional/star coordinates, so run
postprocess_steady_plume_htff26.py with --coords-are-dimensionless --lref ... .

Typical command from the repository root:

mpirun -np 4 python refine_restart_export_fields.py \
  --restart-checkpoint PlumeCase_Brodowicz_Air_reduced/runs/.../base/restart_checkpoint \
  --outdir PlumeCase_Brodowicz_Air_reduced/runs/.../base/refined_export \
  --experiment-index 1 --formulation base \
  --levels 2 --top-fraction 0.30 --force-wire-ring --wire-ring-factor 20

Then postprocess, for example:

python postprocess_steady_plume_htff26.py \
  --temperature-xdmf refined_export/air_temperature_refined_from_checkpoint_step_XXXXX.xdmf \
  --velocity-xdmf refined_export/air_velocity_refined_from_checkpoint_step_XXXXX.xdmf \
  --pressure-xdmf refined_export/air_pressure_refined_from_checkpoint_step_XXXXX.xdmf \
  --heatflux-xdmf refined_export/air_temperature_refined_from_checkpoint_step_XXXXX_heatflux.xdmf \
  --coords-are-dimensionless --lref <Lref> ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

try:
    import dolfin as fenics
except ImportError:  # some installations expose fenics directly
    import fenics  # type: ignore

try:
    from mpi4py import MPI as MPI4Py
except Exception:  # pragma: no cover
    MPI4Py = None


COMM = fenics.MPI.comm_world
RANK = fenics.MPI.rank(COMM)
SIZE = fenics.MPI.size(COMM)


def print0(*args, **kwargs):
    if RANK == 0:
        print(*args, **kwargs, flush=True)


def mkdir0(path: str | Path) -> None:
    if RANK == 0:
        Path(path).mkdir(parents=True, exist_ok=True)
    fenics.MPI.barrier(COMM)


def _mpi4py_comm():
    """Return an mpi4py communicator for legacy DOLFIN variants.

    Some installations expose fenics.MPI.comm_world as a DOLFIN MPICommWrapper
    with .tompi4py(); others already expose an mpi4py Intracomm.
    """
    if hasattr(COMM, "tompi4py"):
        return COMM.tompi4py()
    if hasattr(COMM, "allreduce"):
        return COMM
    return None


def global_min(x: float) -> float:
    comm = _mpi4py_comm()
    if MPI4Py is not None and comm is not None:
        return float(comm.allreduce(float(x), op=MPI4Py.MIN))
    return float(fenics.MPI.min(COMM, float(x)))


def global_max(x: float) -> float:
    comm = _mpi4py_comm()
    if MPI4Py is not None and comm is not None:
        return float(comm.allreduce(float(x), op=MPI4Py.MAX))
    return float(fenics.MPI.max(COMM, float(x)))


def global_sum_int(x: int) -> int:
    comm = _mpi4py_comm()
    if MPI4Py is not None and comm is not None:
        return int(comm.allreduce(int(x), op=MPI4Py.SUM))
    return int(fenics.MPI.sum(COMM, int(x)))


def build_mixed_space(mesh: fenics.Mesh) -> fenics.FunctionSpace:
    """Current solver space: P1 pressure, P2 velocity, P1 temperature."""
    P1 = fenics.FiniteElement("P", mesh.ufl_cell(), 1)
    P2 = fenics.VectorElement("P", mesh.ufl_cell(), 2)
    return fenics.FunctionSpace(mesh, fenics.MixedElement([P1, P2, P1]))


def load_checkpoint(checkpoint_dir: str | Path):
    checkpoint_dir = Path(checkpoint_dir)
    h5_path = checkpoint_dir / "state.h5"
    meta_path = checkpoint_dir / "state.json"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing {h5_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    mesh = fenics.Mesh()
    h5 = fenics.HDF5File(COMM, str(h5_path), "r")
    h5.read(mesh, "/mesh", False)

    W = build_mixed_space(mesh)
    Vp, _ = W.sub(0).collapse(True)
    Vu, _ = W.sub(1).collapse(True)
    VT, _ = W.sub(2).collapse(True)

    p_star = fenics.Function(Vp, name="p_star")
    u_star = fenics.Function(Vu, name="u_star")
    theta_star = fenics.Function(VT, name="theta_star")
    h5.read(p_star, "/p_star")
    h5.read(u_star, "/u_star")
    h5.read(theta_star, "/theta_star")
    h5.close()

    with open(meta_path, "r") as f:
        meta = json.load(f)

    try:
        mesh.bounding_box_tree().build(mesh)
    except Exception:
        pass

    print0(f"[load] checkpoint: {checkpoint_dir}")
    print0(f"[load] cells={mesh.num_cells()}, vertices={mesh.num_vertices()}, ranks={SIZE}")
    print0(f"[load] meta step={meta.get('step')}, time={meta.get('time')}, dt={meta.get('dt')}")
    return mesh, p_star, u_star, theta_star, meta


def local_mesh_bounds(mesh: fenics.Mesh):
    coords = mesh.coordinates()
    if coords.size == 0:
        return np.inf, -np.inf, np.inf, -np.inf
    return coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()


def mesh_bounds(mesh: fenics.Mesh):
    xmin, xmax, ymin, ymax = local_mesh_bounds(mesh)
    return global_min(xmin), global_max(xmax), global_min(ymin), global_max(ymax)


def make_indicator(mesh: fenics.Mesh, theta: fenics.Function, u: fenics.Function,
                   u_weight: float, theta_weight: float, speed_weight: float) -> fenics.Function:
    V0 = fenics.FunctionSpace(mesh, "DG", 0)
    h = fenics.CellDiameter(mesh)
    grad_theta = fenics.sqrt(fenics.inner(fenics.grad(theta), fenics.grad(theta)) + fenics.DOLFIN_EPS)
    grad_u = fenics.sqrt(fenics.inner(fenics.grad(u), fenics.grad(u)) + fenics.DOLFIN_EPS)
    speed = fenics.sqrt(fenics.inner(u, u) + fenics.DOLFIN_EPS)
    eta_expr = h * (grad_theta + fenics.Constant(float(u_weight)) * grad_u)
    eta_expr += fenics.Constant(float(theta_weight)) * fenics.sqrt(theta * theta + fenics.DOLFIN_EPS)
    eta_expr += fenics.Constant(float(speed_weight)) * speed
    return fenics.project(eta_expr, V0, solver_type="mumps")


def quantile_threshold_global(local_values: np.ndarray, keep_top_fraction: float) -> float:
    """
    Conservative MPI threshold: use min of local quantiles so parallel runs mark
    slightly more cells rather than too few in ranks containing the plume.
    """
    if local_values.size == 0:
        local_q = 1.0e100
    else:
        local_q = float(np.quantile(local_values, 1.0 - float(keep_top_fraction)))
    return global_min(local_q)


def mark_for_refinement(mesh: fenics.Mesh, theta: fenics.Function, u: fenics.Function,
                        top_fraction: float, u_weight: float,
                        theta_weight: float, speed_weight: float,
                        force_wire_ring: bool, wire_center_x: float,
                        wire_center_y: float | None, wire_radius: float | None,
                        wire_ring_factor: float) -> fenics.MeshFunction:
    if not (0.0 < float(top_fraction) <= 1.0):
        raise ValueError("--top-fraction must be in (0, 1].")

    markers = fenics.MeshFunction("bool", mesh, mesh.topology().dim())
    markers.set_all(False)

    if float(top_fraction) >= 0.999999:
        for cell in fenics.cells(mesh):
            markers[cell] = True
    else:
        eta = make_indicator(mesh, theta, u, u_weight, theta_weight, speed_weight)
        vals = eta.vector().get_local()
        threshold = quantile_threshold_global(vals, top_fraction)
        V0 = eta.function_space()
        dofmap = V0.dofmap()
        local_vals = eta.vector().get_local()
        for cell in fenics.cells(mesh):
            dof = dofmap.cell_dofs(cell.index())[0]
            if local_vals[dof] >= threshold:
                markers[cell] = True

    if force_wire_ring:
        if wire_center_y is None or wire_radius is None:
            raise ValueError("--force-wire-ring requires experiment-derived or explicit wire center/radius.")
        r_outer = float(wire_ring_factor) * float(wire_radius)
        cx = float(wire_center_x)
        cy = float(wire_center_y)
        for cell in fenics.cells(mesh):
            mp = cell.midpoint()
            d = math.hypot(float(mp.x()) - cx, float(mp.y()) - cy)
            if d <= r_outer:
                markers[cell] = True

    n_local = sum(1 for cell in fenics.cells(mesh) if bool(markers[cell]))
    n_global = global_sum_int(n_local)
    print0(f"[mark] marked {n_global} cells")
    return markers


def interpolate_fields_to_mesh(mesh_new: fenics.Mesh, p_old, u_old, theta_old):
    for f in (p_old, u_old, theta_old):
        try:
            f.set_allow_extrapolation(True)
        except Exception:
            pass
    Vp = fenics.FunctionSpace(mesh_new, "CG", 1)
    Vu = fenics.VectorFunctionSpace(mesh_new, "CG", 2)
    VT = fenics.FunctionSpace(mesh_new, "CG", 1)
    p_new = fenics.interpolate(p_old, Vp)
    u_new = fenics.interpolate(u_old, Vu)
    theta_new = fenics.interpolate(theta_old, VT)
    p_new.rename("p_star", "p_star")
    u_new.rename("u_star", "u_star")
    theta_new.rename("theta_star", "theta_star")
    return p_new, u_new, theta_new



def transfer_fields_to_refined_mesh(mesh_new: fenics.Mesh, p_old, u_old, theta_old, method: str = "adapt"):
    """
    Transfer fields from a parent mesh to its refined child mesh.

    For meshes produced by fenics.refine(parent, markers), fenics.adapt(function, child_mesh)
    is the safest transfer in legacy FEniCS because it uses the refinement relation instead
    of evaluating the old function through a global point-search.  The latter can be unreliable
    in MPI and was the source of the huge nonsense values seen in the exported fields.
    """
    method = str(method).lower()

    if method == "adapt":
        try:
            p_new = fenics.adapt(p_old, mesh_new)
            u_new = fenics.adapt(u_old, mesh_new)
            theta_new = fenics.adapt(theta_old, mesh_new)
            p_new.rename("p_star", "p_star")
            u_new.rename("u_star", "u_star")
            theta_new.rename("theta_star", "theta_star")
            return p_new, u_new, theta_new
        except Exception as e:
            print0(f"[transfer] fenics.adapt failed; falling back to interpolate. Reason: {e}")

    return interpolate_fields_to_mesh(mesh_new, p_old, u_old, theta_old)


def refine_solution(mesh, p, u, theta, args, wire_center_y, wire_radius):
    mesh_i = mesh
    p_i, u_i, theta_i = p, u, theta
    for lev in range(int(args.levels)):
        print0(f"[refine] level {lev + 1}/{args.levels}")
        markers = mark_for_refinement(
            mesh_i, theta_i, u_i,
            top_fraction=args.top_fraction,
            u_weight=args.u_indicator_weight,
            theta_weight=args.theta_indicator_weight,
            speed_weight=args.speed_indicator_weight,
            force_wire_ring=args.force_wire_ring,
            wire_center_x=args.wire_center_x,
            wire_center_y=wire_center_y,
            wire_radius=wire_radius,
            wire_ring_factor=args.wire_ring_factor,
        )
        mesh_new = fenics.refine(mesh_i, markers)
        try:
            mesh_new.bounding_box_tree().build(mesh_new)
        except Exception:
            pass
        p_i, u_i, theta_i = transfer_fields_to_refined_mesh(mesh_new, p_i, u_i, theta_i, args.transfer_method)
        mesh_i = mesh_new
        print0(f"[refine] cells={mesh_i.num_cells()}, vertices={mesh_i.num_vertices()}")
    return mesh_i, p_i, u_i, theta_i


def assign_to_mixed(mesh, p, u, theta) -> fenics.Function:
    W = build_mixed_space(mesh)
    Vp, _ = W.sub(0).collapse(True)
    Vu, _ = W.sub(1).collapse(True)
    VT, _ = W.sub(2).collapse(True)
    p_i = fenics.interpolate(p, Vp)
    u_i = fenics.interpolate(u, Vu)
    theta_i = fenics.interpolate(theta, VT)
    w = fenics.Function(W, name="w")
    fenics.FunctionAssigner(W.sub(0), Vp).assign(w.sub(0), p_i)
    fenics.FunctionAssigner(W.sub(1), Vu).assign(w.sub(1), u_i)
    fenics.FunctionAssigner(W.sub(2), VT).assign(w.sub(2), theta_i)
    w.vector().apply("insert")
    return w


def write_refined_checkpoint(outdir: Path, mesh, p, u, theta, meta, args):
    if not args.write_refined_checkpoint:
        return None
    ckpt = outdir / "restart_checkpoint_refined"
    mkdir0(ckpt)
    w = assign_to_mixed(mesh, p, u, theta)
    p_w, u_w, theta_w = w.split(deepcopy=True)
    h5 = fenics.HDF5File(COMM, str(ckpt / "state.h5"), "w")
    h5.write(mesh, "/mesh")
    h5.write(p_w, "/p_star")
    h5.write(u_w, "/u_star")
    h5.write(theta_w, "/theta_star")
    h5.close()
    meta2 = dict(meta)
    meta2.update({
        "offline_refined": True,
        "offline_refinement_levels": int(args.levels),
        "offline_top_fraction": float(args.top_fraction),
        "parent_checkpoint": os.path.abspath(args.restart_checkpoint),
    })
    if args.dt_factor is not None and "dt" in meta2:
        meta2["dt"] = float(meta2["dt"]) * float(args.dt_factor)
        meta2["offline_dt_factor"] = float(args.dt_factor)
    if RANK == 0:
        with open(ckpt / "state.json", "w") as f:
            json.dump(meta2, f, indent=2)
    fenics.MPI.barrier(COMM)
    print0(f"[write] refined restart checkpoint: {ckpt}")
    return ckpt


def write_xdmf(path: Path, mesh: fenics.Mesh, f: fenics.Function, time_value: float | None = None):
    xdmf = fenics.XDMFFile(COMM, str(path))
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.parameters["rewrite_function_mesh"] = True
    if time_value is None:
        xdmf.write(mesh)
        xdmf.write(f)
    else:
        # Writing mesh first gives the simple single-grid layout your h5py
        # postprocessor expects, while the time value remains available.
        xdmf.write(mesh)
        xdmf.write(f, float(time_value))
    xdmf.close()
    print0(f"[write] {path}")


def local_project(expr, V: fenics.FunctionSpace, name: str | None = None) -> fenics.Function:
    """Fast element-local projection, suitable for DG0 heat-flux export.

    This avoids a global MUMPS solve. For DG spaces, LocalSolver performs
    independent cell-local mass-matrix solves, which is much faster and
    far less memory hungry on heavily refined meshes.
    """
    trial = fenics.TrialFunction(V)
    test = fenics.TestFunction(V)
    a = fenics.inner(trial, test) * fenics.dx
    L = fenics.inner(expr, test) * fenics.dx
    out = fenics.Function(V)
    solver = fenics.LocalSolver(a, L)
    solver.factorize()
    solver.solve_local_rhs(out)
    if name:
        out.rename(name, name)
    return out


def dimensional_fields(mesh, p_star, u_star, theta_star, *, Uref: float, dTref: float,
                       T_inf: float, rho: float, export_order: int):
    print0("[export] interpolating dimensional p/u/T fields")
    Vp = fenics.FunctionSpace(mesh, "CG", int(export_order))
    Vu = fenics.VectorFunctionSpace(mesh, "CG", int(export_order))
    VT = fenics.FunctionSpace(mesh, "CG", int(export_order))

    p_i = fenics.interpolate(p_star, Vp)
    u_i = fenics.interpolate(u_star, Vu)
    th_i = fenics.interpolate(theta_star, VT)

    u_dim = fenics.Function(Vu, name="u_dim")
    u_dim.vector()[:] = u_i.vector()[:] * float(Uref)
    u_dim.vector().apply("insert")

    p_dim = fenics.Function(Vp, name="p_dim")
    p_dim.vector()[:] = p_i.vector()[:] * (float(rho) * float(Uref) ** 2)
    p_dim.vector().apply("insert")

    T_dim = fenics.Function(VT, name="T_dim")
    T_dim.vector()[:] = float(T_inf) + th_i.vector()[:] * float(dTref)
    T_dim.vector().apply("insert")

    print0("[export] dimensional p/u/T fields ready")
    return p_dim, u_dim, T_dim


def compute_heat_flux_local(mesh, T_dim, k: float):
    print0("[export] computing heat flux q=-k*grad(T) with DG0 LocalSolver")
    Vq = fenics.VectorFunctionSpace(mesh, "DG", 0)
    q_heat = local_project(-fenics.Constant(float(k)) * fenics.grad(T_dim), Vq, "q_heat_dim")
    print0("[export] heat flux ready")
    return q_heat


def dimensional_exports(mesh, p_star, u_star, theta_star, *, Uref: float, dTref: float,
                        T_inf: float, rho: float, k: float, export_order: int):
    p_dim, u_dim, T_dim = dimensional_fields(
        mesh, p_star, u_star, theta_star,
        Uref=Uref, dTref=dTref, T_inf=T_inf, rho=rho,
        export_order=export_order,
    )
    q_heat = compute_heat_flux_local(mesh, T_dim, k)
    return p_dim, u_dim, T_dim, q_heat


def maybe_load_experiment(index: int, experiments_json: str | None, schema_json: str | None):
    if index < 0:
        return None, None
    # Use the same project parser/scales when available.
    from utils.parser import parser
    from solver.scales import compute_nondimensional_scales
    if experiments_json is None:
        try:
            from utils.imports import EXPERIMENTS_JSON_PATH
            experiments_json = EXPERIMENTS_JSON_PATH
        except Exception:
            experiments_json = "experiments.json"
    if schema_json is None:
        try:
            from utils.imports import SCHEMA_JSON_PATH
            schema_json = SCHEMA_JSON_PATH
        except Exception:
            schema_json = "schema.json"
    exps = parser(experiments_json_path=experiments_json, schema_json_path=schema_json)
    exp = exps[int(index)]
    sc = compute_nondimensional_scales(exp)
    return exp, sc


def resolve_scales_and_geometry(args, mesh):
    exp, sc = maybe_load_experiment(args.experiment_index, args.experiments_json, args.schema_json)

    if exp is not None:
        props = exp.fluid.properties
        T_inf = float(exp.initial_conditions.temperature)
        rho = float(props["rho"])
        k = float(props["k"])
        dTref = float(sc.dTref)
        Lref = float(sc.Lref)
        if args.formulation.lower() == "abe" and hasattr(sc, "Uref_abe"):
            Uref = float(sc.Uref_abe)
        else:
            Uref = float(sc.Uref)
        wire_radius = (float(exp.dimensions.wire.diameter) / 2.0) / Lref
        # Must match your params_bcs.Hot_wall convention in nondimensional mesh coords.
        wire_center_y = float(exp.dimensions.domain.y_max) / Lref / 10.0 + 11.0 * wire_radius
        print0(f"[scales] loaded experiment {args.experiment_index}: {exp.name}")
        print0(f"[scales] Lref={Lref:.8e}, dTref={dTref:.8e}, Uref={Uref:.8e}, T_inf={T_inf:.8e}")
        print0(f"[wire] center=({args.wire_center_x:.8e}, {wire_center_y:.8e}), r={wire_radius:.8e} in checkpoint coordinates")
        return Uref, dTref, T_inf, rho, k, Lref, wire_center_y, wire_radius

    required = [args.Uref, args.dTref, args.T_inf, args.rho, args.k]
    if any(v is None for v in required):
        raise ValueError(
            "Without --experiment-index you must provide --Uref --dTref --T-inf --rho --k "
            "and optionally --wire-center-y/--wire-radius for wire-ring refinement."
        )
    wire_center_y = args.wire_center_y
    wire_radius = args.wire_radius
    return float(args.Uref), float(args.dTref), float(args.T_inf), float(args.rho), float(args.k), args.lref, wire_center_y, wire_radius



def vector_minmax(f: fenics.Function):
    vals = f.vector().get_local()
    if vals.size == 0:
        lo, hi = np.inf, -np.inf
    else:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    return global_min(lo), global_max(hi)


def sanity_check_nondim(label: str, theta: fenics.Function, u: fenics.Function, args=None):
    th_min, th_max = vector_minmax(theta)
    u_min, u_max = vector_minmax(u)
    print0(f"[sanity] {label}: theta=[{th_min:.6e}, {th_max:.6e}], u_vector_entries=[{u_min:.6e}, {u_max:.6e}]")
    bad = (not np.isfinite([th_min, th_max, u_min, u_max]).all()) or abs(th_min) > 1.0e4 or abs(th_max) > 1.0e4 or abs(u_min) > 1.0e4 or abs(u_max) > 1.0e4
    if bad and (args is None or not getattr(args, "allow_bad_field_ranges", False)):
        raise RuntimeError(
            f"Sanity check failed for {label}. The field ranges are absurd; refusing to write corrupted output. "
            "Discard this refined checkpoint/export and rerun from the original solver restart checkpoint, preferably in serial."
        )


def sanity_check_dimensional(T_dim: fenics.Function, u_dim: fenics.Function, args=None):
    T_min, T_max = vector_minmax(T_dim)
    u_min, u_max = vector_minmax(u_dim)
    print0(f"[sanity] dimensional export: T=[{T_min:.6e}, {T_max:.6e}] K, u_vector_entries=[{u_min:.6e}, {u_max:.6e}] m/s")
    bad = (not np.isfinite([T_min, T_max, u_min, u_max]).all()) or T_min < 0.0 or T_max > 5000.0 or abs(u_min) > 1000.0 or abs(u_max) > 1000.0
    if bad and (args is None or not getattr(args, "allow_bad_field_ranges", False)):
        raise RuntimeError(
            "Dimensional export sanity check failed. Refusing to write corrupted XDMF/HDF5 files."
        )


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--restart-checkpoint", required=True, help="Directory containing state.h5/state.json")
    ap.add_argument("--outdir", required=True, help="Output directory for refined export")
    ap.add_argument("--experiment-index", type=int, default=-1, help="Use project experiments.json to infer scales and wire geometry")
    ap.add_argument("--experiments-json", default=None)
    ap.add_argument("--schema-json", default=None)
    ap.add_argument("--formulation", choices=["base", "abe"], default="base", help="Controls whether Uref or Uref_abe is used when --experiment-index is supplied")

    ap.add_argument("--levels", type=int, default=2, help="Number of local refinement passes")
    ap.add_argument("--top-fraction", type=float, default=0.25, help="Fraction of cells refined each pass; use 1.0 for uniform refinement")
    ap.add_argument("--u-indicator-weight", type=float, default=0.25)
    ap.add_argument("--theta-indicator-weight", type=float, default=0.10)
    ap.add_argument("--speed-indicator-weight", type=float, default=0.05)
    ap.add_argument("--force-wire-ring", action="store_true", help="Always refine cells near the wire")
    ap.add_argument("--wire-ring-factor", type=float, default=16.0, help="Refine cells with distance from wire center <= factor*wire_radius")
    ap.add_argument("--wire-center-x", type=float, default=0.0)
    ap.add_argument("--wire-center-y", type=float, default=None, help="Explicit wire center y in checkpoint mesh coordinates")
    ap.add_argument("--wire-radius", type=float, default=None, help="Explicit wire radius in checkpoint mesh coordinates")

    ap.add_argument("--export-order", type=int, default=1, choices=[1, 2], help="CG order for exported dimensional p/u/T. Use 1 for safest postprocess compatibility.")
    ap.add_argument("--write-refined-checkpoint", action="store_true", help="Also write restart_checkpoint_refined with P1/P2/P1 solver fields")
    ap.add_argument("--skip-heatflux", action="store_true", help="Only write pressure/velocity/temperature XDMF files; skip q=-k grad(T) export")
    ap.add_argument("--dt-factor", type=float, default=None, help="Optional dt multiplier written only to refined checkpoint metadata")
    ap.add_argument("--prefix", default="refined_from_checkpoint")

    # Manual scale fallback, only used without --experiment-index.
    ap.add_argument("--Uref", type=float, default=None)
    ap.add_argument("--dTref", type=float, default=None)
    ap.add_argument("--T-inf", dest="T_inf", type=float, default=None)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--k", type=float, default=None)
    ap.add_argument("--lref", type=float, default=None, help="Not used for writing, but printed for the postprocess command")

    ap.add_argument("--allow-mpi", action="store_true", help="Allow MPI execution. Not recommended for offline mesh-to-mesh transfer in legacy FEniCS; serial is safer.")
    ap.add_argument("--transfer-method", choices=["adapt", "interpolate"], default="adapt", help="How to transfer fields after refinement. adapt is safer for refined child meshes.")
    ap.add_argument("--allow-bad-field-ranges", action="store_true", help="Do not abort if sanity checks find absurd field ranges.")

    args = ap.parse_args()

    if SIZE > 1 and not args.allow_mpi:
        raise RuntimeError(
            "This offline refinement/export script is intentionally serial-safe by default. "
            "Run it without mpirun, e.g. `python refine_restart_export_fields_fixed4.py ...`. "
            "Legacy FEniCS cross-mesh interpolation in MPI can silently corrupt fields. "
            "Use --allow-mpi only if you accept that risk."
        )

    outdir = Path(args.outdir)
    mkdir0(outdir)

    mesh, p, u, theta, meta = load_checkpoint(args.restart_checkpoint)
    Uref, dTref, T_inf, rho, k, Lref, wire_center_y, wire_radius = resolve_scales_and_geometry(args, mesh)
    sanity_check_nondim("loaded checkpoint", theta, u, args)

    if args.levels > 0:
        mesh, p, u, theta = refine_solution(mesh, p, u, theta, args, wire_center_y, wire_radius)
    else:
        print0("[refine] skipped because --levels 0")

    sanity_check_nondim("after refinement/transfer", theta, u, args)

    write_refined_checkpoint(outdir, mesh, p, u, theta, meta, args)

    step = int(meta.get("step", -1))
    step_label = f"{step:05d}" if step >= 0 else "unknown"
    stem = f"{args.prefix}_step_{step_label}"

    p_dim, u_dim, T_dim = dimensional_fields(
        mesh, p, u, theta,
        Uref=Uref, dTref=dTref, T_inf=T_inf, rho=rho,
        export_order=args.export_order,
    )
    sanity_check_dimensional(T_dim, u_dim, args)

    tval = meta.get("time", None)
    tval = float(tval) if tval is not None else None

    p_path = outdir / f"air_pressure_{stem}.xdmf"
    u_path = outdir / f"air_velocity_{stem}.xdmf"
    T_path = outdir / f"air_temperature_{stem}.xdmf"
    q_path = outdir / f"air_temperature_{stem}_heatflux.xdmf"

    print0("[export] writing pressure/velocity/temperature XDMF files")
    write_xdmf(p_path, mesh, p_dim, tval)
    write_xdmf(u_path, mesh, u_dim, tval)
    write_xdmf(T_path, mesh, T_dim, tval)

    if args.skip_heatflux:
        print0("[export] skipped heat-flux XDMF because --skip-heatflux was supplied")
    else:
        q_heat = compute_heat_flux_local(mesh, T_dim, k)
        write_xdmf(q_path, mesh, q_heat, tval)

    xmin, xmax, ymin, ymax = mesh_bounds(mesh)
    print0("\n[done] refined export complete")
    print0(f"[done] mesh coordinate bounds: x=[{xmin:.6e}, {xmax:.6e}], y=[{ymin:.6e}, {ymax:.6e}]")
    print0("[done] postprocess files:")
    print0(f"  temperature: {T_path}")
    print0(f"  velocity:    {u_path}")
    print0(f"  pressure:    {p_path}")
    if not args.skip_heatflux:
        print0(f"  heat flux:   {q_path}")
    if Lref is not None:
        print0(f"[hint] postprocess with: --coords-are-dimensionless --lref {float(Lref):.12g}")


if __name__ == "__main__":
    main()
