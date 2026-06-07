from __future__ import annotations

import os
import json
import shutil
import math
from pathlib import Path
import h5py

from solver import scales
from utils.imports import *
from solver.scales import *
from solver.params_bcs import *


def build_mixed_space_on_mesh(mesh: fenics.Mesh) -> fenics.FunctionSpace:
    """
    Match the current project mixed space:
        W = [P1 pressure, P2 velocity, P1 temperature]
    """
    P1 = fenics.FiniteElement("P", mesh.ufl_cell(), 1)
    P2 = fenics.VectorElement("P", mesh.ufl_cell(), 2)
    return fenics.FunctionSpace(mesh, fenics.MixedElement([P1, P2, P1]))


def compute_uniform_qn_star_value(experiment, scales) -> float:
    """
    Same uniform interface heat-flux logic currently used in main.py.

    q_line [W/m], wire_d [m]
    qsurf_dim = q_line / (pi * d)
    qn_star = qsurf_dim * Lref / (k_air * dTref)
    """
    q_line = float(experiment.initial_conditions.heat_length)
    wire_d = float(experiment.dimensions.wire.diameter)
    k_air_dim = float(experiment.fluid.properties["k"])

    qsurf_dim = q_line / (math.pi * wire_d)
    qn_star = qsurf_dim * float(scales.Lref) / (
        k_air_dim * float(scales.dTref)
    )
    return float(qn_star)


def build_uniform_qn_air(mesh: fenics.Mesh, qn_star_value: float) -> fenics.Function:
    """
    Build DG0 constant qn_air on the current air/star mesh.
    """
    V0 = fenics.FunctionSpace(mesh, "DG", 0)
    qn_air = fenics.Function(V0, name="qn_air")
    qn_air.vector()[:] = float(qn_star_value)
    qn_air.vector().apply("insert")
    return qn_air


def _global_mesh_bounds(mesh: fenics.Mesh):
    """
    Return global xmin, xmax, ymin, ymax for a distributed legacy FEniCS mesh.
    """
    coords = mesh.coordinates()

    if coords.shape[0] == 0:
        local = np.array([np.inf, -np.inf, np.inf, -np.inf], dtype=float)
    else:
        local = np.array(
            [
                coords[:, 0].min(),
                coords[:, 0].max(),
                coords[:, 1].min(),
                coords[:, 1].max(),
            ],
            dtype=float,
        )

    xmin = COMM.allreduce(local[0], op=MPI4Py.MIN)
    xmax = COMM.allreduce(local[1], op=MPI4Py.MAX)
    ymin = COMM.allreduce(local[2], op=MPI4Py.MIN)
    ymax = COMM.allreduce(local[3], op=MPI4Py.MAX)

    return float(xmin), float(xmax), float(ymin), float(ymax)


def rebuild_air_facet_tags(mesh: fenics.Mesh, experiment, scales) -> fenics.MeshFunction:
    """
    Rebuild air facet tags geometrically on the current nondimensional/star mesh.

    Expected important tags:
        OUTER_AIR_TAG = 101
        INTERFACE_TAG = 102
    """
    tdim = mesh.topology().dim()
    ft = fenics.MeshFunction("size_t", mesh, tdim - 1, 0)

    xmin, xmax, ymin, ymax = _global_mesh_bounds(mesh)

    span = max(1.0, xmax - xmin, ymax - ymin)
    tol_outer = 1.0e-8 * span

    r = (float(experiment.dimensions.wire.diameter) / 2.0) / float(scales.Lref)

    # Must match params_bcs.py::Hot_wall
    yc = (
        float(experiment.dimensions.domain.y_max) / float(scales.Lref) / 10.0
        + 11.0 * r
    )

    tol_wire = max(1.0e-12, 1.0e-1 * r)

    class OuterBoundary(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and (
                    fenics.near(x[0], xmin, tol_outer)
                    or fenics.near(x[0], xmax, tol_outer)
                    or fenics.near(x[1], ymin, tol_outer)
                    or fenics.near(x[1], ymax, tol_outer)
                )
            )

    class WireInterface(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and fenics.near(
                    x[0] ** 2 + (x[1] - yc) ** 2 - r * r,
                    0.0,
                    eps=tol_wire,
                )
                and x[1] >= yc - r - 1.0e-12
                and x[1] <= yc + r + 1.0e-12
            )

    # Mark outer first, then wire last so the wire wins if there is overlap.
    OuterBoundary().mark(ft, OUTER_AIR_TAG)
    WireInterface().mark(ft, INTERFACE_TAG)

    local_tags = set(int(v) for v in ft.array())
    all_tags = COMM.gather(local_tags, root=0)

    if is_rank0():
        tags = sorted(set().union(*all_tags))
        print0(f"[AMR] mesh bounds: xmin={xmin:.6e}, xmax={xmax:.6e}, ymin={ymin:.6e}, ymax={ymax:.6e}")
        print0(f"[AMR] rebuilt facet tags: {tags}")

    COMM.Barrier()
    return ft


def mark_cells_for_refinement(
    mesh: fenics.Mesh,
    theta: fenics.Function,
    u: fenics.Function | None = None,
    top_fraction: float = 0.05,
) -> fenics.MeshFunction:
    """
    Mark the top_fraction cells according to a simple plume indicator.

    Primary indicator:
        eta_K = h_K |grad(theta)|

    Optional velocity contribution:
        eta_K = h_K ( |grad(theta)| + 0.25 |grad(u)| )

    This is deliberately simple and robust for legacy FEniCS.
    """
    top_fraction = float(top_fraction)
    if not (0.0 < top_fraction < 1.0):
        raise ValueError(f"top_fraction must be in (0, 1), got {top_fraction}")

    V0 = fenics.FunctionSpace(mesh, "DG", 0)
    h = fenics.CellDiameter(mesh)

    grad_theta_mag = fenics.sqrt(
        fenics.inner(fenics.grad(theta), fenics.grad(theta))
        + fenics.DOLFIN_EPS
    )

    if u is None:
        eta_expr = h * grad_theta_mag
    else:
        grad_u_mag = fenics.sqrt(
            fenics.inner(fenics.grad(u), fenics.grad(u))
            + fenics.DOLFIN_EPS
        )
        eta_expr = h * (grad_theta_mag + fenics.Constant(0.25) * grad_u_mag)

    eta = fenics.project(eta_expr, V0, solver_type="mumps")
    vals_local = eta.vector().get_local()

    if vals_local.size == 0:
        local_threshold = 1.0e100
    else:
        local_threshold = float(np.quantile(vals_local, 1.0 - top_fraction))

    # In MPI, each rank has local vals. A conservative global threshold can be
    # approximated by taking the minimum of local quantiles, marking slightly more
    # cells rather than too few.
    threshold = COMM.allreduce(local_threshold, op=MPI4Py.MIN)

    markers = fenics.MeshFunction("bool", mesh, mesh.topology().dim())
    markers.set_all(False)

    dofmap = V0.dofmap()
    vals = eta.vector().get_local()

    n_marked_local = 0
    for cell in fenics.cells(mesh):
        dof = dofmap.cell_dofs(cell.index())[0]
        if vals[dof] >= threshold:
            markers[cell] = True
            n_marked_local += 1

    n_marked = COMM.allreduce(n_marked_local, op=MPI4Py.SUM)
    print0(f"[AMR] marked {n_marked} cells for refinement; threshold={threshold:.6e}")

    return markers


def load_checkpoint_on_own_mesh(checkpoint_dir: str):
    """
    Load a restart checkpoint using the mesh stored in state.h5.

    This is the AMR-compatible version of load_true_restart_checkpoint(...).
    Your current loader reads p/u/theta into an already-existing W; that is not
    enough after refinement.
    """
    checkpoint_dir = str(checkpoint_dir)
    h5_path = os.path.join(checkpoint_dir, "state.h5")
    meta_path = os.path.join(checkpoint_dir, "state.json")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing restart file: {h5_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing restart metadata: {meta_path}")

    mesh = fenics.Mesh()
    h5 = fenics.HDF5File(MPI.comm_world, h5_path, "r")
    h5.read(mesh, "/mesh", False)

    W = build_mixed_space_on_mesh(mesh)

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

    w = fenics.Function(W)
    w_n = fenics.Function(W)

    assign_p = fenics.FunctionAssigner(W.sub(0), Vp)
    assign_u = fenics.FunctionAssigner(W.sub(1), Vu)
    assign_T = fenics.FunctionAssigner(W.sub(2), VT)

    assign_p.assign(w_n.sub(0), p_star)
    assign_u.assign(w_n.sub(1), u_star)
    assign_T.assign(w_n.sub(2), theta_star)
    w_n.vector().apply("insert")

    w.assign(w_n)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    meta["source"] = "checkpoint_own_mesh"

    try:
        mesh.bounding_box_tree().build(mesh)
    except Exception:
        pass

    print0(
        f"[AMR] loaded checkpoint mesh: "
        f"cells={mesh.num_cells()}, vertices={mesh.num_vertices()}"
    )
    print0(
        f"[AMR] loaded checkpoint state: "
        f"step={meta.get('step')}, time={meta.get('time')}, dt={meta.get('dt')}"
    )

    return mesh, W, w, w_n, meta


def write_checkpoint_with_mesh(
    checkpoint_dir: str,
    mesh: fenics.Mesh,
    w_n: fenics.Function,
    meta: dict,
):
    """
    AMR-compatible checkpoint writer.

    This writes the same datasets as your existing save_restart_checkpoint:
        /mesh
        /p_star
        /u_star
        /theta_star

    It also preserves extra metadata fields, unlike save_restart_checkpoint(...)
    which currently only writes step/time/dt.
    """
    checkpoint_dir = str(checkpoint_dir)
    mkdir0(checkpoint_dir)

    p_star, u_star, theta_star = w_n.split(deepcopy=True)

    h5_path = os.path.join(checkpoint_dir, "state.h5")
    meta_path = os.path.join(checkpoint_dir, "state.json")

    h5 = fenics.HDF5File(mesh.mpi_comm(), h5_path, "w")
    h5.write(mesh, "/mesh")
    h5.write(p_star, "/p_star")
    h5.write(u_star, "/u_star")
    h5.write(theta_star, "/theta_star")
    h5.close()

    if is_rank0():
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    COMM.Barrier()

    print0(f"[AMR] wrote checkpoint: {checkpoint_dir}")


def assign_split_to_mixed(
    W: fenics.FunctionSpace,
    p: fenics.Function,
    u: fenics.Function,
    theta: fenics.Function,
) -> fenics.Function:
    """
    Assign split scalar/vector/scalar fields into mixed W.
    """
    Vp, _ = W.sub(0).collapse(True)
    Vu, _ = W.sub(1).collapse(True)
    VT, _ = W.sub(2).collapse(True)

    # Ensure fields are on exactly the right collapsed spaces.
    p_i = fenics.interpolate(p, Vp)
    u_i = fenics.interpolate(u, Vu)
    theta_i = fenics.interpolate(theta, VT)

    w_new = fenics.Function(W)

    fenics.FunctionAssigner(W.sub(0), Vp).assign(w_new.sub(0), p_i)
    fenics.FunctionAssigner(W.sub(1), Vu).assign(w_new.sub(1), u_i)
    fenics.FunctionAssigner(W.sub(2), VT).assign(w_new.sub(2), theta_i)

    w_new.vector().apply("insert")
    return w_new


def refine_checkpoint_offline(
    input_checkpoint_dir: str,
    output_checkpoint_dir: str,
    top_fraction: float = 0.05,
    levels: int = 1,
    dt_factor: float = 0.25,
):
    """
    Offline migration:

        old restart checkpoint
            -> mark cells
            -> refine mesh
            -> interpolate p/u/theta
            -> write new restart checkpoint

    This function does not require experiment geometry, because it only transfers
    the state. Facet tags and qn_air are rebuilt later during resume.
    """
    input_checkpoint_dir = str(input_checkpoint_dir)
    output_checkpoint_dir = str(output_checkpoint_dir)
    levels = int(levels)

    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")

    mesh, W, w, w_n, meta = load_checkpoint_on_own_mesh(input_checkpoint_dir)

    p_old, u_old, theta_old = w_n.split(deepcopy=True)

    for f in (p_old, u_old, theta_old):
        f.set_allow_extrapolation(True)

    mesh_new = mesh

    for lev in range(levels):
        print0(f"[AMR] offline refinement level {lev + 1}/{levels}")

        markers = mark_cells_for_refinement(
            mesh_new,
            theta_old,
            u=u_old,
            top_fraction=top_fraction,
        )

        mesh_new = fenics.refine(mesh_new, markers)

        try:
            mesh_new.bounding_box_tree().build(mesh_new)
        except Exception:
            pass

        # Prepare fields for another possible level.
        Vp_tmp = fenics.FunctionSpace(mesh_new, "CG", 1)
        Vu_tmp = fenics.VectorFunctionSpace(mesh_new, "CG", 2)
        VT_tmp = fenics.FunctionSpace(mesh_new, "CG", 1)

        for f in (p_old, u_old, theta_old):
            f.set_allow_extrapolation(True)

        p_old = fenics.interpolate(p_old, Vp_tmp)
        u_old = fenics.interpolate(u_old, Vu_tmp)
        theta_old = fenics.interpolate(theta_old, VT_tmp)

        print0(
            f"[AMR] refined mesh now has "
            f"cells={mesh_new.num_cells()}, vertices={mesh_new.num_vertices()}"
        )

    W_new = build_mixed_space_on_mesh(mesh_new)
    w_n_new = assign_split_to_mixed(W_new, p_old, u_old, theta_old)

    meta = dict(meta)
    old_dt = float(meta.get("dt", 1.0e-5))
    meta["dt"] = float(dt_factor) * old_dt
    meta["amr_refined"] = True
    meta["amr_levels_added"] = int(levels)
    meta["amr_top_fraction"] = float(top_fraction)
    meta["amr_parent_checkpoint"] = os.path.abspath(input_checkpoint_dir)
    meta["amr_dt_factor"] = float(dt_factor)

    write_checkpoint_with_mesh(
        output_checkpoint_dir,
        mesh_new,
        w_n_new,
        meta,
    )

    return output_checkpoint_dir


def prepare_loaded_checkpoint_for_base_run(
    checkpoint_dir: str,
    experiment,
):
    """
    Load a checkpoint on its own mesh and rebuild the objects base_version needs.
    """
    scales = compute_nondimensional_scales(experiment)

    sub_mesh_star, W, w, w_n, restart_meta = load_checkpoint_on_own_mesh(checkpoint_dir)

    # The checkpoint mesh is already nondimensional/star-scaled.
    sub_mesh_dim = sub_mesh_star

    sub_ft_star = rebuild_air_facet_tags(sub_mesh_star, experiment, scales)
    sub_dx_star = fenics.Measure("dx", domain=sub_mesh_star)
    sub_ds_star = fenics.Measure("ds", domain=sub_mesh_star, subdomain_data=sub_ft_star)

    sub_ft_dim = sub_ft_star
    sub_dx_dim = sub_dx_star
    sub_ds_dim = sub_ds_star

    qn_star_value = compute_uniform_qn_star_value(experiment, scales)
    qn_air_star = build_uniform_qn_air(sub_mesh_star, qn_star_value)

    psi_p, psi_u, psi_T = fenics.TestFunctions(W)

    p_ufl, u_ufl, theta_ufl = fenics.split(w)

    theta_checkpoint = w_n.sub(2, deepcopy=True)

    mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc = set_param(
        sub_mesh_star,
        theta_checkpoint,
        theta_ufl,
        0.0,
        experiment.fluid.properties["rho"],
        experiment.fluid.properties["beta"],
        experiment,
    )

    return {
        "sub_mesh_star": sub_mesh_star,
        "sub_mesh_dim": sub_mesh_dim,
        "W": W,
        "w": w,
        "w_n": w_n,
        "psi_p": psi_p,
        "psi_u": psi_u,
        "psi_T": psi_T,
        "sub_ft_star": sub_ft_star,
        "sub_dx_star": sub_dx_star,
        "sub_ds_star": sub_ds_star,
        "sub_ft_dim": sub_ft_dim,
        "sub_dx_dim": sub_dx_dim,
        "sub_ds_dim": sub_ds_dim,
        "qn_air_star": qn_air_star,
        "restart_meta": restart_meta,
        "T_air_bc": T_air_bc,
        "T_c": T_c,
        "mu": mu,
        "Pr": Pr,
        "Ra": Ra,
        "f_b": f_b,
        "T_h": T_h,
        "T_ref": T_ref,
    }

def prepare_loaded_checkpoint_for_abe_run(
    checkpoint_dir: str,
    experiment,
):
    """
    Load a checkpoint on its own mesh and rebuild the objects abe_version needs.
    """
    scales = compute_nondimensional_scales(experiment)

    sub_mesh_star, W, w, w_n, restart_meta = load_checkpoint_on_own_mesh(checkpoint_dir)

    # The checkpoint mesh is already nondimensional/star-scaled.
    sub_mesh_dim = sub_mesh_star

    sub_ft_star = rebuild_air_facet_tags(sub_mesh_star, experiment, scales)
    sub_dx_star = fenics.Measure("dx", domain=sub_mesh_star)
    sub_ds_star = fenics.Measure("ds", domain=sub_mesh_star, subdomain_data=sub_ft_star)

    sub_ft_dim = sub_ft_star
    sub_dx_dim = sub_dx_star
    sub_ds_dim = sub_ds_star

    qn_star_value = compute_uniform_qn_star_value(experiment, scales)
    qn_air_star = build_uniform_qn_air(sub_mesh_star, qn_star_value)

    psi_p, psi_u, psi_T = fenics.TestFunctions(W)

    p_ufl, u_ufl, theta_ufl = fenics.split(w)

    theta_checkpoint = w_n.sub(2, deepcopy=True)

    mu, kappa, Pr, Gr, f_b, T_h, T_c, T_ref, T_air_bc = set_param_abe(
        sub_mesh_star,
        theta_checkpoint,
        theta_ufl,
        0.0,
        experiment.fluid.properties["rho"],
        experiment.fluid.properties["beta"],
        experiment,
    )

    return {
        "sub_mesh_star": sub_mesh_star,
        "sub_mesh_dim": sub_mesh_dim,
        "W": W,
        "w": w,
        "w_n": w_n,
        "psi_p": psi_p,
        "psi_u": psi_u,
        "psi_T": psi_T,
        "sub_ft_star": sub_ft_star,
        "sub_dx_star": sub_dx_star,
        "sub_ds_star": sub_ds_star,
        "sub_ft_dim": sub_ft_dim,
        "sub_dx_dim": sub_dx_dim,
        "sub_ds_dim": sub_ds_dim,
        "qn_air_star": qn_air_star,
        "restart_meta": restart_meta,
        "T_air_bc": T_air_bc,
        "T_c": T_c,
        "mu": mu,
        "kappa": kappa,
        "Pr": Pr,
        "Gr": Gr,
        "f_b": f_b,
        "T_h": T_h,
        "T_ref": T_ref,
    }

def mark_cells_for_plume_remesh(
    mesh: fenics.Mesh,
    theta: fenics.Function,
    u: fenics.Function | None,
    experiment,
    scales,
    top_fraction: float = 0.05,
    wire_ring_factor: float = 8.0,
) -> fenics.MeshFunction:
    """
    Mark cells for supervisor-style remeshing on a new coarse mesh.

    Indicator:
        eta = h*|grad(theta)|
            + 0.25*h*|grad(u)|
            + 0.10*|theta|
            + 0.05*|u|

    In addition, force-refine a ring around the wire. This protects the heat-flux
    boundary region even if interpolation smooths the near-wire gradients.
    """
    top_fraction = float(top_fraction)
    if not (0.0 < top_fraction < 1.0):
        raise ValueError(f"top_fraction must be in (0, 1), got {top_fraction}")

    V0 = fenics.FunctionSpace(mesh, "DG", 0)
    h = fenics.CellDiameter(mesh)

    grad_theta_mag = fenics.sqrt(
        fenics.inner(fenics.grad(theta), fenics.grad(theta))
        + fenics.DOLFIN_EPS
    )

    theta_mag = fenics.sqrt(theta * theta + fenics.DOLFIN_EPS)

    eta_expr = h * grad_theta_mag + fenics.Constant(0.10) * theta_mag

    if u is not None:
        grad_u_mag = fenics.sqrt(
            fenics.inner(fenics.grad(u), fenics.grad(u))
            + fenics.DOLFIN_EPS
        )
        u_mag = fenics.sqrt(fenics.inner(u, u) + fenics.DOLFIN_EPS)

        eta_expr += (
            fenics.Constant(0.25) * h * grad_u_mag
            + fenics.Constant(0.05) * u_mag
        )

    eta = fenics.project(eta_expr, V0, solver_type="mumps")
    vals_local = eta.vector().get_local()

    if vals_local.size == 0:
        local_threshold = 1.0e100
    else:
        local_threshold = float(np.quantile(vals_local, 1.0 - top_fraction))

    # Conservative MPI approximation: mark slightly more, not too few.
    threshold = COMM.allreduce(local_threshold, op=MPI4Py.MIN)

    markers = fenics.MeshFunction("bool", mesh, mesh.topology().dim())
    markers.set_all(False)

    dofmap = V0.dofmap()
    vals = eta.vector().get_local()

    # Must match rebuild_air_facet_tags(...) and params_bcs.py::Hot_wall.
    r_wire = (
        float(experiment.dimensions.wire.diameter) / 2.0
    ) / float(scales.Lref)

    yc = (
        float(experiment.dimensions.domain.y_max) / float(scales.Lref) / 10.0
        + 11.0 * r_wire
    )

    ring_radius = float(wire_ring_factor) * r_wire

    n_marked_local = 0

    for cell in fenics.cells(mesh):
        dof = dofmap.cell_dofs(cell.index())[0]
        mp = cell.midpoint()
        x = mp.x()
        y = mp.y()

        dist = math.sqrt(x * x + (y - yc) * (y - yc))

        indicator_mark = vals[dof] >= threshold
        wire_ring_mark = dist <= ring_radius

        if indicator_mark or wire_ring_mark:
            markers[cell] = True
            n_marked_local += 1

    n_marked = COMM.allreduce(n_marked_local, op=MPI4Py.SUM)

    print0(
        f"[REMESH] marked {n_marked} cells; "
        f"threshold={threshold:.6e}, "
        f"wire_ring_radius={ring_radius:.6e}"
    )

    return markers

def remesh_checkpoint_from_coarse_mesh(
    input_checkpoint_dir: str,
    coarse_air_cells_xdmf: str,
    coarse_air_facets_xdmf: str,
    output_checkpoint_dir: str,
    experiment,
    top_fraction: float = 0.05,
    levels: int = 2,
    dt_factor: float = 0.25,
    wire_ring_factor: float = 8.0,
):
    """
    Supervisor-style remeshing workflow:

        old checkpoint on old mesh
            -> load old p/u/theta
            -> read new coarse air mesh
            -> scale new coarse mesh to star coordinates
            -> interpolate old solution onto new coarse mesh
            -> refine new mesh according to transferred plume solution
            -> write checkpoint on the new refined mesh

    The written checkpoint can then be used with:
        --restart-from-checkpoint-mesh <output_checkpoint_dir>
    """
    from utils.geometry import read_mesh
    from utils.transfer import scale_mesh_inplace

    input_checkpoint_dir = str(input_checkpoint_dir)
    output_checkpoint_dir = str(output_checkpoint_dir)
    levels = int(levels)

    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")

    scales = compute_nondimensional_scales(experiment)

    # ------------------------------------------------------------
    # 1. Load old checkpoint on its own old star mesh
    # ------------------------------------------------------------
    old_mesh, old_W, old_w, old_w_n, meta = load_checkpoint_on_own_mesh(
        input_checkpoint_dir
    )

    p_old, u_old, theta_old = old_w_n.split(deepcopy=True)

    for f in (p_old, u_old, theta_old):
        try:
            f.set_allow_extrapolation(True)
        except Exception:
            pass

    print0(
        f"[REMESH] old checkpoint mesh: "
        f"cells={old_mesh.num_cells()}, vertices={old_mesh.num_vertices()}"
    )

    # ------------------------------------------------------------
    # 2. Read new coarse AIR mesh in dimensional coordinates
    # ------------------------------------------------------------
    MESH_NAME = "Grid"

    mesh_new, air_ct, air_ft, _, sub_dx_dim, _, sub_mc_dim, sub_ft_dim = read_mesh(
        coarse_air_cells_xdmf,
        coarse_air_facets_xdmf,
        MESH_NAME,
        PRINT_TAG_SUMMARY,
    )

    print0(
        f"[REMESH] coarse dimensional air mesh: "
        f"cells={mesh_new.num_cells()}, vertices={mesh_new.num_vertices()}"
    )

    # ------------------------------------------------------------
    # 3. Scale new mesh to star coordinates
    # ------------------------------------------------------------
    scale_mesh_inplace(mesh_new, float(scales.Lref))

    try:
        mesh_new.bounding_box_tree().build(mesh_new)
    except Exception:
        pass

    print0(
        f"[REMESH] coarse star air mesh: "
        f"cells={mesh_new.num_cells()}, vertices={mesh_new.num_vertices()}"
    )

    # ------------------------------------------------------------
    # 4. Transfer old fields onto the new coarse mesh
    # ------------------------------------------------------------
    Vp_new = fenics.FunctionSpace(mesh_new, "CG", 1)
    Vu_new = fenics.VectorFunctionSpace(mesh_new, "CG", 2)
    VT_new = fenics.FunctionSpace(mesh_new, "CG", 1)

    p_new = fenics.interpolate(p_old, Vp_new)
    u_new = fenics.interpolate(u_old, Vu_new)
    theta_new = fenics.interpolate(theta_old, VT_new)

    p_new.rename("p_star", "p_star")
    u_new.rename("u_star", "u_star")
    theta_new.rename("theta_star", "theta_star")

    print0(
        f"[REMESH] transferred theta range: "
        f"min={global_vec_min(theta_new):.6e}, "
        f"max={global_vec_max(theta_new):.6e}"
    )
    print0(
        f"[REMESH] transferred |u| vector range proxy: "
        f"min={global_vec_min(u_new):.6e}, "
        f"max={global_vec_max(u_new):.6e}"
    )

    # ------------------------------------------------------------
    # 5. Refine new mesh using solution-based indicator
    # ------------------------------------------------------------
    for lev in range(levels):
        print0(f"[REMESH] refinement level {lev + 1}/{levels}")

        for f in (p_new, u_new, theta_new):
            try:
                f.set_allow_extrapolation(True)
            except Exception:
                pass

        markers = mark_cells_for_plume_remesh(
            mesh=mesh_new,
            theta=theta_new,
            u=u_new,
            experiment=experiment,
            scales=scales,
            top_fraction=top_fraction,
            wire_ring_factor=wire_ring_factor,
        )

        mesh_refined = fenics.refine(mesh_new, markers)

        try:
            mesh_refined.bounding_box_tree().build(mesh_refined)
        except Exception:
            pass

        Vp_ref = fenics.FunctionSpace(mesh_refined, "CG", 1)
        Vu_ref = fenics.VectorFunctionSpace(mesh_refined, "CG", 2)
        VT_ref = fenics.FunctionSpace(mesh_refined, "CG", 1)

        for f in (p_new, u_new, theta_new):
            try:
                f.set_allow_extrapolation(True)
            except Exception:
                pass

        p_new = fenics.interpolate(p_new, Vp_ref)
        u_new = fenics.interpolate(u_new, Vu_ref)
        theta_new = fenics.interpolate(theta_new, VT_ref)

        p_new.rename("p_star", "p_star")
        u_new.rename("u_star", "u_star")
        theta_new.rename("theta_star", "theta_star")

        mesh_new = mesh_refined

        print0(
            f"[REMESH] refined mesh now has "
            f"cells={mesh_new.num_cells()}, vertices={mesh_new.num_vertices()}"
        )
        print0(
            f"[REMESH] theta range after level {lev + 1}: "
            f"min={global_vec_min(theta_new):.6e}, "
            f"max={global_vec_max(theta_new):.6e}"
        )

    # ------------------------------------------------------------
    # 6. Assemble mixed checkpoint state
    # ------------------------------------------------------------
    W_new = build_mixed_space_on_mesh(mesh_new)
    w_n_new = assign_split_to_mixed(W_new, p_new, u_new, theta_new)

    # ------------------------------------------------------------
    # 7. Update metadata and write checkpoint
    # ------------------------------------------------------------
    meta = dict(meta)

    old_dt = float(meta.get("dt", 1.0e-5))
    meta["dt"] = float(dt_factor) * old_dt

    meta["source"] = "remeshed_from_coarse_mesh"
    meta["remeshed"] = True
    meta["remesh_parent_checkpoint"] = os.path.abspath(input_checkpoint_dir)
    meta["remesh_coarse_air_cells"] = os.path.abspath(coarse_air_cells_xdmf)
    meta["remesh_coarse_air_facets"] = os.path.abspath(coarse_air_facets_xdmf)
    meta["remesh_levels"] = int(levels)
    meta["remesh_top_fraction"] = float(top_fraction)
    meta["remesh_wire_ring_factor"] = float(wire_ring_factor)
    meta["remesh_dt_factor"] = float(dt_factor)

    write_checkpoint_with_mesh(
        output_checkpoint_dir,
        mesh_new,
        w_n_new,
        meta,
    )

    print0(f"[REMESH] wrote remeshed checkpoint: {output_checkpoint_dir}")

    return output_checkpoint_dir

def foreign_checkpoint_to_target_checkpoint(
    input_checkpoint_dir: str,
    coarse_air_cells_xdmf: str,
    coarse_air_facets_xdmf: str,
    output_checkpoint_dir: str,
    source_experiment,
    target_experiment,
    top_fraction: float = 0.05,
    levels: int = 2,
    dt_factor: float = 0.05,
    wire_ring_factor: float = 8.0,
):
    """
    Project a checkpoint from one experiment onto another experiment's mesh.

    This is NOT a true restart. It is a foreign-field initial condition.

    Steps:
      1. Load source checkpoint on its own star mesh.
      2. Rescale source mesh coordinates into target-star coordinates.
      3. Convert source nondimensional fields into target nondimensional fields:
             T_dim = Tinf_src + dTref_src * theta_src
             theta_tgt = (T_dim - Tinf_tgt) / dTref_tgt

             u_dim = Uref_src * u_src
             u_tgt = u_dim / Uref_tgt

             p_dim = rho_src * Uref_src^2 * p_src
             p_tgt = p_dim / (rho_tgt * Uref_tgt^2)
      4. Read target coarse air mesh, scale it with target Lref.
      5. Interpolate converted fields onto target mesh.
      6. Optionally refine using the transferred plume field.
      7. Write a normal restart checkpoint usable by --restart-from-checkpoint-mesh.
    """
    from utils.geometry import read_mesh
    from utils.transfer import scale_mesh_inplace

    input_checkpoint_dir = str(input_checkpoint_dir)
    output_checkpoint_dir = str(output_checkpoint_dir)
    levels = int(levels)

    if levels < 0:
        raise ValueError(f"levels must be >= 0, got {levels}")

    sc_src = compute_nondimensional_scales(source_experiment)
    sc_tgt = compute_nondimensional_scales(target_experiment)

    rho_src = float(source_experiment.fluid.properties["rho"])
    rho_tgt = float(target_experiment.fluid.properties["rho"])

    Tinf_src = float(source_experiment.initial_conditions.temperature)
    Tinf_tgt = float(target_experiment.initial_conditions.temperature)

    # ------------------------------------------------------------
    # 1. Load source checkpoint on its own source-star mesh
    # ------------------------------------------------------------
    old_mesh, old_W, old_w, old_w_n, meta = load_checkpoint_on_own_mesh(
        input_checkpoint_dir
    )

    p_src, u_src, theta_src = old_w_n.split(deepcopy=True)

    # ------------------------------------------------------------
    # 2. Put source mesh into target-star coordinates
    #
    # Old checkpoint coordinates are x_src_star = x_dim / Lref_src.
    # Target mesh coordinates will be x_tgt_star = x_dim / Lref_tgt.
    #
    # Therefore:
    #     x_tgt_star = x_src_star * Lref_src / Lref_tgt
    # ------------------------------------------------------------
    coord_factor = float(sc_src.Lref) / float(sc_tgt.Lref)
    old_mesh.coordinates()[:] *= coord_factor

    try:
        old_mesh.bounding_box_tree().build(old_mesh)
    except Exception:
        pass

    for f in (p_src, u_src, theta_src):
        try:
            f.set_allow_extrapolation(True)
        except Exception:
            pass

    # ------------------------------------------------------------
    # 3. Convert source nondimensional fields to target nondimensional fields
    #    on the rescaled source mesh.
    # ------------------------------------------------------------
    Vp_src = p_src.function_space()
    Vu_src = u_src.function_space()
    VT_src = theta_src.function_space()

    p_tgt_on_src = fenics.Function(Vp_src, name="p_star")
    u_tgt_on_src = fenics.Function(Vu_src, name="u_star")
    theta_tgt_on_src = fenics.Function(VT_src, name="theta_star")

    p_factor = (rho_src * float(sc_src.Uref)**2) / (
        rho_tgt * float(sc_tgt.Uref)**2
    )
    u_factor = float(sc_src.Uref) / float(sc_tgt.Uref)
    theta_factor = float(sc_src.dTref) / float(sc_tgt.dTref)
    theta_shift = (Tinf_src - Tinf_tgt) / float(sc_tgt.dTref)

    p_tgt_on_src.vector()[:] = p_factor * p_src.vector()[:]
    u_tgt_on_src.vector()[:] = u_factor * u_src.vector()[:]
    theta_tgt_on_src.vector()[:] = theta_shift + theta_factor * theta_src.vector()[:]

    p_tgt_on_src.vector().apply("insert")
    u_tgt_on_src.vector().apply("insert")
    theta_tgt_on_src.vector().apply("insert")

    for f in (p_tgt_on_src, u_tgt_on_src, theta_tgt_on_src):
        try:
            f.set_allow_extrapolation(True)
        except Exception:
            pass

    print0("[FOREIGN] source -> target scale conversion")
    print0(f"[FOREIGN] coordinate factor Lsrc/Ltgt = {coord_factor:.6e}")
    print0(f"[FOREIGN] p factor                  = {p_factor:.6e}")
    print0(f"[FOREIGN] u factor                  = {u_factor:.6e}")
    print0(f"[FOREIGN] theta factor              = {theta_factor:.6e}")
    print0(f"[FOREIGN] theta shift               = {theta_shift:.6e}")

    # ------------------------------------------------------------
    # 4. Read target coarse air mesh in dimensional coordinates
    # ------------------------------------------------------------
    MESH_NAME = "Grid"

    mesh_new, air_ct, air_ft, _, sub_dx_dim, _, sub_mc_dim, sub_ft_dim = read_mesh(
        coarse_air_cells_xdmf,
        coarse_air_facets_xdmf,
        MESH_NAME,
        PRINT_TAG_SUMMARY,
    )

    print0(
        f"[FOREIGN] target coarse dimensional air mesh: "
        f"cells={mesh_new.num_cells()}, vertices={mesh_new.num_vertices()}"
    )

    # ------------------------------------------------------------
    # 5. Scale target mesh to target-star coordinates
    # ------------------------------------------------------------
    scale_mesh_inplace(mesh_new, float(sc_tgt.Lref))

    try:
        mesh_new.bounding_box_tree().build(mesh_new)
    except Exception:
        pass

    print0(
        f"[FOREIGN] target coarse star air mesh: "
        f"cells={mesh_new.num_cells()}, vertices={mesh_new.num_vertices()}"
    )

    # ------------------------------------------------------------
    # 6. Interpolate converted source fields onto target mesh
    # ------------------------------------------------------------
    Vp_new = fenics.FunctionSpace(mesh_new, "CG", 1)
    Vu_new = fenics.VectorFunctionSpace(mesh_new, "CG", 2)
    VT_new = fenics.FunctionSpace(mesh_new, "CG", 1)

    p_new = fenics.interpolate(p_tgt_on_src, Vp_new)
    u_new = fenics.interpolate(u_tgt_on_src, Vu_new)
    theta_new = fenics.interpolate(theta_tgt_on_src, VT_new)

    p_new.rename("p_star", "p_star")
    u_new.rename("u_star", "u_star")
    theta_new.rename("theta_star", "theta_star")

    print0(
        f"[FOREIGN] transferred theta range: "
        f"min={global_vec_min(theta_new):.6e}, "
        f"max={global_vec_max(theta_new):.6e}"
    )
    print0(
        f"[FOREIGN] transferred u component range proxy: "
        f"min={global_vec_min(u_new):.6e}, "
        f"max={global_vec_max(u_new):.6e}"
    )

    # Optional but recommended: remove arbitrary pressure mean.
    p_mean = fenics.assemble(p_new * fenics.dx(domain=mesh_new)) / fenics.assemble(
        fenics.Constant(1.0) * fenics.dx(domain=mesh_new)
    )
    p_new.vector()[:] -= float(p_mean)
    p_new.vector().apply("insert")

    # ------------------------------------------------------------
    # 7. Refine target mesh using transferred target-scaled fields
    # ------------------------------------------------------------
    for lev in range(levels):
        print0(f"[FOREIGN] refinement level {lev + 1}/{levels}")

        for f in (p_new, u_new, theta_new):
            try:
                f.set_allow_extrapolation(True)
            except Exception:
                pass

        markers = mark_cells_for_plume_remesh(
            mesh=mesh_new,
            theta=theta_new,
            u=u_new,
            experiment=target_experiment,
            scales=sc_tgt,
            top_fraction=top_fraction,
            wire_ring_factor=wire_ring_factor,
        )

        mesh_refined = fenics.refine(mesh_new, markers)

        try:
            mesh_refined.bounding_box_tree().build(mesh_refined)
        except Exception:
            pass

        Vp_ref = fenics.FunctionSpace(mesh_refined, "CG", 1)
        Vu_ref = fenics.VectorFunctionSpace(mesh_refined, "CG", 2)
        VT_ref = fenics.FunctionSpace(mesh_refined, "CG", 1)

        for f in (p_new, u_new, theta_new):
            try:
                f.set_allow_extrapolation(True)
            except Exception:
                pass

        p_new = fenics.interpolate(p_new, Vp_ref)
        u_new = fenics.interpolate(u_new, Vu_ref)
        theta_new = fenics.interpolate(theta_new, VT_ref)

        p_new.rename("p_star", "p_star")
        u_new.rename("u_star", "u_star")
        theta_new.rename("theta_star", "theta_star")

        mesh_new = mesh_refined

        print0(
            f"[FOREIGN] refined mesh now has "
            f"cells={mesh_new.num_cells()}, vertices={mesh_new.num_vertices()}"
        )

    # ------------------------------------------------------------
    # 8. Assemble and write checkpoint
    # ------------------------------------------------------------
    W_new = build_mixed_space_on_mesh(mesh_new)
    w_n_new = assign_split_to_mixed(W_new, p_new, u_new, theta_new)

    meta = dict(meta)
    old_dt = float(meta.get("dt", 1.0e-5))

    meta["step"] = 0
    meta["time"] = 0.0
    meta["dt"] = float(dt_factor) * old_dt
    meta["source"] = "foreign_projected_checkpoint"
    meta["foreign_projected"] = True
    meta["foreign_parent_checkpoint"] = os.path.abspath(input_checkpoint_dir)
    meta["foreign_source_experiment"] = source_experiment.name
    meta["foreign_target_experiment"] = target_experiment.name
    meta["foreign_coordinate_factor"] = coord_factor
    meta["foreign_p_factor"] = p_factor
    meta["foreign_u_factor"] = u_factor
    meta["foreign_theta_factor"] = theta_factor
    meta["foreign_theta_shift"] = theta_shift
    meta["foreign_dt_factor"] = float(dt_factor)

    write_checkpoint_with_mesh(
        output_checkpoint_dir,
        mesh_new,
        w_n_new,
        meta,
    )

    print0(f"[FOREIGN] wrote projected checkpoint: {output_checkpoint_dir}")
    return output_checkpoint_dir

def checkpoint_from_xdmf_snapshots(
    output_checkpoint_dir: str,
    mesh_xdmf: str,
    p_xdmf: str,
    u_xdmf: str,
    T_xdmf: str,
    experiment,
    step: int = 0,
    time_value: float = 0.0,
    dt_value: float = 1.0e-6,
):
    """
    Build a normal restart checkpoint from dimensional visualization XDMF/H5 files.

    Expected input fields:
        p_xdmf : dimensional pressure [Pa] or equivalent project output pressure
        u_xdmf : dimensional velocity [m/s]
        T_xdmf : dimensional absolute temperature [K]

    Output checkpoint contains nondimensional:
        /mesh       star-scaled mesh
        /p_star
        /u_star
        /theta_star
    """

    import xml.etree.ElementTree as ET
    from utils.geometry import read_mesh
    from utils.transfer import scale_mesh_inplace

    output_checkpoint_dir = str(output_checkpoint_dir)

    scales = compute_nondimensional_scales(experiment)
    T_ambient = float(292.96)  # K, for nondimensional temperature conversion

    def infer_attr_name(xdmf_path):
        root = ET.parse(xdmf_path).getroot()
        for node in root.iter():
            if node.tag.endswith("Attribute"):
                name = node.attrib.get("Name", "").strip()
                if name:
                    return name
        raise RuntimeError(f"Could not infer Attribute Name from {xdmf_path}")

    def _first_dataitem_for_attribute(xdmf_path):
        """
        Return the HDF5 file and dataset path referenced by the first Attribute
        in a visualization-style XDMF file.

        Typical XDMF text looks like:
            air_temperature_transient_72000.h5:/VisualisationVector/0
        """
        root = ET.parse(xdmf_path).getroot()

        for attr in root.iter():
            if not attr.tag.endswith("Attribute"):
                continue

            attr_name = attr.attrib.get("Name", "").strip()

            for dataitem in attr.iter():
                if not dataitem.tag.endswith("DataItem"):
                    continue

                text = (dataitem.text or "").strip()
                if ".h5:" in text or ".hdf5:" in text:
                    h5_name, h5_dataset = text.split(":", 1)

                    h5_name = h5_name.strip()
                    h5_dataset = h5_dataset.strip()

                    if not os.path.isabs(h5_name):
                        h5_name = os.path.join(os.path.dirname(xdmf_path), h5_name)

                    return attr_name, h5_name, h5_dataset

        raise RuntimeError(f"Could not find HDF5 DataItem in {xdmf_path}")


    def _read_visualization_h5_array(xdmf_path):
        attr_name, h5_path, h5_dataset = _first_dataitem_for_attribute(xdmf_path)

        print0(
            f"[XDMF->CHK] reading visualization field "
            f"attr='{attr_name}', h5='{h5_path}', dataset='{h5_dataset}'"
        )

        with h5py.File(h5_path, "r") as h5:
            data = np.asarray(h5[h5_dataset])

        return data


    def read_xdmf_scalar_cg1(xdmf_path, function):
        """
        Read scalar vertex visualization data into a CG1 scalar Function.

        Must be run on the same mesh as the XDMF geometry.
        Recommended: run this reconstruction step in serial.
        """
        V = function.function_space()
        mesh = V.mesh()

        data = _read_visualization_h5_array(xdmf_path)

        if data.ndim == 2:
            if data.shape[1] != 1:
                raise RuntimeError(
                    f"Expected scalar data in {xdmf_path}, got shape {data.shape}"
                )
            data = data[:, 0]

        n_vertices = mesh.num_vertices()
        if data.shape[0] != n_vertices:
            raise RuntimeError(
                f"Scalar data size mismatch for {xdmf_path}: "
                f"data has {data.shape[0]} rows, mesh has {n_vertices} vertices. "
                f"This usually means the field XDMF and mesh XDMF are not from the same run."
            )

        v2d = fenics.vertex_to_dof_map(V)
        function.vector()[:] = 0.0
        function.vector()[v2d] = data
        function.vector().apply("insert")

        return function


    def read_xdmf_vector_cg1(xdmf_path, function):
        """
        Read vector vertex visualization data into a vector CG1 Function.

        If the target velocity space later needs to be P2, assign_split_to_mixed()
        will interpolate this CG1 velocity into the mixed P2 velocity space.
        """
        Vu = function.function_space()
        mesh = Vu.mesh()

        data = _read_visualization_h5_array(xdmf_path)

        if data.ndim != 2 or data.shape[1] < 2:
            raise RuntimeError(
                f"Expected vector data with at least two columns in {xdmf_path}, "
                f"got shape {data.shape}"
            )

        n_vertices = mesh.num_vertices()
        if data.shape[0] != n_vertices:
            raise RuntimeError(
                f"Vector data size mismatch for {xdmf_path}: "
                f"data has {data.shape[0]} rows, mesh has {n_vertices} vertices. "
                f"This usually means the field XDMF and mesh XDMF are not from the same run."
            )

        V1 = fenics.FunctionSpace(mesh, "CG", 1)
        ux = fenics.Function(V1)
        uy = fenics.Function(V1)

        v2d = fenics.vertex_to_dof_map(V1)

        ux.vector()[:] = 0.0
        uy.vector()[:] = 0.0

        ux.vector()[v2d] = data[:, 0]
        uy.vector()[v2d] = data[:, 1]

        ux.vector().apply("insert")
        uy.vector().apply("insert")

        assigner = fenics.FunctionAssigner(Vu, [V1, V1])
        assigner.assign(function, [ux, uy])
        function.vector().apply("insert")

        return function

    # ------------------------------------------------------------------
    # 1. Read old dimensional mesh.
    # ------------------------------------------------------------------
    mesh = fenics.Mesh()

    with fenics.XDMFFile(COMM, mesh_xdmf) as xdmf:
        xdmf.read(mesh)

    try:
        mesh.bounding_box_tree().build(mesh)
    except Exception:
        pass

    print0(
        f"[XDMF->CHK] read dimensional mesh: "
        f"cells={mesh.num_cells()}, vertices={mesh.num_vertices()}"
    )

    # ------------------------------------------------------------------
    # 2. Read dimensional fields on the old dimensional mesh.
    # ------------------------------------------------------------------
    Vp_dim = fenics.FunctionSpace(mesh, "CG", 1)
    Vu_dim = fenics.VectorFunctionSpace(mesh, "CG", 1)
    VT_dim = fenics.FunctionSpace(mesh, "CG", 1)

    p_dim = fenics.Function(Vp_dim)
    u_dim = fenics.Function(Vu_dim)
    T_dim = fenics.Function(VT_dim)

    read_xdmf_scalar_cg1(p_xdmf, p_dim)
    read_xdmf_vector_cg1(u_xdmf, u_dim)
    read_xdmf_scalar_cg1(T_xdmf, T_dim)

    print0(
        f"[XDMF->CHK] loaded fields: "
        f"T_min={global_vec_min(T_dim):.6e}, "
        f"T_max={global_vec_max(T_dim):.6e}, "
        f"u_min={global_vec_min(u_dim):.6e}, "
        f"u_max={global_vec_max(u_dim):.6e}"
    )

    # ------------------------------------------------------------------
    # 3. Convert fields to nondimensional variables.
    # ------------------------------------------------------------------
    p_star_dim = fenics.Function(Vp_dim)
    u_star_dim = fenics.Function(Vu_dim)
    theta_dim = fenics.Function(VT_dim)

    # p_star_dim.vector()[:] = p_dim.vector()[:] / float(scales.Pref)
    # u_star_dim.vector()[:] = u_dim.vector()[:] / float(scales.Uref)
    p_star_dim.vector()[:] = p_dim.vector()[:] / float(scales.rho * scales.Uref_abe**2)
    u_star_dim.vector()[:] = u_dim.vector()[:] / float(scales.Uref_abe)
    theta_dim.vector()[:] = (T_dim.vector()[:] - T_ambient) / float(scales.dTref)

    p_star_dim.vector().apply("insert")
    u_star_dim.vector().apply("insert")
    theta_dim.vector().apply("insert")

    p_star_dim.rename("p_star", "p_star")
    u_star_dim.rename("u_star", "u_star")
    theta_dim.rename("theta_star", "theta_star")

    print0(
        f"[XDMF->CHK] nondimensional ranges: "
        f"theta_min={global_vec_min(theta_dim):.6e}, "
        f"theta_max={global_vec_max(theta_dim):.6e}, "
        f"u_star_min={global_vec_min(u_star_dim):.6e}, "
        f"u_star_max={global_vec_max(u_star_dim):.6e}"
    )

    # ------------------------------------------------------------------
    # 4. Scale mesh to star coordinates before writing checkpoint.
    # ------------------------------------------------------------------
    scale_mesh_inplace(mesh, float(scales.Lref))

    try:
        mesh.bounding_box_tree().build(mesh)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 5. Assemble mixed state and write checkpoint.
    # ------------------------------------------------------------------
    W = build_mixed_space_on_mesh(mesh)
    w_n = assign_split_to_mixed(W, p_star_dim, u_star_dim, theta_dim)

    meta = {
        "step": int(step),
        "time": float(time_value),
        "dt": float(dt_value),
        "source": "reconstructed_from_xdmf_snapshots",
        "xdmf_mesh": os.path.abspath(mesh_xdmf),
        "xdmf_pressure": os.path.abspath(p_xdmf),
        "xdmf_velocity": os.path.abspath(u_xdmf),
        "xdmf_temperature": os.path.abspath(T_xdmf),
    }

    write_checkpoint_with_mesh(
        output_checkpoint_dir,
        mesh,
        w_n,
        meta,
    )

    print0(f"[XDMF->CHK] wrote reconstructed checkpoint: {output_checkpoint_dir}")
    return output_checkpoint_dir

