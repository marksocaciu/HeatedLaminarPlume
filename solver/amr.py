from __future__ import annotations

import os
import json
import shutil
from pathlib import Path

from utils.imports import *
from solver.scales import compute_nondimensional_scales
from solver.params_bcs import set_bcs


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


def rebuild_air_facet_tags(mesh: fenics.Mesh, experiment, scales) -> fenics.MeshFunction:
    """
    Rebuild air facet tags geometrically on the current nondimensional/star mesh.

    This is essential after refine(mesh, markers), because the old facet MeshFunction
    is not automatically valid on the refined mesh.

    Tags follow utils.imports:
        SYMMETRY_AIR_TAG = 100
        OUTER_AIR_TAG   = 101
        INTERFACE_TAG   = 102

    Your current set_bcs(...) uses geometric SubDomains for BCs, but the weak heat
    flux term uses sub_ds(INTERFACE_TAG), so INTERFACE_TAG must be correct.
    """
    tdim = mesh.topology().dim()
    ft = fenics.MeshFunction("size_t", mesh, tdim - 1, 0)

    r = (float(experiment.dimensions.wire.diameter) / 2.0) / float(scales.Lref)

    # This matches Hot_wall in solver/params_bcs.py
    yc = (
        float(experiment.dimensions.domain.y_max) / float(scales.Lref) / 10.0
        + 11.0 * r
    )

    x_min = float(experiment.dimensions.domain.x_min) / float(scales.Lref)
    x_max = float(experiment.dimensions.domain.x_max) / float(scales.Lref)
    y_min = float(experiment.dimensions.domain.y_min) / float(scales.Lref)
    y_max = float(experiment.dimensions.domain.y_max) / float(scales.Lref)

    # A slightly relaxed tolerance helps after refinement.
    tol_outer = 1.0e-9
    tol_wire = max(1.0e-12, 1.0e-1 * r)

    class WireInterface(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return (
                on_boundary
                and fenics.near(x[0] ** 2 + (x[1] - yc) ** 2 - r * r, 0.0, eps=tol_wire)
                and x[1] >= yc - r - 1.0e-12
                and x[1] <= yc + r + 1.0e-12
            )

    class WestBoundary(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[0], x_min, tol_outer)

    class EastBoundary(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[0], x_max, tol_outer)

    class SouthBoundary(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[1], y_min, tol_outer)

    class NorthBoundary(fenics.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[1], y_max, tol_outer)

    # For your current equations, both left/right/top/bottom can be treated as
    # outer air for diagnostic flux integration.
    WestBoundary().mark(ft, OUTER_AIR_TAG)
    EastBoundary().mark(ft, OUTER_AIR_TAG)
    SouthBoundary().mark(ft, OUTER_AIR_TAG)
    NorthBoundary().mark(ft, OUTER_AIR_TAG)

    # Mark wire last, so it overrides any accidental default/outer marking.
    WireInterface().mark(ft, INTERFACE_TAG)

    if is_rank0():
        tags = set(ft.array())
        print0(f"[AMR] rebuilt facet tags: {tags}")

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

    Returns:
        sub_mesh_star, sub_mesh_dim,
        W, w, w_n,
        psi_p, psi_u, psi_T,
        sub_ft_star, sub_dx_star, sub_ds_star,
        sub_ft_dim, sub_dx_dim, sub_ds_dim,
        qn_air_star,
        restart_meta,
        T_air_bc, T_c, mu, Pr, Ra, f_b, T_h, T_ref
    """
    from solver.solver import solver

    scales = compute_nondimensional_scales(experiment)

    sub_mesh_star, W, w, w_n, restart_meta = load_checkpoint_on_own_mesh(checkpoint_dir)

    # In your current code, the air mesh is scaled in place and then used as both
    # sub_mesh_dim and sub_mesh_star. For a checkpoint restart, the mesh is already
    # star/nondimensional, so keep the same object.
    sub_mesh_dim = sub_mesh_star

    sub_ft_star = rebuild_air_facet_tags(sub_mesh_star, experiment, scales)
    sub_dx_star = fenics.Measure("dx", domain=sub_mesh_star)
    sub_ds_star = fenics.Measure("ds", domain=sub_mesh_star, subdomain_data=sub_ft_star)

    # Existing postprocessing/diagnostics expect these names too.
    sub_ft_dim = sub_ft_star
    sub_dx_dim = sub_dx_star
    sub_ds_dim = sub_ds_star

    qn_star_value = compute_uniform_qn_star_value(experiment, scales)
    qn_air_star = build_uniform_qn_air(sub_mesh_star, qn_star_value)

    # Recreate solver constants/BC placeholders by calling solver(...) with the
    # checkpoint temperature as initial theta. Then immediately overwrite w/w_n
    # with the checkpoint state again to be safe.
    _, _, _, _, _, _, _, _, _, psi_p, psi_u, psi_T, \
        mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc = solver(
            sub_mesh_star,
            w_n.sub(2, deepcopy=True),
            0.0,
            experiment.fluid.properties["rho"],
            experiment.fluid.properties["beta"],
            experiment,
        )

    # solver(...) creates fresh w/w_n; restore checkpoint-loaded ones.
    # W itself is equivalent, but keep the one from checkpoint load.
    psi_p, psi_u, psi_T = fenics.TestFunctions(W)

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
