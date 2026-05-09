from curses.ascii import SUB
from email.mime import base
import csv
import json

from matplotlib.pyplot import step

from utils.imports import *
from utils.geometry import *
from utils.material import *
from utils.parser import *
from utils.plot import *
from solver.solver import *
from solver.initial import *
from solver.biot import *
from solver.params_bcs import *
from solver.scales import *
from utils.results import *
from utils.transfer import *
from solver.base_solver import *
from solver.abe_solver import *
from solver.temp_solver import *
from solver.amr import *

from dataclasses import dataclass
from pathlib import Path

@dataclass
class MeshPaths:
    run_root: Path
    geo: Path
    msh: Path
    full_cells: Path
    full_facets: Path
    air_cells: Path
    air_facets: Path

def make_mesh_paths(run_root):
    run_root = Path(run_root)
    return MeshPaths(
        run_root=run_root,
        geo=run_root / "geom.geo",
        msh=run_root / "plume.msh",
        full_cells=run_root / "full_cells.xdmf",
        full_facets=run_root / "full_facets.xdmf",
        air_cells=run_root / "air_cells.xdmf",
        air_facets=run_root / "air_facets.xdmf",
    )

def make_run_root(experiment_name: str, mode: str, reuse_existing: str = "") -> str:
    """
    Create one shared run directory for all MPI ranks.

    Rank 0 chooses the directory name, then broadcasts it.
    """
    if reuse_existing:
        run_root = reuse_existing
    else:
        if is_rank0():
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pid = os.getpid()
            run_root = os.path.join(
                experiment_name,
                "runs",
                f"{mode}_{stamp}_pid{pid}",
            )
        else:
            run_root = None

        run_root = COMM.bcast(run_root, root=0)

    if is_rank0():
        os.makedirs(run_root, exist_ok=True)

    COMM.Barrier()
    return run_root

def _infer_xdmf_attribute_name(xdmf_path: str) -> str:
    import xml.etree.ElementTree as ET

    root = ET.parse(xdmf_path).getroot()
    for attr in root.iter():
        if attr.tag.endswith('Attribute'):
            name = attr.attrib.get('Name', '').strip()
            if name:
                return name
    raise RuntimeError(f"Could not infer XDMF Attribute Name from {xdmf_path}")

def _load_checkpoint_snapshot_from_xdmf(xdmf_path: str, function):
    attr_name = _infer_xdmf_attribute_name(xdmf_path)
    with fenics.XDMFFile(function.function_space().mesh().mpi_comm(), xdmf_path) as xdmf:
        try:
            xdmf.read_checkpoint(function, attr_name, 0)
        except RuntimeError:
            xdmf.read(function)
    return function

def _last_transient_step_from_history(history_csv_path: str) -> int:
    if not history_csv_path or not os.path.exists(history_csv_path):
        return -1
    last_step = -1

    with open(history_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                last_step = int(row["step"])
            except Exception:
                continue
    return last_step

def load_true_restart_checkpoint(checkpoint_dir: str, W, w, w_n):
    h5_path = os.path.join(checkpoint_dir, "state.h5")
    meta_path = os.path.join(checkpoint_dir, "state.json")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing restart file: {h5_path}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing restart metadata: {meta_path}")

    Vp_star, _ = W.sub(0).collapse(True)
    Vu_star, _ = W.sub(1).collapse(True)
    VT_star, _ = W.sub(2).collapse(True)

    p_star = fenics.Function(Vp_star)
    u_star = fenics.Function(Vu_star)
    theta_star = fenics.Function(VT_star)

    h5 = fenics.HDF5File(W.mesh().mpi_comm(), h5_path, "r")
    h5.read(p_star, "/p_star")
    h5.read(u_star, "/u_star")
    h5.read(theta_star, "/theta_star")
    h5.close()

    assign_p = fenics.FunctionAssigner(W.sub(0), Vp_star)
    assign_u = fenics.FunctionAssigner(W.sub(1), Vu_star)
    assign_T = fenics.FunctionAssigner(W.sub(2), VT_star)

    assign_p.assign(w_n.sub(0), p_star)
    assign_u.assign(w_n.sub(1), u_star)
    assign_T.assign(w_n.sub(2), theta_star)

    w_n.vector().apply("insert")

    copy_state(w, w_n)

    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    meta["source"] = "true_checkpoint"
    return w, w_n, meta

def approximate_restart_from_last_saved_transient(
    run_root: str,
    mode_subdir: str,
    sub_mesh_dim,
    sub_mesh_star,
    W,
    w,
    w_n,
    scales,
    T_ambient: float,
    rho_air: float,
    fallback_dt: float,
):
    """
    Rebuild an approximate restart state from the latest saved dimensional transient
    XDMF snapshots plus transient_history.csv.

    This is intended for one-off recovery of an interrupted run when no true restart
    checkpoint was written.
    """
    import csv

    base_dir = os.path.join(run_root, mode_subdir)
    history_csv = os.path.join(run_root, 'transient_history.csv')
    step = _last_transient_step_from_history(history_csv)
    if step < 0:
        raise RuntimeError(f"No usable transient_history.csv found at {history_csv}")

    p_xdmf = os.path.join(base_dir, f'air_pressure_transient_{step:05d}.xdmf')
    u_xdmf = os.path.join(base_dir, f'air_velocity_transient_{step:05d}.xdmf')
    T_xdmf = os.path.join(base_dir, f'air_temperature_transient_{step:05d}.xdmf')
    for path in (p_xdmf, u_xdmf, T_xdmf):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing transient snapshot file: {path}")

    # Recover time / dt from history.
    time_value = 0.0
    dt_value = float(fallback_dt)
    with open(history_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row['step']) == step:
                    time_value = float(row['time'])
                    dt_value = float(row['dt'])
            except Exception:
                continue

    # Read dimensional fields from the last snapshot.
    Vp_dim = fenics.FunctionSpace(sub_mesh_dim, 'CG', 1)
    Vu_dim = fenics.VectorFunctionSpace(sub_mesh_dim, 'CG', 2)
    VT_dim = fenics.FunctionSpace(sub_mesh_dim, 'CG', 1)

    p_dim = fenics.Function(Vp_dim)
    u_dim = fenics.Function(Vu_dim)
    T_dim = fenics.Function(VT_dim)

    _load_checkpoint_snapshot_from_xdmf(p_xdmf, p_dim)
    _load_checkpoint_snapshot_from_xdmf(u_xdmf, u_dim)
    _load_checkpoint_snapshot_from_xdmf(T_xdmf, T_dim)

    # Convert back to nondimensional variables on the dimensional submesh.
    p_star_dim = fenics.Function(Vp_dim)
    u_star_dim = fenics.Function(Vu_dim)
    theta_dim = fenics.Function(VT_dim)

    p_star_dim.vector()[:] = p_dim.vector()[:] / float(scales.Pref)
    u_star_dim.vector()[:] = u_dim.vector()[:] / float(scales.Uref)
    theta_dim.vector()[:] = (T_dim.vector()[:] - float(T_ambient)) / float(scales.dTref)
    p_star_dim.vector().apply('insert')
    u_star_dim.vector().apply('insert')
    theta_dim.vector().apply('insert')

    # Interpolate onto the current star mesh collapsed subspaces.
    Vp_star, _ = W.sub(0).collapse(True)
    Vu_star, _ = W.sub(1).collapse(True)
    VT_star, _ = W.sub(2).collapse(True)

    for fn in (p_star_dim, u_star_dim, theta_dim):
        try:
            fn.set_allow_extrapolation(True)
        except Exception:
            pass

    p_star = fenics.interpolate(p_star_dim, Vp_star)
    u_star = fenics.interpolate(u_star_dim, Vu_star)
    theta_star = fenics.interpolate(theta_dim, VT_star)

    assign_p = fenics.FunctionAssigner(W.sub(0), Vp_star)
    assign_u = fenics.FunctionAssigner(W.sub(1), Vu_star)
    assign_T = fenics.FunctionAssigner(W.sub(2), VT_star)

    assign_p.assign(w_n.sub(0), p_star)
    assign_u.assign(w_n.sub(1), u_star)
    assign_T.assign(w_n.sub(2), theta_star)
    w_n.vector().apply('insert')
    copy_state(w, w_n)

    meta = {
        'step': int(step),
        'time': float(time_value),
        'dt': float(dt_value),
        'source': 'approximate_xdmf_restart',
    }
    print0('Loaded approximate restart from transient snapshots:')
    print0(f"  step = {meta['step']}")
    print0(f"  time = {meta['time']:.6e}")
    print0(f"  dt   = {meta['dt']:.6e}")
    return w, w_n, meta

def check_interface_power(sub_ds, sub_ft, qn_air, scales, experiment, interface_tag=INTERFACE_TAG):
    # 1) dimensionalize qn_air: qn_dim [W/m^2]
    k_inf = float(experiment.fluid.properties["k"])  # use experiment value (not global)
    qn_dim = qn_air * fenics.Constant(k_inf * float(scales.dTref) / float(scales.Lref))

    # 2) integrate over the *half* interface boundary (your mesh is half-domain)
    # QL_half = fenics.assemble(qn_dim * sub_ds(interface_tag))  # [W/m]
    QL_full = fenics.assemble(qn_dim * sub_ds(interface_tag))  # [W/m]
    # QL_half = QL_half * scales.Lref
    # QL_full = 2.0 * QL_half                                   # mirror symmetry -> full wire

    # 3) interface length checks (helps debug tagging/completeness)
    # L_half = fenics.assemble(fenics.Constant(1.0) * sub_ds(interface_tag))  # [m]
    L_full = fenics.assemble(fenics.Constant(1.0) * sub_ds(interface_tag))  # [m]
    # L_half = L_half * scales.Lref
    # L_full = 2.0 * L_half

    # 4) expected values
    QL_target = float(experiment.initial_conditions.heat_length)  # [W/m]
    d = float(experiment.dimensions.wire.diameter)
    qsurf_target = QL_target / (math.pi * d)  # [W/m^2]
    # qsurf_avg = (QL_half / L_half) if L_half > 0 else float("nan")
    qsurf_avg = (QL_full / L_full) if L_full > 0 else float("nan")

    print0("=== Interface power conservation ===")
    print0(f"interface_tag = {interface_tag}")
    # print0(f"Interface length (half)  L_half  = {L_half:.6e} m")
    print0(f"Interface length (full)  L_full  = {L_full:.6e} m")
    # print0(f"Recovered power (half)   QL_half = {QL_half:.6e} W/m")
    print0(f"Recovered power (full)   QL_full = {QL_full:.6e} W/m")
    print0(f"Target power (paper)     QL      = {QL_target:.6e} W/m")
    print0(f"Relative error (full)    = {(QL_full-QL_target)/QL_target:.3%}")

    print0("--- Flux magnitude sanity ---")
    print0(f"Target mean q''          = {qsurf_target:.6e} W/m^2  (QL/(pi*d))")
    print0(f"Recovered mean q'' (half)= {qsurf_avg:.6e} W/m^2  (QL_half/L_half)")
    print0("===================================")

def under_relax_scalar(new_f, old_f, omega):
    out = fenics.Function(new_f.function_space(), name=new_f.name())
    out.vector()[:] = (1.0 - omega) * old_f.vector()[:] + omega * new_f.vector()[:]
    out.vector().apply("insert")
    return out

def under_relax_mixed_component(new_f, old_f, omega):
    out = fenics.Function(new_f.function_space(), name=new_f.name())
    out.vector()[:] = (1.0 - omega) * old_f.vector()[:] + omega * new_f.vector()[:]
    out.vector().apply("insert")
    return out

def relative_update(new_f, old_f):
    diff = new_f.vector().copy()
    diff.axpy(-1.0, old_f.vector())
    return diff.norm("l2") / (new_f.vector().norm("l2") + 1.0e-14)

# def base_version_new(experiment: Experiment):
#     """
#     New conjugate skeleton:

#       1) solve T_full on parent mesh
#       2) restrict T_full -> theta_air
#       3) solve air-only flow on air submesh
#       4) extend u_air -> parent mesh
#       5) solve updated T_full with air-only convection
#       6) iterate
#     """
#     run_root = make_run_root(experiment.name, "base")
#     GEOM_FILE = geometry_template(
#         wire_radius=experiment.dimensions.wire.diameter / 2,
#         output_path=experiment.name,
#         xmax=experiment.dimensions.domain.x_max,
#         ymax=experiment.dimensions.domain.y_max,
#         resolution=250,
#     )
#     MSH_FILE = experiment.name + "/plume.msh"
#     TRIG_XDMF_PATH = run_root + "/plume.xdmf"
#     FACETS_XDMF_PATH = run_root + "/plume_mt.xdmf"
#     OUTPUT_XDMF_PATH_WIRE = run_root + "/base/wire_temperature.xdmf"
#     OUTPUT_XDMF_PATH_TEMP = run_root + "/base/temperature.xdmf"
#     OUTPUT_XDMF_PATH_AIR_T = run_root + "/base/air_temperature.xdmf"
#     OUTPUT_XDMF_PATH_AIR_P = run_root + "/base/air_pressure.xdmf"
#     OUTPUT_XDMF_PATH_AIR_V = run_root + "/base/air_velocity.xdmf"
#     OUTPUT_XDMF_PATH_AIR_PVT = run_root + "/base/air_pvt.xdmf"
#     MESH_NAME = "Grid"
#     ELEM = "triangle"

#     # ------------------------------------------------------------------
#     # 0) Build full mesh and dimensional air submesh
#     # ------------------------------------------------------------------
#     generate_mesh(GEOM_FILE, MSH_FILE, TRIG_XDMF_PATH, FACETS_XDMF_PATH)

#     mesh, ct, ft, domains, dx, boundary_markers, mc, mf = read_mesh(
#         TRIG_XDMF_PATH, FACETS_XDMF_PATH, "Grid", PRINT_TAG_SUMMARY
#     )

#     # Dimensional air submesh
#     sub_mesh_dim, sub_ft_dim, sub_dx_dim, sub_ds_dim = create_submesh(mesh, mc, mf, AIR_TAG)

#     scales = compute_nondimensional_scales(experiment)
#     print0(scales)

#     T_ambient = float(experiment.initial_conditions.temperature)
#     dTref = float(scales.dTref)

#     # ------------------------------------------------------------------
#     # 1) Thermal initialization on FULL DIMENSIONAL parent mesh
#     # ------------------------------------------------------------------
#     print0("Computing initial full-domain thermal field...")
#     heat_volume = volume_heat_source(experiment)
#     print0(f"Using wire volumetric heating: {heat_volume:.6e} W/m^3")

#     T_full, k_func = solve_full_temperature(
#         mesh=mesh,
#         mc=mc,
#         mf=mf,
#         experiment=experiment,
#         heat_volume=heat_volume,
#         output_xdmf_path=OUTPUT_XDMF_PATH_TEMP,
#         u_full=None,
#         include_convection=False,
#         T_prev=None,
#         pseudo_dt=None,
#         max_material_iters=6,
#         material_tol=1.0e-8,
#     )

#     # Restrict dimensional T_full -> dimensional air submesh -> nondim theta on dim submesh
#     T_air_dim, theta_air_dim = restrict_full_temperature_to_air_submesh(
#         T_full=T_full,
#         sub_mesh_dim=sub_mesh_dim,
#         T_ambient=T_ambient,
#         dTref=dTref,
#     )

#     print0(f"Initial T_full min/max [K]: {T_full.vector().min():.6e}, {T_full.vector().max():.6e}")
#     print0(f"Initial theta_air_dim min/max [-]: {theta_air_dim.vector().min():.6e}, {theta_air_dim.vector().max():.6e}")

#     # ------------------------------------------------------------------
#     # 2) Keep qn_air ONLY as diagnostic
#     # ------------------------------------------------------------------
#     qn_air = flux_continuity(T_full, k_func, mesh, sub_mesh_dim, sub_ft_dim, mc, scales)
#     check_interface_power(sub_ds_dim, sub_ft_dim, qn_air, scales, experiment)

#     biot_air_h_eff, biot_air_Bi = biot(
#         sub_mesh_dim,
#         sub_ft_dim,
#         T_full,
#         qn_air,
#         T_ambient,
#         experiment.wire.properties["k"],
#         experiment.dimensions.wire.diameter,
#     )

#     # print0(f"Diagnostic h_eff [air side]: {biot_air_h_eff:.6e}")
#     # print0(f"Diagnostic Bi [air side]:    {biot_air_Bi:.6e}")

#     # ------------------------------------------------------------------
#     # 3) Scale parent mesh to STAR coordinates
#     #
#     # Your current main path already does this before solving on the air
#     # star mesh.  [oai_citation:6‡main.py](sediment://file_00000000fb8872468f86b32172903ceb)
#     # ------------------------------------------------------------------
#     Lref = float(scales.Lref)
#     scale_mesh_inplace(mesh, Lref)
#     scale_mesh_inplace(sub_mesh_dim, Lref)

#     # rebuild STAR air submesh from scaled parent mesh
#     sub_mesh_star, sub_ft_star, sub_dx_star, sub_ds_star = create_submesh(mesh, mc, mf, AIR_TAG)

#     # transfer theta from dim-submesh to star-submesh by sampling
#     theta_air_star = transfer_scalar_to_new_submesh_by_sampling(
#         theta_air_dim,
#         sub_mesh_star,
#         name="theta_air_star",
#     )

#     print0(f"Initial theta_air_star min/max [-]: {theta_air_star.vector().min():.6e}, {theta_air_star.vector().max():.6e}")
    
#     # ------------------------------------------------------------------
#     # 4) Air-flow startup (Stokes / no momentum convection)
#     # ------------------------------------------------------------------
#     print0("Starting air-only Stokes startup...")
#     startup = solve_air_stokes_startup(
#         sub_mesh_star=sub_mesh_star,
#         sub_dx_star=sub_dx_star,
#         sub_ft_star=sub_ft_star,
#         experiment=experiment,
#         theta_air_star=theta_air_star,
#         relaxation=1.0,
#     )

#     p_air = startup["p"]
#     u_air = startup["u"]
#     w_air = startup["w"]

#     print0(f"[startup] converged = {startup['converged']} | n_iter = {startup['n_iter']}")

#     # ------------------------------------------------------------------
#     # 5) Outer segregated coupling loop
#     # ------------------------------------------------------------------
#     max_outer_it = 20
#     tol_outer = 1.0e-5
#     omega_T = 0.5
#     omega_u = 0.4

#     convection_schedule = [0.10, 0.30, 0.60, 1.00]

#     T_full_old = fenics.Function(T_full.function_space(), name="T_full_old")
#     T_full_old.assign(T_full)

#     u_air_old = fenics.Function(u_air.function_space(), name="u_air_old")
#     u_air_old.assign(u_air)

#     for cscale in convection_schedule:
#         print0(f"\n=== Convection continuation: cscale = {cscale:.2f} ===")

#         for outer_it in range(max_outer_it):
#             print0(f"\n--- Outer iteration {outer_it:02d} @ convection scale {cscale:.2f} ---")

#             # ----------------------------------------------------------
#             # A) Extend air velocity to the FULL STAR parent mesh
#             # ----------------------------------------------------------
#             u_full_star = extend_air_velocity_to_parent_mesh_by_point_eval(
#                 u_air_star=u_air,
#                 mesh_star=mesh,
#                 mc=mc,
#             )

#             # ----------------------------------------------------------
#             # B) Solve full-domain temperature on parent mesh (STAR geometry,
#             #    but still dimensional T values)
#             # ----------------------------------------------------------
#             T_full_new, k_func = solve_full_temperature(
#                 mesh=mesh,
#                 mc=mc,
#                 mf=mf,
#                 experiment=experiment,
#                 heat_volume=heat_volume,
#                 output_xdmf_path=None,
#                 u_full=u_full_star,
#                 include_convection=True,
#                 T_prev=T_full_old,
#                 pseudo_dt=None,     # you can add pseudo-time later if needed
#                 max_material_iters=4,
#                 material_tol=1.0e-8,
#             )

#             # under-relax T_full
#             T_full_relaxed = under_relax_scalar(T_full_new, T_full_old, omega_T)

#             # ----------------------------------------------------------
#             # C) Restrict updated T_full -> dim/star air theta
#             #
#             # Since the parent mesh is now already scaled, the old dim-submesh
#             # has also been scaled in place, so sampling still works.
#             # ----------------------------------------------------------
#             T_air_dim_cur, theta_air_dim_cur = restrict_full_temperature_to_air_submesh(
#                 T_full=T_full_relaxed,
#                 sub_mesh_dim=sub_mesh_dim,
#                 T_ambient=T_ambient,
#                 dTref=dTref,
#             )

#             theta_air_star_new = transfer_scalar_to_new_submesh_by_sampling(
#                 theta_air_dim_cur,
#                 sub_mesh_star,
#                 name="theta_air_star",
#             )

#             # ----------------------------------------------------------
#             # D) Solve air-only flow
#             # ----------------------------------------------------------
#             air_sol = solve_air_flow_problem(
#                 sub_mesh_star=sub_mesh_star,
#                 sub_dx_star=sub_dx_star,
#                 sub_ft_star=sub_ft_star,
#                 experiment=experiment,
#                 theta_air_star=theta_air_star_new,
#                 w_init=w_air,
#                 include_convection=(cscale > 0.0),
#                 convection_scale=cscale,
#                 relaxation=0.5,
#                 maxit=20,
#                 atol=1.0e-9,
#                 rtol=1.0e-8,
#             )

#             p_air_new = air_sol["p"]
#             u_air_new = air_sol["u"]
#             w_air_new = air_sol["w"]

#             print0(f"[air solve] converged = {air_sol['converged']} | n_iter = {air_sol['n_iter']}")

#             # under-relax velocity only
#             u_air_relaxed = under_relax_mixed_component(u_air_new, u_air_old, omega_u)

#             # ----------------------------------------------------------
#             # E) Compute outer convergence
#             # ----------------------------------------------------------
#             relT = relative_update(T_full_relaxed, T_full_old)
#             relU = relative_update(u_air_relaxed, u_air_old)

#             print0(f"[outer] relT = {relT:.3e} | relU = {relU:.3e}")

#             # update accepted states
#             T_full_old.assign(T_full_relaxed)
#             u_air_old.assign(u_air_relaxed)
#             T_full.assign(T_full_relaxed)
#             u_air.assign(u_air_relaxed)
#             p_air.assign(p_air_new)
#             theta_air_star.assign(theta_air_star_new)
#             w_air.assign(w_air_new)

#             # optional updated diagnostics
#             try:
#                 qn_air = flux_continuity(T_full, k_func, mesh, sub_mesh_dim, sub_ft_dim, mc, scales)
#                 check_interface_power(sub_ds_dim, sub_ft_dim, qn_air, scales, experiment)
#             except Exception as err:
#                 print0(f"[diagnostic] qn_air update skipped: {err}")

#             if max(relT, relU) < tol_outer:
#                 print0(f"[outer] converged at iteration {outer_it:02d} for cscale={cscale:.2f}")
#                 break

#     # ------------------------------------------------------------------
#     # 6) Save final outputs
#     # ------------------------------------------------------------------
#     try:
#         with fenics.XDMFFile(sub_mesh_star.mpi_comm(), OUTPUT_XDMF_PATH_U) as xdmf_u:
#             xdmf_u.write(sub_mesh_star)
#             xdmf_u.write(u_air)

#         with fenics.XDMFFile(sub_mesh_star.mpi_comm(), OUTPUT_XDMF_PATH_P) as xdmf_p:
#             xdmf_p.write(sub_mesh_star)
#             xdmf_p.write(p_air)

#         with fenics.XDMFFile(mesh.mpi_comm(), OUTPUT_XDMF_PATH_TEMP) as xdmf_t:
#             xdmf_t.write(mesh)
#             xdmf_t.write(T_full)
#     except Exception as err:
#         print0(f"[output] warning: final write failed: {err}")

#     return {
#         "T_full": T_full,
#         "k_func": k_func,
#         "u_air": u_air,
#         "p_air": p_air,
#         "theta_air_star": theta_air_star,
#     }

def base_version(
        experiment: Experiment,
        restart_from_last_transient: bool = False,
        existing_run_root: str = "",
        steady_from_last_transient: bool = False,
        restart_from_checkpoint_mesh: str = "",
    ):
    mkdir0(experiment.name)

    run_root = make_run_root(
        experiment.name,
        "base",
        reuse_existing=existing_run_root,
    )

    if is_rank0():
        os.makedirs(os.path.join(run_root, "base"), exist_ok=True)
    COMM.Barrier()

    GEOM_FILE = geometry_template(
        wire_radius=experiment.dimensions.wire.diameter / 2,
        output_path=run_root,
        xmax=experiment.dimensions.domain.x_max,
        ymax=experiment.dimensions.domain.y_max,
        resolution=100,
    )

    MSH_FILE = run_root + "/plume.msh"

    TRIG_XDMF_PATH = run_root + "/full_cells.xdmf"
    FACETS_XDMF_PATH = run_root + "/full_facets.xdmf"

    AIR_TRIG_XDMF_PATH = run_root + "/air_cells.xdmf"
    AIR_FACETS_XDMF_PATH = run_root + "/air_facets.xdmf"

    OUTPUT_XDMF_PATH_WIRE = run_root + "/base/wire_temperature.xdmf"
    OUTPUT_XDMF_PATH_TEMP = run_root + "/base/temperature.xdmf"
    OUTPUT_XDMF_PATH_AIR_T = run_root + "/base/air_temperature.xdmf"
    OUTPUT_XDMF_PATH_AIR_P = run_root + "/base/air_pressure.xdmf"
    OUTPUT_XDMF_PATH_AIR_V = run_root + "/base/air_velocity.xdmf"
    OUTPUT_XDMF_PATH_AIR_PVT = run_root + "/base/air_pvt.xdmf"

    MESH_NAME = "Grid"
    ELEM = "triangle"

    # Rank 0 generates:
    #   full_cells.xdmf/full_facets.xdmf
    #   air_cells.xdmf/air_facets.xdmf
    # All ranks wait at the barrier inside generate_mesh().
    mesh_files_exist = (
        os.path.exists(TRIG_XDMF_PATH)
        and os.path.exists(FACETS_XDMF_PATH)
        and os.path.exists(AIR_TRIG_XDMF_PATH)
        and os.path.exists(AIR_FACETS_XDMF_PATH)
    )

    if existing_run_root and mesh_files_exist:
        print0("Using existing mesh files; not regenerating gmsh mesh.")
        COMM.Barrier()
    else:
        generate_mesh(
            GEOM_FILE,
            MSH_FILE,
            TRIG_XDMF_PATH,
            FACETS_XDMF_PATH,
            AIR_TRIG_XDMF_PATH=AIR_TRIG_XDMF_PATH,
            AIR_FACETS_XDMF_PATH=AIR_FACETS_XDMF_PATH,
            AIR_TAG_VALUE=AIR_TAG,
            ELEM=ELEM,
            PRUNE_Z=True,
        )

    # Full dimensional mesh.
    # Keep this for your full-domain initial thermal solve.
    mesh, ct, ft, domains, dx, boundary_markers, mc, mf = read_mesh(
        TRIG_XDMF_PATH,
        FACETS_XDMF_PATH,
        MESH_NAME,
        PRINT_TAG_SUMMARY,
    )

    # Air-only dimensional mesh.
    # This replaces:
    #     sub_mesh_dim, sub_ft_dim, sub_dx_dim, sub_ds_dim = create_submesh(...)
    sub_mesh_dim, air_ct, air_ft, _, sub_dx_dim, _, sub_mc_dim, sub_ft_dim = read_mesh(
        AIR_TRIG_XDMF_PATH,
        AIR_FACETS_XDMF_PATH,
        MESH_NAME,
        PRINT_TAG_SUMMARY,
    )
    
    sub_dx_dim = fenics.Measure(
        "dx",
        domain=sub_mesh_dim,
        subdomain_data=sub_mc_dim,
    )
    sub_ds_dim = fenics.Measure(
        "ds",
        domain=sub_mesh_dim,
        subdomain_data=sub_ft_dim,
    )
    
    scales = compute_nondimensional_scales(experiment)
    print0(scales)
    print0(f"Uref   = {scales.Uref:.6e} m/s")
    print0(f"Uplume = {scales.Uplume:.6e} m/s")
    print0(f"Uref/Uplume = {scales.Uref_over_Uplume:.6e}")
    print0(f"Lref   = {scales.Lref:.6e} m")
    print0(f"Lplume = {scales.Lplume:.6e} m")
    print0(f"dTref  = {scales.dTref:.6e} K")
    print0(f"dTline = {scales.dTline:.6e} K")
    
    # --- 3) conduction initial guess (dim parent mesh)
    print0("Computing initial guess for temperature field...")
    heat_volume = volume_heat_source(experiment)
    print0(f"Using heat volume: {heat_volume} W/m^3")

    T_full, k_func = initial_guess(mesh, mc, mf, OUTPUT_XDMF_PATH_TEMP,
                                    heat_volume, experiment, dx)

    # --- 4) restrict/interpolate to submesh (dim) → theta_full_dim
    # (DO NOT project across meshes; interpolate T_full onto submesh space first)
    T_ambient = float(experiment.initial_conditions.temperature)
    dTref = float(scales.dTref)

    V_air_dim = fenics.FunctionSpace(sub_mesh_dim, "CG", 1)

    # Allow evaluation just outside due to tolerance / boundary issues
    T_full.set_allow_extrapolation(True)

    T_air_dim = fenics.interpolate(T_full, V_air_dim)

    theta_full_dim = fenics.Function(V_air_dim)
    theta_full_dim.vector()[:] = (T_air_dim.vector()[:] - T_ambient) / dTref
    theta_full_dim.vector().apply("insert")
    theta_full_dim.rename("theta_full", "theta_full")

    print0("theta min/max:", global_vec_min(theta_full_dim), global_vec_max(theta_full_dim))
    print0("T_dim  min/max:", global_vec_min(T_air_dim), global_vec_max(T_air_dim))
    print0("T_nondim min/max:", global_vec_min(theta_full_dim), global_vec_max(theta_full_dim))

    # --- (still dimensional) compute qn_air using your current routine
    # MPI-safe interface heat flux.
    #
    # The old flux_continuity(...) path internally creates a wire SubMesh,
    # which is not robust under MPI. For the MPI base path, impose the
    # equivalent uniform heat flux on the air-side wire interface.
    #
    # For a full circular wire per unit length:
    #     Q_L = q'' * pi * d
    # so:
    #     q'' = Q_L / (pi*d)
    #
    # Nondimensional:
    #     qn_star = q'' * Lref / (k_air*dTref)
    q_line = float(experiment.initial_conditions.heat_length)
    wire_d = float(experiment.dimensions.wire.diameter)
    k_air_dim = float(experiment.fluid.properties["k"])

    qsurf_dim = q_line / (math.pi * wire_d)
    qn_star_value = qsurf_dim * float(scales.Lref) / (
        k_air_dim * float(scales.dTref)
    )

    V0_air = fenics.FunctionSpace(sub_mesh_dim, "DG", 0)
    qn_air = fenics.Function(V0_air, name="qn_air")
    qn_air.vector()[:] = qn_star_value
    qn_air.vector().apply("insert")

    print0("Using MPI-safe uniform interface heat flux:")
    print0(f"  q_line    = {q_line:.6e} W/m")
    print0(f"  qsurf_dim = {qsurf_dim:.6e} W/m^2")
    print0(f"  qn_star   = {qn_star_value:.6e}")

    # Diagnostic: power conservation on DIMENSIONAL interface measure
    check_interface_power(sub_ds_dim, sub_ft_dim, qn_air, scales, experiment)
    
    # Optional diagnostic: Biot numbers should use dimensional geometry/fields
    # biot_air_h_eff, biot_air_Bi = biot(
    #     sub_mesh_dim, sub_ft_dim, T_full, qn_air,
    #     T_ambient, experiment.wire.properties["k"],
    #     experiment.dimensions.wire.diameter
    # )

    print0(f"Initial max temperature: {global_vec_max(T_full):.2f} K")
    print0(f"Initial min temperature: {global_vec_min(T_full):.2f} K")
    print0(f"Initial max theta (dim-submesh): {global_vec_max(theta_full_dim):.6e}")
    print0(f"Initial min theta (dim-submesh): {global_vec_min(theta_full_dim):.6e}")

    # --- 5) scale mesh coordinates dim -> star
    Lref = float(scales.Lref)
    scale_mesh_inplace(mesh, Lref)
    scale_mesh_inplace(sub_mesh_dim, Lref)

    try:
        mesh.bounding_box_tree().build(mesh)
        sub_mesh_dim.bounding_box_tree().build(sub_mesh_dim)
    except Exception:
        pass

    COMM.Barrier()

    # The air mesh has been scaled in place, so it is now the star mesh.
    # No SubMesh reconstruction is needed.
    sub_mesh_star = sub_mesh_dim
    sub_ft_star = sub_ft_dim
    sub_dx_star = fenics.Measure(
        "dx",
        domain=sub_mesh_star,
        subdomain_data=sub_mc_dim,
    )
    sub_ds_star = fenics.Measure(
        "ds",
        domain=sub_mesh_star,
        subdomain_data=sub_ft_star,
    )

    qn_air_star = qn_air

    theta_full_star = solve_air_initial_theta(
        air_mesh=sub_mesh_star,
        air_facet_markers=sub_ft_star,
        air_ds=sub_ds_star,
        qn_air=qn_air_star,
        interface_tag=INTERFACE_TAG,
        cold_tags=(101, 103),
    )

    print0(f"Initial max theta (star-submesh): {global_vec_max(theta_full_star):.6e}")
    print0(f"Initial min theta (star-submesh): {global_vec_min(theta_full_star):.6e}")
    print0(f"Rho_air: {experiment.fluid.properties['rho']}")
    print0(f"Beta_air: {experiment.fluid.properties['beta']}")
    
    # Solving the problem
    print0("Starting solver...")
    W, w, p, u, T, w_n, p_n, u_n, T_n, psi_p, psi_u, psi_T, \
    mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc = solver(
        sub_mesh_star,
        theta_full_star,      # <-- nondimensional theta on star mesh
        0.0,
        experiment.fluid.properties["rho"],
        experiment.fluid.properties["beta"],
        experiment
    )

    # Optional restart
    restart_meta = {"step": 0, "time": 0.0, "dt": 1.0e-5, "source": "fresh_start"}
    if restart_from_last_transient or steady_from_last_transient:
        checkpoint_dir = os.path.join(run_root, "base", "restart_checkpoint")
        try:
            if os.path.exists(os.path.join(checkpoint_dir, "state.h5")):
                print0("Attempting restart from true checkpoint...")
                w, w_n, restart_meta = load_true_restart_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    W=W,
                    w=w,
                    w_n=w_n,
                )
            else:
                print0("Attempting restart from last saved transient snapshot...")
                w, w_n, restart_meta = approximate_restart_from_last_saved_transient(
                    run_root=run_root,
                    mode_subdir="base",
                    sub_mesh_dim=sub_mesh_dim,
                    sub_mesh_star=sub_mesh_star,
                    W=W,
                    w=w,
                    w_n=w_n,
                    scales=scales,
                    T_ambient=T_ambient,
                    rho_air=experiment.fluid.properties["rho"],
                    fallback_dt=1.0e-4,
                )

            print0(
                f"Restart recovered from {restart_meta['source']}: "
                f"step={restart_meta['step']}, "
                f"time={restart_meta['time']:.6e}, "
                f"dt={restart_meta['dt']:.6e}"
            )

        except Exception as exc:
            print0("Restart request could not be satisfied.")
            print0(f"Reason: {exc}")
            print0("Falling back to fresh transient start from steady state.")
            copy_state(w, w_n)
            restart_meta = {"step": 0, "time": 0.0, "dt": 1.0e-4, "source": "fresh_start"}

    if not restart_from_last_transient and not steady_from_last_transient and restart_from_checkpoint_mesh == "":
        # Use Stokes initial guess for better convergene
        print0("Solving Stokes problem for initial guess...")
        w_n = stokes_initial_guess(
            experiment=experiment,
            u_n=u_n, u=u, T_n=T_n, T=T, p=p,
            W=W, w=w,
            psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
            mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
            sub_dx=sub_dx_star, sub_ds=sub_ds_star, sub_ft=sub_ft_star, qn_air=qn_air_star,
            w_n=w_n,
            lambdas=( 0.05, 0.1, 0.3)
        )
    
    # Solve the full nonlinear problem with previous initial guess
    print0("Starting checks")
    w_t = w.copy(deepcopy=True)
    w_t =solve_thermal_sign_check(experiment=experiment,W=W,w=w_t,
                                mu=mu,Pr=Pr,
                                sub_dx=sub_dx_star,sub_ds=sub_ds_star,sub_ft=sub_ft_star,
                                qn_air=qn_air_star,T_c=T_c,T_air_bc=T_air_bc,w_n=w_n)
    w_p = w.copy(deepcopy=True)
    w_p = solve_buoyancy_sign_check(experiment=experiment,W=W,w=w_p,
                                  mu=mu,Pr=Pr,
                                  sub_dx=sub_dx_star,sub_ds=sub_ds_star,sub_ft=sub_ft_star,
                                  qn_air=qn_air_star,T_c=T_c,T_air_bc=T_air_bc,w_n=w_n)
    print0("Checks complete")

    # w = solve_steady_newton_continuation(
    #     experiment=experiment,
    #     u_n=u_n, u=u, T_n=T_n, T=T, p=p,
    #     W=W, w=w,
    #     psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
    #     mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
    #     sub_dx=sub_dx_star, sub_ds=sub_ds_star, sub_ft=sub_ft_star, qn_air=qn_air_star,
    #     w_n=w_n,
    #     lambdas=[0.1, 0.2], #0.03, 0.04, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00],
    #     relaxation_schedule=(0.9, 0.7, 0.5),# 0.4, 0.35, 0.30, 0.27, 0.25, 0.22, 0.20, 0.15, 0.10),
    #     stokes_startup=False,
    #     sub_mesh_star=sub_mesh_star,
    #     sub_mesh_dim=sub_mesh_dim,
    #     p_path=OUTPUT_XDMF_PATH_AIR_P,
    #     u_path=OUTPUT_XDMF_PATH_AIR_V,
    #     T_path=OUTPUT_XDMF_PATH_AIR_T
    # )

    save_obj = (
        sub_ds_star,
        sub_mesh_dim,
        scales,
        OUTPUT_XDMF_PATH_AIR_P,
        OUTPUT_XDMF_PATH_AIR_V,
        OUTPUT_XDMF_PATH_AIR_T
    )

    SUPG = True
    if steady_from_last_transient:
        print0("\n=== Steady Newton solve from last transient checkpoint ===")

        w = solve_steady_from_loaded_checkpoint(
            experiment=experiment,
            W=W,
            w=w,
            w_n=w_n,
            psi_p=psi_p,
            psi_u=psi_u,
            psi_T=psi_T,
            mu=mu,
            Pr=Pr,
            f_b=f_b,
            T_c=T_c,
            T_air_bc=T_air_bc,
            sub_dx=sub_dx_star,
            sub_ds=sub_ds_star,
            sub_ft=sub_ft_star,
            qn_air=qn_air_star,
            sub_mesh_star=sub_mesh_star,
            sub_mesh_dim=sub_mesh_dim,
            scales=scales,
            T_ambient=T_ambient,
            rho_air=experiment.fluid.properties["rho"],
            p_path=OUTPUT_XDMF_PATH_AIR_P,
            u_path=OUTPUT_XDMF_PATH_AIR_V,
            T_path=OUTPUT_XDMF_PATH_AIR_T,
            checkpoint_meta=restart_meta,
            SUPG=SUPG,
        )
        print0("Steady-from-transient branch complete.")
        return

    elif not restart_from_last_transient and restart_from_checkpoint_mesh == "":
        w, info = solve_ptc_continuation(
            experiment,
            W, w, w_n,
            psi_p, psi_u, psi_T,
            mu, Pr, f_b, T_c, T_air_bc,
            sub_dx_star, sub_ds_star, sub_ft_star, qn_air_star,
            run_root=run_root,
            dtau_init=1e-4,
            dtau_min=1e-8,
            dtau_max=1e-2,
            stage_max_steps=40,
            final_stage_max_steps=2000,
            update_tol=1e-8,
            residual_tol=1e-8,
            # save_obj=save_obj
            save_obj=None,
            SUPG=SUPG,
        )

        if info.get("status") == "continuation_failed":
            print0("failed_stage:", info.get("failed_stage"))
            last = info.get("last_stage_info", {})
            print0("last_stage_status:", last.get("status"))
            print0("accepted_steps:", last.get("accepted_steps"))
            print0("rejected_steps:", last.get("rejected_steps"))
            print0("final_dtau:", last.get("final_dtau"))
            print0("final_rel_update:", last.get("final_rel_update"))
            print0("final_steady_residual:", last.get("final_steady_residual"))
        else:
            print0("accepted_steps:", info.get("accepted_steps"))
            print0("rejected_steps:", info.get("rejected_steps"))
            print0("final_dtau:", info.get("final_dtau"))
            print0("final_rel_update:", info.get("final_rel_update"))
            print0("final_steady_residual:", info.get("final_steady_residual"))
    
    # Split nondimensional solution
    p_star, u_star, theta = w.split(deepcopy=True)

    # Dimensionalize fields (note: mesh is star; dimensionalize handles scaling)
    u_dim, p_dim, T_dim = dimensionalize_fields(
        sub_mesh_star, u_star, p_star, theta,
        scales.Uplume, scales.dTref, T_ambient,
        experiment.fluid.properties["rho"]
    )

    # Optional diagnostic: Biot numbers should use dimensional geometry/fields
    try:
        _,_, biots = biot_wrap(
            sub_mesh=sub_mesh_dim,
            sub_ft=sub_ft_dim,
            sub_ds=sub_ds_dim,
            T_air_dim=T_air_dim,
            qn_air=qn_air,
            scales=scales,
            T_ref=T_ambient,
            k_wire=experiment.wire.properties["k"],
            wire_diameter=experiment.dimensions.wire.diameter,
            characteristic_length="radius",
            return_local_field=True
        )
        print0(f"Biot number stats: min={global_vec_min(biots):.6e}, max={global_vec_max(biots):.6e}")
    except Exception:
        print0("Biot diagnostic failed; check dimensional geometry/fields and interface tagging.")
    # print0(f"Effective Biot number after steady solve: Bi_air = {biot_air_Bi:.6e}")

    dt_start=1.0e-4
    if restart_from_last_transient:
        dt_start = restart_meta["dt"]
        print0(f"Starting transient with dt={dt_start:.6e} from restart source: {restart_meta['source']}")

    if restart_from_checkpoint_mesh != "":
        print0(f"Restarting from checkpoint-owned mesh: {restart_from_checkpoint_mesh}")

        loaded = prepare_loaded_checkpoint_for_base_run(
            checkpoint_dir=restart_from_checkpoint_mesh,
            experiment=experiment,
        )

        sub_mesh_star = loaded["sub_mesh_star"]
        sub_mesh_dim = loaded["sub_mesh_dim"]

        W = loaded["W"]
        w = loaded["w"]
        w_n = loaded["w_n"]

        psi_p = loaded["psi_p"]
        psi_u = loaded["psi_u"]
        psi_T = loaded["psi_T"]

        sub_ft_star = loaded["sub_ft_star"]
        sub_dx_star = loaded["sub_dx_star"]
        sub_ds_star = loaded["sub_ds_star"]

        sub_ft_dim = loaded["sub_ft_dim"]
        sub_dx_dim = loaded["sub_dx_dim"]
        sub_ds_dim = loaded["sub_ds_dim"]

        qn_air_star = loaded["qn_air_star"]
        qn_air = qn_air_star

        restart_meta = loaded["restart_meta"]

        mu = loaded["mu"]
        Pr = loaded["Pr"]
        Ra = loaded["Ra"]
        f_b = loaded["f_b"]
        T_c = loaded["T_c"]
        T_air_bc = loaded["T_air_bc"]

        T_ambient = float(experiment.initial_conditions.temperature)
        restart_from_last_transient = True
        # Continue directly with transient.
        print0(
            f"Loaded AMR checkpoint: step={restart_meta['step']}, "
            f"time={restart_meta['time']:.6e}, dt={restart_meta['dt']:.6e}"
        )
            
    if not steady_from_last_transient:
        w, transient_info = run_post_continuation_transient(
                experiment=experiment,
                W=W,
                w=w,
                w_n=w_n,
                psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
                mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
                sub_dx=sub_dx_star, sub_ds=sub_ds_star, sub_ft=sub_ft_star, qn_air=qn_air_star,
                sub_mesh_star=sub_mesh_star,
                sub_mesh_dim=sub_mesh_dim,
                sub_ft_dim=sub_ft_dim,
                sub_ds_dim=sub_ds_dim,
                scales=scales,
                p_path=OUTPUT_XDMF_PATH_AIR_P,
                u_path=OUTPUT_XDMF_PATH_AIR_V,
                T_path=OUTPUT_XDMF_PATH_AIR_T,
                dt_start=dt_start,
                dt_growth=1.9,
                dt_cut=0.8,
                dt_hard_cut=0.5,
                dt_min=1.0e-5,
                dt_max=1.0,
                t_end=15000.0,
                n_steps=200000,
                save_every=100,
                max_retries_per_step=8,
                rel_update_easy=1.0e-3,
                rel_update_hard=1.0e-3,
                rel_update_reject=1.0e-2,
                steady_window=25,
                steady_rel_tol=1.0e-5,
                steady_update_tol=1.0e-6,
                start_time=restart_meta["time"],
                start_step=restart_meta["step"],
                history_csv_path=run_root + "/transient_history.csv",
                SUPG=SUPG,
                restart_recovered=restart_from_last_transient,
                restart_step=restart_meta["step"],
            )

        print0("transient status:", transient_info["status"])
        print0("transient steps:", transient_info["n_steps"])

    # Split nondimensional solution
    p_star, u_star, theta = w.split(deepcopy=True)

    # Dimensionalize fields (note: mesh is star; dimensionalize handles scaling)
    u_dim, p_dim, T_dim = dimensionalize_fields(
        sub_mesh_star, u_star, p_star, theta,
        scales.Uplume, scales.dTref, T_ambient,
        experiment.fluid.properties["rho"]
    )
    k_air = fenics.Constant(experiment.fluid.properties["k"])
    q_heat, q_mag = compute_heat_flux_dim(T_dim, k_air)
    q_out = OUTPUT_XDMF_PATH_AIR_T.split(".xdmf")[0] + f"_heatflux.xdmf"
    qmag_out = OUTPUT_XDMF_PATH_AIR_T.split(".xdmf")[0] + f"_heatflux_mag.xdmf"

    # on sub_mesh_star, with theta (nondim)
    k_inf = float(experiment.fluid.properties["k"])  # use experiment value (not global)
    n = fenics.FacetNormal(sub_mesh_star)
    dTref = float(scales.dTref)

    # Example: if facet tags exist for TOP and FAR boundaries
    Q_far  = -k_inf*dTref * fenics.assemble(fenics.dot(fenics.grad(theta), n) * sub_ds_star(OUTER_AIR_TAG))
    print0(f"Heat flux through outer air boundary: Q_far = {Q_far:.6e} W/m")


    # plotting + output
    # plot_mesh(T_dim, title="Temperature field", label="Temperature (K)",
    #             cmap="coolwarm", colorbar=True)
    # plot_mesh(theta, title="Temperature field nondimensional", label="Temperature (nondim)",
    #             cmap="coolwarm", colorbar=True)
    # plot_mesh(u_dim, title="Velocity magnitude", label="Velocity (m/s)",
    #             cmap="coolwarm", colorbar=True, mode="glyphs")
    # plot_mesh(p_dim, title="Pressure field", label="Pressure (Pa)",
    #             cmap="coolwarm", colorbar=True)

    save_experiment(OUTPUT_XDMF_PATH_AIR_P, sub_mesh_dim, [p_dim])
    save_experiment(OUTPUT_XDMF_PATH_AIR_V, sub_mesh_dim, [u_dim])
    save_experiment(OUTPUT_XDMF_PATH_AIR_T, sub_mesh_dim, [T_dim])
    save_experiment(q_out, sub_mesh_dim, [q_heat])
    save_experiment(qmag_out, sub_mesh_dim, [q_mag])
    # save_experiment(OUTPUT_XDMF_PATH_AIR_PVT, sub_mesh, [p,u,T])

    # Example: Brodowicz-style heights 1, 4, 8 cm above wire center
    y0_m_list = [0.01, 0.04, 0.08]
    hmin_star = sub_mesh_star.hmin()   # dimensionless
    hmax_star = sub_mesh_star.hmax()
    eps_m = 3 * 0.5*(hmin_star + hmax_star) * scales.Lref
    # eps_m = 2 * hmin_star * scales.Lref
    flux_rows = plane_fluxes_slab_star(
        sub_mesh_star,
        u_star, theta,                   # your returned nondim u and theta
        y0_m_list,
        scales=scales,
        rho=experiment.fluid.properties["rho"],
        cp=experiment.fluid.properties["cp"],
        k=experiment.fluid.properties["k"],
        eps_m=eps_m             # e.g. 1 mm slab half-thickness (tune to mesh)
    )

    for (y0_m, Qconv, Qcond, Qtot, mdot) in flux_rows:
        print0(f"y0={y0_m:.3f} m: Qconv={Qconv:.6e} W/m, Qcond={Qcond:.6e} W/m, "
            f"Qtot={Qtot:.6e} W/m, mdot={mdot:.6e} kg/(s·m)")
        
    out_dir=Path.cwd()
    csv_path = os.path.join(out_dir,run_root, "base", "plane_fluxes.csv")
    write_header = not os.path.exists(csv_path)

    if is_rank0():
        with open(csv_path, "a", newline="") as f:
            wcsv = csv.writer(f)
            if write_header:
                wcsv.writerow([
                    "time",
                    "y0_m",
                    "Qconv_W_per_m",
                    "Qcond_W_per_m",
                    "Qtot_W_per_m",
                    "mdot_kg_per_s_per_m",
                ])
            t = 0
            for (y0_m, Qconv, Qcond, Qtot, mdot) in flux_rows:
                wcsv.writerow([float(t), y0_m, Qconv, Qcond, Qtot, mdot])

    COMM.Barrier()

def temperature_dependent_version(experiment: Experiment, restart_from_last_transient: bool = False, existing_run_root: str = ""):
    run_root = make_run_root(experiment.name, "temp", reuse_existing=existing_run_root)
    GEOM_FILE = geometry_template(
        wire_radius=experiment.dimensions.wire.diameter / 2,
        output_path=experiment.name,
        xmax=experiment.dimensions.domain.x_max,
        ymax=experiment.dimensions.domain.y_max,
        resolution=250,
    )
    MSH_FILE = experiment.name + "/plume.msh"
    TRIG_XDMF_PATH = run_root + "/plume.xdmf"
    FACETS_XDMF_PATH = run_root + "/plume_mt.xdmf"
    OUTPUT_XDMF_PATH_WIRE = run_root + "/t_dep_mat/wire_temperature.xdmf"
    OUTPUT_XDMF_PATH_TEMP = run_root + "/t_dep_mat/temperature.xdmf"
    OUTPUT_XDMF_PATH_AIR_T = run_root + "/t_dep_mat/air_temperature.xdmf"
    OUTPUT_XDMF_PATH_AIR_P = run_root + "/t_dep_mat/air_pressure.xdmf"
    OUTPUT_XDMF_PATH_AIR_V = run_root + "/t_dep_mat/air_velocity.xdmf"
    OUTPUT_XDMF_PATH_AIR_PVT = run_root + "/t_dep_mat/air_pvt.xdmf"
    MESH_NAME = "Grid"
    ELEM = "triangle"                             # use uniform heat generation

    # Generate and read mesh
    generate_mesh(GEOM_FILE, MSH_FILE, TRIG_XDMF_PATH, FACETS_XDMF_PATH)
    
    # --- 1) read mesh (dim)  [already]
    mesh, ct, ft, domains, dx, boundary_markers, mc, mf = read_mesh(
        TRIG_XDMF_PATH, FACETS_XDMF_PATH, MESH_NAME, PRINT_TAG_SUMMARY
    )

    # --- 2) create submesh (dim)
    sub_mesh_dim, sub_ft_dim, sub_dx_dim, sub_ds_dim = create_submesh(mesh, mc, mf, AIR_TAG)

    scales = compute_nondimensional_scales(experiment)
    print0(scales)
    print0(f"Uref   = {scales.Uref:.6e} m/s")
    print0(f"Uplume = {scales.Uplume:.6e} m/s")
    print0(f"Uref/Uplume = {scales.Uref_over_Uplume:.6e}")
    print0(f"Lref   = {scales.Lref:.6e} m")
    print0(f"Lplume = {scales.Lplume:.6e} m")
    print0(f"dTref  = {scales.dTref:.6e} K")
    print0(f"dTline = {scales.dTline:.6e} K")

    # --- 3) conduction initial guess (dim parent mesh)
    print0("Computing initial guess for temperature field...")
    heat_volume = volume_heat_source(experiment)
    print0(f"Using heat volume: {heat_volume} W/m^3")

    T_full, k_func = initial_guess(mesh, mc, mf, OUTPUT_XDMF_PATH_TEMP,
                                    heat_volume, experiment, dx)

    # --- 4) restrict/interpolate to submesh (dim) → theta_full_dim
    # (DO NOT project across meshes; interpolate T_full onto submesh space first)
    T_ambient = float(experiment.initial_conditions.temperature)
    theta_ambient = 0.0   # nondim ambient temperature
    dTref = float(scales.dTref)

    V_air_dim = fenics.FunctionSpace(sub_mesh_dim, "CG", 1)

    # Allow evaluation just outside due to tolerance / boundary issues
    T_full.set_allow_extrapolation(True)

    T_air_dim = fenics.interpolate(T_full, V_air_dim)

    theta_full_dim = fenics.Function(V_air_dim)
    theta_full_dim.vector()[:] = (T_air_dim.vector()[:] - T_ambient) / dTref
    theta_full_dim.vector().apply("insert")
    theta_full_dim.rename("theta_full", "theta_full")

    print0("theta min/max:", theta_full_dim.vector().min(), theta_full_dim.vector().max())
    print0("T_dim  min/max:", T_air_dim.vector().min(), T_air_dim.vector().max())
    print0("T_nondim min/max:", theta_full_dim.vector().min(), theta_full_dim.vector().max())

    # --- (still dimensional) compute qn_air using your current routine
    qn_air = flux_continuity(T_full, k_func, mesh, sub_mesh_dim, sub_ft_dim, mc, scales)

    # Diagnostic: power conservation on DIMENSIONAL interface measure
    check_interface_power(sub_ds_dim, sub_ft_dim, qn_air, scales, experiment)

    # Optional diagnostic: Biot numbers should use dimensional geometry/fields
    biot_air_h_eff, biot_air_Bi = biot(
        sub_mesh_dim, sub_ft_dim, T_full, qn_air,
        T_ambient, experiment.wire.properties["k"],
        experiment.dimensions.wire.diameter
    )

    print0(f"Initial max temperature: {T_full.vector().max():.2f} K")
    print0(f"Initial min temperature: {T_full.vector().min():.2f} K")
    print0(f"Initial max theta (dim-submesh): {theta_full_dim.vector().max():.6e}")
    print0(f"Initial min theta (dim-submesh): {theta_full_dim.vector().min():.6e}")
    theta_ambient=theta_full_dim.vector().min()
    
    # --- 5) scale parent mesh coordinates (dim→star)
    Lref = float(scales.Lref)
    scale_mesh_inplace(mesh, Lref)
    scale_mesh_inplace(sub_mesh_dim, Lref)

    # --- 6) recreate submesh + measures on scaled parent mesh
    sub_mesh_star, sub_ft_star, sub_dx_star, sub_ds_star, theta_full_star, qn_air_star = \
    build_star_submesh_and_transfer(
        mesh=mesh,
        mc=mc,
        mf=mf,
        air_tag=AIR_TAG,
        theta_full_dim=theta_full_dim,
        qn_air=qn_air,
    )

    print0(f"Initial max theta (star-submesh): {theta_full_star.vector().max():.6e}")
    print0(f"Initial min theta (star-submesh): {theta_full_star.vector().min():.6e}")
    print0(f"Rho_air: {experiment.fluid.properties['rho']}")
    print0(f"Beta_air: {experiment.fluid.properties['beta']}")

    # Define temperature-dependent material model for air
    fluid_material = TemperatureDependentMaterial(
        mesh=sub_mesh_star,
        T_ref=experiment.initial_conditions.temperature,
        mu_ref=experiment.fluid.properties["mu"],
        cp_ref=experiment.fluid.properties["cp"],
        k_ref=experiment.fluid.properties["k"],
        beta_ref=experiment.fluid.properties["beta"],
        rho_ref=experiment.fluid.properties["rho"],
        # table_file="materials/air_properties_coolprop.csv"   # or None
        table_file="materials/spindle_oil_estimated_table.csv"   # or None
    )
    

    # Solving the problem
    print0("Starting solver...")
    W, w, p, u, T, w_n, p_n, u_n, T_n, psi_p, psi_u, psi_T, \
    mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc = solver(
        sub_mesh_star,
        theta_full_star,      # <-- nondimensional theta on star mesh
        0.0,
        experiment.fluid.properties["rho"],
        experiment.fluid.properties["beta"],
        experiment
    )

    
    # Use Stokes initial guess for better convergene
    print0("Solving Stokes problem for initial guess...")
    update_material_from_mixed_nondimensional_temperature(
        fluid_material=fluid_material,
        w_mixed=w_n,
        scales=scales,
        T_ambient=theta_ambient,
    )
    # Optional restart
    restart_meta = {"step": 0, "time": 0.0, "dt": 1.0e-5, "source": "fresh_start"}
    if restart_from_last_transient:
        checkpoint_dir = os.path.join(run_root, "base", "restart_checkpoint")
        try:
            if os.path.exists(os.path.join(checkpoint_dir, "state.h5")):
                print0("Attempting restart from true checkpoint...")
                w, w_n, restart_meta = load_true_restart_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    W=W,
                    w=w,
                    w_n=w_n,
                )
            else:
                print0("Attempting restart from last saved transient snapshot...")
                w, w_n, restart_meta = approximate_restart_from_last_saved_transient(
                    run_root=run_root,
                    mode_subdir="base",
                    sub_mesh_dim=sub_mesh_dim,
                    sub_mesh_star=sub_mesh_star,
                    W=W,
                    w=w,
                    w_n=w_n,
                    scales=scales,
                    T_ambient=T_ambient,
                    rho_air=experiment.fluid.properties["rho"],
                    fallback_dt=1.0e-4,
                )

            print0(
                f"Restart recovered from {restart_meta['source']}: "
                f"step={restart_meta['step']}, "
                f"time={restart_meta['time']:.6e}, "
                f"dt={restart_meta['dt']:.6e}"
            )

        except Exception as exc:
            print0("Restart request could not be satisfied.")
            print0(f"Reason: {exc}")
            print0("Falling back to fresh transient start from steady state.")
            copy_state(w, w_n)
            restart_meta = {"step": 0, "time": 0.0, "dt": 1.0e-4, "source": "fresh_start"}

    # if not restart_from_last_transient:
    #     # Use Stokes initial guess for better convergene
    #     print0("Solving Stokes problem for initial guess...")
    #     w_n = stokes_initial_guess(
    #         experiment=experiment,
    #         u_n=u_n, u=u, T_n=T_n, T=T, p=p,
    #         W=W, w=w,
    #         psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
    #         mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
    #         sub_dx=sub_dx_star, sub_ds=sub_ds_star, sub_ft=sub_ft_star, qn_air=qn_air_star,
    #         w_n=w_n,
    #         lambdas=( 0.05, 0.1, 0.3)
    #     )
    #     # theta_ambient = w_n.sub(2).vector().min()  # update ambient temperature based on initial guess
    #     update_material_from_mixed_nondimensional_temperature(
    #         fluid_material=fluid_material,
    #         w_mixed=w_n,
    #         scales=scales,
    #         T_ambient=T_ambient,
    #     )
    

    print0("Starting checks")
    w_t = w.copy(deepcopy=True)
    w_t =solve_thermal_sign_check(experiment=experiment,W=W,w=w_t,
                                mu=mu,Pr=Pr,
                                sub_dx=sub_dx_star,sub_ds=sub_ds_star,sub_ft=sub_ft_star,
                                qn_air=qn_air_star,T_c=T_c,T_air_bc=T_air_bc,w_n=w_n)
    w_p = w.copy(deepcopy=True)
    w_p = solve_buoyancy_sign_check(experiment=experiment,W=W,w=w_p,
                                  mu=mu,Pr=Pr,
                                  sub_dx=sub_dx_star,sub_ds=sub_ds_star,sub_ft=sub_ft_star,
                                  qn_air=qn_air_star,T_c=T_c,T_air_bc=T_air_bc,w_n=w_n)
    print0("Checks complete")

    # w = temp_dep_solver(F,w, boundary_conditions, JF, w_n, fluid_material)
    # w = solve_temp_newton_continuation(
    #     experiment=experiment,
    #     u_n=u_n, u=u, T_n=T_n, T=T, p=p,
    #     W=W, w=w,
    #     psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
    #     mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
    #     sub_dx=sub_dx_star, sub_ds=sub_ds_star, sub_ft=sub_ft_star, qn_air=qn_air_star,
    #     w_n=w_n,
    #     lambdas=[0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00],
    #     relaxation_schedule=(0.9, 0.7, 0.5), #0.4, 0.35, 0.30, 0.8, 0.27, 0.25, 0.22, 0.20, 0.7, 0.15, 0.10, 0.05, 0.02, 0.01),
    #     stokes_startup=False,
    #     sub_mesh_star=sub_mesh_star,
    #     sub_mesh_dim=sub_mesh_dim,
    #     p_path=OUTPUT_XDMF_PATH_AIR_P,
    #     u_path=OUTPUT_XDMF_PATH_AIR_V,
    #     T_path=OUTPUT_XDMF_PATH_AIR_T,
    #     fluid_material=fluid_material,
    # )

    save_obj = (
        sub_ds_star,
        sub_mesh_dim,
        scales,
        OUTPUT_XDMF_PATH_AIR_P,
        OUTPUT_XDMF_PATH_AIR_V,
        OUTPUT_XDMF_PATH_AIR_T
    )
    
    if not restart_from_last_transient:
        w, info = solve_ptc_temp_continuation(
            experiment=experiment,
            W=W,
            w=w,
            w_n=w_n,
            T_ambient=T_ambient,
            psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
            mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
            sub_dx=sub_dx_star, sub_ds=sub_ds_star, sub_ft=sub_ft_star, qn_air=qn_air_star,
            fluid_material=fluid_material,
            run_root=run_root,
            save_obj=None,
        )
        update_material_from_mixed_nondimensional_temperature(
            fluid_material=fluid_material,
            w_mixed=w,
            scales=scales,
            T_ambient=T_ambient,
        )

        if info.get("status") == "continuation_failed":
            print0("failed_stage:", info.get("failed_stage"))
            last = info.get("last_stage_info", {})
            print0("last_stage_status:", last.get("status"))
            print0("accepted_steps:", last.get("accepted_steps"))
            print0("rejected_steps:", last.get("rejected_steps"))
            print0("final_dtau:", last.get("final_dtau"))
            print0("final_rel_update:", last.get("final_rel_update"))
            print0("final_steady_residual:", last.get("final_steady_residual"))
        else:
            print0("accepted_steps:", info.get("accepted_steps"))
            print0("rejected_steps:", info.get("rejected_steps"))
            print0("final_dtau:", info.get("final_dtau"))
            print0("final_rel_update:", info.get("final_rel_update"))
            print0("final_steady_residual:", info.get("final_steady_residual"))
    
    # Split nondimensional solution
    p_star, u_star, theta = w.split(deepcopy=True)

    # Dimensionalize fields (note: mesh is star; dimensionalize handles scaling)
    u_dim, p_dim, T_dim = dimensionalize_fields(
        sub_mesh_star, u_star, p_star, theta,
        scales.Uplume, scales.dTref, T_ambient,
        experiment.fluid.properties["rho"]
    )

    # Optional diagnostic: Biot numbers should use dimensional geometry/fields
    try:
        _,_, biots = biot_wrap(
            sub_mesh=sub_mesh_dim,
            sub_ft=sub_ft_dim,
            sub_ds=sub_ds_dim,
            T_air_dim=T_air_dim,
            qn_air=qn_air,
            scales=scales,
            T_ref=T_ambient,
            k_wire=experiment.wire.properties["k"],
            wire_diameter=experiment.dimensions.wire.diameter,
            characteristic_length="radius",
            return_local_field=True
        )
        print0(f"Biot number stats: min={biots.vector().min():.6e}, max={biots.vector().max():.6e}")
    except Exception:
        print0("Biot diagnostic failed; check dimensional geometry/fields and interface tagging.")
    # print0(f"Effective Biot number after steady solve: Bi_air = {biot_air_Bi:.6e}")
    
    w, transient_info = run_post_temp_continuation_transient(
            experiment=experiment,
            fluid_material=fluid_material,
            W=W,
            w=w,
            w_n=w_n,
            psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
            mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
            sub_dx=sub_dx_star, sub_ds=sub_ds_star, sub_ft=sub_ft_star, qn_air=qn_air_star,
            sub_mesh_star=sub_mesh_star,
            sub_mesh_dim=sub_mesh_dim,
            sub_ft_dim=sub_ft_dim,
            sub_ds_dim=sub_ds_dim,
            scales=scales,
            T_ambient=T_ambient,
            p_path=OUTPUT_XDMF_PATH_AIR_P,
            u_path=OUTPUT_XDMF_PATH_AIR_V,
            T_path=OUTPUT_XDMF_PATH_AIR_T,
            dt_start=1.0e-2,
            dt_growth=1.2,
            dt_cut=0.5,
            dt_hard_cut=0.8,
            dt_min=1.0e-5,
            dt_max=1.0,
            t_end=15000.0,
            step_max=20000,
            save_every=20,
            max_retries_per_step=8,
            rel_update_easy=1.0e-3,
            rel_update_hard=5.0e-3,
            rel_update_reject=2.0e-2,
            steady_window=25,
            steady_rel_tol=5.0e-3,
            steady_update_tol=1.0e-4,
            start_time=restart_meta["time"],
            start_step=restart_meta["step"],
            history_csv_path=run_root + "/transient_history.csv",
        )

    print0("transient status:", transient_info["status"])
    print0("transient steps:", transient_info["n_steps"])


    # Split nondimensional solution
    p_star, u_star, theta = w.split(deepcopy=True)

    # Dimensionalize fields (note: mesh is star; dimensionalize handles scaling)
    u_dim, p_dim, T_dim = dimensionalize_fields(
        sub_mesh_star, u_star, p_star, theta,
        scales.Uplume, scales.dTref, T_ambient,
        experiment.fluid.properties["rho"]
    )

    # on sub_mesh_star, with theta (nondim)
    k_inf = float(experiment.fluid.properties["k"])  # use experiment value (not global)
    n = fenics.FacetNormal(sub_mesh_star)
    dTref = float(scales.dTref)

    # Example: if facet tags exist for TOP and FAR boundaries
    Q_far  = -k_inf*dTref * fenics.assemble(fenics.dot(fenics.grad(theta), n) * sub_ds_star(OUTER_AIR_TAG))
    print0(f"Heat flux through outer air boundary: Q_far = {Q_far:.6e} W/m")


    # plotting + output
    plot_mesh(T_dim, title="Temperature field", label="Temperature (K)",
                cmap="coolwarm", colorbar=True)
    plot_mesh(theta, title="Temperature field", label="Temperature (nondim)",
                cmap="coolwarm", colorbar=True)
    plot_mesh(u_dim, title="Velocity magnitude", label="Velocity (m/s)",
                cmap="coolwarm", colorbar=True, mode="glyphs")
    plot_mesh(p_dim, title="Pressure field", label="Pressure (Pa)",
                cmap="coolwarm", colorbar=True)

    save_experiment(OUTPUT_XDMF_PATH_AIR_P, sub_mesh_star, [p_dim])
    save_experiment(OUTPUT_XDMF_PATH_AIR_V, sub_mesh_star, [u_dim])
    save_experiment(OUTPUT_XDMF_PATH_AIR_T, sub_mesh_star, [T_dim])

    # Example: Brodowicz-style heights 1, 4, 8 cm above wire center
    y0_m_list = [0.01, 0.04, 0.08]
    hmin_star = sub_mesh_star.hmin()   # dimensionless
    hmax_star = sub_mesh_star.hmax()
    eps_m = 3 * 0.5*(hmin_star + hmax_star) * scales.Lref
    # eps_m = 2 * hmin_star * scales.Lref
    flux_rows = plane_fluxes_slab_star(
        sub_mesh_star,
        u, T,                   # your returned nondim u and theta
        y0_m_list,
        scales=scales,
        rho=experiment.fluid.properties["rho"],
        cp=experiment.fluid.properties["cp"],
        k=experiment.fluid.properties["k"],
        eps_m=eps_m             # e.g. 1 mm slab half-thickness (tune to mesh)
    )

    for (y0_m, Qconv, Qcond, Qtot, mdot) in flux_rows:
        print0(f"y0={y0_m:.3f} m: Qconv={Qconv:.6e} W/m, Qcond={Qcond:.6e} W/m, "
            f"Qtot={Qtot:.6e} W/m, mdot={mdot:.6e} kg/(s·m)")
        
    out_dir=Path.cwd()
    csv_path = os.path.join(out_dir,run_root, "temp", "plane_fluxes.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        wcsv = csv.writer(f)
        if write_header:
            wcsv.writerow(["time", "y0_m", "Qconv_W_per_m", "Qcond_W_per_m", "Qtot_W_per_m", "mdot_kg_per_s_per_m"])
        t=0
        for (y0_m, Qconv, Qcond, Qtot, mdot) in flux_rows:
            wcsv.writerow([float(t), y0_m, Qconv, Qcond, Qtot, mdot])

def abs_version(experiment: Experiment, restart_from_last_transient: bool = False, existing_run_root: str = ""):
    mkdir0(experiment.name)
    run_root = make_run_root(experiment.name, "abs", reuse_existing=existing_run_root)

    if is_rank0():
        os.makedirs(os.path.join(run_root, "base"), exist_ok=True)
    COMM.Barrier()

    GEOM_FILE = geometry_template(
        wire_radius=experiment.dimensions.wire.diameter / 2,
        output_path=experiment.name,
        xmax=experiment.dimensions.domain.x_max,
        ymax=experiment.dimensions.domain.y_max,
        resolution=100,
    )
    
    MSH_FILE = run_root + "/plume.msh"
    TRIG_XDMF_PATH = run_root + "/full_cells.xdmf"
    FACETS_XDMF_PATH = run_root + "/full_facets.xdmf"
    AIR_TRIG_XDMF_PATH = run_root + "/air_cells.xdmf"
    AIR_FACETS_XDMF_PATH = run_root + "/air_facets.xdmf"
    OUTPUT_XDMF_PATH_WIRE = run_root + "/abs/wire_temperature.xdmf"
    OUTPUT_XDMF_PATH_TEMP = run_root + "/abs/temperature.xdmf"
    OUTPUT_XDMF_PATH_AIR_T = run_root + "/abs/air_temperature.xdmf"
    OUTPUT_XDMF_PATH_AIR_P = run_root + "/abs/air_pressure.xdmf"
    OUTPUT_XDMF_PATH_AIR_V = run_root + "/abs/air_velocity.xdmf"
    OUTPUT_XDMF_PATH_AIR_PVT = run_root + "/abs/air_pvt.xdmf"
    MESH_NAME = "Grid"
    ELEM = "triangle"

   # Rank 0 generates:
    #   full_cells.xdmf/full_facets.xdmf
    #   air_cells.xdmf/air_facets.xdmf
    # All ranks wait at the barrier inside generate_mesh().
    mesh_files_exist = (
        os.path.exists(TRIG_XDMF_PATH)
        and os.path.exists(FACETS_XDMF_PATH)
        and os.path.exists(AIR_TRIG_XDMF_PATH)
        and os.path.exists(AIR_FACETS_XDMF_PATH)
    )

    if existing_run_root and mesh_files_exist:
        print0("Using existing mesh files; not regenerating gmsh mesh.")
        COMM.Barrier()
    else:
        generate_mesh(
            GEOM_FILE,
            MSH_FILE,
            TRIG_XDMF_PATH,
            FACETS_XDMF_PATH,
            AIR_TRIG_XDMF_PATH=AIR_TRIG_XDMF_PATH,
            AIR_FACETS_XDMF_PATH=AIR_FACETS_XDMF_PATH,
            AIR_TAG_VALUE=AIR_TAG,
            ELEM=ELEM,
            PRUNE_Z=True,
        )

    # Full dimensional mesh.
    # Keep this for your full-domain initial thermal solve.
    mesh, ct, ft, domains, dx, boundary_markers, mc, mf = read_mesh(
        TRIG_XDMF_PATH,
        FACETS_XDMF_PATH,
        MESH_NAME,
        PRINT_TAG_SUMMARY,
    )

    # Air-only dimensional mesh.
    # This replaces:
    #     sub_mesh_dim, sub_ft_dim, sub_dx_dim, sub_ds_dim = create_submesh(...)
    sub_mesh_dim, air_ct, air_ft, _, sub_dx_dim, _, sub_mc_dim, sub_ft_dim = read_mesh(
        AIR_TRIG_XDMF_PATH,
        AIR_FACETS_XDMF_PATH,
        MESH_NAME,
        PRINT_TAG_SUMMARY,
    )
    
    sub_dx_dim = fenics.Measure(
        "dx",
        domain=sub_mesh_dim,
        subdomain_data=sub_mc_dim,
    )
    sub_ds_dim = fenics.Measure(
        "ds",
        domain=sub_mesh_dim,
        subdomain_data=sub_ft_dim,
    )
    
    scales = compute_nondimensional_scales(experiment)
    print0(scales)
    print0(f"Uref   = {scales.Uref:.6e} m/s")
    print0(f"Uplume = {scales.Uplume:.6e} m/s")
    print0(f"Uref/Uplume = {scales.Uref_over_Uplume:.6e}")
    print0(f"Lref   = {scales.Lref:.6e} m")
    print0(f"Lplume = {scales.Lplume:.6e} m")
    print0(f"dTref  = {scales.dTref:.6e} K")
    print0(f"dTline = {scales.dTline:.6e} K")

    # --- 3) conduction initial guess (dim parent mesh)
    print0("Computing initial guess for temperature field...")
    heat_volume = volume_heat_source(experiment)
    print0(f"Using heat volume: {heat_volume} W/m^3")

    T_full, k_func = initial_guess(mesh, mc, mf, OUTPUT_XDMF_PATH_TEMP,
                                    heat_volume, experiment, dx)

    # --- 4) restrict/interpolate to submesh (dim) → theta_full_dim
    # (DO NOT project across meshes; interpolate T_full onto submesh space first)
    T_ambient = float(experiment.initial_conditions.temperature)
    dTref = float(scales.dTref)

    V_air_dim = fenics.FunctionSpace(sub_mesh_dim, "CG", 1)

    # Allow evaluation just outside due to tolerance / boundary issues
    T_full.set_allow_extrapolation(True)

    T_air_dim = fenics.interpolate(T_full, V_air_dim)

    theta_full_dim = fenics.Function(V_air_dim)
    theta_full_dim.vector()[:] = (T_air_dim.vector()[:] - T_ambient) / dTref
    theta_full_dim.vector().apply("insert")
    theta_full_dim.rename("theta_full", "theta_full")

    print0("theta min/max:", global_vec_min(theta_full_dim), global_vec_max(theta_full_dim))
    print0("T_dim  min/max:", global_vec_min(T_air_dim), global_vec_max(T_air_dim))
    print0("T_nondim min/max:", global_vec_min(theta_full_dim), global_vec_max(theta_full_dim))

    # --- (still dimensional) compute qn_air using your current routine
    qn_air = flux_continuity(T_full, k_func, mesh, sub_mesh_dim, sub_ft_dim, mc, scales)

    # Diagnostic: power conservation on DIMENSIONAL interface measure
    check_interface_power(sub_ds_dim, sub_ft_dim, qn_air, scales, experiment)

    # Optional diagnostic: Biot numbers should use dimensional geometry/fields
    # biot_air_h_eff, biot_air_Bi = biot(
    #     sub_mesh_dim, sub_ft_dim, T_full, qn_air,
    #     T_ambient, experiment.wire.properties["k"],
    #     experiment.dimensions.wire.diameter
    # )

    print0(f"Initial max temperature: {global_vec_max(T_full):.2f} K")
    print0(f"Initial min temperature: {global_vec_min(T_full):.2f} K")
    print0(f"Initial max theta (dim-submesh): {global_vec_max(theta_full_dim):.6e}")
    print0(f"Initial min theta (dim-submesh): {global_vec_min(theta_full_dim):.6e}")

    # --- 5) scale parent mesh coordinates (dim→star)
    Lref = float(scales.Lref)
    scale_mesh_inplace(mesh, Lref)
    scale_mesh_inplace(sub_mesh_dim, Lref)

    try:
        mesh.bounding_box_tree().build(mesh)
        sub_mesh_dim.bounding_box_tree().build(sub_mesh_dim)
    except Exception:
        pass

    COMM.Barrier()

    # The air mesh has been scaled in place, so it is now the star mesh.
    # No SubMesh reconstruction is needed.
    sub_mesh_star = sub_mesh_dim
    sub_ft_star = sub_ft_dim
    sub_dx_star = fenics.Measure(
        "dx",
        domain=sub_mesh_star,
        subdomain_data=sub_mc_dim,
    )
    sub_ds_star = fenics.Measure(
        "ds",
        domain=sub_mesh_star,
        subdomain_data=sub_ft_star,
    )

    qn_air_star = qn_air

    theta_full_star = solve_air_initial_theta(
        air_mesh=sub_mesh_star,
        air_facet_markers=sub_ft_star,
        air_ds=sub_ds_star,
        qn_air=qn_air_star,
        interface_tag=INTERFACE_TAG,
        cold_tags=(101, 103),
    )

    print0(f"Initial max theta (star-submesh): {global_vec_max(theta_full_star):.6e}")
    print0(f"Initial min theta (star-submesh): {global_vec_min(theta_full_star):.6e}")
    print0(f"Rho_air: {experiment.fluid.properties['rho']}")
    print0(f"Beta_air: {experiment.fluid.properties['beta']}")
    
    # Solving the problem
    print0("Starting solver...")
    W, w, p, u, T, w_n, p_n, u_n, T_n, psi_p, psi_u, psi_T, \
    mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc = solver(
        sub_mesh_star,
        theta_full_star,      # <-- nondimensional theta on star mesh
        0.0,
        experiment.fluid.properties["rho"],
        experiment.fluid.properties["beta"],
        experiment
    )

    # Optional restart
    restart_meta = {"step": 0, "time": 0.0, "dt": 1.0e-4, "source": "fresh_start"}
    if restart_from_last_transient:
        checkpoint_dir = os.path.join(run_root, "abs", "restart_checkpoint")
        try:
            if os.path.exists(os.path.join(checkpoint_dir, "state.h5")):
                print0("Attempting restart from true checkpoint...")
                w, w_n, restart_meta = load_true_restart_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    W=W,
                    w=w,
                    w_n=w_n,
                )
            else:
                print0("Attempting restart from last saved transient snapshot...")
                w, w_n, restart_meta = approximate_restart_from_last_saved_transient(
                    run_root=run_root,
                    mode_subdir="base",
                    sub_mesh_dim=sub_mesh_dim,
                    sub_mesh_star=sub_mesh_star,
                    W=W,
                    w=w,
                    w_n=w_n,
                    scales=scales,
                    T_ambient=T_ambient,
                    rho_air=experiment.fluid.properties["rho"],
                    fallback_dt=1.0e-4,
                )

            print0(
                f"Restart recovered from {restart_meta['source']}: "
                f"step={restart_meta['step']}, "
                f"time={restart_meta['time']:.6e}, "
                f"dt={restart_meta['dt']:.6e}"
            )

        except Exception as exc:
            print0("Restart request could not be satisfied.")
            print0(f"Reason: {exc}")
            print0("Falling back to fresh transient start from steady state.")
            copy_state(w, w_n)
            restart_meta = {"step": 0, "time": 0.0, "dt": 1.0e-5, "source": "fresh_start"}

    if not restart_from_last_transient:
        # Use Stokes initial guess for better convergene
        print0("Solving Stokes problem for initial guess...")
        w_n = stokes_initial_guess(
            experiment=experiment,
            u_n=u_n, u=u, T_n=T_n, T=T, p=p,
            W=W, w=w,
            psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
            mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
            sub_dx=sub_dx_star, sub_ds=sub_ds_star, sub_ft=sub_ft_star, qn_air=qn_air_star,
            w_n=w_n,
            lambdas=( 0.05, 0.1, 0.3)
        )

    # Solve the full nonlinear problem with previous initial guess
    print0("Starting checks")
    w_t = w.copy(deepcopy=True)
    w_t =solve_thermal_sign_check(experiment=experiment,W=W,w=w_t,
                                mu=mu,Pr=Pr,
                                sub_dx=sub_dx_star,sub_ds=sub_ds_star,sub_ft=sub_ft_star,
                                qn_air=qn_air_star,T_c=T_c,T_air_bc=T_air_bc,w_n=w_n)
    w_p = w.copy(deepcopy=True)
    w_p = solve_buoyancy_sign_check(experiment=experiment,W=W,w=w_p,
                                  mu=mu,Pr=Pr,
                                  sub_dx=sub_dx_star,sub_ds=sub_ds_star,sub_ft=sub_ft_star,
                                  qn_air=qn_air_star,T_c=T_c,T_air_bc=T_air_bc,w_n=w_n)
    print0("Checks complete")

    # w = solve_ABE_newton_continuation(
    #     experiment=experiment,
    #     u_n=u_n, u=u, T_n=T_n, T=T, p=p,
    #     W=W, w=w,
    #     psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
    #     mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
    #     sub_dx=sub_dx_star, sub_ds=sub_ds_star, sub_ft=sub_ft_star, qn_air=qn_air_star,
    #     w_n=w_n,
    #     lambdas=[0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00],
    #     relaxation_schedule=(0.9, 0.7, 0.5), #0.4, 0.35, 0.30, 0.8, 0.27, 0.25, 0.22, 0.20, 0.7, 0.15, 0.10, 0.05, 0.02, 0.01),
    #     stokes_startup=False,
    #     sub_mesh_star=sub_mesh_star,
    #     sub_mesh_dim=sub_mesh_dim,
    #     p_path=OUTPUT_XDMF_PATH_AIR_P,
    #     u_path=OUTPUT_XDMF_PATH_AIR_V,
    #     T_path=OUTPUT_XDMF_PATH_AIR_T
    # )

    save_obj = (
        sub_ds_star,
        sub_mesh_dim,
        scales,
        OUTPUT_XDMF_PATH_AIR_P,
        OUTPUT_XDMF_PATH_AIR_V,
        OUTPUT_XDMF_PATH_AIR_T
    )
    SUPG = False

    if not restart_from_last_transient:
        w, info = solve_ptc_abe_continuation(
            experiment,
            run_root,
            W, w, w_n,
            psi_p, psi_u, psi_T,
            mu, Pr, f_b, T_c, T_air_bc,
            sub_dx_star, sub_ds_star, sub_ft_star, qn_air_star,
            dtau_init=1e-5,
            dtau_min=1e-8,
            dtau_max=1e-2,
            stage_max_steps=40,
            final_stage_max_steps=2000,
            update_tol=1e-8,
            residual_tol=1e-8,
            # save_obj=save_obj
            save_obj=None
        )

        if info.get("status") == "continuation_failed":
            print0("failed_stage:", info.get("failed_stage"))
            last = info.get("last_stage_info", {})
            print0("last_stage_status:", last.get("status"))
            print0("accepted_steps:", last.get("accepted_steps"))
            print0("rejected_steps:", last.get("rejected_steps"))
            print0("final_dtau:", last.get("final_dtau"))
            print0("final_rel_update:", last.get("final_rel_update"))
            print0("final_steady_residual:", last.get("final_steady_residual"))
        else:
            print0("accepted_steps:", info.get("accepted_steps"))
            print0("rejected_steps:", info.get("rejected_steps"))
            print0("final_dtau:", info.get("final_dtau"))
            print0("final_rel_update:", info.get("final_rel_update"))
            print0("final_steady_residual:", info.get("final_steady_residual"))
    
    # Split nondimensional solution
    p_star, u_star, theta = w.split(deepcopy=True)

    # Dimensionalize fields (note: mesh is star; dimensionalize handles scaling)
    u_dim, p_dim, T_dim = dimensionalize_fields(
        sub_mesh_star, u_star, p_star, theta,
        scales.Uplume, scales.dTref, T_ambient,
        experiment.fluid.properties["rho"]
    )

    # Optional diagnostic: Biot numbers should use dimensional geometry/fields
    try:
        _,_, biots = biot_wrap(
            sub_mesh=sub_mesh_dim,
            sub_ft=sub_ft_dim,
            sub_ds=sub_ds_dim,
            T_air_dim=T_air_dim,
            qn_air=qn_air,
            scales=scales,
            T_ref=T_ambient,
            k_wire=experiment.wire.properties["k"],
            wire_diameter=experiment.dimensions.wire.diameter,
            characteristic_length="radius",
            return_local_field=True
        )
        print0(f"Biot number stats: min={biots.vector().min():.6e}, max={biots.vector().max():.6e}")
    except Exception:
        print0("Biot diagnostic failed; check dimensional geometry/fields and interface tagging.")
    # print0(f"Effective Biot number after steady solve: Bi_air = {biot_air_Bi:.6e}")

    dt_start=1.0e-4
    if restart_from_last_transient:
        dt_start = restart_meta["dt"]
        print0(f"Starting transient with dt={dt_start:.6e} from restart source: {restart_meta['source']}")
    
    w, transient_info = run_post_abe_continuation_transient(
            experiment=experiment,
            W=W,
            w=w,
            w_n=w_n,
            psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
            mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
            sub_dx=sub_dx_star, sub_ds=sub_ds_star, sub_ft=sub_ft_star, qn_air=qn_air_star,
            sub_mesh_star=sub_mesh_star,
            sub_mesh_dim=sub_mesh_dim,
            sub_ft_dim=sub_ft_dim,
            sub_ds_dim=sub_ds_dim,
            scales=scales,
            p_path=OUTPUT_XDMF_PATH_AIR_P,
            u_path=OUTPUT_XDMF_PATH_AIR_V,
            T_path=OUTPUT_XDMF_PATH_AIR_T,
            dt_start=dt_start,
            dt_growth=1.1,
            dt_cut=0.8,
            dt_hard_cut=0.5,
            dt_min=1.0e-5,
            dt_max=1.0,
            t_end=15000.0,
            n_steps=200000,
            save_every=20,
            max_retries_per_step=8,
            rel_update_easy=1.0e-3,
            rel_update_hard=1.0e-3,
            rel_update_reject=2.0e-2,
            steady_window=25,
            steady_rel_tol=1.0e-5,
            steady_update_tol=1.0e-6,
            start_time=restart_meta["time"],
            start_step=restart_meta["step"],
            history_csv_path=run_root + "/transient_history.csv",
            restart_recovered=restart_from_last_transient,
            restart_step=restart_meta["step"],
        )

    print0("transient status:", transient_info["status"])
    print0("transient steps:", transient_info["n_steps"])

    # Split nondimensional solution
    p_star, u_star, theta = w.split(deepcopy=True)

    # Dimensionalize fields (note: mesh is star; dimensionalize handles scaling)
    u_dim, p_dim, T_dim = dimensionalize_fields(
        sub_mesh_star, u_star, p_star, theta,
        scales.Uplume, scales.dTref, T_ambient,
        experiment.fluid.properties["rho"]
    )

    # plotting + output
    # plot_mesh(T_dim, title="Temperature field", label="Temperature (K)",
    #             cmap="coolwarm", colorbar=True)
    # plot_mesh(theta, title="Temperature field", label="Temperature (nondim)",
    #             cmap="coolwarm", colorbar=True)
    # plot_mesh(u_dim, title="Velocity magnitude", label="Velocity (m/s)",
    #             cmap="coolwarm", colorbar=True, mode="glyphs")
    # plot_mesh(p_dim, title="Pressure field", label="Pressure (Pa)",
    #             cmap="coolwarm", colorbar=True)

    save_experiment(OUTPUT_XDMF_PATH_AIR_P, sub_mesh_star, [p_dim])
    save_experiment(OUTPUT_XDMF_PATH_AIR_V, sub_mesh_star, [u_dim])
    save_experiment(OUTPUT_XDMF_PATH_AIR_T, sub_mesh_star, [T_dim])

    # Example: Brodowicz-style heights 1, 4, 8 cm above wire center
    y0_m_list = [0.01, 0.04, 0.08]
    hmin_star = sub_mesh_star.hmin()   # dimensionless
    hmax_star = sub_mesh_star.hmax()
    eps_m = 3 * 0.5*(hmin_star + hmax_star) * scales.Lref
    # eps_m = 2 * hmin_star * scales.Lref
    flux_rows = plane_fluxes_slab_star(
        sub_mesh_star,
        u, T,                   # your returned nondim u and theta
        y0_m_list,
        scales=scales,
        rho=experiment.fluid.properties["rho"],
        cp=experiment.fluid.properties["cp"],
        k=experiment.fluid.properties["k"],
        eps_m=eps_m             # e.g. 1 mm slab half-thickness (tune to mesh)
    )

    for (y0_m, Qconv, Qcond, Qtot, mdot) in flux_rows:
        print0(f"y0={y0_m:.3f} m: Qconv={Qconv:.6e} W/m, Qcond={Qcond:.6e} W/m, "
            f"Qtot={Qtot:.6e} W/m, mdot={mdot:.6e} kg/(s·m)")
    
    out_dir=Path.cwd()
    csv_path = os.path.join(out_dir,run_root, "abs", "plane_fluxes.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        wcsv = csv.writer(f)
        if write_header:
            wcsv.writerow(["time", "y0_m", "Qconv_W_per_m", "Qcond_W_per_m", "Qtot_W_per_m", "mdot_kg_per_s_per_m"])
        t=0
        for (y0_m, Qconv, Qcond, Qtot, mdot) in flux_rows:
            wcsv.writerow([float(t), y0_m, Qconv, Qcond, Qtot, mdot])


def main():
    # Parse command line arguments
    argparser = argparse.ArgumentParser(description="Heated Laminar Plume Simulation")
    argparser.add_argument(
        "--experiment-index",
        type=int,
        default=1,
        help="Index of the experiment to run from experiments.json",
    )
    argparser.add_argument("--restart-from-last-transient", action="store_true")
    argparser.add_argument("--existing-run-root", type=str, default="")
    argparser.add_argument("--steady-from-last-transient", action="store_true")

    argparser.add_argument(
        "--refine-restart-checkpoint",
        type=str,
        default="",
        help="Offline AMR: input restart checkpoint directory containing state.h5/state.json",
    )

    argparser.add_argument(
        "--refined-checkpoint-out",
        type=str,
        default="",
        help="Offline AMR: output checkpoint directory for the refined restart",
    )

    argparser.add_argument(
        "--amr-top-fraction",
        type=float,
        default=0.05,
        help="Fraction of cells to refine during offline AMR",
    )

    argparser.add_argument(
        "--amr-levels",
        type=int,
        default=1,
        help="Number of offline AMR refinement levels",
    )

    argparser.add_argument(
        "--amr-dt-factor",
        type=float,
        default=0.25,
        help="Factor applied to checkpoint dt after AMR interpolation",
    )

    argparser.add_argument(
        "--restart-from-checkpoint-mesh",
        type=str,
        default="",
        help="Restart from a checkpoint using the mesh stored in state.h5",
    )
    
    args = argparser.parse_args()
    args.experiment_index = max(0, args.experiment_index)
    experiment_list = parser(experiments_json_path=EXPERIMENTS_JSON_PATH, schema_json_path=SCHEMA_JSON_PATH)
    experiment = experiment_list[args.experiment_index]
    print0(f"Running experiment: {experiment.name}")

    if args.refine_restart_checkpoint:
        if not args.refined_checkpoint_out:
            raise ValueError(
                "--refined-checkpoint-out is required when using "
                "--refine-restart-checkpoint"
            )

        refine_checkpoint_offline(
            input_checkpoint_dir=args.refine_restart_checkpoint,
            output_checkpoint_dir=args.refined_checkpoint_out,
            top_fraction=args.amr_top_fraction,
            levels=args.amr_levels,
            dt_factor=args.amr_dt_factor,
        )
        print0("Offline AMR checkpoint migration completed.")
        return

    base_version(
        experiment,
        restart_from_last_transient=args.restart_from_last_transient,
        existing_run_root=args.existing_run_root,
        steady_from_last_transient=args.steady_from_last_transient,
        restart_from_checkpoint_mesh=args.restart_from_checkpoint_mesh,
    )
    # base_version_new(experiment)  
    # temperature_dependent_version(experiment)
    # abs_version(
    #     experiment,
    #     restart_from_last_transient=args.restart_from_last_transient,
    #     existing_run_root=args.existing_run_root
    # )


if __name__ == "__main__":
    main()
    # mpirun -np 48 --use-hwthread-cpus python main.py --experiment-index 1 --restart-from-last-transient --existing-run-root PlumeCase_Brodowicz_Air/runs/abs_20260423_195242_pid2592222
    # mpirun -np 48 --use-hwthread-cpus python main.py --experiment-index 1 --existing-run-root  PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866/ --restart-from-checkpoint-mesh PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866/base/restart_checkpoint_amr
    # python main.py --experiment-index 1 --refine-restart-checkpoint  PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866/base/restart_checkpoint --refined-checkpoint-out PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866/base/restart_checkpoint_amr --amr-top-fraction 0.1 --amr-levels 2 --amr-dt-factor 0.25
