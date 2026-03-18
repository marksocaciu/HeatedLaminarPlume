from solver import scales
from utils.imports import *
from solver.solver import *
from solver.params_bcs import *


def pseudo_transient_rescue(
    experiment: Experiment,
    W: fenics.FunctionSpace,
    w: fenics.Function,
    w_n: fenics.Function,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b, T_c, T_air_bc,
    sub_dx, sub_ds, sub_ft, qn_air,
    lam: float,
    conv_scale: float,
    dtau_schedule=(1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    steps_per_dtau: int = 6,
    update_tol: float = 1e-8,
    retry_newton_every: int = 3,
    relaxation: float = 1.0,
):
    """
    Pseudo-transient rescue started from the provided seed state w_n.

    In the adaptive continuation workflow, w_n should usually be the
    failed Newton iterate at the difficult continuation stage.
    """
    scales = compute_nondimensional_scales(experiment)
    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

    w_prev = fenics.Function(W)
    copy_state(w, w_n)
    copy_state(w_prev, w_n)

    ptc_step = 0
    for dtau in dtau_schedule:
        print(f"    -> PTC block with dtau={dtau:.3e}")
        for _ in range(steps_per_dtau):
            ptc_step += 1
            copy_state(w_prev, w)

            F_ptc, JF_ptc = build_ptc_problem(
                W=W, w=w, w_prev=w_prev,
                psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
                mu=mu, Pr=Pr, f_b=f_b,
                sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
                dtau=dtau,
                buoyancy_scale=lam,
                qn_scale=lam,
                include_convection=(conv_scale > 0.0),
                convection_scale=conv_scale,
            )

            try:
                base_solver(
                    F_ptc, w, boundary_conditions, JF_ptc,
                    relaxation=relaxation,
                    maxit=20,
                    atol=1e-9,
                    rtol=1e-8,
                )
            except RuntimeError as err:
                print(f"    PTC step failed at dtau={dtau:.3e}: {err}")
                return False

            delta = w.vector().copy()
            delta.axpy(-1.0, w_prev.vector())
            rel_update = delta.norm("l2") / (w.vector().norm("l2") + 1e-14)
            print(f"    PTC step {ptc_step:03d}: rel_update={rel_update:.3e}")

            copy_state(w_n, w)

            if rel_update < update_tol:
                print("    PTC update is small; handing control back to steady Newton.")
                return True

            if (ptc_step % retry_newton_every) == 0:
                print("    PTC block produced a new seed; retry steady Newton now.")
                return True

    return True

def solve_steady_newton_continuation(
    experiment: Experiment,
    u_n: fenics.Function, u: fenics.Function, T_n: fenics.Function, T: fenics.Function, p: fenics.Function,
    W: fenics.FunctionSpace, w: fenics.Function,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b, T_c, T_air_bc,
    sub_dx, sub_ds, sub_ft, qn_air,
    w_n: fenics.Function,
    lambdas=None,
    relaxation_schedule=(0.9, 0.7, 0.5),
    stokes_startup=True,
    sub_mesh_star=None,
    sub_mesh_dim=None,
    p_path: str = "",
    u_path: str = "",
    T_path: str = "",
):
    """
    Steady continuation solve with monotone convection ramping.

    For each lambda:
      1) solve Stokes / no-momentum-convection stage
      2) ramp convection monotonically with a coarse ladder
      3) if a target convection scale fails, bisect only that interval

    This avoids the wasteful pattern:
      direct full solve -> many failed relaxations -> fallback continuation
    """
    if lambdas is None:
        lambdas = [0.05, 0.10, 0.20, 0.40, 0.70, 1.00]

    w.vector()[:] = w_n.vector()
    w.vector().apply("insert")

    scales = compute_nondimensional_scales(experiment)
    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

    conv_targets = conv_stage_sequence()

    for lam in lambdas[:1]:
        print(f"\n=== Newton continuation lambda = {lam:.2f} ===")

        # optional output of current accepted state before advancing lambda
        if sub_mesh_star is not None and sub_mesh_dim is not None:
            p_star, u_star, theta = w_n.split(deepcopy=True)

            u_dim, p_dim, T_dim = dimensionalize_fields(
                sub_mesh_star, u_star, p_star, theta,
                scales.Uref, scales.dTref, T_ambient,
                experiment.fluid.properties["rho"]
            )

            p_out = p_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
            u_out = u_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
            t_out = T_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"

            save_experiment(p_out, sub_mesh_dim, [p_dim])
            save_experiment(u_out, sub_mesh_dim, [u_dim])
            save_experiment(t_out, sub_mesh_dim, [T_dim])

        # ------------------------------------------------------------
        # Stage 1: Stokes / zero momentum convection
        # ------------------------------------------------------------
        print("  --- stage: stokes ---")

        F_stokes, JF_stokes = build_nonlinear_problem(
            W=W, w=w,
            psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
            mu=mu, Pr=Pr, f_b=f_b,
            sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
            buoyancy_scale=lam,
            qn_scale=lam,
            include_convection=False,
            convection_scale=0.0,
        )

        ok, used_relax, last_error = try_newton_stage_FJF_outside(
            F=F_stokes,
            JF=JF_stokes,
            w=w,
            w_n=w_n,
            boundary_conditions=boundary_conditions,
            relaxation_schedule=relaxation_schedule,
            stage_name=f"lambda_{lam:.2f}_stokes",
        )

        if not ok:
            raise RuntimeError(
                f"Continuation Newton failed at lambda={lam:.2f} during stokes stage. "
                f"Last error: {last_error}"
            )

        accept_current_state(w, w_n)
        print(f"  stokes accepted at lambda={lam:.2f} (relaxation={used_relax:.3f})")

        # ------------------------------------------------------------
        # Stage 2: monotone convection advance
        # ------------------------------------------------------------
        accepted_conv = 0.0
        for target_conv,lamda in conv_targets:
            accepted_conv, _, _ = advance_convection_monotone(
                W=W, w=w,
                psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
                mu=mu, Pr=Pr, f_b=f_b,
                sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
                boundary_conditions=boundary_conditions,
                w_n=w_n,
                buoyancy_scale=lamda,
                start_conv=accepted_conv,
                target_conv=target_conv,
                relaxation_schedule=relaxation_schedule,
                min_interval=0.005,          # tighten/loosen as needed
                max_local_bisections=3,
            )

        print(f"  lambda={lam:.2f} completed with accepted conv_scale={accepted_conv:.4f}")

        # keep working function synchronized with accepted state
        w.vector()[:] = w_n.vector()
        w.vector().apply("insert")

    return w

def solve_steady_newton_continuation_with_pts(
    experiment: Experiment,
    u_n: fenics.Function, u: fenics.Function, T_n: fenics.Function, T: fenics.Function, p: fenics.Function,
    W: fenics.FunctionSpace, w: fenics.Function,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b, T_c, T_air_bc,
    sub_dx, sub_ds, sub_ft, qn_air,
    w_n: fenics.Function,
    lambdas=None,
    relaxation_schedule=(0.2, 0.1, 0.05),
    stokes_startup=True,
    sub_mesh_star=None,
    sub_mesh_dim=None,
    p_path: str = "",
    u_path: str = "",
    T_path: str = "",
):
    """
      Hybrid steady solve:
      - primary method: damped Newton continuation
      - on a failed convection substage: bisect the failed step
      - if the bisected step still fails: pseudo-transient rescue
      - then retry steady Newton from the rescued state

    This keeps the target solution steady, while using pseudo-time marching only as a
    globalization device near difficult continuation points.

    For each lambda:
      1) solve a no-momentum-convection stage
      2) solve the full nonlinear stage
    and promote the converged solution after each successful stage.
    """
    if lambdas is None:
        lambdas = [0.05, 0.10, 0.20, 0.40, 0.70, 1.00]

    # w.vector()[:] = w_n.vector()
    # w.vector().apply("insert")
    copy_state(w, w_n)
    scales = compute_nondimensional_scales(experiment)
    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)
    max_bisection_depth = 3

    for lam in lambdas:
        print(f"\n=== Newton continuation lambda = {lam:.2f} ===")
        
        if sub_mesh_star is not None and sub_mesh_dim is not None and p_path and u_path and T_path:
            p_star, u_star, theta = w.split(deepcopy=True)
            u_dim, p_dim, T_dim = dimensionalize_fields(
                sub_mesh_star, u_star, p_star, theta,
                scales.Uref, scales.dTref, T_ambient,
                experiment.fluid.properties["rho"],
            )
            p_path_l = p_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
            v_path_l = u_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
            t_path_l = T_path.split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}.xdmf"
            save_experiment(p_path_l, sub_mesh_dim, [p_dim])
            save_experiment(v_path_l, sub_mesh_dim, [u_dim])
            save_experiment(t_path_l, sub_mesh_dim, [T_dim])

        stage_grid = build_stage_grid(lam)
        accepted_conv_scale = 0.0
        i = 0
        
        while i < len(stage_grid):
            target_conv_scale = stage_grid[i]
            include_convection = target_conv_scale > 0.0

            if i == 0 and target_conv_scale == 0.0:
                print("  --- stage: Stokes / zero-convection stage ---")
            else:
                print(f"  --- stage: convection_scale = {target_conv_scale:.6f} ---")

            stage_seed = fenics.Function(W)
            copy_state(stage_seed, w_n)

            F, JF = build_nonlinear_problem(
                W=W, w=w,
                psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
                mu=mu, Pr=Pr, f_b=f_b,
                sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
                buoyancy_scale=lam,
                qn_scale=lam,
                include_convection=include_convection,
                convection_scale=target_conv_scale,
            )

            ok, used_relax, err = try_newton_stage_FJF_outside(
                F=F, JF =JF, w=w, w_n=stage_seed,
                boundary_conditions=boundary_conditions,
                relaxation_schedule=relaxation_schedule,
                maxit=20,
                atol=1e-9,
                rtol=1e-8
            )

            if ok:
                copy_state(w_n, w)
                accepted_conv_scale = target_conv_scale
                print(
                    f"  accepted stage at lambda={lam:.2f}, conv_scale={target_conv_scale:.6f}, "
                    f"relaxation={used_relax:.3f}"
                )
                i += 1
                continue

            print(
                f"  steady Newton stalled between conv_scale={accepted_conv_scale:.6f} "
                f"and {target_conv_scale:.6f} at lambda={lam:.2f}."
            )

            inserted = False
            depth = 0
            left = accepted_conv_scale
            right = target_conv_scale
            while depth < max_bisection_depth and (right - left) > 1e-2:
                mid = 0.5 * (left + right)
                print(f"  trying adaptive midpoint conv_scale={mid:.6f}")

                ok_mid, used_relax_mid, err_mid = try_newton_stage_FJF_outside(
                    F=F, JF =JF, w=w, w_n=stage_seed,
                    boundary_conditions=boundary_conditions,
                    relaxation_schedule=relaxation_schedule,
                    maxit=20,
                    atol=1e-9,
                    rtol=1e-8
                )

                if ok_mid:
                    copy_state(w_n, w)
                    accepted_conv_scale = mid
                    stage_grid.insert(i, mid)
                    print(
                        f"  inserted successful midpoint conv_scale={mid:.6f} "
                        f"with relaxation={used_relax_mid:.3f}"
                    )
                    inserted = True
                    break

                right = mid
                depth += 1

            if inserted:
                continue

            print("  launching pseudo-transient rescue...")
            rescued = pseudo_transient_rescue(
                experiment=experiment,
                W=W,
                w=w,
                w_n=w_n,
                psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
                mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
                sub_dx=sub_dx, sub_ds=sub_ds, sub_ft=sub_ft, qn_air=qn_air,
                lam=lam,
                conv_scale=accepted_conv_scale,
                dtau_schedule=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
                steps_per_dtau=6,
                update_tol=1e-9,
                retry_newton_every=3,
                relaxation=1.0,
            )

            if not rescued:
                raise RuntimeError(
                    f"Pseudo-transient rescue failed at lambda={lam:.2f} near conv_scale={accepted_conv_scale:.6f}."
                )

            rescue_seed = fenics.Function(W)
            copy_state(rescue_seed, w_n)
            ok_retry, used_relax_retry, err_retry = try_newton_stage_FJF_outside(
                F=F, JF =JF, w=w, w_n=stage_seed,
                boundary_conditions=boundary_conditions,
                relaxation_schedule=relaxation_schedule,
                maxit=20,
                atol=1e-9,
                rtol=1e-8
            )

            if ok_retry:
                copy_state(w_n, w)
                accepted_conv_scale = target_conv_scale
                print(
                    f"  stage recovered after PTC at lambda={lam:.2f}, "
                    f"conv_scale={target_conv_scale:.6f}, relaxation={used_relax_retry:.3f}"
                )
                i += 1
                continue

            raise RuntimeError(
                f"Hybrid continuation failed at lambda={lam:.2f}, conv_scale={target_conv_scale:.6f}. "
                f"Last steady Newton error: {err_retry if err_retry is not None else err}"
            )
        
    return w

from utils.imports import *
from solver.solver import *
from solver.params_bcs import *


def _vector_relative_update(w_new: fenics.Function, w_old: fenics.Function) -> float:
    dw = w_new.vector().copy()
    dw.axpy(-1.0, w_old.vector())
    return dw.norm("l2") / (w_new.vector().norm("l2") + 1e-14)


def _steady_residual_norm(
    W,
    w,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b,
    sub_dx, sub_ds, qn_air,
    boundary_conditions,
    buoyancy_scale=1.0,
    qn_scale=1.0,
    include_convection=True,
    convection_scale=1.0,
) -> float:
    F_steady, _ = build_nonlinear_problem(
        W=W, w=w,
        psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
        mu=mu, Pr=Pr, f_b=f_b,
        sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
        buoyancy_scale=buoyancy_scale,
        qn_scale=qn_scale,
        include_convection=include_convection,
        convection_scale=convection_scale,
    )

    r = fenics.assemble(F_steady)
    for bc in boundary_conditions:
        bc.apply(r)

    return r.norm("l2")


def _collect_observables(w: fenics.Function) -> dict:
    p_f, u_f, T_f = w.split(deepcopy=True)
    u_vec = u_f.vector().get_local()
    T_vec = T_f.vector().get_local()
    p_vec = p_f.vector().get_local()

    return {
        "u_l2": u_f.vector().norm("l2"),
        "T_l2": T_f.vector().norm("l2"),
        "p_l2": p_f.vector().norm("l2"),
        "u_max_abs": float(np.max(np.abs(u_vec))) if u_vec.size else 0.0,
        "T_max": float(np.max(T_vec)) if T_vec.size else 0.0,
        "T_min": float(np.min(T_vec)) if T_vec.size else 0.0,
        "p_max_abs": float(np.max(np.abs(p_vec))) if p_vec.size else 0.0,
    }


def _history_window_increasing(values, window=5):
    if len(values) < window:
        return False
    tail = values[-window:]
    return all(tail[i] > tail[i - 1] for i in range(1, len(tail)))


def _history_window_nondecreasing(values, window=5):
    if len(values) < window:
        return False
    tail = values[-window:]
    return all(tail[i] >= tail[i - 1] for i in range(1, len(tail)))

def _assembled_residual_norm(F_form, boundary_conditions):
    r = fenics.assemble(F_form)
    for bc in boundary_conditions:
        bc.apply(r)
    return r.norm("l2")

def _safe_eval_scalar(f, x, y):
    try:
        val = f(x, y)
        if isinstance(val, (tuple, list)):
            return float(val[0])
        return float(val)
    except Exception:
        return float("nan")


def _safe_eval_vector_component(f, x, y, comp=1):
    try:
        val = f(x, y)
        return float(val[comp])
    except Exception:
        return float("nan")


def _collect_ptc_probe_diagnostics(w, sub_dx, probe_ys, x_probe=0.0):
    """
    Collect a compact set of scalar diagnostics for PTC monitoring.
    """
    p_f, u_f, T_f = w.split(deepcopy=True)

    u_l2 = float(u_f.vector().norm("l2"))
    T_l2 = float(T_f.vector().norm("l2"))
    kinetic_energy = float(fenics.assemble(0.5 * fenics.inner(u_f, u_f) * sub_dx))

    data = {
        "u_l2": u_l2,
        "T_l2": T_l2,
        "kinetic_energy": kinetic_energy,
    }

    for y in probe_ys:
        tag = str(y).replace(".", "p")
        data[f"uy_y{tag}"] = _safe_eval_vector_component(u_f, x_probe, y, comp=1)
        data[f"T_y{tag}"] = _safe_eval_scalar(T_f, x_probe, y)

    return data


def _init_ptc_csv(log_path, probe_ys):
    fieldnames = [
        "step",
        "pseudo_time",
        "dtau",
        "rel_update",
        "steady_residual",
        "u_l2",
        "T_l2",
        "kinetic_energy",
    ]

    for y in probe_ys:
        tag = str(y).replace(".", "p")
        fieldnames.append(f"uy_y{tag}")
    for y in probe_ys:
        tag = str(y).replace(".", "p")
        fieldnames.append(f"T_y{tag}")

    # os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

    return fieldnames


def _append_ptc_csv(log_path, fieldnames, row):
    with open(log_path, "a", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writerow(row)

def solve_pseudo_transient_continuation_problem(
    experiment: Experiment,
    W: fenics.FunctionSpace,
    w: fenics.Function,
    w_n: fenics.Function,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b, T_c, T_air_bc,
    sub_dx, sub_ds, sub_ft, qn_air,
    dtau_init=1e-6,
    dtau_min=1e-8,
    dtau_max=1e-4,
    max_steps=300,
    update_tol=1e-8,
    residual_tol=1e-8,
    warmup_steps=20,
    ptc_relaxation=1.0,
    ptc_max_newton_it=20,
    ptc_atol=1e-9,
    ptc_rtol=1e-8,
    steady_polish=True,
    steady_relaxation_schedule=(1.0, 0.7, 0.5),
    polish_maxit=20,
    growth_factor=1.20,
    shrink_factor=0.50,
    drift_window=5,
    residual_improve_factor=0.95,
    residual_worsen_factor=1.02,
    residual_check_every=5,
    log_every=5,
    probe_ys=(2.0, 5.0, 10.0, 15.0),
    x_probe=0.0,
):
    """
    Conservative pseudo-transient march for the full coupled target problem.

    Intended usage:
      - startup state w_n already comes from your linear Stokes / thermal continuation
      - this routine then marches the *full* nonlinear problem at lambda = 1
      - dtau adaptation is based mainly on the steady-residual trend
    """
    scales = compute_nondimensional_scales(experiment)
    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

    # Accepted state is always stored in w_n
    copy_state(w, w_n)

    w_prev = fenics.Function(W)
    accepted_steps = 0
    rejected_steps = 0
    dtau = float(dtau_init)

    history = []
    rel_hist = []
    res_hist = []
    T_hist = []

    info = {
        "status": "not_converged",
        "n_steps": 0,
        "accepted_steps": 0,
        "rejected_steps": 0,
        "final_dtau": dtau,
        "final_rel_update": None,
        "final_steady_residual": None,
        "steady_polished": False,
        "history": history,
    }

    pseudo_time = 0.0

    log_path = "ptc_history.csv"
    csv_fieldnames = _init_ptc_csv(log_path, probe_ys)

    prev_steady_res = None
    prev_rel_update = None

    dtau = float(dtau_init)
    dtau_c = fenics.Constant(dtau)

    buoyancy_scale_c = fenics.Constant(1.0)
    qn_scale_c = fenics.Constant(1.0)
    convection_scale_c = fenics.Constant(1.0)

    copy_state(w, w_n)
    copy_state(w_prev, w_n)

    F_ptc, JF_ptc = build_ptc_problem(
        W=W, w=w, w_prev=w_prev,
        psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
        mu=mu, Pr=Pr, f_b=f_b,
        sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
        dtau=dtau_c,
        buoyancy_scale=buoyancy_scale_c,
        qn_scale=qn_scale_c,
        include_convection=True,
        convection_scale=convection_scale_c,
    )

    F_steady, JF_steady = build_nonlinear_problem(
        W=W, w=w,
        psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
        mu=mu, Pr=Pr, f_b=f_b,
        sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
        buoyancy_scale=buoyancy_scale_c,
        qn_scale=qn_scale_c,
        include_convection=True,
        convection_scale=convection_scale_c,
    )

    for step in range(1, max_steps + 1):
        copy_state(w_prev, w_n)
        copy_state(w, w_n)

        print(f"\n=== PTC step {step:04d} | dtau={dtau:.3e} ===")


        try:
            base_solver(
                F_ptc, w, boundary_conditions, JF_ptc,
                relaxation=ptc_relaxation,
                maxit=ptc_max_newton_it,
                atol=ptc_atol,
                rtol=ptc_rtol,
            )
        except RuntimeError as err:
            rejected_steps += 1
            dtau = max(dtau_min, shrink_factor * dtau)
            dtau_c.assign(dtau)
            print(f"PTC Newton failed. Shrinking dtau -> {dtau:.3e}")
            print(f"Failure reason: {err}")

            if dtau <= dtau_min * (1.0 + 1e-12):
                info["status"] = "failed_dtau_min"
                info["n_steps"] = step
                info["accepted_steps"] = accepted_steps
                info["rejected_steps"] = rejected_steps
                info["final_dtau"] = dtau
                return w_n, info

            continue

        rel_update = _vector_relative_update(w, w_prev)
        # steady_res = _steady_residual_norm(
        #     W=W, w=w,
        #     psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
        #     mu=mu, Pr=Pr, f_b=f_b,
        #     sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
        #     boundary_conditions=boundary_conditions,
        #     buoyancy_scale=1.0,
        #     qn_scale=1.0,
        #     include_convection=True,
        #     convection_scale=1.0,
        # )
        steady_res = _assembled_residual_norm(F_steady, boundary_conditions)
        obs = _collect_observables(w)

        history.append({
            "step": step,
            "dtau": dtau,
            "rel_update": rel_update,
            "steady_residual": steady_res,
            **obs,
        })
        rel_hist.append(rel_update)
        res_hist.append(steady_res)
        T_hist.append(obs["T_l2"])

        print(
            f"Accepted PTC step {step:04d}: "
            f"rel_update={rel_update:.3e}, "
            f"steady_residual={steady_res:.3e}, "
            f"||u||={obs['u_l2']:.3e}, "
            f"||T||={obs['T_l2']:.3e}"
        )

        # Accept step
        copy_state(w_n, w)
        accepted_steps += 1
        pseudo_time += dtau
        diag = _collect_ptc_probe_diagnostics(
                w=w,
                sub_dx=sub_dx,
                probe_ys=probe_ys,
                x_probe=x_probe,
            )

        row = {
            "step": step,
            "pseudo_time": pseudo_time,
            "dtau": dtau,
            "rel_update": rel_update,
            "steady_residual": steady_res,
            **diag,
        }
        _append_ptc_csv(log_path, csv_fieldnames, row)

        # Main steady-state criterion
        if rel_update < update_tol and steady_res < residual_tol:
            print("PTC reached steady-state tolerances.")

            if steady_polish:
                F_steady, JF_steady = build_nonlinear_problem(
                    W=W, w=w,
                    psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
                    mu=mu, Pr=Pr, f_b=f_b,
                    sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
                    buoyancy_scale=1.0,
                    qn_scale=1.0,
                    include_convection=True,
                    convection_scale=1.0,
                )

                ok, used_relax, last_error = try_newton_stage_FJF_outside(
                    F=F_steady,
                    JF=JF_steady,
                    w=w,
                    w_n=w_n,
                    boundary_conditions=boundary_conditions,
                    relaxation_schedule=steady_relaxation_schedule,
                    maxit=polish_maxit,
                    atol=1e-10,
                    rtol=1e-9,
                    stage_name="final_steady_polish",
                )

                if ok:
                    copy_state(w_n, w)
                    info["steady_polished"] = True
                    print(f"Final steady polish succeeded with relaxation={used_relax:.3f}")
                else:
                    print(f"Final steady polish failed: {last_error}")

            info["status"] = "steady"
            info["n_steps"] = step
            info["accepted_steps"] = accepted_steps
            info["rejected_steps"] = rejected_steps
            info["final_dtau"] = dtau
            info["final_rel_update"] = rel_update
            info["final_steady_residual"] = steady_res
            copy_state(w, w_n)
            return w, info

        # Drift / non-steady detector
        if step >= max(warmup_steps, drift_window + 2):
            initial_res = res_hist[0]
            no_real_residual_drop = steady_res > 0.95 * initial_res
            rel_is_rising = _history_window_increasing(rel_hist, window=drift_window)
            T_is_rising = _history_window_nondecreasing(T_hist, window=drift_window)

            if no_real_residual_drop and rel_is_rising and T_is_rising:
                print("PTC appears to be drifting instead of relaxing to a steady state.")
                info["status"] = "drifting_or_not_steady"
                info["n_steps"] = step
                info["accepted_steps"] = accepted_steps
                info["rejected_steps"] = rejected_steps
                info["final_dtau"] = dtau
                info["final_rel_update"] = rel_update
                info["final_steady_residual"] = steady_res
                copy_state(w, w_n)
                return w, info

        # Conservative dtau controller
        if step < warmup_steps:
            # fixed dtau warmup
            pass
        else:
            if prev_steady_res is not None and prev_rel_update is not None:
                improved = (
                    steady_res < residual_improve_factor * prev_steady_res
                    and rel_update <= prev_rel_update
                )
                worsened = steady_res > residual_worsen_factor * prev_steady_res

                if worsened:
                    dtau = max(dtau_min, shrink_factor * dtau)
                    dtau_c.assign(dtau)
                    print(f"Steady residual worsened -> shrink dtau to {dtau:.3e}")
                elif improved:
                    dtau = min(dtau_max, growth_factor * dtau)
                    dtau_c.assign(dtau)
                    print(f"Steady residual improved -> grow dtau to {dtau:.3e}")
                else:
                    print("Keeping dtau unchanged.")
            else:
                print("Keeping dtau unchanged.")

        prev_steady_res = steady_res
        prev_rel_update = rel_update

        

    copy_state(w, w_n)

    info["status"] = "max_steps_reached"
    info["n_steps"] = max_steps
    info["accepted_steps"] = accepted_steps
    info["rejected_steps"] = rejected_steps
    info["final_dtau"] = dtau
    info["final_rel_update"] = rel_hist[-1] if rel_hist else None
    info["final_steady_residual"] = res_hist[-1] if res_hist else None

    return w, info


def solve_ptc_stage(
    experiment: Experiment,
    W: fenics.FunctionSpace,
    w: fenics.Function,
    w_n: fenics.Function,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b, T_c, T_air_bc,
    sub_dx, sub_ds, sub_ft, qn_air,
    buoyancy_scale=1.0,
    qn_scale=1.0,
    convection_scale=1.0,
    dtau_init=1e-5,
    dtau_min=1e-8,
    dtau_max=1e-4,
    max_steps=100,
    update_tol=1e-8,
    residual_tol=1e-8,
    warmup_steps=10,
    ptc_relaxation=1.0,
    ptc_max_newton_it=20,
    ptc_atol=1e-9,
    ptc_rtol=1e-8,
    steady_polish=False,
    steady_relaxation_schedule=(1.0, 0.7, 0.5),
    polish_maxit=20,
    growth_factor=1.20,
    shrink_factor=0.50,
    drift_window=5,
    residual_improve_factor=0.95,
    residual_worsen_factor=1.02,
    residual_check_every=5,
    log_every=5,
    probe_ys=(2.0, 5.0, 10.0, 15.0),
    x_probe=0.0,
    stage_name="ptc_stage",
    strict_steady=True,
):
    """
    Solve one pseudo-transient continuation stage with fixed
    buoyancy/heat/convection scales.

    strict_steady=True:
        require both update and steady residual tolerances

    strict_steady=False:
        allow a looser "stage_relaxed" exit to be used as an initial
        condition for the next continuation stage
    """
    scales = compute_nondimensional_scales(experiment)
    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

    # Accepted state lives in w_n
    copy_state(w, w_n)

    w_prev = fenics.Function(W)
    accepted_steps = 0
    rejected_steps = 0

    dtau = float(dtau_init)
    pseudo_time = 0.0

    history = []
    rel_hist = []
    res_hist = []
    T_hist = []

    info = {
        "status": "not_converged",
        "stage_name": stage_name,
        "n_steps": 0,
        "accepted_steps": 0,
        "rejected_steps": 0,
        "final_dtau": dtau,
        "final_rel_update": None,
        "final_steady_residual": None,
        "steady_polished": False,
        "history": history,
        "buoyancy_scale": float(buoyancy_scale),
        "qn_scale": float(qn_scale),
        "convection_scale": float(convection_scale),
    }

    prev_steady_res = None
    prev_rel_update = None

    dtau_c = fenics.Constant(dtau)
    buoyancy_scale_c = fenics.Constant(float(buoyancy_scale))
    qn_scale_c = fenics.Constant(float(qn_scale))
    convection_scale_c = fenics.Constant(float(convection_scale))

    copy_state(w_prev, w_n)

    # Build reusable forms once
    F_ptc, JF_ptc = build_ptc_problem(
        W=W, w=w, w_prev=w_prev,
        psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
        mu=mu, Pr=Pr, f_b=f_b,
        sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
        dtau=dtau_c,
        buoyancy_scale=buoyancy_scale_c,
        qn_scale=qn_scale_c,
        include_convection=True,
        convection_scale=convection_scale_c,
    )

    F_steady, JF_steady = build_nonlinear_problem(
        W=W, w=w,
        psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
        mu=mu, Pr=Pr, f_b=f_b,
        sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
        buoyancy_scale=buoyancy_scale_c,
        qn_scale=qn_scale_c,
        include_convection=True,
        convection_scale=convection_scale_c,
    )

    safe_stage_name = stage_name.replace(" ", "_").replace("/", "_")
    log_path = os.path.join(
        experiment.name,
        "time_step",
        "base",
        f"ptc_history_{safe_stage_name}.csv",
    )
    csv_fieldnames = _init_ptc_csv(log_path, probe_ys)

    print("\n" + "-" * 72)
    print(f"Starting PTC stage: {stage_name}")
    print(f"  buoyancy_scale   = {float(buoyancy_scale):.3f}")
    print(f"  qn_scale         = {float(qn_scale):.3f}")
    print(f"  convection_scale = {float(convection_scale):.3f}")
    print(f"  strict_steady    = {strict_steady}")
    print("-" * 72)

    for step in range(1, max_steps + 1):
        copy_state(w_prev, w_n)
        copy_state(w, w_n)

        print(f"\n=== {stage_name} | step {step:04d} | dtau={dtau:.3e} ===")

        try:
            base_solver(
                F_ptc, w, boundary_conditions, JF_ptc,
                relaxation=ptc_relaxation,
                maxit=ptc_max_newton_it,
                atol=ptc_atol,
                rtol=ptc_rtol,
            )
        except RuntimeError as err:
            rejected_steps += 1
            dtau = max(dtau_min, shrink_factor * dtau)
            dtau_c.assign(dtau)

            print(f"PTC Newton failed. Shrinking dtau -> {dtau:.3e}")
            print(f"Failure reason: {err}")

            if dtau <= dtau_min * (1.0 + 1e-12):
                info["status"] = "failed_dtau_min"
                info["n_steps"] = step
                info["accepted_steps"] = accepted_steps
                info["rejected_steps"] = rejected_steps
                info["final_dtau"] = dtau
                return w_n, info

            continue

        rel_update = _vector_relative_update(w, w_prev)

        if step == 1 or step % residual_check_every == 0:
            steady_res = _assembled_residual_norm(F_steady, boundary_conditions)
        else:
            steady_res = prev_steady_res if prev_steady_res is not None else float("nan")

        if step == 1 or step % log_every == 0:
            diag = _collect_ptc_probe_diagnostics(
                w=w,
                sub_dx=sub_dx,
                probe_ys=probe_ys,
                x_probe=x_probe,
            )
        else:
            diag = {
                "u_l2": float("nan"),
                "T_l2": float("nan"),
                "kinetic_energy": float("nan"),
            }
            for y in probe_ys:
                tag = str(y).replace(".", "p")
                diag[f"uy_y{tag}"] = float("nan")
                diag[f"T_y{tag}"] = float("nan")

        history.append({
            "step": step,
            "dtau": dtau,
            "rel_update": rel_update,
            "steady_residual": steady_res,
            **diag,
        })
        rel_hist.append(rel_update)
        res_hist.append(steady_res if np.isfinite(steady_res) else np.nan)
        T_hist.append(diag["T_l2"] if np.isfinite(diag["T_l2"]) else np.nan)

        print(
            f"Accepted {stage_name} step {step:04d}: "
            f"rel_update={rel_update:.3e}, "
            f"steady_residual={steady_res:.3e}, "
            f"||u||={diag['u_l2']:.3e}, "
            f"||T||={diag['T_l2']:.3e}"
        )

        # accept step
        copy_state(w_n, w)
        accepted_steps += 1
        pseudo_time += dtau

        if step == 1 or step % log_every == 0:
            row = {
                "step": step,
                "pseudo_time": pseudo_time,
                "dtau": dtau,
                "rel_update": rel_update,
                "steady_residual": steady_res,
                **diag,
            }
            _append_ptc_csv(log_path, csv_fieldnames, row)

        # stage acceptance
        if strict_steady:
            stage_converged = (
                rel_update < update_tol and
                np.isfinite(steady_res) and
                steady_res < residual_tol
            )
        else:
            # looser acceptance for intermediate continuation stages
            stage_converged = (
                rel_update < max(update_tol, 1e-4)
            )

        if stage_converged:
            print(f"{stage_name}: stage convergence criterion satisfied.")

            if strict_steady and steady_polish:
                ok, used_relax, last_error = try_newton_stage_FJF_outside(
                    F=F_steady,
                    JF=JF_steady,
                    w=w,
                    w_n=w_n,
                    boundary_conditions=boundary_conditions,
                    relaxation_schedule=steady_relaxation_schedule,
                    maxit=polish_maxit,
                    atol=1e-10,
                    rtol=1e-9,
                    stage_name=f"{stage_name}_final_steady_polish",
                )

                if ok:
                    copy_state(w_n, w)
                    info["steady_polished"] = True
                    print(f"Final steady polish succeeded with relaxation={used_relax:.3f}")
                else:
                    print(f"Final steady polish failed: {last_error}")

            info["status"] = "steady" if strict_steady else "stage_relaxed"
            info["n_steps"] = step
            info["accepted_steps"] = accepted_steps
            info["rejected_steps"] = rejected_steps
            info["final_dtau"] = dtau
            info["final_rel_update"] = rel_update
            info["final_steady_residual"] = steady_res
            copy_state(w, w_n)
            return w, info

        # drift detector
        if step >= max(warmup_steps, drift_window + 2):
            finite_res = [v for v in res_hist if np.isfinite(v)]
            finite_T = [v for v in T_hist if np.isfinite(v)]

            if len(finite_res) >= 1 and len(finite_T) >= drift_window:
                initial_res = finite_res[0]
                no_real_residual_drop = steady_res > 0.98 * initial_res
                rel_is_rising = _history_window_increasing(rel_hist, window=drift_window)
                T_is_rising = _history_window_nondecreasing(finite_T, window=drift_window)

                if no_real_residual_drop and rel_is_rising and T_is_rising:
                    print(f"{stage_name}: drifting instead of relaxing to steady state.")
                    info["status"] = "drifting_or_not_steady"
                    info["n_steps"] = step
                    info["accepted_steps"] = accepted_steps
                    info["rejected_steps"] = rejected_steps
                    info["final_dtau"] = dtau
                    info["final_rel_update"] = rel_update
                    info["final_steady_residual"] = steady_res
                    copy_state(w, w_n)
                    return w, info

        # conservative dtau controller
        if step < warmup_steps:
            pass
        else:
            if prev_steady_res is not None and prev_rel_update is not None and np.isfinite(steady_res):
                improved = (
                    steady_res < residual_improve_factor * prev_steady_res and
                    rel_update <= prev_rel_update
                )
                worsened = steady_res > residual_worsen_factor * prev_steady_res

                if worsened:
                    dtau = max(dtau_min, shrink_factor * dtau)
                    dtau_c.assign(dtau)
                    print(f"Steady residual worsened -> shrink dtau to {dtau:.3e}")
                elif improved:
                    dtau = min(dtau_max, growth_factor * dtau)
                    dtau_c.assign(dtau)
                    print(f"Steady residual improved -> grow dtau to {dtau:.3e}")
                else:
                    print("Keeping dtau unchanged.")
            else:
                print("Keeping dtau unchanged.")

        if np.isfinite(steady_res):
            prev_steady_res = steady_res
        prev_rel_update = rel_update

    copy_state(w, w_n)

    info["status"] = "max_steps_reached"
    info["n_steps"] = max_steps
    info["accepted_steps"] = accepted_steps
    info["rejected_steps"] = rejected_steps
    info["final_dtau"] = dtau
    info["final_rel_update"] = rel_hist[-1] if rel_hist else None
    finite_res = [v for v in res_hist if np.isfinite(v)]
    info["final_steady_residual"] = finite_res[-1] if finite_res else None

    return w, info

def solve_ptc_continuation(
    experiment: Experiment,
    W: fenics.FunctionSpace,
    w: fenics.Function,
    w_n: fenics.Function,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b, T_c, T_air_bc,
    sub_dx, sub_ds, sub_ft, qn_air,
    stages=None,
    save_obj = None,
    dtau_init=1e-5,
    dtau_min=1e-8,
    dtau_max=1e-4,
    stage_max_steps=40,
    final_stage_max_steps=150,
    update_tol=1e-8,
    residual_tol=1e-8,
):
    """
    Continuation in (lambda, convection_scale), using PTC as the inner stage solver.
    """
    if stages is None:
        stages = [
            {"name": "L0.10_C0.00", "lambda": 0.10, "conv": 0.00, "strict": False},
            {"name": "L0.30_C0.00", "lambda": 0.30, "conv": 0.00, "strict": False},
            {"name": "L0.50_C0.10", "lambda": 0.50, "conv": 0.10, "strict": False},
            {"name": "L0.70_C0.20", "lambda": 0.70, "conv": 0.20, "strict": False},
            {"name": "L1.00_C0.30", "lambda": 1.00, "conv": 0.30, "strict": False},
            {"name": "L1.00_C0.50", "lambda": 1.00, "conv": 0.50, "strict": False},
            {"name": "L1.00_C0.70", "lambda": 1.00, "conv": 0.70, "strict": False},
            {"name": "L1.00_C0.85", "lambda": 1.00, "conv": 0.85, "strict": False},
            {"name": "L1.00_C0.90", "lambda": 1.00, "conv": 0.90, "strict": False},
            {"name": "L1.00_C0.95", "lambda": 1.00, "conv": 0.95, "strict": False},
            {"name": "L1.00_C1.00", "lambda": 1.00, "conv": 1.00, "strict": True},
        ]

    continuation_history = []

    for k, stage in enumerate(stages, start=1):
        stage_name = stage["name"]
        lam = float(stage["lambda"])
        conv = float(stage["conv"])
        strict = bool(stage.get("strict", False))

        print("\n" + "=" * 72)
        print(f"PTC continuation stage {k}/{len(stages)}: {stage_name}")
        print(f"  lambda            = {lam:.3f}")
        print(f"  convection_scale  = {conv:.3f}")
        print(f"  strict_steady     = {strict}")
        print("=" * 72)

        stage_steps = final_stage_max_steps if strict else stage_max_steps

        w, stage_info = solve_ptc_stage(
            experiment=experiment,
            W=W,
            w=w,
            w_n=w_n,
            psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
            mu=mu, Pr=Pr, f_b=f_b, T_c=T_c, T_air_bc=T_air_bc,
            sub_dx=sub_dx, sub_ds=sub_ds, sub_ft=sub_ft, qn_air=qn_air,
            buoyancy_scale=lam,
            qn_scale=lam,
            convection_scale=conv,
            dtau_init=dtau_init,
            dtau_min=dtau_min,
            dtau_max=dtau_max,
            max_steps=stage_steps,
            update_tol=update_tol,
            residual_tol=residual_tol,
            warmup_steps=10,
            residual_check_every=5,
            log_every=5,
            stage_name=stage_name,
            strict_steady=strict,
            steady_polish=strict,
            ptc_atol=5e-7,
            ptc_rtol=5e-7,
            ptc_max_newton_it=20,
        )

        continuation_history.append({
            "stage": stage_name,
            "lambda": lam,
            "conv": conv,
            "status": stage_info["status"],
            "final_rel_update": stage_info.get("final_rel_update"),
            "final_steady_residual": stage_info.get("final_steady_residual"),
            "accepted_steps": stage_info.get("accepted_steps"),
        })

        print(
            f"Stage {stage_name} finished with status={stage_info['status']}, "
            f"rel_update={stage_info.get('final_rel_update')}, "
            f"steady_residual={stage_info.get('final_steady_residual')}"
        )

        if strict:
            ok = (stage_info["status"] == "steady")
        else:
            ok = stage_info["status"] in ("steady", "stage_relaxed")

        if not ok:
            return w, {
                "status": "continuation_failed",
                "failed_stage": stage_name,
                "history": continuation_history,
                "last_stage_info": stage_info,
            }

        copy_state(w_n, w)

        # save state
        if save_obj is not None:
            p_star, u_star, theta = w_n.split(deepcopy=True)

            u_dim, p_dim, T_dim = dimensionalize_fields(
                save_obj[1], u_star, p_star, theta,
                save_obj[3].Uref, save_obj[3].dTref, T_ambient,
                experiment.fluid.properties["rho"]
            )

            p_out = save_obj[4].split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}" + f"_conv_{int(conv*100):03d}.xdmf"
            u_out = save_obj[5].split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}" + f"_conv_{int(conv*100):03d}.xdmf"
            t_out = save_obj[6].split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}" + f"_conv_{int(conv*100):03d}.xdmf"

            save_experiment(p_out, save_obj[1], [p_dim])
            save_experiment(u_out, save_obj[1], [u_dim])
            save_experiment(t_out, save_obj[1], [T_dim])

    return w, {
        "status": "continuation_complete",
        "history": continuation_history,
    }
