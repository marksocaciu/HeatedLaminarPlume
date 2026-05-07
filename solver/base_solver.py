from re import S
import json

from matplotlib.pyplot import step

from utils.imports import *
from solver.solver import *
from solver.params_bcs import *
from solver.scales import *
from solver.biot import *
from utils.results import *


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
        print0(f"    -> PTC block with dtau={dtau:.3e}")
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
                experiment=experiment,
                scales=scales,
                outlet_penalty=1.0e-3,
                backflow_beta=5.0e-1,
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
                print0(f"    PTC step failed at dtau={dtau:.3e}: {err}")
                return False

            delta = w.vector().copy()
            delta.axpy(-1.0, w_prev.vector())
            rel_update = delta.norm("l2") / (w.vector().norm("l2") + 1e-14)
            print0(f"    PTC step {ptc_step:03d}: rel_update={rel_update:.3e}")

            copy_state(w_n, w)

            if rel_update < update_tol:
                print0("    PTC update is small; handing control back to steady Newton.")
                return True

            if (ptc_step % retry_newton_every) == 0:
                print0("    PTC block produced a new seed; retry steady Newton now.")
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
    SUPG=False,
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
        print0(f"\n=== Newton continuation lambda = {lam:.2f} ===")

        # optional output of current accepted state before advancing lambda
        if sub_mesh_star is not None and sub_mesh_dim is not None:
            p_star, u_star, theta = w_n.split(deepcopy=True)

            u_dim, p_dim, T_dim = dimensionalize_fields(
                sub_mesh_star, u_star, p_star, theta,
                scales.Uplume, scales.dTref, T_ambient,
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
        print0("  --- stage: stokes ---")

        F_stokes, JF_stokes = build_nonlinear_problem(
            W=W, w=w,
            psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
            mu=mu, Pr=Pr, f_b=f_b,
            sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
            buoyancy_scale=lam,
            qn_scale=lam,
            include_convection=False,
            convection_scale=0.0,
            SUPG=SUPG,
            # experiment=experiment,
            # scales=scales,
            outlet_penalty=1.0e-3,
            backflow_beta=5.0e-1,
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
        print0(f"  stokes accepted at lambda={lam:.2f} (relaxation={used_relax:.3f})")

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

        print0(f"  lambda={lam:.2f} completed with accepted conv_scale={accepted_conv:.4f}")

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
    SUPG=False,
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
        print0(f"\n=== Newton continuation lambda = {lam:.2f} ===")
        
        if sub_mesh_star is not None and sub_mesh_dim is not None and p_path and u_path and T_path:
            p_star, u_star, theta = w.split(deepcopy=True)
            u_dim, p_dim, T_dim = dimensionalize_fields(
                sub_mesh_star, u_star, p_star, theta,
                scales.Uplume, scales.dTref, T_ambient,
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
                print0("  --- stage: Stokes / zero-convection stage ---")
            else:
                print0(f"  --- stage: convection_scale = {target_conv_scale:.6f} ---")

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
                SUPG=SUPG,
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
                print0(
                    f"  accepted stage at lambda={lam:.2f}, conv_scale={target_conv_scale:.6f}, "
                    f"relaxation={used_relax:.3f}"
                )
                i += 1
                continue

            print0(
                f"  steady Newton stalled between conv_scale={accepted_conv_scale:.6f} "
                f"and {target_conv_scale:.6f} at lambda={lam:.2f}."
            )

            inserted = False
            depth = 0
            left = accepted_conv_scale
            right = target_conv_scale
            while depth < max_bisection_depth and (right - left) > 1e-2:
                mid = 0.5 * (left + right)
                print0(f"  trying adaptive midpoint conv_scale={mid:.6f}")

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
                    print0(
                        f"  inserted successful midpoint conv_scale={mid:.6f} "
                        f"with relaxation={used_relax_mid:.3f}"
                    )
                    inserted = True
                    break

                right = mid
                depth += 1

            if inserted:
                continue

            print0("  launching pseudo-transient rescue...")
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
                print0(
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
    SUPG=False,
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
    csv_fieldnames = init_ptc_csv(log_path, probe_ys)

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
        # experiment=experiment,
        # scales=scales,
        outlet_penalty=1.0e-3,
        backflow_beta=5.0e-1,
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
        SUPG=SUPG,
        # experiment=experiment,
        # scales=scales,
        outlet_penalty=1.0e-3,
        backflow_beta=5.0e-1,
    )

    for step in range(1, max_steps + 1):
        copy_state(w_prev, w_n)
        copy_state(w, w_n)

        print0(f"\n=== PTC step {step:04d} | dtau={dtau:.3e} ===")


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
            print0(f"PTC Newton failed. Shrinking dtau -> {dtau:.3e}")
            print0(f"Failure reason: {err}")

            if dtau <= dtau_min * (1.0 + 1e-12):
                info["status"] = "failed_dtau_min"
                info["n_steps"] = step
                info["accepted_steps"] = accepted_steps
                info["rejected_steps"] = rejected_steps
                info["final_dtau"] = dtau
                return w_n, info

            continue

        rel_update = vector_relative_update(w, w_prev)
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
        steady_res = assembled_residual_norm(F_steady, boundary_conditions)
        obs = collect_observables(w)

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

        print0(
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
        diag = collect_ptc_probe_diagnostics(
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
        append_ptc_csv(log_path, csv_fieldnames, row)

        # Main steady-state criterion
        if rel_update < update_tol and steady_res < residual_tol:
            print0("PTC reached steady-state tolerances.")

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
                    SUPG=SUPG,
                    # experiment=experiment,
                    # scales=scales,
                    outlet_penalty=1.0e-3,
                    backflow_beta=5.0e-1,
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
                    print0(f"Final steady polish succeeded with relaxation={used_relax:.3f}")
                else:
                    print0(f"Final steady polish failed: {last_error}")

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
            rel_is_rising = history_window_increasing(rel_hist, window=drift_window)
            T_is_rising = history_window_nondecreasing(T_hist, window=drift_window)

            if no_real_residual_drop and rel_is_rising and T_is_rising:
                print0("PTC appears to be drifting instead of relaxing to a steady state.")
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
                    print0(f"Steady residual worsened -> shrink dtau to {dtau:.3e}")
                elif improved:
                    dtau = min(dtau_max, growth_factor * dtau)
                    dtau_c.assign(dtau)
                    print0(f"Steady residual improved -> grow dtau to {dtau:.3e}")
                else:
                    print0("Keeping dtau unchanged.")
            else:
                print0("Keeping dtau unchanged.")

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

def _theta_to_dimensional_temperature(
    theta_src: fenics.Function,
    scales,
    T_ambient: float,
):
    """
    Convert nondimensional theta on the air submesh to dimensional temperature.

    T_dim = T_ambient + dTref * theta
    """
    T_dim = theta_src.copy(deepcopy=True)
    T_dim.rename("T_dim_material", "T_dim_material")
    T_dim.vector()[:] *= float(scales.dTref)

    ones = theta_src.copy(deepcopy=True)
    ones.vector()[:] = 1.0
    T_dim.vector().axpy(float(T_ambient), ones.vector())
    T_dim.vector().apply("insert")
    return T_dim

def _update_material_from_mixed_temperature(
    w_mixed: fenics.Function,
    fluid_material,
    scales,
    T_ambient: float,
):
    """
    Update temperature-dependent material fields from the current mixed state.

    The temperature component of the mixed state is nondimensional theta on the
    air submesh. The material model is updated with dimensional temperature.
    """
    theta = w_mixed.sub(2, deepcopy=True)
    T_dim = _theta_to_dimensional_temperature(theta, scales, T_ambient)
    fluid_material.update(T_dim)
    return theta, T_dim

def _material_outer_picard(
    w: fenics.Function,
    boundary_conditions,
    F,
    JF,
    fluid_material,
    scales,
    T_ambient: float,
    material_max_it: int = 8,
    material_rtol: float = 1.0e-6,
    newton_relaxation: float = 0.9,
    newton_maxit: int = 20,
    newton_atol: float = 1.0e-9,
    newton_rtol: float = 1.0e-8,
):
    """
    Frozen-coefficient outer iteration.

    Coefficient Functions inside F/JF are assumed to be mutable objects owned by
    ``fluid_material``. That means the forms do not need to be rebuilt, as long as
    the material model updates those Functions in place.
    """
    theta_old = w.sub(2, deepcopy=True)
    rel = float("inf")
    for it in range(int(material_max_it)):
        _update_material_from_mixed_temperature(w, fluid_material, scales, T_ambient)

        base_solver(
            F, w, boundary_conditions, JF,
            relaxation=newton_relaxation,
            maxit=newton_maxit,
            atol=newton_atol,
            rtol=newton_rtol,
        )

        theta_new = w.sub(2, deepcopy=True)
        diff = (theta_new.vector() - theta_old.vector()).norm("l2")
        norm = theta_old.vector().norm("l2") + 1.0e-14
        rel = diff / norm
        print0(f"[material loop {it}] rel ||ΔT|| = {rel:.3e}")

        if rel < float(material_rtol):
            return w, rel, it + 1

        theta_old.assign(theta_new)

    return w, rel, int(material_max_it)

def solve_ptc_stage(
    experiment: Experiment,
    run_root,
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
    residual_check_every=1,
    log_every=1,
    probe_ys=(2.0, 5.0, 10.0, 15.0),
    x_probe=0.0,
    stage_name="ptc_stage",
    strict_steady=False,
    material_max_it=8,
    material_rtol=1.0e-6,
    SUPG=False,
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
        SUPG=SUPG,
        # experiment=experiment,
        # scales=scales,
        outlet_penalty=1.0e-3,
        backflow_beta=5.0e-1,
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
        SUPG=SUPG,
        # experiment=experiment,
        # scales=scales,
        outlet_penalty=1.0e-3,
        backflow_beta=5.0e-1,
    )

    # safe_stage_name = stage_name.replace(" ", "_").replace("/", "_")
    # log_path = os.path.join(
    #     run_root,
    #     "base",
    #     f"ptc_history_{safe_stage_name}.csv",
    # )
    # csv_fieldnames = init_ptc_csv(log_path, probe_ys)

    print0("\n" + "-" * 72)
    print0(f"Starting PTC stage: {stage_name}")
    print0(f"  buoyancy_scale   = {float(buoyancy_scale):.3f}")
    print0(f"  qn_scale         = {float(qn_scale):.3f}")
    print0(f"  convection_scale = {float(convection_scale):.3f}")
    print0(f"  strict_steady    = {strict_steady}")
    print0("-" * 72)

    for step in range(1, max_steps + 1):
        copy_state(w_prev, w_n)
        copy_state(w, w_n)

        print0(f"\n=== {stage_name} | step {step:04d} | dtau={dtau:.3e} ===")

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

            print0(f"PTC Newton failed. Shrinking dtau -> {dtau:.3e}")
            print0(f"Failure reason: {err}")

            if dtau <= dtau_min * (1.0 + 1e-12):
                info["status"] = "failed_dtau_min"
                info["n_steps"] = step
                info["accepted_steps"] = accepted_steps
                info["rejected_steps"] = rejected_steps
                info["final_dtau"] = dtau
                return w_n, info

            continue

        rel_update = vector_relative_update(w, w_prev)

        if step == 1 or step % residual_check_every == 0:
            steady_res = assembled_residual_norm(F_steady, boundary_conditions)
        else:
            steady_res = prev_steady_res if prev_steady_res is not None else float("nan")

        if step == 1 or step % log_every == 0:
            diag = collect_ptc_probe_diagnostics(
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

        print0(
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
            # append_ptc_csv(log_path, csv_fieldnames, row)

        # stage acceptance
        # For intermediate continuation stages, do not accept merely because the
        # pseudo-time update is small. Also require either a meaningful drop in
        # the steady residual relative to the start of the stage, or an absolute
        # residual small enough to be a credible seed for the next stage.
        finite_res = [v for v in res_hist if np.isfinite(v)]
        initial_stage_res = finite_res[0] if finite_res else np.nan
        stage_residual_ratio = (
            steady_res / initial_stage_res
            if np.isfinite(steady_res) and np.isfinite(initial_stage_res) and initial_stage_res > 0.0
            else np.nan
        )

        if strict_steady:
            stage_converged = (
                rel_update < update_tol and
                np.isfinite(steady_res) and
                steady_res < residual_tol
            )
        else:
            intermediate_update_tol = max(update_tol, 1e-4)
            intermediate_abs_residual_tol = 0.5 + 5.0 * float(convection_scale)
            intermediate_required_drop = 0.95  # require at least 5% drop

            stage_converged = (
                rel_update < intermediate_update_tol and
                np.isfinite(steady_res) and
                (
                    steady_res < intermediate_abs_residual_tol or
                    (
                        np.isfinite(stage_residual_ratio) and
                        stage_residual_ratio < intermediate_required_drop
                    )
                )
            )

        if stage_converged:
            print0(f"{stage_name}: stage convergence criterion satisfied.")

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
                    print0(f"Final steady polish succeeded with relaxation={used_relax:.3f}")
                else:
                    print0(f"Final steady polish failed: {last_error}")

            info["status"] = "steady" if strict_steady else "stage_relaxed"
            info["n_steps"] = step
            info["accepted_steps"] = accepted_steps
            info["rejected_steps"] = rejected_steps
            info["final_dtau"] = dtau
            info["final_rel_update"] = rel_update
            info["final_steady_residual"] = steady_res
            copy_state(w, w_n)
            return w, info

        # Relaxed-stage plateau acceptor:
        # if updates are already tiny, the residual is modest for the current
        # continuation level, and the residual has flattened out across the most
        # recent accepted steps, accept this stage as a usable seed instead of
        # forcing it to march until max_steps.
        if not strict_steady and step >= max(warmup_steps, 6):
            finite_recent = [v for v in res_hist[-5:] if np.isfinite(v)]
            if len(finite_recent) >= 4 and np.isfinite(steady_res):
                rmin_recent = min(finite_recent)
                rmax_recent = max(finite_recent)
                plateau_ref = max(abs(initial_stage_res), 1.0) if np.isfinite(initial_stage_res) else 1.0
                residual_plateaued = (rmax_recent - rmin_recent) <= 0.02 * plateau_ref

                relaxed_abs_cap = max(
                    0.5 + 5.0 * float(convection_scale),
                    1.05 * plateau_ref + 1e-12,
                )

                if (
                    rel_update < max(update_tol, 1e-4) and
                    steady_res <= relaxed_abs_cap and
                    residual_plateaued
                ):
                    print0(f"{stage_name}: relaxed-stage residual plateau detected; accepting stage as usable seed.")
                    info["status"] = "stage_relaxed"
                    info["n_steps"] = step
                    info["accepted_steps"] = accepted_steps
                    info["rejected_steps"] = rejected_steps
                    info["final_dtau"] = dtau
                    info["final_rel_update"] = rel_update
                    info["final_steady_residual"] = steady_res
                    copy_state(w, w_n)
                    return w, info

        # drift / plateau detector
        if step >= max(warmup_steps, drift_window + 2):
            finite_res = [v for v in res_hist if np.isfinite(v)]
            finite_T = [v for v in T_hist if np.isfinite(v)]

            if len(finite_res) >= 1 and len(finite_T) >= drift_window:
                initial_res = finite_res[0]
                no_real_residual_drop = steady_res > 0.98 * initial_res
                rel_is_rising = history_window_increasing(rel_hist, window=drift_window)
                T_is_rising = history_window_nondecreasing(finite_T, window=drift_window)

                if no_real_residual_drop and rel_is_rising and T_is_rising:
                    print0(f"{stage_name}: drifting instead of relaxing to steady state.")
                    info["status"] = "drifting_or_not_steady"
                    info["n_steps"] = step
                    info["accepted_steps"] = accepted_steps
                    info["rejected_steps"] = rejected_steps
                    info["final_dtau"] = dtau
                    info["final_rel_update"] = rel_update
                    info["final_steady_residual"] = steady_res
                    copy_state(w, w_n)
                    return w, info

            # Stronger plateau detector for the final strict-steady stage:
            # if the steady residual stays nearly flat for a sustained window,
            # abort instead of marching indefinitely with tiny pseudo-time updates.
            if strict_steady and len(finite_res) >= 6 and np.isfinite(steady_res):
                recent_res = finite_res[-6:]
                rmin = min(recent_res)
                rmax = max(recent_res)
                plateau_band_abs = rmax - rmin
                plateau_ref = max(abs(steady_res), 1.0)
                if steady_res > 1.0 and plateau_band_abs <= 0.01 * plateau_ref:
                    print0(f"{stage_name}: steady residual plateau detected; accepting current state.")
                    info["status"] = "steady_residual_plateau_accept"
                    info["n_steps"] = step
                    info["accepted_steps"] = accepted_steps
                    info["rejected_steps"] = rejected_steps
                    info["final_dtau"] = dtau
                    info["final_rel_update"] = rel_update
                    info["final_steady_residual"] = steady_res
                    copy_state(w, w_n)
                    return w, info
        # dtau controller
        if step < warmup_steps:
            pass
        else:
            if prev_steady_res is not None and prev_rel_update is not None and np.isfinite(steady_res):
                improved = (
                    steady_res < residual_improve_factor * prev_steady_res and
                    rel_update <= prev_rel_update
                )
                worsened = steady_res > residual_worsen_factor * prev_steady_res
                plateauing = (
                    abs(steady_res - prev_steady_res) <= 0.01 * max(abs(prev_steady_res), 1.0)
                    and rel_update < prev_rel_update
                )

                if worsened:
                    dtau = max(dtau_min, shrink_factor * dtau)
                    dtau_c.assign(dtau)
                    print0(f"Steady residual worsened -> shrink dtau to {dtau:.3e}")
                elif improved:
                    dtau = min(dtau_max, growth_factor * dtau)
                    dtau_c.assign(dtau)
                    print0(f"Steady residual improved -> grow dtau to {dtau:.3e}")
                elif strict_steady and plateauing:
                    dtau = max(dtau_min, shrink_factor * dtau)
                    dtau_c.assign(dtau)
                    print0(f"Steady residual plateau with shrinking updates -> shrink dtau to {dtau:.3e}")
                else:
                    print0("Keeping dtau unchanged.")
            else:
                print0("Keeping dtau unchanged.")

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
    run_root = None,
    dtau_init=1e-5,
    dtau_min=1e-8,
    dtau_max=1e-4,
    stage_max_steps=40,
    final_stage_max_steps=150,
    update_tol=1e-8,
    residual_tol=1e-8,
    SUPG=False,
):
    """
    Continuation in (lambda, convection_scale), using PTC as the inner stage solver.
    """
    if stages is None:
        stages = [
            {"name": "L0.30_C0.05", "lambda": 0.10, "conv": 0.05, "strict": False},
            {"name": "L0.30_C0.10", "lambda": 0.30, "conv": 0.10, "strict": False},
            {"name": "L0.50_C0.10", "lambda": 0.50, "conv": 0.10, "strict": False},
            {"name": "L0.70_C0.20", "lambda": 0.70, "conv": 0.20, "strict": False},
            {"name": "L0.85_C0.20", "lambda": 0.85, "conv": 0.20, "strict": False},
            {"name": "L1.00_C0.20", "lambda": 1.00, "conv": 0.20, "strict": False},
            {"name": "L1.00_C0.25", "lambda": 1.00, "conv": 0.25, "strict": False},
            {"name": "L1.00_C0.30", "lambda": 1.00, "conv": 0.30, "strict": False},
            {"name": "L1.00_C0.35", "lambda": 1.00, "conv": 0.35, "strict": False},
            {"name": "L1.00_C0.40", "lambda": 1.00, "conv": 0.40, "strict": False},
            {"name": "L1.00_C0.45", "lambda": 1.00, "conv": 0.45, "strict": False},
            {"name": "L1.00_C0.50", "lambda": 1.00, "conv": 0.50, "strict": False},
            {"name": "L1.00_C0.70", "lambda": 1.00, "conv": 0.70, "strict": False},
            {"name": "L1.00_C0.80", "lambda": 1.00, "conv": 0.80, "strict": False},
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

        print0("\n" + "=" * 72)
        print0(f"PTC continuation stage {k}/{len(stages)}: {stage_name}")
        print0(f"  lambda            = {lam:.3f}")
        print0(f"  convection_scale  = {conv:.3f}")
        print0(f"  strict_steady     = {strict}")
        print0("=" * 72)

        stage_steps = final_stage_max_steps if strict else stage_max_steps

        w, stage_info = solve_ptc_stage(
            experiment=experiment,
            run_root=run_root,
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
            warmup_steps=5,
            residual_check_every=1,
            log_every=1,
            stage_name=stage_name,
            strict_steady=strict,
            steady_polish=strict,
            ptc_atol=1e-7,
            ptc_rtol=1e-7,
            ptc_max_newton_it=20,
            SUPG=SUPG,
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

        print0(
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
                save_obj[2].Uplume, save_obj[2].dTref, T_ambient,
                experiment.fluid.properties["rho"]
            )

            p_out = save_obj[3].split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}" + f"_conv_{int(conv*100):03d}.xdmf"
            u_out = save_obj[4].split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}" + f"_conv_{int(conv*100):03d}.xdmf"
            t_out = save_obj[5].split(".xdmf")[0] + f"_lambda_{int(lam*100):03d}" + f"_conv_{int(conv*100):03d}.xdmf"

            save_experiment(p_out, save_obj[1], [p_dim])
            save_experiment(u_out, save_obj[1], [u_dim])
            save_experiment(t_out, save_obj[1], [T_dim])

    return w, {
        "status": "continuation_complete",
        "history": continuation_history,
    }

def run_post_continuation_transient(
    experiment: Experiment,
    W: fenics.FunctionSpace,
    w: fenics.Function,
    w_n: fenics.Function,
    psi_p, psi_u, psi_T,
    mu, Pr, f_b, T_c, T_air_bc,
    sub_dx, sub_ds, sub_ft, qn_air,
    sub_mesh_star=None,
    sub_mesh_dim=None,
    sub_ft_dim=None,
    sub_ds_dim=None,
    scales=None,
    p_path: str = "",
    u_path: str = "",
    T_path: str = "",
    probe_heights_m=(0.01, 0.04, 0.08),
    dt_start: float = 1.0e-2,
    dt_growth: float = 1.2,
    dt_cut: float = 0.5,
    dt_hard_cut: float = 0.8,
    dt_min: float = 1.0e-5,
    dt_max: float = 1.0,
    t_end: float = 100.0,
    step_max: int = 200000,
    n_steps: int = 1000000,
    save_every: int = 50,
    relaxation: float = 1.0,
    max_newton_it: int = 20,
    max_retries_per_step: int = 8,
    atol: float = 1.0e-9,
    rtol: float = 1.0e-11,
    rel_update_easy: float = 1.0e-3,
    rel_update_hard: float = 5.0e-3,
    rel_update_reject: float = 2.0e-2,
    newton_easy_iters: int = 6,
    newton_hard_iters: int = 10,
    steady_window: int = 25,
    steady_rel_tol: float = 1.0e-5,
    steady_update_tol: float = 1.0e-6,
    diagnostic_every: int = 1,
    history_csv_path: str = "",
    SUPG=False,
    start_time: float = 0.0,
    start_step: int = 0,
    restart_recovered=False,
    restart_step:int = 0,
):
    """
    Long-time backward-Euler transient workflow with rollback, adaptive timestep
    control, diagnostics, and stopping criteria.

    The method starts from the accepted continuation state in ``w_n`` and advances the
    fully coupled target problem (lambda=1, convection_scale=1) in physical time.
    Steps are accepted only after a successful Newton solve and basic sanity checks.
    On failure the state is rolled back, ``dt`` is reduced, and the same physical time
    is retried.
    """
    if scales is None:
        scales = compute_nondimensional_scales(experiment)

    if n_steps is not None:
        step_max = int(n_steps)

    boundary_conditions = set_bcs(W, sub_ft, T_air_bc, T_c, experiment, scales)

    copy_state(w, w_n)
    w_prev = fenics.Function(W)
    w_last_accepted = fenics.Function(W)
    copy_state(w_last_accepted, w_n)

    dt = float(dt_start)
    t = float(start_time)
    step = int(start_step)  
    accepted_steps = 0
    rejected_steps = 0
    history = []
    status = "transient_complete"
    restart_settle_steps = 10
    restart_rel_update_reject = 5.0e-2
    restart_rel_u_reject = 2.0e-1
    restart_rel_theta_reject = 1.0e-1
    restart_u_abs_max = 1.0e3
    restart_theta_min = -1.0
    restart_theta_max = 20.0

    x_probe = max(1.0e-8, 2.0 * float(sub_mesh_star.hmin())) if sub_mesh_star is not None else 1.0e-8
    
    if sub_mesh_star is not None:
        for comp in w_n.split(deepcopy=False):
            try:
                comp.set_allow_extrapolation(True)
            except Exception:
                pass

    def _sample_probes(sol_w):
        probes = {}
        try:
            _, u_star, theta_star = sol_w.split(deepcopy=True)
            u_star.set_allow_extrapolation(True)
            theta_star.set_allow_extrapolation(True)

            for y_m in probe_heights_m:
                if sub_mesh_star is None or scales is None:
                    probes[f"uy_{y_m:.3f}m"] = float("nan")
                    probes[f"theta_{y_m:.3f}m"] = float("nan")
                    continue

                y_star = float(y_m) / float(scales.Lref)
                try:
                    u_val = safe_eval_vector_component(u_star, x_probe, y_star, comp=1)
                    theta_val = safe_eval_scalar(theta_star, x_probe, y_star)
                    probes[f"uy_{y_m:.3f}m"] = float(u_val)
                    probes[f"theta_{y_m:.3f}m"] = float(theta_val)
                except Exception:
                    probes[f"uy_{y_m:.3f}m"] = float("nan")
                    probes[f"theta_{y_m:.3f}m"] = float("nan")
        except Exception:
            for y_m in probe_heights_m:
                probes[f"uy_{y_m:.3f}m"] = float("nan")
                probes[f"theta_{y_m:.3f}m"] = float("nan")
        return probes

    def _safe_float_norm(vec):
        try:
            val = float(vec.norm("l2"))
        except Exception:
            return float("nan")
        return val if np.isfinite(val) else float("nan")

    def _relative_function_update(new_f, old_f):
        delta_f = new_f.vector().copy()
        delta_f.axpy(-1.0, old_f.vector())
        return delta_f.norm("l2") / (new_f.vector().norm("l2") + 1.0e-14)

    def _candidate_component_diagnostics(w_candidate, w_old):
        diag = {
            "rel_p": float("nan"),
            "rel_u": float("nan"),
            "rel_theta": float("nan"),
            "p_min": float("nan"),
            "p_max": float("nan"),
            "u_max": float("nan"),
            "theta_min": float("nan"),
            "theta_max": float("nan"),
        }

        try:
            p_new, u_new, theta_new = w_candidate.split(deepcopy=True)
            p_old, u_old, theta_old = w_old.split(deepcopy=True)

            diag["rel_p"] = _relative_function_update(p_new, p_old)
            diag["rel_u"] = _relative_function_update(u_new, u_old)
            diag["rel_theta"] = _relative_function_update(theta_new, theta_old)

            diag["p_min"] = global_vec_min(p_new)
            diag["p_max"] = global_vec_max(p_new)
            diag["theta_min"] = global_vec_min(theta_new)
            diag["theta_max"] = global_vec_max(theta_new)

            if sub_mesh_star is not None:
                Vmag = fenics.FunctionSpace(sub_mesh_star, "CG", 1)
                u_mag = fenics.project(
                    fenics.sqrt(fenics.inner(u_new, u_new)),
                    Vmag,
                    solver_type="mumps",
                )
                diag["u_max"] = global_vec_max(u_mag)

        except Exception as err:
            print0(f"Candidate diagnostic failed: {err}")

        return diag

    def _compute_integral_diagnostics(sol_w):
        diag = {
            "Q_interface_W_per_m": float("nan"),
            "Q_far_W_per_m": float("nan"),
            "heat_imbalance_rel": float("nan"),
            "theta_max": float("nan"),
            "u_max": float("nan"),
        }
        if sub_mesh_star is None:
            return diag

        try:
            _, u_star, theta = sol_w.split(deepcopy=True)
            k_inf = float(experiment.fluid.properties["k"])
            qn_dim = qn_air * fenics.Constant(float(scales.qsurf))
            n = fenics.FacetNormal(sub_mesh_star)
            Q_interface = float(fenics.assemble(qn_dim * sub_ds(INTERFACE_TAG)) * float(scales.Lref))
            Q_far = float(-k_inf * float(scales.dTref) * fenics.assemble(fenics.dot(fenics.grad(theta), n) * sub_ds(OUTER_AIR_TAG)))
            diag["Q_interface_W_per_m"] = Q_interface
            diag["Q_far_W_per_m"] = Q_far
            if abs(Q_interface) > 1.0e-14:
                diag["heat_imbalance_rel"] = abs(Q_interface - Q_far) / abs(Q_interface)
            diag["theta_max"] = global_vec_max(theta)
            try:
                Vmag = fenics.FunctionSpace(sub_mesh_star, "CG", 1)
                umag = fenics.project(fenics.sqrt(fenics.inner(u_star, u_star)), Vmag, solver_type="mumps")
                diag["u_max"] = global_vec_max(umag)
            except Exception:
                diag["u_max"] = float("nan")
        except Exception:
            pass
        return diag

    def _write_history_csv(path, rows):
        if not path or not rows:
            return
        import csv
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        if is_rank0():
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        COMM.Barrier()

    def _window_mean(values):
        vals = [float(v) for v in values if np.isfinite(v)]
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    def _statistically_steady(rows):
        if len(rows) < 2 * steady_window:
            return False, float("nan")
        prev = rows[-2 * steady_window:-steady_window]
        curr = rows[-steady_window:]
        keys = [f"uy_{y:.3f}m" for y in probe_heights_m] + [f"theta_{y:.3f}m" for y in probe_heights_m]
        drifts = []
        for key in keys:
            m_prev = _window_mean([r.get(key, float("nan")) for r in prev])
            m_curr = _window_mean([r.get(key, float("nan")) for r in curr])
            if np.isfinite(m_prev) and np.isfinite(m_curr):
                drifts.append(abs(m_curr - m_prev) / (abs(m_curr) + 1.0e-14))
        if not drifts:
            return False, float("nan")
        mean_update = _window_mean([r.get("rel_update", float("nan")) for r in curr])
        max_drift = max(drifts)
        return (max_drift < steady_rel_tol and np.isfinite(mean_update) and mean_update < steady_update_tol), max_drift

    print0("\n" + "=" * 72)
    print0("Starting post-continuation transient branch")
    print0(f"  dt_start   = {dt_start:.3e}")
    print0(f"  dt_min     = {dt_min:.3e}")
    print0(f"  dt_max     = {dt_max:.3e}")
    print0(f"  dt_growth  = {dt_growth:.3e}")
    print0(f"  dt_cut     = {dt_cut:.3e}")
    print0(f"  t_end      = {t_end:.3e}")
    print0(f"  step_max   = {step_max}")
    print0(f"  save_every = {save_every}")
    print0("  target     = full coupled transient (lambda=1, convection=1)")
    print0("=" * 72)

    while step < int(step_max) and t < float(t_end):
        trial_success = False
        local_retry = 0
        last_error = None

        while not trial_success:
            copy_state(w_prev, w_n)
            copy_state(w, w_n)

            print0(f"\n=== transient step {step + 1:04d} | t={t:.6e} | dt={dt:.3e} | retry={local_retry} ===")

            F_tr, JF_tr = build_ptc_problem(
                W=W,
                w=w,
                w_prev=w_prev,
                psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
                mu=mu, Pr=Pr, f_b=f_b,
                sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
                dtau=dt,
                buoyancy_scale=1.0,
                qn_scale=1.0,
                include_convection=True,
                convection_scale=1.0,
                SUPG=SUPG,
                # experiment=experiment,
                # scales=scales,
            )

            try:
                _, n_newton, _ = base_solver(
                    F_tr, w, boundary_conditions, JF_tr,
                    relaxation=relaxation,
                    maxit=max_newton_it,
                    atol=atol,
                    rtol=rtol,
                    return_meta=True,
                )

                delta = w.vector().copy()
                delta.axpy(-1.0, w_n.vector())
                rel_update = delta.norm("l2") / (w.vector().norm("l2") + 1.0e-14)
                mixed_norm = _safe_float_norm(w.vector())
                finite_ok = np.isfinite(mixed_norm)

                in_restart_settle = (
                    bool(restart_recovered)
                    and restart_step is not None
                    and step < int(restart_step) + int(restart_settle_steps)
                )

                rel_update_limit = restart_rel_update_reject if in_restart_settle else rel_update_reject

                candidate_diag = _candidate_component_diagnostics(w, w_n)

                print0(
                    "candidate diagnostics: "
                    f"rel_mix={rel_update:.3e}, "
                    f"rel_p={candidate_diag['rel_p']:.3e}, "
                    f"rel_u={candidate_diag['rel_u']:.3e}, "
                    f"rel_theta={candidate_diag['rel_theta']:.3e}, "
                    f"p=[{candidate_diag['p_min']:.3e}, {candidate_diag['p_max']:.3e}], "
                    f"|u|max={candidate_diag['u_max']:.3e}, "
                    f"theta=[{candidate_diag['theta_min']:.3e}, {candidate_diag['theta_max']:.3e}], "
                    f"limit={rel_update_limit:.3e}, "
                    f"restart_settle={in_restart_settle}"
                )

                if not finite_ok or not np.isfinite(rel_update):
                    raise RuntimeError(
                        f"transient step rejected by sanity check: non-finite state, "
                        f"rel_update={rel_update:.3e}, ||w||={mixed_norm}"
                    )

                if rel_update > rel_update_limit:
                    raise RuntimeError(
                        f"transient step rejected by sanity check: "
                        f"rel_update={rel_update:.3e}, "
                        f"limit={rel_update_limit:.3e}, "
                        f"||w||={mixed_norm}"
                    )

                if in_restart_settle:
                    if (
                        np.isfinite(candidate_diag["rel_u"])
                        and candidate_diag["rel_u"] > restart_rel_u_reject
                    ):
                        raise RuntimeError(
                            f"restart-settle sanity rejection: "
                            f"rel_u={candidate_diag['rel_u']:.3e}, "
                            f"limit={restart_rel_u_reject:.3e}"
                        )

                    if (
                        np.isfinite(candidate_diag["rel_theta"])
                        and candidate_diag["rel_theta"] > restart_rel_theta_reject
                    ):
                        raise RuntimeError(
                            f"restart-settle sanity rejection: "
                            f"rel_theta={candidate_diag['rel_theta']:.3e}, "
                            f"limit={restart_rel_theta_reject:.3e}"
                        )

                    if (
                        np.isfinite(candidate_diag["u_max"])
                        and candidate_diag["u_max"] > restart_u_abs_max
                    ):
                        raise RuntimeError(
                            f"restart-settle sanity rejection: "
                            f"|u|max={candidate_diag['u_max']:.3e}, "
                            f"limit={restart_u_abs_max:.3e}"
                        )

                    if (
                        np.isfinite(candidate_diag["theta_min"])
                        and candidate_diag["theta_min"] < restart_theta_min
                    ):
                        raise RuntimeError(
                            f"restart-settle sanity rejection: "
                            f"theta_min={candidate_diag['theta_min']:.3e}, "
                            f"limit={restart_theta_min:.3e}"
                        )

                    if (
                        np.isfinite(candidate_diag["theta_max"])
                        and candidate_diag["theta_max"] > restart_theta_max
                    ):
                        raise RuntimeError(
                            f"restart-settle sanity rejection: "
                            f"theta_max={candidate_diag['theta_max']:.3e}, "
                            f"limit={restart_theta_max:.3e}"
                        )

                trial_success = True
                copy_state(w_last_accepted, w)

            except RuntimeError as err:
                last_error = str(err)
                rejected_steps += 1
                local_retry += 1
                copy_state(w, w_n)
                copy_state(w_prev, w_n)

                dt = max(float(dt_min), float(dt) * float(dt_cut))
                print0(f"Rejected transient step {step + 1:04d}: {err}")
                print0(f"  -> rolling back to last accepted state and reducing dt to {dt:.3e}")

                if dt <= float(dt_min) + 1.0e-30:
                    status = "dt_underflow"
                    break
                if local_retry > int(max_retries_per_step):
                    status = "too_many_retries"
                    break

        if not trial_success:
            print0(f"Transient branch stopping with status={status}")
            if last_error is not None:
                print0(f"  last_error={last_error}")
            break

        copy_state(w_n, w_last_accepted)
        t += dt
        step += 1
        accepted_steps += 1

        row = {
            "step": step,
            "time": t,
            "dt": dt,
            "newton_iterations": int(n_newton) if n_newton is not None else -1,
            "rel_update": float(rel_update),
            "rel_p": float(candidate_diag.get("rel_p", float("nan"))),
            "rel_u": float(candidate_diag.get("rel_u", float("nan"))),
            "rel_theta": float(candidate_diag.get("rel_theta", float("nan"))),
            "candidate_u_max": float(candidate_diag.get("u_max", float("nan"))),
            "candidate_theta_min": float(candidate_diag.get("theta_min", float("nan"))),
            "candidate_theta_max": float(candidate_diag.get("theta_max", float("nan"))),
        }
        row.update(_sample_probes(w_n))
        if diagnostic_every > 0 and (step % diagnostic_every == 0):
            row.update(_compute_integral_diagnostics(w_n))
        history.append(row)

        probe_str = ", ".join(
            f"{k}={v:.3e}" for k, v in row.items()
            if k.startswith("uy_") or k.startswith("theta_")
        )
        print0(
            f"Accepted transient step {step:04d}: rel_update={rel_update:.3e}, "
            f"newton_iterations={n_newton}, t={t:.6e}"
        )
        if probe_str:
            print0(f"  probes: {probe_str}")

        if history_csv_path:
            _write_history_csv(history_csv_path, history)

        if (
            save_every > 0 and step % save_every == 0 and
            sub_mesh_star is not None and sub_mesh_dim is not None and
            p_path and u_path and T_path
        ):
            p_star, u_star, theta = w_n.split(deepcopy=True)
            u_dim, p_dim, T_dim = dimensionalize_fields(
                sub_mesh_star, u_star, p_star, theta,
                scales.Uplume, scales.dTref, T_ambient,
                experiment.fluid.properties["rho"],
            )
            k_air = fenics.Constant(experiment.fluid.properties["k"])
            q_heat, q_mag = compute_heat_flux_dim(T_dim, k_air)
            q_out = T_path.split(".xdmf")[0] + f"_heatflux_transient_{step:05d}.xdmf"
            qmag_out = T_path.split(".xdmf")[0] + f"_heatflux_mag_transient_{step:05d}.xdmf"

            p_out = p_path.split(".xdmf")[0] + f"_transient_{step:05d}.xdmf"
            u_out = u_path.split(".xdmf")[0] + f"_transient_{step:05d}.xdmf"
            t_out = T_path.split(".xdmf")[0] + f"_transient_{step:05d}.xdmf"

            # --- Effective Grashof number diagnostic ---
            theta_max = float(global_vec_max(theta))
            theta_min = float(global_vec_min(theta))

            dT_eff = scales.dTref * max(theta_max, 0.0)

            props = experiment.fluid.properties
            g = float(props.get("g", 9.81))
            beta = float(props["beta"])
            nu = float(scales.nu)

            L_eff = float(scales.Lref)   # keep consistent with your reference Gr/Ra definition

            Gr_eff = g * beta * dT_eff * L_eff**3 / (nu**2)
            Ra_eff = Gr_eff * float(scales.Pr)

            print0(
                f"  snapshot diagnostics: "
                f"theta_min={theta_min:.6e}, theta_max={theta_max:.6e}, "
                f"dT_eff={dT_eff:.6e} K, "
                f"Gr_eff={Gr_eff:.6e}, Ra_eff={Ra_eff:.6e}"
            )

            J_dim = compute_entropy_flux_dim(
                mesh=sub_mesh_dim,
                u_dim=u_dim,
                T_dim=T_dim,
                rho=experiment.fluid.properties["rho"],
                cp=experiment.fluid.properties["cp"],
                k=experiment.fluid.properties["k"],
                T_inf=T_ambient,
                degree=1,
                family="DG",
            )
            J_out = T_path.split(".xdmf")[0] + f"_entropy_flux_transient_{step:05d}.xdmf"
            
            # save_experiment(p_out, sub_mesh_dim, [p_dim])
            save_experiment(u_out, sub_mesh_dim, [u_dim])
            save_experiment(t_out, sub_mesh_dim, [T_dim])
            save_experiment(q_out, sub_mesh_dim, [q_heat])
            save_experiment(J_out, sub_mesh_dim, [J_dim])
            # save_experiment(qmag_out, sub_mesh_dim, [q_mag])

            Lref_dim = float(scales.Lref)
            plane_fluxes = compute_horizontal_plane_heat_fluxes(
                u_dim=u_dim,
                T_dim=T_dim,
                sub_mesh_dim=sub_mesh_dim,
                experiment=experiment,
                y_planes_m=(0.01, 0.02, 0.04, 0.08),
                T_ref=T_ambient,
                nx=400,
                half_domain_symmetric=True,
            )

            flux_row = {
                "step": step,
                "time": t,
                "dt": dt,
            }
            flux_row.update(plane_fluxes)

            append_plane_flux_csv(
                os.path.join(os.path.dirname(T_path), "plane_fluxes.csv"),
                flux_row
            )

            checkpoint_dir = os.path.join(os.path.dirname(T_path), "restart_checkpoint")
            save_restart_checkpoint(
                checkpoint_dir=checkpoint_dir,
                mesh_star=sub_mesh_star,
                w_n=w_n,
                step=step,
                time_value=t,
                dt_value=dt,
            )
        
            if sub_ft_dim is not None and sub_ds_dim is not None:
                try:
                    _,_, biots = biot_wrap(
                        sub_mesh=sub_mesh_dim,
                        sub_ft=sub_ft_dim,
                        sub_ds=sub_ds_dim,
                        T_air_dim=T_dim,
                        qn_air=qn_air,
                        scales=scales,
                        T_ref=T_ambient,
                        k_wire=experiment.wire.properties["k"],
                        wire_diameter=experiment.dimensions.wire.diameter,
                        characteristic_length="radius",
                        return_local_field=True
                    )
                    print0(f"Biot number stats: min={float(global_vec_min(biots)):.6e}, max={float(global_vec_max(biots)):.6e}")
                except Exception as err:
                    print0(f"Biot diagnostic skipped at step {step:04d}: {err}")

        is_steady, max_drift = _statistically_steady(history)
        if is_steady:
            status = "statistically_steady"
            print0(
                f"Transient stopping criterion satisfied: statistically steady "
                f"(max window drift={max_drift:.3e})."
            )
            break
        
        _, u_star, theta = w_n.split(deepcopy=True)
        dt_cfl = cfl_limited_dt(
            sub_mesh_star, u_star,
            cfl_target=1.0,
            safety=0.9,
            dt_min=dt_min,
            dt_max=dt_max
        )

        in_restart_settle_after_accept = (
            bool(restart_recovered)
            and restart_step is not None
            and step <= int(restart_step) + int(restart_settle_steps)
        )

        if in_restart_settle_after_accept:
            # During the restart-settling window, do not grow dt.
            # Let the migrated state relax onto the new mesh/discretization first.
            if rel_update >= float(rel_update_hard):
                dt = max(float(dt_min), float(dt) * float(dt_hard_cut))
            else:
                dt = min(float(dt), float(dt_max))
        else:
            if n_newton is not None and n_newton <= int(newton_easy_iters) and rel_update <= float(rel_update_easy):
                dt = min(float(dt_max), float(dt) * float(dt_growth))
            elif n_newton is not None and (n_newton >= int(newton_hard_iters) or rel_update >= float(rel_update_hard)):
                dt = max(float(dt_min), float(dt) * float(dt_hard_cut))
            else:
                dt = min(float(dt), float(dt_max))

        dt = min(dt_cfl, dt)

    if step >= int(step_max) and status == "transient_complete":
        status = "step_limit_reached"
    if t >= float(t_end) and status == "transient_complete":
        status = "final_time_reached"

    return w, {
        "status": status,
        "n_steps": step,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "final_dt": dt,
        "final_time": t,
        "history": history,
    }

def solve_steady_from_loaded_checkpoint(
    experiment: Experiment,
    W: fenics.FunctionSpace,
    w: fenics.Function,
    w_n: fenics.Function,
    psi_p,
    psi_u,
    psi_T,
    mu,
    Pr,
    f_b,
    T_c,
    T_air_bc,
    sub_dx,
    sub_ds,
    sub_ft,
    qn_air,
    sub_mesh_star,
    sub_mesh_dim,
    scales,
    T_ambient: float,
    rho_air: float,
    p_path: str,
    u_path: str,
    T_path: str,
    checkpoint_meta: dict,
    SUPG: bool = False,
):
    """
    Load state is assumed already copied into both w and w_n.
    This solves the steady residual only:
        no dt
        no u_n
        no T_n
        no pseudo-time terms
    """

    boundary_conditions = set_bcs(
        W,
        sub_ft,
        T_air_bc,
        T_c,
        experiment,
        scales,
    )

    # Ensure current Newton iterate starts exactly from accepted restart state.
    copy_state(w, w_n)

    print0("Building full steady residual from checkpoint initial guess...")
    F_steady, JF_steady = build_nonlinear_problem(
        W=W,
        w=w,
        psi_p=psi_p,
        psi_u=psi_u,
        psi_T=psi_T,
        mu=mu,
        Pr=Pr,
        f_b=f_b,
        sub_dx=sub_dx,
        sub_ds=sub_ds,
        qn_air=qn_air,
        buoyancy_scale=1.0,
        qn_scale=1.0,
        include_convection=True,
        convection_scale=1.0,
        SUPG=SUPG,
        outlet_penalty=0.0,
        backflow_beta=0.0,
    )

    try:
        print0("Trying direct steady Newton from transient checkpoint...")
        w, n_iter, converged = base_solver(
            F_steady,
            w,
            boundary_conditions,
            JF_steady,
            relaxation=0.7,
            maxit=30,
            atol=1.0e-9,
            rtol=1.0e-8,
            return_meta=True,
        )

        print0(
            f"Direct steady Newton converged={converged}, "
            f"iterations={n_iter}"
        )
    except RuntimeError as direct_err:
        print0(f"Direct steady Newton failed: {direct_err}")
        print0("Retrying with damped relaxation values...")

        last_err = direct_err
        ok = False

        for relaxation in (0.5, 0.3, 0.15):
            copy_state(w, w_n)

            try:
                print0(f"  retry steady Newton with relaxation={relaxation:.2f}")
                w, n_iter, converged = base_solver(
                    F_steady,
                    w,
                    boundary_conditions,
                    JF_steady,
                    relaxation=relaxation,
                    maxit=30,
                    atol=1.0e-9,
                    rtol=1.0e-8,
                    return_meta=True,
                )

                print0(
                    f"Direct steady Newton converged={converged}, "
                    f"iterations={n_iter}"
                )
                ok = True
                break
            except RuntimeError as err:
                last_err = err
                print0(f"  retry failed: {err}")

        if not ok:
            raise RuntimeError(
                "Steady Newton from transient checkpoint failed for all relaxation attempts."
            ) from last_err

    copy_state(w_n, w)

    print0("Steady Newton accepted. Saving dimensional fields...")

    p_star, u_star, theta = w.split(deepcopy=True)

    u_dim, p_dim, T_dim = dimensionalize_fields(
        sub_mesh_star,
        u_star,
        p_star,
        theta,
        scales.Uplume,
        scales.dTref,
        T_ambient,
        rho_air,
    )

    stem = f"steady_from_transient_step_{int(checkpoint_meta.get('step', 0)):05d}"

    p_out = p_path.split(".xdmf")[0] + f"_{stem}.xdmf"
    u_out = u_path.split(".xdmf")[0] + f"_{stem}.xdmf"
    T_out = T_path.split(".xdmf")[0] + f"_{stem}.xdmf"

    save_experiment(p_out, sub_mesh_dim, [p_dim])
    save_experiment(u_out, sub_mesh_dim, [u_dim])
    save_experiment(T_out, sub_mesh_dim, [T_dim])

    steady_checkpoint_dir = os.path.join(
        os.path.dirname(T_path),
        "steady_from_transient_checkpoint",
    )

    save_restart_checkpoint(
        checkpoint_dir=steady_checkpoint_dir,
        mesh_star=sub_mesh_star,
        w_n=w_n,
        step=int(checkpoint_meta.get("step", 0)),
        time_value=float(checkpoint_meta.get("time", 0.0)),
        dt_value=0.0,
    )

    print0(f"Saved steady pressure:    {p_out}")
    print0(f"Saved steady velocity:    {u_out}")
    print0(f"Saved steady temperature: {T_out}")
    print0(f"Saved steady checkpoint:  {steady_checkpoint_dir}")

    return w
