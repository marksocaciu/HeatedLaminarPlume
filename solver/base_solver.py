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

    for lam in lambdas:
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
        for target_conv in conv_targets:
            accepted_conv, _, _ = advance_convection_monotone(
                W=W, w=w,
                psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
                mu=mu, Pr=Pr, f_b=f_b,
                sub_dx=sub_dx, sub_ds=sub_ds, qn_air=qn_air,
                boundary_conditions=boundary_conditions,
                w_n=w_n,
                buoyancy_scale=lam,
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
                convection_scale=accepted_conv_scale,
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
