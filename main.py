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


def check_interface_power(sub_ds, sub_ft, qn_air, scales, experiment, interface_tag=INTERFACE_TAG):
    # 1) dimensionalize qn_air: qn_dim [W/m^2]
    k_inf = float(experiment.fluid.properties["k"])  # use experiment value (not global)
    qn_dim = qn_air * fenics.Constant(k_inf * float(scales.dTref) / float(scales.Lref))

    # 2) integrate over the *half* interface boundary (your mesh is half-domain)
    QL_half = fenics.assemble(qn_dim * sub_ds(interface_tag))  # [W/m]
    # QL_half = QL_half * scales.Lref
    QL_full = 2.0 * QL_half                                   # mirror symmetry -> full wire

    # 3) interface length checks (helps debug tagging/completeness)
    L_half = fenics.assemble(fenics.Constant(1.0) * sub_ds(interface_tag))  # [m]
    # L_half = L_half * scales.Lref
    L_full = 2.0 * L_half

    # 4) expected values
    QL_target = float(experiment.initial_conditions.heat_length)  # [W/m]
    d = float(experiment.dimensions.wire.diameter)
    qsurf_target = QL_target / (math.pi * d)  # [W/m^2]
    qsurf_avg = (QL_half / L_half) if L_half > 0 else float("nan")

    print("=== Interface power conservation ===")
    print(f"interface_tag = {interface_tag}")
    print(f"Interface length (half)  L_half  = {L_half:.6e} m")
    print(f"Interface length (full)  L_full  = {L_full:.6e} m")
    print(f"Recovered power (half)   QL_half = {QL_half:.6e} W/m")
    print(f"Recovered power (full)   QL_full = {QL_full:.6e} W/m")
    print(f"Target power (paper)     QL      = {QL_target:.6e} W/m")
    print(f"Relative error (full)    = {(QL_full-QL_target)/QL_target:.3%}")

    print("--- Flux magnitude sanity ---")
    print(f"Target mean q''          = {qsurf_target:.6e} W/m^2  (QL/(pi*d))")
    print(f"Recovered mean q'' (half)= {qsurf_avg:.6e} W/m^2  (QL_half/L_half)")
    print("===================================")



def base_version(experiment: Experiment):
    GEOM_FILE = geometry_template(
        wire_radius=experiment.dimensions.wire.diameter / 2,
        output_path=experiment.name,
        xmax=experiment.dimensions.domain.x_max,
        ymax=experiment.dimensions.domain.y_max
    )
    MSH_FILE = experiment.name + "/plume.msh"
    TRIG_XDMF_PATH = experiment.name + "/plume.xdmf"
    FACETS_XDMF_PATH = experiment.name + "/plume_mt.xdmf"
    OUTPUT_XDMF_PATH_WIRE = experiment.name + "/base/wire_temperature.xdmf"
    OUTPUT_XDMF_PATH_TEMP = experiment.name + "/base/temperature.xdmf"
    OUTPUT_XDMF_PATH_AIR_T = experiment.name + "/base/air_temperature.xdmf"
    OUTPUT_XDMF_PATH_AIR_P = experiment.name + "/base/air_pressure.xdmf"
    OUTPUT_XDMF_PATH_AIR_V = experiment.name + "/base/air_velocity.xdmf"
    OUTPUT_XDMF_PATH_AIR_PVT = experiment.name + "/base/air_pvt.xdmf"
    MESH_NAME = "Grid"
    ELEM = "triangle"

    # Generate and read mesh
    generate_mesh(GEOM_FILE, MSH_FILE, TRIG_XDMF_PATH, FACETS_XDMF_PATH)
    
    # --- 1) read mesh (dim)  [already]
    mesh, ct, ft, domains, dx, boundary_markers, mc, mf = read_mesh(
        TRIG_XDMF_PATH, FACETS_XDMF_PATH, MESH_NAME, PRINT_TAG_SUMMARY
    )

    # --- 2) create submesh (dim)
    sub_mesh_dim, sub_ft_dim, sub_dx_dim, sub_ds_dim = create_submesh(mesh, mc, mf, AIR_TAG)

    scales = compute_nondimensional_scales(experiment)
    print(scales)

    # --- 3) conduction initial guess (dim parent mesh)
    print("Computing initial guess for temperature field...")
    heat_volume = volume_heat_source(experiment)
    print(f"Using heat volume: {heat_volume} W/m^3")

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

    print("theta min/max:", theta_full_dim.vector().min(), theta_full_dim.vector().max())
    print("T_dim  min/max:", T_air_dim.vector().min(), T_air_dim.vector().max())
    print("T_nondim min/max:", theta_full_dim.vector().min(), theta_full_dim.vector().max())

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

    print(f"Initial max temperature: {T_full.vector().max():.2f} K")
    print(f"Initial min temperature: {T_full.vector().min():.2f} K")
    print(f"Initial max theta (dim-submesh): {theta_full_dim.vector().max():.6e}")
    print(f"Initial min theta (dim-submesh): {theta_full_dim.vector().min():.6e}")

    # --- 5) scale parent mesh coordinates (dim→star)
    Lref = float(scales.Lref)
    mesh.coordinates()[:] /= Lref

    # IMPORTANT: also scale the OLD dim-submesh in-place so theta_full_dim now
    # lives on star-coordinates (needed for safe interpolation between submeshes)
    sub_mesh_dim.coordinates()[:] /= Lref

    # --- 6) recreate submesh + measures on scaled parent mesh
    sub_mesh_star, sub_ft_star, sub_dx_star, sub_ds_star = create_submesh(mesh, mc, mf, AIR_TAG)

    # --- 7) transfer theta_full_dim onto the new scaled submesh
    V_air_star = fenics.FunctionSpace(sub_mesh_star, "CG", 1)
    V0_star = fenics.FunctionSpace(sub_mesh_star, "DG", 0)
    qn_air_star = fenics.Function(V0_star)
    qn_air.set_allow_extrapolation(True)

    dof_coords = V0_star.tabulate_dof_coordinates().reshape((-1, sub_mesh_star.geometry().dim()))
    vals = [qn_air(x) for x in dof_coords]
    qn_air_star.vector()[:] = vals
    qn_air_star.vector().apply("insert")

    # Build a plain Function on the star submesh
    theta_full_star = fenics.Function(V_air_star)
    theta_full_dim.set_allow_extrapolation(True)

    # Sample theta_full_dim at star-submesh DOF coordinates
    dof_coords = V_air_star.tabulate_dof_coordinates().reshape((-1, sub_mesh_star.geometry().dim()))
    values = []

    for x in dof_coords:
        values.append(theta_full_dim(x))

    theta_full_star.vector()[:] = values
    theta_full_star.vector().apply("insert")
    theta_full_star.rename("theta_full", "theta_full")

    print(f"Initial max theta (star-submesh): {theta_full_star.vector().max():.6e}")
    print(f"Initial min theta (star-submesh): {theta_full_star.vector().min():.6e}")
    print(f"Rho_air: {experiment.fluid.properties['rho']}")
    print(f"Beta_air: {experiment.fluid.properties['beta']}")
    

    # Solving the problem
    print("Starting solver...")
    W, w, p, u, T, w_n, p_n, u_n, T_n, psi_p, psi_u, psi_T, \
    mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc = solver(
        sub_mesh_star,
        theta_full_star,      # <-- nondimensional theta on star mesh
        0.0,
        experiment.fluid.properties["rho"],
        experiment.fluid.properties["beta"],
        experiment
    )

    # F, w, boundary_conditions, JF, w_n = nonlinear_solver(
    #     experiment, u_n, u, T_n, T, p, W, w,
    #     psi_p, psi_u, psi_T,
    #     mu, Pr, f_b, T_c, T_air_bc,
    #     sub_dx_star, sub_ds_star, sub_ft_star, qn_air_star,
    #     w_n
    # )

    # w = base_solver(F, w, boundary_conditions, JF)
    
    w = solve_steady_picard(experiment=experiment, W=W, w=w, w_n=w_n,
                            psi_p=psi_p, psi_u=psi_u, psi_T=psi_T,
                            mu=mu, Pr=Pr, Ra=Ra, sub_dx=sub_dx_star, sub_ds=sub_ds_star, sub_ft=sub_ft_star, qn_air=qn_air_star)

    # Split nondimensional solution
    p_star, u_star, theta = w.split(deepcopy=True)

    # Dimensionalize fields (note: mesh is star; dimensionalize handles scaling)
    u_dim, p_dim, T_dim = dimensionalize_fields(
        sub_mesh_star, u_star, p_star, theta,
        scales.Uref, scales.dTref, T_ambient,
        experiment.fluid.properties["rho"]
    )

    # on sub_mesh_star, with theta (nondim)
    k_inf = float(experiment.fluid.properties["k"])  # use experiment value (not global)
    n = fenics.FacetNormal(sub_mesh_star)
    dTref = float(scales.dTref)

    # Example: if facet tags exist for TOP and FAR boundaries
    Q_far  = -k_inf*dTref * fenics.assemble(fenics.dot(fenics.grad(theta), n) * sub_ds_star(OUTER_AIR_TAG))
    print(f"Heat flux through outer air boundary: Q_far = {Q_far:.6e} W/m")


    # plotting + output
    plot_mesh(T_dim, title="Temperature field", label="Temperature (K)",
                cmap="coolwarm", colorbar=True)
    plot_mesh(theta, title="Temperature field nondimensional", label="Temperature (nondim)",
                cmap="coolwarm", colorbar=True)
    plot_mesh(u_dim, title="Velocity magnitude", label="Velocity (m/s)",
                cmap="coolwarm", colorbar=True, mode="glyphs")
    plot_mesh(p_dim, title="Pressure field", label="Pressure (Pa)",
                cmap="coolwarm", colorbar=True)

    save_experiment(OUTPUT_XDMF_PATH_AIR_P, sub_mesh_dim, [p_dim])
    save_experiment(OUTPUT_XDMF_PATH_AIR_V, sub_mesh_dim, [u_dim])
    save_experiment(OUTPUT_XDMF_PATH_AIR_T, sub_mesh_dim, [T_dim])
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
        print(f"y0={y0_m:.3f} m: Qconv={Qconv:.6e} W/m, Qcond={Qcond:.6e} W/m, "
            f"Qtot={Qtot:.6e} W/m, mdot={mdot:.6e} kg/(s·m)")
        
    out_dir=Path.cwd()
    csv_path = os.path.join(out_dir,experiment.name, "base", "plane_fluxes.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        wcsv = csv.writer(f)
        if write_header:
            wcsv.writerow(["time", "y0_m", "Qconv_W_per_m", "Qcond_W_per_m", "Qtot_W_per_m", "mdot_kg_per_s_per_m"])
        t=0
        for (y0_m, Qconv, Qcond, Qtot, mdot) in flux_rows:
            wcsv.writerow([float(t), y0_m, Qconv, Qcond, Qtot, mdot])

def temperature_dependent_version(experiment: Experiment):
    GEOM_FILE = geometry_template(
        wire_radius=experiment.dimensions.wire.diameter / 2,
        output_path=experiment.name,
        xmax=experiment.dimensions.domain.x_max,
        ymax=experiment.dimensions.domain.y_max
    )
    MSH_FILE = experiment.name + "/plume.msh"
    TRIG_XDMF_PATH = experiment.name + "/plume.xdmf"
    FACETS_XDMF_PATH = experiment.name + "/plume_mt.xdmf"
    OUTPUT_XDMF_PATH_WIRE = experiment.name + "/t_dep_mat/wire_temperature.xdmf"
    OUTPUT_XDMF_PATH_TEMP = experiment.name + "/t_dep_mat/temperature.xdmf"
    OUTPUT_XDMF_PATH_AIR_T = experiment.name + "/t_dep_mat/air_temperature.xdmf"
    OUTPUT_XDMF_PATH_AIR_P = experiment.name + "/t_dep_mat/air_pressure.xdmf"
    OUTPUT_XDMF_PATH_AIR_V = experiment.name + "/t_dep_mat/air_velocity.xdmf"
    OUTPUT_XDMF_PATH_AIR_PVT = experiment.name + "/t_dep_mat/air_pvt.xdmf"
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
    print(scales)

    # --- 3) conduction initial guess (dim parent mesh)
    print("Computing initial guess for temperature field...")
    heat_volume = volume_heat_source(experiment)
    print(f"Using heat volume: {heat_volume} W/m^3")

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

    print("theta min/max:", theta_full_dim.vector().min(), theta_full_dim.vector().max())
    print("T_dim  min/max:", T_air_dim.vector().min(), T_air_dim.vector().max())
    print("T_nondim min/max:", theta_full_dim.vector().min(), theta_full_dim.vector().max())

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

    print(f"Initial max temperature: {T_full.vector().max():.2f} K")
    print(f"Initial min temperature: {T_full.vector().min():.2f} K")
    print(f"Initial max theta (dim-submesh): {theta_full_dim.vector().max():.6e}")
    print(f"Initial min theta (dim-submesh): {theta_full_dim.vector().min():.6e}")

    # --- 5) scale parent mesh coordinates (dim→star)
    Lref = float(scales.Lref)
    mesh.coordinates()[:] /= Lref

    # IMPORTANT: also scale the OLD dim-submesh in-place so theta_full_dim now
    # lives on star-coordinates (needed for safe interpolation between submeshes)
    sub_mesh_dim.coordinates()[:] /= Lref

    # --- 6) recreate submesh + measures on scaled parent mesh
    sub_mesh_star, sub_ft_star, sub_dx_star, sub_ds_star = create_submesh(mesh, mc, mf, AIR_TAG)

    # --- 7) transfer theta_full_dim onto the new scaled submesh
    V_air_star = fenics.FunctionSpace(sub_mesh_star, "CG", 1)

    # Build a plain Function on the star submesh
    theta_full_star = fenics.Function(V_air_star)
    theta_full_dim.set_allow_extrapolation(True)

    # Sample theta_full_dim at star-submesh DOF coordinates
    dof_coords = V_air_star.tabulate_dof_coordinates().reshape((-1, sub_mesh_star.geometry().dim()))
    values = []

    for x in dof_coords:
        values.append(theta_full_dim(x))

    theta_full_star.vector()[:] = values
    theta_full_star.vector().apply("insert")
    theta_full_star.rename("theta_full", "theta_full")

    print(f"Initial max theta (star-submesh): {theta_full_star.vector().max():.6e}")
    print(f"Initial min theta (star-submesh): {theta_full_star.vector().min():.6e}")
    print(f"Rho_air: {experiment.fluid.properties['rho']}")
    print(f"Beta_air: {experiment.fluid.properties['beta']}")

        # Define temperature-dependent material model for air
    fluid_material = TemperatureDependentMaterial(
        mesh=sub_mesh_star,
        T_ref=experiment.initial_conditions.temperature,
        mu_ref=experiment.fluid.properties["mu"],
        cp_ref=experiment.fluid.properties["cp"],
        k_ref=experiment.fluid.properties["k"],
        beta_ref=experiment.fluid.properties["beta"],
        rho_ref=experiment.fluid.properties["rho"],
        table_file="materials/air_properties_coolprop.csv"   # or None
    )
    

    # Solving the problem
    print("Starting solver...")
    W, w, p, u, T, w_n, p_n, u_n, T_n, psi_p, psi_u, psi_T, \
    mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc = solver(
        sub_mesh_star,
        theta_full_star,      # <-- nondimensional theta on star mesh
        0.0,
        experiment.fluid.properties["rho"],
        experiment.fluid.properties["beta"],
        experiment
    )

    F, w, boundary_conditions, JF, w_n = nonlinear_solver(
        experiment, u_n, u, T_n, T, p, W, w,
        psi_p, psi_u, psi_T,
        mu, Pr, f_b, T_c, T_air_bc,
        sub_dx_star, sub_ds_star, sub_ft_star, qn_air,
        w_n
    )
    w = temp_dep_solver(F,w, boundary_conditions, JF, w_n, fluid_material)

    # Split nondimensional solution
    p_star, u_star, theta = w.split(deepcopy=True)

    # Dimensionalize fields (note: mesh is star; dimensionalize handles scaling)
    u_dim, p_dim, T_dim = dimensionalize_fields(
        sub_mesh_star, u_star, p_star, theta,
        scales.Uref, scales.dTref, T_ambient,
        experiment.fluid.properties["rho"]
    )

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
        print(f"y0={y0_m:.3f} m: Qconv={Qconv:.6e} W/m, Qcond={Qcond:.6e} W/m, "
            f"Qtot={Qtot:.6e} W/m, mdot={mdot:.6e} kg/(s·m)")
        
    out_dir=Path.cwd()
    csv_path = os.path.join(out_dir,experiment.name, "t_dep_mat", "plane_fluxes.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        wcsv = csv.writer(f)
        if write_header:
            wcsv.writerow(["time", "y0_m", "Qconv_W_per_m", "Qcond_W_per_m", "Qtot_W_per_m", "mdot_kg_per_s_per_m"])
        t=0
        for (y0_m, Qconv, Qcond, Qtot, mdot) in flux_rows:
            wcsv.writerow([float(t), y0_m, Qconv, Qcond, Qtot, mdot])


def abs_version(experiment: Experiment):
    GEOM_FILE = geometry_template(
        wire_radius=experiment.dimensions.wire.diameter / 2,
        output_path=experiment.name,
        xmax=experiment.dimensions.domain.x_max,
        ymax=experiment.dimensions.domain.y_max
    )
    MSH_FILE = experiment.name + "/plume.msh"
    TRIG_XDMF_PATH = experiment.name + "/plume.xdmf"
    FACETS_XDMF_PATH = experiment.name + "/plume_mt.xdmf"
    OUTPUT_XDMF_PATH_WIRE = experiment.name + "/abs/wire_temperature.xdmf"
    OUTPUT_XDMF_PATH_TEMP = experiment.name + "/abs/temperature.xdmf"
    OUTPUT_XDMF_PATH_AIR_T = experiment.name + "/abs/air_temperature.xdmf"
    OUTPUT_XDMF_PATH_AIR_P = experiment.name + "/abs/air_pressure.xdmf"
    OUTPUT_XDMF_PATH_AIR_V = experiment.name + "/abs/air_velocity.xdmf"
    OUTPUT_XDMF_PATH_AIR_PVT = experiment.name + "/abs/air_pvt.xdmf"
    MESH_NAME = "Grid"
    ELEM = "triangle"

    # Generate and read mesh
    generate_mesh(GEOM_FILE, MSH_FILE, TRIG_XDMF_PATH, FACETS_XDMF_PATH)
    
    # --- 1) read mesh (dim)  [already]
    mesh, ct, ft, domains, dx, boundary_markers, mc, mf = read_mesh(
        TRIG_XDMF_PATH, FACETS_XDMF_PATH, MESH_NAME, PRINT_TAG_SUMMARY
    )

    # --- 2) create submesh (dim)
    sub_mesh_dim, sub_ft_dim, sub_dx_dim, sub_ds_dim = create_submesh(mesh, mc, mf, AIR_TAG)

    scales = compute_nondimensional_scales(experiment)
    print(scales)

    # --- 3) conduction initial guess (dim parent mesh)
    print("Computing initial guess for temperature field...")
    heat_volume = volume_heat_source(experiment)
    print(f"Using heat volume: {heat_volume} W/m^3")

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

    print("theta min/max:", theta_full_dim.vector().min(), theta_full_dim.vector().max())
    print("T_dim  min/max:", T_air_dim.vector().min(), T_air_dim.vector().max())
    print("T_nondim min/max:", theta_full_dim.vector().min(), theta_full_dim.vector().max())

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

    print(f"Initial max temperature: {T_full.vector().max():.2f} K")
    print(f"Initial min temperature: {T_full.vector().min():.2f} K")
    print(f"Initial max theta (dim-submesh): {theta_full_dim.vector().max():.6e}")
    print(f"Initial min theta (dim-submesh): {theta_full_dim.vector().min():.6e}")

    # --- 5) scale parent mesh coordinates (dim→star)
    Lref = float(scales.Lref)
    mesh.coordinates()[:] /= Lref

    # IMPORTANT: also scale the OLD dim-submesh in-place so theta_full_dim now
    # lives on star-coordinates (needed for safe interpolation between submeshes)
    sub_mesh_dim.coordinates()[:] /= Lref

    # --- 6) recreate submesh + measures on scaled parent mesh
    sub_mesh_star, sub_ft_star, sub_dx_star, sub_ds_star = create_submesh(mesh, mc, mf, AIR_TAG)

    # --- 7) transfer theta_full_dim onto the new scaled submesh
    V_air_star = fenics.FunctionSpace(sub_mesh_star, "CG", 1)

    # Build a plain Function on the star submesh
    theta_full_star = fenics.Function(V_air_star)
    theta_full_dim.set_allow_extrapolation(True)

    # Sample theta_full_dim at star-submesh DOF coordinates
    dof_coords = V_air_star.tabulate_dof_coordinates().reshape((-1, sub_mesh_star.geometry().dim()))
    values = []

    for x in dof_coords:
        values.append(theta_full_dim(x))

    theta_full_star.vector()[:] = values
    theta_full_star.vector().apply("insert")
    theta_full_star.rename("theta_full", "theta_full")

    print(f"Initial max theta (star-submesh): {theta_full_star.vector().max():.6e}")
    print(f"Initial min theta (star-submesh): {theta_full_star.vector().min():.6e}")
    print(f"Rho_air: {experiment.fluid.properties['rho']}")
    print(f"Beta_air: {experiment.fluid.properties['beta']}")
    

    # Solving the problem
    print("Starting solver...")
   
    W, w, p, u, T, w_n, p_n, u_n, T_n, psi_p, psi_u, psi_T, \
    mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc = solver(
        sub_mesh_star,
        theta_full_star,      # <-- nondimensional theta on star mesh
        0.0,
        experiment.fluid.properties["rho"],
        experiment.fluid.properties["beta"],
        experiment
    )

    F, w, boundary_conditions, JF, w_n = nonlinear_solver_ABE(
        experiment, u_n, u, T_n, T, p, W, w,
        psi_p, psi_u, psi_T,
        mu, Pr, f_b, T_c, T_air_bc,
        sub_dx_star, sub_ds_star, sub_ft_star, qn_air,
        w_n,
        fenics.Constant(scales.fEc)
    )

    w = base_solver(F, w, boundary_conditions, JF)

    # Split nondimensional solution
    p_star, u_star, theta = w.split(deepcopy=True)

    # Dimensionalize fields (note: mesh is star; dimensionalize handles scaling)
    u_dim, p_dim, T_dim = dimensionalize_fields(
        sub_mesh_star, u_star, p_star, theta,
        scales.Uref, scales.dTref, T_ambient,
        experiment.fluid.properties["rho"]
    )

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
        print(f"y0={y0_m:.3f} m: Qconv={Qconv:.6e} W/m, Qcond={Qcond:.6e} W/m, "
            f"Qtot={Qtot:.6e} W/m, mdot={mdot:.6e} kg/(s·m)")
    
    out_dir=Path.cwd()
    csv_path = os.path.join(out_dir,experiment.name, "abs", "plane_fluxes.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        wcsv = csv.writer(f)
        if write_header:
            wcsv.writerow(["time", "y0_m", "Qconv_W_per_m", "Qcond_W_per_m", "Qtot_W_per_m", "mdot_kg_per_s_per_m"])
        t=0
        for (y0_m, Qconv, Qcond, Qtot, mdot) in flux_rows:
            wcsv.writerow([float(t), y0_m, Qconv, Qcond, Qtot, mdot])


def abs_temperature_dependent_version(experiment: Experiment):
    pass

def main():
    # Parse command line arguments
    argparser = argparse.ArgumentParser(description="Heated Laminar Plume Simulation")
    argparser.add_argument(
        "--experiment-index",
        type=int,
        default=1,
        help="Index of the experiment to run from experiments.json",
    )
    args = argparser.parse_args()
    args.experiment_index = max(0, args.experiment_index)
    experiment_list = parser(experiments_json_path=EXPERIMENTS_JSON_PATH, schema_json_path=SCHEMA_JSON_PATH)
    experiment = experiment_list[args.experiment_index]
    print(f"Running experiment: {experiment.name}")

    base_version(experiment)
    # temperature_dependent_version(experiment)
    # abs_version(experiment)
    # abs_temperature_dependent_version(experiment)


if __name__ == "__main__":
    main()
