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
    mesh, ct, ft, domains, dx, boundary_markers, mc, mf = read_mesh(TRIG_XDMF_PATH, FACETS_XDMF_PATH, MESH_NAME, PRINT_TAG_SUMMARY)
    sub_mesh, sub_ft, sub_dx, sub_ds = create_submesh(mesh,mc,mf,AIR_TAG)
    # plot_mesh(mesh, title="Full Mesh")
    # plot_mesh(sub_mesh, title="Sub-Mesh")
    scales = compute_nondimensional_scales(experiment)
    print(scales)


    # Initial guess for solver
    print("Computing initial guess for temperature field...")
    heat_volume = volume_heat_source(experiment)
    print(f"Using heat volume: {heat_volume} W/m^3")
    T_full, k_func = initial_guess(mesh,mc,mf,OUTPUT_XDMF_PATH_TEMP,heat_volume,experiment,dx)
    qn_air = flux_continuity(T_full, k_func, mesh, sub_mesh, sub_ft, mc, scales)
    V_air = fenics.FunctionSpace(sub_mesh, "CG", 1)
    theta_full = fenics.project(
        (T_full - fenics.Constant(experiment.initial_conditions.temperature)) / fenics.Constant(scales.dTref),
        V_air
    )
    theta_full.rename("theta_full", "theta_full")




    print(f"Initial max temperature: {T_full.vector().max():.2f} K")
    print(f"Initial min temperature: {T_full.vector().min():.2f} K")
    print(f"Initial max theta: {theta_full.vector().max():.2f}")
    print(f"Initial min theta: {theta_full.vector().min():.2f}")
    print(f"Rho_air: {experiment.fluid.properties['rho']}")
    print(f"Beta_air: {experiment.fluid.properties['beta']}")

    # Solving the problem
    print("Starting solver...")
    W, w, p, u, T, w_n, p_n, u_n, T_n, psi_p, psi_u, psi_T, mu, Pr, Ra, f_b, T_h, T_c, T_ref, T_air_bc = solver(sub_mesh, theta_full, 0.0, experiment.fluid.properties["rho"], experiment.fluid.properties["beta"], experiment)
    w = nonlinear_solver(experiment, u_n,u,T_n,T, p, W, w,
                         psi_p, psi_u, psi_T,
                         mu, Pr, f_b, T_c, T_air_bc,
                         sub_dx, sub_ds, sub_ft, qn_air,
                         w_n)
    biot_air_h_eff, biot_air_Bi = biot(sub_mesh, sub_ft, T_full, qn_air, experiment.initial_conditions.temperature, experiment.wire.properties["k"], experiment.dimensions.wire.diameter) 


    # p, u, T = fenics.split(w)
    p_star, u_star, theta = w.split(deepcopy=True) # T is nondim theta here
    # p_star, u_star, theta = fenics.split(w) # T is nondim theta here

    # T = theta*scales.dTref + experiment.initial_conditions.temperature  # dimensionalize
    # u = u*scales.Uref  # dimensionalize

    u, p, T = dimensionalize_fields(sub_mesh, u_star, p_star, theta, scales.Uref, scales.dTref, experiment.initial_conditions.temperature, experiment.fluid.properties["rho"])

    plot_mesh(T, title="Temperature field", label="Temperature (K)", cmap = "coolwarm", colorbar=True)
    plot_mesh(theta, title="Temperature field", label="Temperature (nondim)", cmap = "coolwarm", colorbar=True)
    plot_mesh(u, title="Velocity magnitude", label="Velocity (m/s)", cmap = "coolwarm", colorbar=True, mode="glyphs")
    plot_mesh(p, title="Pressure field", label="Pressure (Pa)", cmap = "coolwarm", colorbar=True)

    print(type(w.split()[0]))  # <class 'dolfin.function.function.Function'>
    save_experiment(OUTPUT_XDMF_PATH_AIR_P, sub_mesh, [p])
    save_experiment(OUTPUT_XDMF_PATH_AIR_V, sub_mesh, [u])
    save_experiment(OUTPUT_XDMF_PATH_AIR_T, sub_mesh, [T])
    # save_experiment(OUTPUT_XDMF_PATH_AIR_PVT, sub_mesh, [p,u,T])

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

    pass

def abs_version(experiment: Experiment):
    pass

def abs_temperature_dependent_version(experiment: Experiment):
    pass

def main():
    # Parse command line arguments
    argparser = argparse.ArgumentParser(description="Heated Laminar Plume Simulation")
    argparser.add_argument(
        "--experiment-index",
        type=int,
        default=0,
        help="Index of the experiment to run from experiments.json",
    )
    args = argparser.parse_args()
    args.experiment_index = max(0, args.experiment_index)
    experiment_list = parser(experiments_json_path=EXPERIMENTS_JSON_PATH, schema_json_path=SCHEMA_JSON_PATH)
    experiment = experiment_list[args.experiment_index]
    print(f"Running experiment: {experiment.name}")

    base_version(experiment)
    temperature_dependent_version(experiment)
    abs_version(experiment)
    abs_temperature_dependent_version(experiment)


if __name__ == "__main__":
    main()
