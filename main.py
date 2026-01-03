from utils.imports import *
from utils.geometry import *
from utils.material import *
from utils.parser import *
from utils.plot import *
from solver.solver import *
from solver.initial import *
from solver.biot import *

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
    experiment_list = parser()
    experiment = experiment_list[args.experiment_index]

    GEOM_FILE = geometry_template(
        wire_radius=experiment.dimensions.wire.diameter / 2,
        output_path=experiment.name
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
    plot_mesh(mesh, title="Full Mesh")
    plot_mesh(sub_mesh, title="Sub-Mesh")

    # Initial guess for solver
    T_full = initial_guess(mesh,mc,mf,OUTPUT_XDMF_PATH_TEMP)
    qn_air = flux_continuity(T_full, mesh, sub_mesh, sub_ft, mc)

    # Solving the problem
    w = solver(sub_mesh, sub_dx, sub_ds, T_full, qn_air, experiment.material.rho_air, experiment.material.beta_air, experiment.environment.T_ambient)
    biot_air_h_eff, biot_air_Bi = biot(sub_mesh, sub_ft, T_full, qn_air, experiment.environment.T_ambient, experiment.material.k_wire, experiment.dimensions.wire.diameter) 


    p, u, T = w.split()

    plot_mesh(T, title="Temperature field", cmap = "coolwarm", colorbar=True)
    plot_mesh(u, title="Velocity magnitude", cmap = "coolwarm", colorbar=True, mode="glyphs")
    plot_mesh(p, title="Pressure field", cmap = "coolwarm", colorbar=True)

    print(type(w.split()[0]))  # <class 'dolfin.function.function.Function'>
    save_experiment(OUTPUT_XDMF_PATH_AIR_P, sub_mesh, [p])
    save_experiment(OUTPUT_XDMF_PATH_AIR_V, sub_mesh, [u])
    save_experiment(OUTPUT_XDMF_PATH_AIR_T, sub_mesh, [T])
    save_experiment(OUTPUT_XDMF_PATH_AIR_PVT, sub_mesh, [p,u,T])


if __name__ == "__main__":
    main()
