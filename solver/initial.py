from utils.imports import *
from utils.geometry import *
from utils.material import *
from utils.plot import *
from fenics import *
from dolfin import *
from solver.scales import *

def initial_guess(mesh,mc,mf, OUTPUT_XDMF_PATH_TEMP, heat_volume, experiment,dx) -> Tuple[fenics.Function, fenics.Function]:
    # -----------------------------------------
    # Function spaces
    # -----------------------------------------
    V_T_full = fenics.FunctionSpace(mesh, "Lagrange", 1)

    T = fenics.TrialFunction(V_T_full)
    v = fenics.TestFunction(V_T_full)

    # DG0 fields for k and q
    V0 = fenics.FunctionSpace(mesh, "DG", 0)
    k_func = fenics.Function(V0)
    q_func = fenics.Function(V0)

    # -----------------------------------------
    # Fill cellwise values (using MeshFunction ct)
    # -----------------------------------------
    # ct: MeshFunction("size_t", mesh, mesh.topology().dim())
    # ct.array() gives tag per cell in order of mesh.cells()

    k_vals = np.full(mc.array().shape, experiment.fluid.properties["k"], dtype=float)
    k_vals[mc.array() == WIRE_TAG] = k_of_T(T_ambient)

    q_vals = np.full(mc.array().shape, experiment.fluid.properties["q"], dtype=float)
    q_vals[mc.array() == WIRE_TAG] = heat_volume
    # Assign into DG0 functions
    k_func.vector()[:] = k_vals
    q_func.vector()[:] = q_vals

    # -----------------------------------------
    # Boundary data
    # -----------------------------------------
    T_inf = fenics.Constant(experiment.initial_conditions.temperature)
    h     = fenics.Constant(experiment.wire.properties["h"])

    # -----------------------------------------
    # Measures
    # -----------------------------------------
    dx = fenics.Measure("dx", domain=mesh, subdomain_data=mc)
    ds = fenics.Measure("ds", domain=mesh, subdomain_data=mf)   # mf = MeshFunction for facets

    # -----------------------------------------
    # Weak form
    # -----------------------------------------
    a_T = (k_func * inner(grad(T), grad(v))) * dx \
        + h * T * v * ds(OUTER_AIR_TAG)

    L_T = (q_func * v) * dx \
        + h * T_inf * v * ds(OUTER_AIR_TAG)

    # -----------------------------------------
    # Optional: no Dirichlet BCs
    # -----------------------------------------
    bcs_T = []

    # -----------------------------------------
    # Solve
    # -----------------------------------------
    T_full = fenics.Function(V_T_full)
    fenics.solve(a_T == L_T, T_full, bcs_T,
        solver_parameters={
            "linear_solver": "mumps"
        })

    for it in range(max_it):
        T_full_new =  fenics.Function(V_T_full)
        h.assign(h_of_T(T_full.vector().max(),T_ambient,D_wire))

        k_vals[mc.array() == WIRE_TAG] = k_of_T(T_full.vector().max())   # updates DG0 mu/Pr/... on sub_mesh
        k_func.vector()[:] = k_vals

        a_T = (k_func * inner(grad(T), grad(v))) * dx \
        + h * T * v * ds(OUTER_AIR_TAG)

        L_T = (q_func * v) * dx \
        + h * T_inf * v * ds(OUTER_AIR_TAG)

        fenics.solve(a_T == L_T, T_full_new, bcs_T,
        solver_parameters={
            "linear_solver": "mumps"
        })


        # convergence check on temperature (choose your norm)
        diff = (T_full_new.vector() - T_full.vector()).norm("l2")
        norm = T_full.vector().norm("l2") + 1e-14
        rel  = diff / norm

        print(f"[material loop {it}] rel ||ΔT|| = {rel:.3e}")

        T_full.assign(T_full_new)
        if rel < rtol:
            break




    T_full.rename("T_conduction_full", "")
    print(max(T_full.vector()))

    # -----------------------------------------
    # Save result
    # -----------------------------------------
    save_experiment(OUTPUT_XDMF_PATH_TEMP, mesh, [T_full])
    # plot_mesh(T_full, title="Temperature Distribution in Wire and Air", cmap = "coolwarm", colorbar=True)

    return T_full, k_func

def flux_continuity(T_full: fenics.Function,
                    k_func: fenics.Function,
                    mesh: fenics.Mesh,
                    sub_mesh: fenics.Mesh,
                    sub_ft: fenics.MeshFunction,
                    mc: fenics.MeshFunction, 
                    sc: NondimScales) -> fenics.Function:
    # -----------------------------------------
    # preparing for flux continuity
    # -----------------------------------------
    Vg = fenics.VectorFunctionSpace(mesh, "DG", 0)
    gradT_DG0 = fenics.project(grad(T_full), Vg)   # T_full from your conduction solve

    wire_mesh = fenics.SubMesh(mesh, mc, WIRE_TAG)
    bbt_wire = fenics.BoundingBoxTree()
    bbt_wire.build(wire_mesh)

    V0_air = FunctionSpace(sub_mesh, "DG", 0)
    qn_air = Function(V0_air)
    qn_air.vector().zero()

    # To average if a cell touches the interface via multiple facets
    counts = qn_air.vector().copy()
    counts.zero()

    tdim = sub_mesh.topology().dim()
    sub_mesh.init(tdim-1, tdim)
    mesh.init(tdim-1, tdim)

    for f in facets(sub_mesh):
        if sub_ft[f] != INTERFACE_TAG:
            continue

        c_air = list(cells(f))[0]
        c_air_idx = c_air.index()

        # Air outward normal on this interface facet (points from air to wire if your submesh is air-only)
        n_air = f.normal().array()
        n_air /= np.linalg.norm(n_air)

        x = f.midpoint().array()

        # Shift slightly into the wire side to avoid ambiguity exactly on the interface
        # Use a tiny length scale based on air cell diameter
        eps = 1e-6 * float(c_air.circumradius())
        x_in_wire = x + eps * n_air  # outward from air

        # Find which wire cell contains this shifted point
        # (returns (cell_id, distance); if not found, it may return a large distance)
        cid = bbt_wire.compute_first_entity_collision(Point(*x_in_wire))

        if cid < 0:
            # If normal orientation is opposite, try the other side
            x_in_wire = x - eps * n_air
            cid = bbt_wire.compute_first_entity_collision(Point(*x_in_wire))
            if cid < 0:
                continue  # give up on this facet

        c_wire = Cell(wire_mesh, cid)

        # Map wire submesh cell back to parent mesh cell index
        parent_wire = wire_mesh.data().array("parent_cell_indices", mesh.topology().dim())
        parent_cid = int(parent_wire[c_wire.index()])
        c_parent = Cell(mesh, parent_cid)

        # Evaluate cellwise grad(T) and k in that parent cell (DG0 values)
        gT = gradT_DG0(c_parent.midpoint())
        k_w = k_func(c_parent.midpoint())

        # qn = -k_w * (gT[0]*n_air[0] + gT[1]*n_air[1])
        qn_dim = k_w * (gT[0]*n_air[0] + gT[1]*n_air[1])
        qn_star = qn_dim * (sc.Lref / (k_air * sc.dTref))
        qn = qn_star

        # Accumulate into the adjacent air cell (DG0)
        qn_air.vector()[c_air_idx] += qn
        counts[c_air_idx] += 1.0

    # Average per cell
    qn_air.vector()[:] = qn_air.vector().get_local() / np.maximum(counts.get_local(), 1.0)
    return qn_air

# initial.py

from utils.imports import *
from utils.geometry import *
from utils.material import *
from utils.plot import *
from fenics import *
from dolfin import *
from solver.scales import *

def build_cellwise_k_q_fields(mesh, mc, experiment, heat_volume, T_wire_ref=None):
    """
    Build DG0 cellwise conductivity/source fields on the full parent mesh.

    Parameters
    ----------
    mesh : parent mesh
    mc   : cell markers on parent mesh
    experiment
    heat_volume : volumetric heating [W/m^3] applied only in WIRE_TAG cells
    T_wire_ref : optional scalar temperature [K] used for wire conductivity update

    Returns
    -------
    k_func : DG0 Function [W/m/K]
    q_func : DG0 Function [W/m^3]
    """
    V0 = fenics.FunctionSpace(mesh, "DG", 0)
    k_func = fenics.Function(V0, name="k_full")
    q_func = fenics.Function(V0, name="q_full")

    k_air = float(experiment.fluid.properties["k"])
    q_air = float(experiment.fluid.properties.get("q", 0.0))

    if T_wire_ref is None:
        T_wire_ref = float(experiment.initial_conditions.temperature)

    # NOTE:
    # This mirrors your current approach: one scalar wire conductivity evaluation.
    # Later you may want a local wire-temperature-dependent update.
    k_wire = float(k_of_T(T_wire_ref))

    cell_tags = mc.array()
    k_vals = np.full(cell_tags.shape, k_air, dtype=float)
    q_vals = np.full(cell_tags.shape, q_air, dtype=float)

    k_vals[cell_tags == WIRE_TAG] = k_wire
    q_vals[cell_tags == WIRE_TAG] = float(heat_volume)

    k_func.vector()[:] = k_vals
    q_func.vector()[:] = q_vals
    k_func.vector().apply("insert")
    q_func.vector().apply("insert")

    return k_func, q_func


def solve_full_temperature(
    mesh,
    mc,
    mf,
    experiment,
    heat_volume,
    output_xdmf_path=None,
    u_full=None,
    include_convection=False,
    T_prev=None,
    pseudo_dt=None,
    max_material_iters=5,
    material_tol=1.0e-8,
):
    """
    Solve the scalar temperature problem on the FULL parent mesh:

        - div(k grad T) + u·grad(T) = q'''    in air+wire,
        with convection ONLY in AIR_TAG cells.

    This is the central thermal solve of the new conjugate structure.

    Parameters
    ----------
    mesh, mc, mf : full parent mesh + markers
    experiment
    heat_volume : wire volumetric heating [W/m^3]
    output_xdmf_path : optional path for output
    u_full : vector field on parent mesh, zero in wire, air velocity in air
    include_convection : whether to include convection on AIR_TAG
    T_prev : optional previous iterate for pseudo-time stabilization
    pseudo_dt : optional pseudo-time parameter (dimensional or artificial)
    max_material_iters : outer fixed-point update count for wire k and h
    material_tol : convergence tolerance for thermal fixed-point loop

    Returns
    -------
    T_full : CG1 Function on parent mesh [K]
    k_func : DG0 conductivity field
    """
    V_T = fenics.FunctionSpace(mesh, "CG", 1)
    T = fenics.TrialFunction(V_T)
    v = fenics.TestFunction(V_T)

    dx = fenics.Measure("dx", domain=mesh, subdomain_data=mc)
    ds = fenics.Measure("ds", domain=mesh, subdomain_data=mf)

    T_inf = fenics.Constant(float(experiment.initial_conditions.temperature))
    h_bc = fenics.Constant(float(experiment.wire.properties["h"]))

    # initial guess for conductivity
    T_wire_ref = float(experiment.initial_conditions.temperature)
    k_func, q_func = build_cellwise_k_q_fields(
        mesh=mesh,
        mc=mc,
        experiment=experiment,
        heat_volume=heat_volume,
        T_wire_ref=T_wire_ref,
    )

    T_full = fenics.Function(V_T, name="T_full")
    if T_prev is not None:
        T_full.assign(T_prev)
    else:
        T_full.interpolate(T_inf)

    # If convection is requested but no parent-mesh velocity is provided,
    # fall back to zero convection.
    if u_full is None:
        V_u = fenics.VectorFunctionSpace(mesh, "CG", 1)
        u_full = fenics.Function(V_u, name="u_full_zero")
        u_full.vector().zero()
        u_full.vector().apply("insert")
        include_convection = False

    bcs_T = []  # keep as in your current formulation unless you later choose Dirichlet far-field

    for it in range(max_material_iters):
        T_old = fenics.Function(V_T)
        T_old.assign(T_full)

        # Optional update of wire-side effective h, mirroring your current style
        # based on the current maximum temperature.
        T_max = float(T_old.vector().max())
        T_amb = float(experiment.initial_conditions.temperature)
        D_wire = float(experiment.dimensions.wire.diameter)
        h_bc.assign(float(h_of_T(T_max, T_amb, D_wire)))

        # Update wire conductivity
        k_func, q_func = build_cellwise_k_q_fields(
            mesh=mesh,
            mc=mc,
            experiment=experiment,
            heat_volume=heat_volume,
            T_wire_ref=T_max,
        )

        # Base conduction term over full domain
        a_T = k_func * inner(grad(T), grad(v)) * dx

        # Convection only in AIR_TAG
        if include_convection:
            a_T += dot(u_full, grad(T)) * v * dx(AIR_TAG)

        # Keep your current outer Robin closure
        a_T += h_bc * T * v * ds(OUTER_AIR_TAG)

        L_T = q_func * v * dx + h_bc * T_inf * v * ds(OUTER_AIR_TAG)

        # Optional pseudo-time stabilization:
        #   (T - T_prev)/pseudo_dt
        if pseudo_dt is not None and T_prev is not None:
            dtau = fenics.Constant(float(pseudo_dt))
            a_T += (1.0 / dtau) * T * v * dx
            L_T += (1.0 / dtau) * T_prev * v * dx

        fenics.solve(
            a_T == L_T,
            T_full,
            bcs_T,
            solver_parameters={"linear_solver": "mumps"},
        )

        diff = T_full.vector().copy()
        diff.axpy(-1.0, T_old.vector())
        rel = diff.norm("l2") / (T_full.vector().norm("l2") + 1.0e-14)

        print(f"[full thermal] material iter {it:02d} | rel_update = {rel:.3e}")

        if rel < material_tol:
            break

    T_full.rename("T_full", "T_full")

    if output_xdmf_path:
        try:
            with fenics.XDMFFile(mesh.mpi_comm(), output_xdmf_path) as xdmf:
                xdmf.write(mesh)
                xdmf.write(T_full)
        except Exception as err:
            print(f"[full thermal] warning: could not write XDMF output: {err}")

    return T_full, k_func
