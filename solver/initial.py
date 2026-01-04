from utils.imports import *
from utils.geometry import *
from utils.material import *
from utils.plot import *
from fenics import *
from dolfin import *

def initial_guess(mesh,mc,mf, OUTPUT_XDMF_PATH_TEMP, heat_volume) -> fenics.Function:
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

    k_vals = np.full(mc.array().shape, k_air, dtype=float)
    k_vals[mc.array() == WIRE_TAG] = k_of_T(T_ambient)

    q_vals = np.full(mc.array().shape, q_air, dtype=float)
    q_vals[mc.array() == WIRE_TAG] = heat_volume
    # Assign into DG0 functions
    k_func.vector()[:] = k_vals
    q_func.vector()[:] = q_vals

    # -----------------------------------------
    # Boundary data
    # -----------------------------------------
    T_inf = fenics.Constant(T_ambient)
    h     = fenics.Constant(h_conv)

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
            "linear_solver": "lu"
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
            "linear_solver": "lu"
        })


        # convergence check on temperature (choose your norm)
        diff = (T_full_new.vector() - T_full.vector()).norm("l2")
        norm = T_full.vector().norm("l2") + 1e-14
        rel  = diff / norm

        print(f"[material loop {it}] rel ||ΔT|| = {rel:.3e}")

        if rel < rtol:
            break

        T_full.assign(T_full_new)



    T_full.rename("T_conduction_full", "")
    print(max(T_full.vector()))

    # -----------------------------------------
    # Save result
    # -----------------------------------------
    save_experiment(OUTPUT_XDMF_PATH_TEMP, mesh, [T_full])
    plot_mesh(T_full, title="Temperature Distribution in Wire and Air", cmap = "coolwarm", colorbar=True)

    return T_full, k_func

def flux_continuity(T_full: fenics.Function,
                    k_func: fenics.Function,
                    mesh: fenics.Mesh,
                    sub_mesh: fenics.Mesh,
                    sub_ft: fenics.MeshFunction,
                    mc: fenics.MeshFunction) -> fenics.Function:
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

    n_eps = 1e-10  # will be scaled below (we’ll improve this using h)

    # A small shift distance into the wire, based on local mesh size
    h_air = CellDiameter(sub_mesh)

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

        qn = -k_w * (gT[0]*n_air[0] + gT[1]*n_air[1])

        # Accumulate into the adjacent air cell (DG0)
        qn_air.vector()[c_air_idx] += qn
        counts[c_air_idx] += 1.0

    # Average per cell
    qn_air.vector()[:] = qn_air.vector().get_local() / np.maximum(counts.get_local(), 1.0)
    return qn_air
