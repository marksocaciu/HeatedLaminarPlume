from utils.imports import *

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    print(set(cell_data))
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data.astype(np.int32)]})
    return out_mesh

def save_experiment(OUTPUT_XDMF_PATH, mesh, sol_list):
    encoding = XDMFFile.Encoding.ASCII
    xdmf = XDMFFile(MPI.comm_world, OUTPUT_XDMF_PATH)
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.write(mesh, encoding=encoding)

    for sol in sol_list:
        xdmf.write(sol)

    if MPI.comm_world.rank == 0:
        print("Solved heat equation on wire submesh. Output:", OUTPUT_XDMF_PATH)

def generate_mesh(GEOM_FILE, MSH_FILE,
                  TRIG_XDMF_PATH, FACETS_XDMF_PATH,
                  ELEM="triangle", PRUNE_Z=True):

    subprocess.run(f"gmsh -nopopup -nt 12 {GEOM_FILE}", shell=True, check=True)

    print("Converting MSH to XDMF...")
    msh = meshio.read(MSH_FILE)

    element_mesh = create_mesh(msh, ELEM, prune_z=PRUNE_Z)
    facet_mesh   = create_mesh(msh, "line", prune_z=PRUNE_Z)

    meshio.write(TRIG_XDMF_PATH, element_mesh)
    meshio.write(FACETS_XDMF_PATH, facet_mesh)

def read_mesh(TRIG_XDMF_PATH, FACETS_XDMF_PATH,
              MESH_NAME="mesh", PRINT_TAG_SUMMARY=True):

    # -------------------------
    # Read cell mesh
    # -------------------------
    mesh = fenics.Mesh()
    with fenics.XDMFFile(MPI.comm_world, TRIG_XDMF_PATH) as xdmf:
        xdmf.read(mesh)  # reads geometry + topology

    # create cell MeshFunction
    tdim = mesh.topology().dim()
    mvc_ct = fenics.MeshValueCollection("size_t", mesh, tdim)
    mvc_ft = fenics.MeshValueCollection("size_t", mesh, tdim)
    with fenics.XDMFFile(MPI.comm_world, TRIG_XDMF_PATH) as xdmf:
        xdmf.read(mvc_ct)  # reads geometry + topology

    with fenics.XDMFFile(MPI.comm_world, FACETS_XDMF_PATH) as xdmf:
        xdmf.read(mvc_ft)  # reads facet tags
        
    mf = fenics.cpp.mesh.MeshFunctionSizet(mesh, mvc_ft)
    mc = fenics.cpp.mesh.MeshFunctionSizet(mesh, mvc_ct)
    domains = fenics.MeshFunction("size_t", mesh, tdim)
    dx = fenics.Measure("dx",domain=mesh, subdomain_data=mf)
    boundary_markers = fenics.MeshFunction("size_t", mesh, tdim - 1)
    
    # -------------------------
    # Print summary
    # -------------------------
    if PRINT_TAG_SUMMARY and MPI.comm_world.rank == 0:
        ct = set(mc.array())
        print("Cell tags in the mesh:", ct)
        ft = set(mf.array())
        ft = ft - {18446744073709551615}  # remove default tag 18446744073709551615
        print("Facet tags in the mesh:", ft)

    return mesh, ct, ft, domains, dx, boundary_markers, mc, mf

def create_submesh(mesh, mc, mf, tag):
    tdim = mesh.topology().dim()

    # # --- Preferred: MeshView (keeps parent mappings, very useful for transferring tags)
    # try:
    #     air_mesh = fenics.MeshView.create(mc, AIR_TAG)
    # except Exception:
    #     print(" --- Fallback: SubMesh (works, but transferring facet tags is more manual")
    #     air_mesh = fenics.SubMesh(mesh, mc, AIR_TAG)
    
    air_mesh = SubMesh(mesh, mc, AIR_TAG)
    air_mesh.init(tdim-1, tdim)  # ensure facet-cell connectivity exists
    air_mf = MeshFunction("size_t", air_mesh, tdim-1, 0)

    parent_cell_indices = air_mesh.data().array("parent_cell_indices", tdim)

    mesh.init(tdim - 1, tdim)
    mesh.init(tdim, tdim - 1)

    for f in facets(air_mesh):
        # Pick the (single) adjacent air cell
        c_air = list(cells(f))[0]
        c_air_index = c_air.index()

        # Corresponding parent cell
        c_parent_index = parent_cell_indices[c_air_index]
        c_parent = Cell(mesh, c_parent_index)

        # Find matching local facet
        for local_f in range(c_parent.num_entities(tdim - 1)):
            parent_facet = Facet(mesh, c_parent.entities(tdim - 1)[local_f])
            if parent_facet.midpoint().distance(f.midpoint()) < DOLFIN_EPS:
                air_mf[f] = mf[parent_facet]
                break

    
    dx_air = Measure("dx", domain=air_mesh)
    ds_air = Measure("ds", domain=air_mesh, subdomain_data=air_mf)

    return air_mesh, air_mf, dx_air, ds_air




def geometry_template(
    wire_radius: float,
    output_path: str | Path,
    xmax: Optional[float] = None,
    ymax: Optional[float] = None,
    template_geo_name: str = "geom.geo",
    resolution: Optional[int] = 150
) -> Path:
    """
    Load a .geo template located in the same directory as this script, set the wire radius
    by updating `R_placeholder`, write a modified .geo to disk, and optionally generate a .msh.

    Parameters
    ----------
    wire_radius:
        Value to assign to `R_placeholder`.
    output_path:
        If ends with ".geo": write the modified .geo there and do NOT mesh.
        If ends with ".msh": generate a mesh and write it there; also write a sibling ".geo"
        next to it (same stem) for traceability.
    template_geo_name:
        Filename of the template .geo located next to this script.
    resolution:
        If provided, overwrites `resolution_placeholder`.
    mesh_dim:
        Mesh dimension to generate (2 for your geometry).
    smoothing:
        If provided, sets Mesh.Smoothing (integer). If None, do not override.
    verbose:
        If True, prints Gmsh messages to the terminal.

    Returns
    -------
    (modified_geo_path, msh_path_or_none)
    """
    output_path = Path(output_path)

    # Template is next to this script
    template_path = Path.cwd()/ template_geo_name
    print(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Template .geo not found: {template_path}")

    geo = template_path.read_text(encoding="utf-8")

    # Replace placeholders (anchored to beginning of line for safety)
    geo, n1 = re.subn(
        r"(?m)^\s*R_placeholder\s*=\s*[^;]*;",
        f"R_placeholder = {wire_radius};",
        geo,
        count=1,
    )
    if n1 != 1:
        raise ValueError("Could not uniquely replace 'R_placeholder = ...;' in the .geo template.")

    if resolution is not None:
        geo, n2 = re.subn(
            r"(?m)^\s*resolution_placeholder\s*=\s*[^;]*;",
            f"resolution_placeholder = {int(resolution)};",
            geo,
            count=1,
        )
        if n2 != 1:
            raise ValueError("Could not uniquely replace 'resolution_placeholder = ...;' in the .geo template.")

    if xmax is not None and xmax != 0.0:
        print(f"Replacing xmax... {xmax}")
        geo, n3 = re.subn(
            r"(?m)^\s*w =\s*[0-9]+ \* R;",
            f"w = {float(xmax)};",
            geo,
            count=1,
        )
        print(n3)
        if n3 != 1:
            raise ValueError("Could not uniquely replace 'w = ...;' in the .geo template.")

    if ymax is not None and ymax != 0.0:
        print(f"Replacing ymax... {ymax}")
        geo, n4 = re.subn(
            r"(?m)^\s*h =\s*[0-9]+ \* R;",
            f"h = {float(ymax)};",
            geo,
            count=1,
        )
        if n4 != 1:
            raise ValueError("Could not uniquely replace 'h = ...;' in the .geo template.")

    # Strip directives that are inconvenient/dangerous when using the Python API
    # (we will generate and write from Python)
    # geo = re.sub(r"(?m)^\s*Mesh\s+\d+\s*;\s*$", "", geo)
    # geo = re.sub(r'(?m)^\s*Save\s+"[^"]*"\s*;\s*$', "", geo)
    # geo = re.sub(r"(?m)^\s*Exit\s*;\s*$", "", geo)

    output_path = Path.cwd()/ output_path / "geom.geo"
    print(output_path)
    # Decide where to write the modified .geo
    if output_path.suffix.lower() == ".geo":
        modified_geo_path = output_path
    else:
        raise ValueError("output_path must end with '.geo'")

    modified_geo_path.parent.mkdir(parents=True, exist_ok=True)
    modified_geo_path.write_text(geo, encoding="utf-8")

    return modified_geo_path
