from operator import is_

from utils.imports import *

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    print0(set(cell_data))
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data.astype(np.int32)]})
    return out_mesh

def create_tagged_submesh_from_msh(
    msh,
    domain_tag,
    cell_type="triangle",
    facet_type="line",
    prune_z=True,
):
    """
    Extract a standalone meshio mesh for one physical cell tag.

    This is the MPI-safe replacement for doing

        fenics.SubMesh(mesh, mc, AIR_TAG)

    after the mesh has already been distributed by FEniCS.

    Rank 0 extracts the air mesh from the serial .msh file, writes it to XDMF,
    and then all MPI ranks collectively read the resulting standalone air mesh.
    """
    cells = msh.get_cells_type(cell_type)
    cell_tags = msh.get_cell_data("gmsh:physical", cell_type).astype(np.int32)

    keep_cells = cell_tags == int(domain_tag)
    if not np.any(keep_cells):
        raise RuntimeError(
            f"No {cell_type} cells found with physical tag {domain_tag}"
        )

    kept_cells_old = cells[keep_cells]
    kept_cell_tags = cell_tags[keep_cells]

    used_points = np.unique(kept_cells_old.reshape(-1))

    old_to_new = -np.ones(msh.points.shape[0], dtype=np.int64)
    old_to_new[used_points] = np.arange(len(used_points), dtype=np.int64)

    points = msh.points[used_points]
    if prune_z:
        points = points[:, :2]

    kept_cells_new = old_to_new[kept_cells_old]

    cell_mesh = meshio.Mesh(
        points=points,
        cells={cell_type: kept_cells_new},
        cell_data={"name_to_read": [kept_cell_tags]},
    )

    facet_cells = msh.get_cells_type(facet_type)
    facet_tags = msh.get_cell_data("gmsh:physical", facet_type).astype(np.int32)

    mapped_facets = old_to_new[facet_cells]
    keep_facets = np.all(mapped_facets >= 0, axis=1)

    kept_facets_new = mapped_facets[keep_facets]
    kept_facet_tags = facet_tags[keep_facets]

    if kept_facets_new.size == 0:
        raise RuntimeError(
            f"No {facet_type} facets retained for physical cell tag {domain_tag}. "
            "Check that your .geo defines Physical Line groups on the air boundary."
        )

    facet_mesh = meshio.Mesh(
        points=points,
        cells={facet_type: kept_facets_new},
        cell_data={"name_to_read": [kept_facet_tags]},
    )

    print0(f"Extracted physical cell tag {domain_tag}:")
    print0(f"  cell tags  = {set(kept_cell_tags.tolist())}")
    print0(f"  facet tags = {set(kept_facet_tags.tolist())}")

    return cell_mesh, facet_mesh

def save_experiment(OUTPUT_XDMF_PATH, mesh, sol_list, time_value=None):
    """
    Write one or more FEniCS Functions to XDMF.

    If ``time_value`` is provided, it is passed to ``XDMFFile.write(function, t)``
    so ParaView/VisIt see the snapshot as a time-labelled dataset instead of a
    timeless field.  This keeps the old call signature backwards-compatible: all
    existing calls without ``time_value`` behave as before.
    """
    encoding = XDMFFile.Encoding.ASCII
    xdmf = XDMFFile(MPI.comm_world, OUTPUT_XDMF_PATH)
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.write(mesh, encoding=encoding)

    for sol in sol_list:
        if time_value is None:
            xdmf.write(sol)
        else:
            xdmf.write(sol, float(time_value))

    xdmf.close()

    if MPI.comm_world.rank == 0:
        if time_value is None:
            print0("Solved heat equation on wire submesh. Output:", OUTPUT_XDMF_PATH)
        else:
            print0("Solved heat equation on wire submesh. Output:", OUTPUT_XDMF_PATH, "time=", float(time_value))

def generate_mesh(
    GEOM_FILE,
    MSH_FILE,
    TRIG_XDMF_PATH,
    FACETS_XDMF_PATH,
    AIR_TRIG_XDMF_PATH=None,
    AIR_FACETS_XDMF_PATH=None,
    AIR_TAG_VALUE=None,
    ELEM="triangle",
    PRUNE_Z=True,
):
    """
    MPI-safe mesh generation.

    Rank 0:
      - runs gmsh
      - converts full .msh to XDMF
      - optionally writes an AIR_TAG-only XDMF mesh

    All ranks:
      - wait at the final barrier before collective FEniCS reading.
    """
    comm = COMM
    rank = comm.rank

    if rank == 0:
        os.makedirs(os.path.dirname(str(MSH_FILE)), exist_ok=True)

        print0("Running gmsh...")
        subprocess.run(
            [
                "gmsh",
                "-2",
                str(GEOM_FILE),
                "-format",
                "msh2",
                "-o",
                str(MSH_FILE),
            ],
            check=True,
        )

        print0("Converting MSH to XDMF...")
        msh = meshio.read(MSH_FILE)

        element_mesh = create_mesh(msh, ELEM, prune_z=PRUNE_Z)
        facet_mesh = create_mesh(msh, "line", prune_z=PRUNE_Z)

        os.makedirs(os.path.dirname(str(TRIG_XDMF_PATH)), exist_ok=True)
        meshio.write(TRIG_XDMF_PATH, element_mesh)
        meshio.write(FACETS_XDMF_PATH, facet_mesh)

        if AIR_TRIG_XDMF_PATH is not None or AIR_FACETS_XDMF_PATH is not None:
            if AIR_TRIG_XDMF_PATH is None or AIR_FACETS_XDMF_PATH is None:
                raise ValueError(
                    "Both AIR_TRIG_XDMF_PATH and AIR_FACETS_XDMF_PATH must be provided."
                )
            if AIR_TAG_VALUE is None:
                raise ValueError("AIR_TAG_VALUE must be provided for air mesh extraction.")

            print0(f"Extracting air-only mesh with AIR_TAG={AIR_TAG_VALUE}...")
            air_element_mesh, air_facet_mesh = create_tagged_submesh_from_msh(
                msh,
                domain_tag=AIR_TAG_VALUE,
                cell_type=ELEM,
                facet_type="line",
                prune_z=PRUNE_Z,
            )

            os.makedirs(os.path.dirname(str(AIR_TRIG_XDMF_PATH)), exist_ok=True)
            meshio.write(AIR_TRIG_XDMF_PATH, air_element_mesh)
            meshio.write(AIR_FACETS_XDMF_PATH, air_facet_mesh)

    comm.Barrier()
    
def read_mesh(
    TRIG_XDMF_PATH,
    FACETS_XDMF_PATH,
    MESH_NAME="mesh",
    PRINT_TAG_SUMMARY=True,
):
    mesh = fenics.Mesh()

    with fenics.XDMFFile(MPI.comm_world, TRIG_XDMF_PATH) as xdmf:
        xdmf.read(mesh)

    tdim = mesh.topology().dim()

    mvc_ct = fenics.MeshValueCollection("size_t", mesh, tdim)
    with fenics.XDMFFile(MPI.comm_world, TRIG_XDMF_PATH) as xdmf:
        try:
            xdmf.read(mvc_ct, "name_to_read")
        except Exception:
            xdmf.read(mvc_ct)

    mvc_ft = fenics.MeshValueCollection("size_t", mesh, tdim - 1)
    with fenics.XDMFFile(MPI.comm_world, FACETS_XDMF_PATH) as xdmf:
        try:
            xdmf.read(mvc_ft, "name_to_read")
        except Exception:
            xdmf.read(mvc_ft)

    mc = fenics.cpp.mesh.MeshFunctionSizet(mesh, mvc_ct)
    mf = fenics.cpp.mesh.MeshFunctionSizet(mesh, mvc_ft)

    domains = fenics.MeshFunction("size_t", mesh, tdim)
    dx = fenics.Measure("dx", domain=mesh, subdomain_data=mc)
    boundary_markers = fenics.MeshFunction("size_t", mesh, tdim - 1)

    local_cell_tags = set(mc.array())
    local_facet_tags = set(mf.array()) - {18446744073709551615}

    all_cell_tags = COMM.allgather(local_cell_tags)
    all_facet_tags = COMM.allgather(local_facet_tags)

    ct = set().union(*all_cell_tags)
    ft = set().union(*all_facet_tags)

    if PRINT_TAG_SUMMARY and is_rank0():
        print0("Cell tags in the mesh:", ct)
        print0("Facet tags in the mesh:", ft)

    return mesh, ct, ft, domains, dx, boundary_markers, mc, mf

def create_submesh(mesh, mc, mf, tag):
    tdim = mesh.topology().dim()

    # # --- Preferred: MeshView (keeps parent mappings, very useful for transferring tags)
    # try:
    #     air_mesh = fenics.MeshView.create(mc, AIR_TAG)
    # except Exception:
    #     print0(" --- Fallback: SubMesh (works, but transferring facet tags is more manual")
    #     air_mesh = fenics.SubMesh(mesh, mc, AIR_TAG)
    
    air_mesh = SubMesh(mesh, mc, tag)
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

    print0(f"Submesh with tag {tag}: and facet tags {set(air_mf.array())}")

    return air_mesh, air_mf, dx_air, ds_air


def create_tagged_submesh_from_msh(
    msh,
    domain_tag,
    cell_type="triangle",
    facet_type="line",
    prune_z=True,
):
    """
    MPI-safe mesh extraction using meshio.

    Extracts all cells with physical tag == domain_tag from the serial gmsh mesh,
    preserves line facets whose endpoints belong to that extracted cell mesh,
    and returns two standalone meshio meshes:
      - cell mesh
      - facet mesh

    This is the replacement for FEniCS SubMesh(...) in MPI runs.
    """
    cells = msh.get_cells_type(cell_type)
    cell_tags = msh.get_cell_data("gmsh:physical", cell_type).astype(np.int32)

    keep_cells = cell_tags == int(domain_tag)
    if not np.any(keep_cells):
        raise RuntimeError(
            f"No {cell_type} cells found with physical tag {domain_tag}"
        )

    kept_cells_old = cells[keep_cells]
    kept_cell_tags = cell_tags[keep_cells]

    used_points = np.unique(kept_cells_old.reshape(-1))

    old_to_new = -np.ones(msh.points.shape[0], dtype=np.int64)
    old_to_new[used_points] = np.arange(len(used_points), dtype=np.int64)

    points = msh.points[used_points]
    if prune_z:
        points = points[:, :2]

    kept_cells_new = old_to_new[kept_cells_old]

    cell_mesh = meshio.Mesh(
        points=points,
        cells={cell_type: kept_cells_new},
        cell_data={"name_to_read": [kept_cell_tags]},
    )

    facet_cells = msh.get_cells_type(facet_type)
    facet_tags = msh.get_cell_data("gmsh:physical", facet_type).astype(np.int32)

    mapped_facets = old_to_new[facet_cells]
    keep_facets = np.all(mapped_facets >= 0, axis=1)

    kept_facets_new = mapped_facets[keep_facets]
    kept_facet_tags = facet_tags[keep_facets]

    if kept_facets_new.size == 0:
        raise RuntimeError(
            f"No {facet_type} facets retained for physical cell tag {domain_tag}. "
            "Check that your .geo defines Physical Line groups on the air boundary."
        )

    facet_mesh = meshio.Mesh(
        points=points,
        cells={facet_type: kept_facets_new},
        cell_data={"name_to_read": [kept_facet_tags]},
    )

    print0(f"Extracted physical cell tag {domain_tag}:")
    print0(f"  cell tags  = {set(kept_cell_tags.tolist())}")
    print0(f"  facet tags = {set(kept_facet_tags.tolist())}")

    return cell_mesh, facet_mesh

def geometry_template(
    wire_radius: float,
    output_path: str | Path,
    xmax: Optional[float] = None,
    ymax: Optional[float] = None,
    template_geo_name: str = "geom.geo",
    resolution: Optional[int] = 100
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

    if is_rank0():
        # Template is next to this script
        template_path = Path.cwd()/ template_geo_name
        print0(template_path)
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
            print0(f"Replacing xmax... {xmax}")
            geo, n3 = re.subn(
                r"(?m)^\s*w =\s*[0-9]+ \* R;",
                f"w = {float(xmax)};",
                geo,
                count=1,
            )
            # print0(n3)
            if n3 != 1:
                raise ValueError("Could not uniquely replace 'w = ...;' in the .geo template.")

        if ymax is not None and ymax != 0.0:
            print0(f"Replacing ymax... {ymax}")
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
        print0(output_path)
        # Decide where to write the modified .geo
        if output_path.suffix.lower() == ".geo":
            modified_geo_path = output_path
        else:
            raise ValueError("output_path must end with '.geo'")

        modified_geo_path.parent.mkdir(parents=True, exist_ok=True)
        modified_geo_path.write_text(geo, encoding="utf-8")

        result = modified_geo_path
    else:
        result = None

    result = COMM.bcast(result, root=0)

    COMM.Barrier()

    return result
