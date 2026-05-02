from fenics import *
import numpy as np


def mark_bad_cells(mesh, thresholds=(0.20, 0.10, 0.05), output="mesh_quality.xdmf"):
    """
    Mark cells by mesh quality using the radius ratio.

    Marker convention:
        0 : good cell
        1 : radius_ratio < thresholds[0]
        2 : radius_ratio < thresholds[1]
        3 : radius_ratio < thresholds[2]

    Lower radius ratio = worse cell quality.
    """

    print0("=== Basic mesh info ===")
    print0(f"Topological dim : {mesh.topology().dim()}")
    print0(f"Geometric dim   : {mesh.geometry().dim()}")
    print0(f"# cells         : {mesh.num_cells()}")
    print0(f"# vertices      : {mesh.num_vertices()}")
    print0(f"hmin            : {mesh.hmin():.6e}")
    print0(f"hmax            : {mesh.hmax():.6e}")

    rr = MeshQuality.radius_ratios(mesh).array()
    qmin, qmax = MeshQuality.radius_ratio_min_max(mesh)

    print0("\n=== Radius ratio stats ===")
    print0(f"min    : {qmin:.6e}")
    print0(f"max    : {qmax:.6e}")
    print0(f"mean   : {np.mean(rr):.6e}")
    print0(f"median : {np.median(rr):.6e}")

    for t in thresholds:
        nbad = np.sum(rr < t)
        print0(f"cells with radius_ratio < {t:4.2f}: {nbad} / {len(rr)}")

    # Cell markers for visualization
    cell_markers = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

    # Optional DG0 field with the actual radius ratio values
    V0 = FunctionSpace(mesh, "DG", 0)
    rr_fun = Function(V0)
    rr_fun.vector()[:] = rr

    t1, t2, t3 = thresholds

    for cell in cells(mesh):
        q = rr[cell.index()]
        marker = 0
        if q < t1:
            marker = 1
        if q < t2:
            marker = 2
        if q < t3:
            marker = 3
        cell_markers[cell] = marker

    print0("\n=== Marker legend ===")
    print0("0 : good")
    print0(f"1 : radius_ratio < {t1}")
    print0(f"2 : radius_ratio < {t2}")
    print0(f"3 : radius_ratio < {t3}")

    # Write both the markers and the raw radius ratio field
    with XDMFFile(mesh.mpi_comm(), output) as xdmf:
        xdmf.parameters["flush_output"] = True
        xdmf.parameters["functions_share_mesh"] = True
        xdmf.write(mesh)
        xdmf.write(cell_markers)
        xdmf.write(rr_fun, 0.0)

    print0(f"\nWrote mesh quality data to: {output}")

    return cell_markers, rr_fun


if __name__ == "__main__":
    # Choose one of these loading methods

    # XML:
    # mesh = Mesh("mesh.xml")

    # XDMF:
    mesh = Mesh()
    with XDMFFile("PlumeCase_Fuji_Air/runs/base_20260422_125618_pid21155/plume.xdmf") as infile:
        infile.read(mesh)

    mark_bad_cells(
        mesh,
        thresholds=(0.20, 0.10, 0.05),
        output="mesh_quality.xdmf"
    )
