import argparse
import json
import os

import fenics
from mpi4py import MPI


def read_legacy_checkpoint(old_checkpoint_dir):
    h5_path = os.path.join(old_checkpoint_dir, "state.h5")
    meta_path = os.path.join(old_checkpoint_dir, "state.json")

    mesh = fenics.Mesh()
    h5 = fenics.HDF5File(mesh.mpi_comm(), h5_path, "r")
    h5.read(mesh, "/mesh", False)

    Q = fenics.FunctionSpace(mesh, "CG", 1)
    V = fenics.VectorFunctionSpace(mesh, "CG", 2)

    p = fenics.Function(Q, name="p_star_legacy")
    u = fenics.Function(V, name="u_star_legacy")
    theta = fenics.Function(Q, name="theta_star_legacy")

    h5.read(p, "/p_star")
    h5.read(u, "/u_star")
    h5.read(theta, "/theta_star")
    h5.close()

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        meta = {}

    return mesh, p, u, theta, meta


def read_new_air_mesh(air_cells_xdmf):
    mesh = fenics.Mesh()
    with fenics.XDMFFile(MPI.COMM_SELF, air_cells_xdmf) as xdmf:
        xdmf.read(mesh)
    return mesh


def migrate(old_checkpoint_dir, air_cells_xdmf, new_checkpoint_dir):
    old_mesh, old_p, old_u, old_theta, old_meta = read_legacy_checkpoint(
        old_checkpoint_dir
    )

    new_mesh = read_new_air_mesh(args.air_cells)
    scale_mesh_inplace(new_mesh, args.Lref)

    Q_new = fenics.FunctionSpace(new_mesh, "CG", 1)
    V_new = fenics.VectorFunctionSpace(new_mesh, "CG", 2)

    p_new = fenics.interpolate(old_p, Q_new)
    u_new = fenics.interpolate(old_u, V_new)
    theta_new = fenics.interpolate(old_theta, Q_new)

    os.makedirs(new_checkpoint_dir, exist_ok=True)

    h5_path = os.path.join(new_checkpoint_dir, "state.h5")
    meta_path = os.path.join(new_checkpoint_dir, "state.json")

    h5 = fenics.HDF5File(new_mesh.mpi_comm(), h5_path, "w")
    h5.write(new_mesh, "/mesh")
    h5.write(p_new, "/p_star")
    h5.write(u_new, "/u_star")
    h5.write(theta_new, "/theta_star")
    h5.close()

    meta = {
        "restart_format": "airmesh_v1",
        "migrated_from": "legacy_serial_submesh",
        "old_checkpoint_dir": old_checkpoint_dir,
        "air_cells_xdmf": air_cells_xdmf,
        "step": int(old_meta.get("step", 0)),
        "time": float(old_meta.get("time", 0.0)),
        "dt": float(old_meta.get("dt", 0.0)),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Migration complete.")
    print(f"  old checkpoint: {old_checkpoint_dir}")
    print(f"  air mesh:       {air_cells_xdmf}")
    print(f"  new checkpoint: {new_checkpoint_dir}")
    print(f"  theta min/max:  {theta_new.vector().min()} {theta_new.vector().max()}")

def scale_mesh_inplace(mesh, Lref):
    mesh.coordinates()[:] /= float(Lref)
    mesh.bounding_box_tree().build(mesh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--air-cells", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--Lref", type=float, required=True)
    args = parser.parse_args()

    migrate(
        old_checkpoint_dir=args.old_checkpoint,
        air_cells_xdmf=args.air_cells,
        new_checkpoint_dir=args.new_checkpoint,
    )
#     python migrate_legacy_restart_to_airmesh.py \
#   --old-checkpoint /path/to/old/checkpoint \
#   --air-cells /path/to/new/run/air_cells.xdmf \
#   --new-checkpoint /path/to/new/run/restart_from_legacy
