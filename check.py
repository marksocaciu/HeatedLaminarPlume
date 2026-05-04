import argparse
import json
import os
import fenics
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_dir")
args = parser.parse_args()

h5_path = os.path.join(args.checkpoint_dir, "state.h5")
meta_path = os.path.join(args.checkpoint_dir, "state.json")

mesh = fenics.Mesh()
h5 = fenics.HDF5File(MPI.COMM_SELF, h5_path, "r")
h5.read(mesh, "/mesh", False)

Q = fenics.FunctionSpace(mesh, "CG", 1)
V = fenics.VectorFunctionSpace(mesh, "CG", 2)

p = fenics.Function(Q)
u = fenics.Function(V)
theta = fenics.Function(Q)

h5.read(p, "/p_star")
h5.read(u, "/u_star")
h5.read(theta, "/theta_star")
h5.close()

u_mag = fenics.project(fenics.sqrt(fenics.inner(u, u)), Q)

print("mesh cells:", mesh.num_cells())
print("mesh vertices:", mesh.num_vertices())
print("p min/max:", p.vector().min(), p.vector().max())
print("|u| min/max:", u_mag.vector().min(), u_mag.vector().max())
print("theta min/max:", theta.vector().min(), theta.vector().max())

if os.path.exists(meta_path):
    with open(meta_path) as f:
        print(json.dumps(json.load(f), indent=2))
