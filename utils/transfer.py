from utils.imports import *

import fenics


def scale_mesh_inplace(mesh, Lref):
    mesh.coordinates()[:] /= float(Lref)
    mesh.bounding_box_tree().build(mesh)


def transfer_cg1_function(src_fun, dst_mesh):
    """
    Transfer a CG1 scalar field from src_fun.mesh() to dst_mesh.
    Uses compiled FEniCS interpolation instead of Python DOF loops.
    """
    V_dst = fenics.FunctionSpace(dst_mesh, "CG", 1)
    src_fun.set_allow_extrapolation(True)
    dst_fun = fenics.interpolate(src_fun, V_dst)
    dst_fun.vector().apply("insert")
    return dst_fun


def transfer_dg0_function(src_fun, dst_mesh):
    """
    Transfer a DG0 scalar field from src_fun.mesh() to dst_mesh.
    """
    V_dst = fenics.FunctionSpace(dst_mesh, "DG", 0)
    src_fun.set_allow_extrapolation(True)
    dst_fun = fenics.interpolate(src_fun, V_dst)
    dst_fun.vector().apply("insert")
    return dst_fun


def build_star_submesh_and_transfer(
    mesh,
    mc,
    mf,
    air_tag,
    theta_full_dim,
    qn_air,
):
    """
    Assumes:
      - mesh has already been scaled to star coordinates
      - theta_full_dim lives on the already-scaled old air submesh
      - qn_air lives on the already-scaled old air submesh / interface-compatible mesh

    Returns:
      sub_mesh_star, sub_ft_star, sub_dx_star, sub_ds_star,
      theta_full_star, qn_air_star
    """
    from utils.geometry import create_submesh

    sub_mesh_star, sub_ft_star, sub_dx_star, sub_ds_star = create_submesh(mesh, mc, mf, air_tag)

    theta_full_star = transfer_cg1_function(theta_full_dim, sub_mesh_star)
    theta_full_star.rename("theta_full", "theta_full")

    qn_air_star = transfer_dg0_function(qn_air, sub_mesh_star)

    return sub_mesh_star, sub_ft_star, sub_dx_star, sub_ds_star, theta_full_star, qn_air_star
