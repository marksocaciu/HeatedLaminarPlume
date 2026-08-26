#!/usr/bin/env python
"""Generate a restart-compatible LegacyPlume state from a learned surrogate.

Run this with the same legacy FEniCS 2019.2 environment used by main.py.  The
script is deliberately serial: checkpoint construction is an offline operation,
and serial DOF assignment avoids partition-dependent learned-state files.
"""

from __future__ import print_function

import argparse
import os

import numpy as np
import fenics
from mpi4py import MPI

from solver.amr import (assign_split_to_mixed, build_mixed_space_on_mesh,
                        write_checkpoint_with_mesh)
from solver.params_bcs import set_bcs
from solver.scales import compute_nondimensional_scales
from utils.geometry import read_mesh
from utils.parser import parser
from utils.transfer import scale_mesh_inplace

from surrogate.model import NpzSurrogate


def _nested(obj, path, default=None):
    value = obj
    for part in path.split("."):
        if isinstance(value, dict):
            if part not in value:
                return default
            value = value[part]
        else:
            if not hasattr(value, part):
                return default
            value = getattr(value, part)
    return value


def _first_number(experiment, paths, default=None):
    for path in paths:
        value = _nested(experiment, path)
        if value is not None:
            return float(value)
    if default is not None:
        return float(default)
    raise KeyError("Experiment has none of: %s" % ", ".join(paths))


def experiment_feature_values(experiment, scales):
    """Return named constant controls available to the learned model."""
    diameter = _first_number(experiment, ("dimensions.wire.diameter",))
    values = {
        "pr": scales.Pr,
        "ra": scales.Ra,
        "gr": scales.Gr,
        "gr_wire": scales.Gr_wire,
        "wire_diameter": diameter,
        "diameter": diameter,
        "lref": scales.Lref,
        "dtref": scales.dTref,
        "uref": scales.Uref,
        "uplume": scales.Uplume,
        "q_line": scales.QL,
        "ql": scales.QL,
        "q_surface": scales.qsurf,
        "x_min_star": _first_number(
            experiment, ("dimensions.domain.x_min",), 0.0) / scales.Lref,
        "x_max_star": _first_number(
            experiment, ("dimensions.domain.x_max",)) / scales.Lref,
        "y_min_star": _first_number(
            experiment, ("dimensions.domain.y_min",), 0.0) / scales.Lref,
        "y_max_star": _first_number(
            experiment, ("dimensions.domain.y_max",)) / scales.Lref,
    }
    # Common training-script spellings.
    values.update({"Pr": values["pr"], "Ra": values["ra"],
                   "Gr": values["gr"], "Gr_wire": values["gr_wire"],
                   "QL": values["ql"]})
    return values


def build_features(coords, feature_names, experiment, scales):
    constants = experiment_feature_values(experiment, scales)
    x_star, y_star = coords[:, 0], coords[:, 1]
    coordinate_values = {
        "x": x_star, "y": y_star, "x_star": x_star, "y_star": y_star,
        "x_hat": x_star, "y_hat": y_star,
    }
    columns = []
    unknown = []
    for name in feature_names:
        name_lower = name.lower()
        if name_lower in coordinate_values:
            columns.append(coordinate_values[name_lower])
        elif name in constants and constants[name] is not None:
            columns.append(np.full(coords.shape[0], constants[name], dtype=float))
        elif name_lower in constants and constants[name_lower] is not None:
            columns.append(np.full(coords.shape[0], constants[name_lower], dtype=float))
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            "Cannot construct surrogate features %s from this experiment. "
            "Rename them to supported coordinate/control names or add them to "
            "experiment_feature_values()." % ", ".join(unknown)
        )
    return np.column_stack(columns)


def _predict_column(model, coords, experiment, scales, aliases, default=None):
    output = model.predict(build_features(
        coords, model.feature_names, experiment, scales))
    index = model.target_index(*aliases)
    if index is None:
        if default is None:
            raise ValueError("Surrogate targets do not contain any of: %s" %
                             ", ".join(aliases))
        return np.full(coords.shape[0], float(default), dtype=float)
    return output[:, index]


def _function_from_values(space, values, name):
    function = fenics.Function(space, name=name)
    local_size = function.vector().local_size()
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != local_size:
        raise RuntimeError("%s prediction count %d != local DOF count %d" %
                           (name, values.size, local_size))
    function.vector().set_local(values)
    function.vector().apply("insert")
    return function


def predict_state(mesh_star, experiment, model, clip_theta=True):
    W = build_mixed_space_on_mesh(mesh_star)
    Vp, _ = W.sub(0).collapse(True)
    Vu, _ = W.sub(1).collapse(True)
    VT, _ = W.sub(2).collapse(True)
    Vux, _ = Vu.sub(0).collapse(True)
    Vuy, _ = Vu.sub(1).collapse(True)
    scales = compute_nondimensional_scales(experiment)

    def coords(space):
        return space.tabulate_dof_coordinates().reshape(
            (-1, mesh_star.geometry().dim()))

    p_values = _predict_column(
        model, coords(Vp), experiment, scales,
        ("p_star", "p", "pressure_star"), default=0.0)
    ux_values = _predict_column(
        model, coords(Vux), experiment, scales,
        ("u_star", "ux_star", "u", "ux"))
    uy_values = _predict_column(
        model, coords(Vuy), experiment, scales,
        ("v_star", "uy_star", "v", "uy"))
    theta_values = _predict_column(
        model, coords(VT), experiment, scales,
        ("theta_star", "theta", "temperature_star"))
    if clip_theta:
        theta_values = np.maximum(theta_values, 0.0)

    p_star = _function_from_values(Vp, p_values, "p_star")
    ux_star = _function_from_values(Vux, ux_values, "ux_star")
    uy_star = _function_from_values(Vuy, uy_values, "uy_star")
    u_star = fenics.Function(Vu, name="u_star")
    fenics.FunctionAssigner(Vu, [Vux, Vuy]).assign(
        u_star, [ux_star, uy_star])
    theta_star = _function_from_values(VT, theta_values, "theta_star")

    w_n = assign_split_to_mixed(W, p_star, u_star, theta_star)
    # Reuse the solver's geometric no-slip, ambient-theta, and pressure-pin BCs.
    for bc in set_bcs(W, None, theta_star, 0.0, experiment, scales):
        bc.apply(w_n.vector())
    w_n.vector().apply("insert")
    return W, w_n


def _select_experiment(experiments, index, name):
    if name:
        matches = [exp for exp in experiments if str(exp.name) == name]
        if len(matches) != 1:
            raise ValueError("Expected one experiment named %r; found %d" %
                             (name, len(matches)))
        return matches[0]
    if index < 0 or index >= len(experiments):
        raise IndexError("Experiment index %d outside [0, %d)" %
                         (index, len(experiments)))
    return experiments[index]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments-json", required=True)
    ap.add_argument("--schema-json", required=True)
    choice = ap.add_mutually_exclusive_group(required=True)
    choice.add_argument("--experiment-index", type=int)
    choice.add_argument("--experiment")
    ap.add_argument("--model", required=True)
    ap.add_argument("--air-cells",
                    help="Dimensional air-only cells XDMF for this experiment")
    ap.add_argument("--air-facets",
                    help="Matching dimensional air-only facets XDMF")
    ap.add_argument("--mesh-run-root",
                    help="Generate a new experiment mesh here using main.py's "
                         "coarse-remesh utility instead of supplying XDMFs")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dt", type=float, default=1.0e-6)
    ap.add_argument("--allow-negative-theta", action="store_true")
    args = ap.parse_args()

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Generate surrogate checkpoints in serial (no mpirun)")
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive")
    supplied_mesh = bool(args.air_cells or args.air_facets)
    if supplied_mesh and not (args.air_cells and args.air_facets):
        raise ValueError("--air-cells and --air-facets must be supplied together")
    if supplied_mesh == bool(args.mesh_run_root):
        raise ValueError("Supply either --mesh-run-root or the air XDMF pair")

    experiments = parser(
        experiments_json_path=args.experiments_json,
        schema_json_path=args.schema_json,
    )
    experiment = _select_experiment(
        experiments, args.experiment_index if args.experiment_index is not None else 0,
        args.experiment)
    scales = compute_nondimensional_scales(experiment)
    model = NpzSurrogate(args.model)

    if args.mesh_run_root:
        # Import lazily because main.py loads the complete solver stack.
        from main import generate_coarse_remesh_files
        air_cells, air_facets = generate_coarse_remesh_files(
            experiment=experiment,
            coarse_run_root=args.mesh_run_root,
        )
    else:
        air_cells, air_facets = args.air_cells, args.air_facets

    mesh, _, _, _, _, _, _, _ = read_mesh(
        air_cells, air_facets, "Grid", True)
    scale_mesh_inplace(mesh, float(scales.Lref))
    mesh.bounding_box_tree().build(mesh)

    _, w_n = predict_state(
        mesh, experiment, model, clip_theta=not args.allow_negative_theta)
    meta = {
        "step": 0,
        "time": 0.0,
        "dt": float(args.dt),
        "source": "surrogate_initial_guess",
        "experiment": str(experiment.name),
        "surrogate_model": os.path.abspath(args.model),
        "surrogate_features": model.feature_names,
        "surrogate_targets": model.target_names,
        "air_cells_source": os.path.abspath(air_cells),
        "air_facets_source": os.path.abspath(air_facets),
        "mesh_coordinates": "nondimensional_by_Lref",
        "Lref": float(scales.Lref),
        "dTref": float(scales.dTref),
        "Uref": float(scales.Uref),
    }
    write_checkpoint_with_mesh(args.output, mesh, w_n, meta)
    print("Wrote surrogate restart checkpoint to %s" % args.output)


if __name__ == "__main__":
    main()
