#!/usr/bin/env python
"""Build a fixed-size supervised dataset from trusted steady checkpoints."""
from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np

try:
    from .common import (
        INPUT_NAMES, TARGET_NAMES, case_features, case_geometry,
        length_reference, load_project_experiments, load_registry,
        ml_coordinates, resolve_checkpoint_dir,
    )
except (ImportError, ValueError):
    from common import (
        INPUT_NAMES, TARGET_NAMES, case_features, case_geometry,
        length_reference, load_project_experiments, load_registry,
        ml_coordinates, resolve_checkpoint_dir,
    )


def _fenics():
    try:
        import fenics
    except ImportError:
        import dolfin as fenics
    return fenics


def _build_mixed_space(fenics, mesh):
    p1 = fenics.FiniteElement("P", mesh.ufl_cell(), 1)
    p2 = fenics.VectorElement("P", mesh.ufl_cell(), 2)
    return fenics.FunctionSpace(mesh, fenics.MixedElement([p1, p2, p1]))


def load_checkpoint(checkpoint_dir):
    """Load fields using precisely the P1/P2/P1 checkpoint layout of solver.py."""
    fenics = _fenics()
    mesh = fenics.Mesh()
    h5 = fenics.HDF5File(mesh.mpi_comm(), os.path.join(checkpoint_dir, "state.h5"), "r")
    h5.read(mesh, "/mesh", False)
    mixed = _build_mixed_space(fenics, mesh)
    vp, _ = mixed.sub(0).collapse(True)
    vu, _ = mixed.sub(1).collapse(True)
    vt, _ = mixed.sub(2).collapse(True)
    pressure = fenics.Function(vp, name="p_star")
    velocity = fenics.Function(vu, name="u_star")
    theta = fenics.Function(vt, name="theta_star")
    h5.read(pressure, "/p_star")
    h5.read(velocity, "/u_star")
    h5.read(theta, "/theta_star")
    h5.close()
    with open(os.path.join(checkpoint_dir, "state.json"), "r") as stream:
        metadata = json.load(stream)
    mesh.bounding_box_tree().build(mesh)
    return mesh, theta, velocity, metadata


def uniform_mesh_points(mesh, count, rng, cell_mask=None):
    """Draw points uniformly by triangle area (not by mesh vertex density)."""
    coordinates = np.asarray(mesh.coordinates(), dtype=np.float64)
    triangles = np.asarray(mesh.cells(), dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Only two-dimensional triangular legacy FEniCS meshes are supported")
    vertices = coordinates[triangles, :2]
    twice_area = np.abs(
        (vertices[:, 1, 0] - vertices[:, 0, 0]) * (vertices[:, 2, 1] - vertices[:, 0, 1])
        - (vertices[:, 2, 0] - vertices[:, 0, 0]) * (vertices[:, 1, 1] - vertices[:, 0, 1])
    )
    if cell_mask is not None:
        cell_ids_available = np.flatnonzero(np.asarray(cell_mask, dtype=bool))
        vertices = vertices[cell_ids_available]
        twice_area = twice_area[cell_ids_available]
    else:
        cell_ids_available = np.arange(len(triangles), dtype=np.int64)
    if not np.any(twice_area > 0.0):
        raise ValueError("Checkpoint mesh contains no positive-area triangles")
    local_ids = rng.choice(len(vertices), size=int(count), p=twice_area / twice_area.sum())
    cell_ids = cell_ids_available[local_ids]
    r1 = np.sqrt(rng.random_sample(int(count)))
    r2 = rng.random_sample(int(count))
    selected_vertices = coordinates[triangles[cell_ids], :2]
    a = selected_vertices[:, 0]
    b = selected_vertices[:, 1]
    c = selected_vertices[:, 2]
    return (1.0 - r1)[:, None] * a + (r1 * (1.0 - r2))[:, None] * b + (r1 * r2)[:, None] * c


def regionally_sample(mesh, geometry, lref, count, rng, sampling):
    near_fraction = float(sampling.get("near_wire_fraction", 0.35))
    plume_fraction = float(sampling.get("plume_fraction", 0.35))
    if near_fraction < 0.0 or plume_fraction < 0.0 or near_fraction + plume_fraction > 1.0:
        raise ValueError("near_wire_fraction and plume_fraction must be nonnegative and sum to <= 1")
    coordinates = np.asarray(mesh.coordinates(), dtype=np.float64)
    triangles = np.asarray(mesh.cells(), dtype=np.int64)
    centroids = coordinates[triangles, :2].mean(axis=1)
    coords = ml_coordinates(centroids, geometry, lref)
    near_radius = float(sampling.get("near_wire_Rwire", 4.0))
    plume_half_width = float(sampling.get("plume_half_width_X", 0.12))
    y_wire = geometry["y_w"] / geometry["H"]
    cell_masks = [
        coords[:, 2] <= near_radius,
        (coords[:, 2] > near_radius) & (np.abs(coords[:, 0]) <= plume_half_width) & (coords[:, 1] >= y_wire),
        np.ones(len(triangles), dtype=bool),
    ]
    quotas = [int(round(count * near_fraction)), int(round(count * plume_fraction))]
    quotas.append(count - quotas[0] - quotas[1])
    chunks = []
    deficit = 0
    for cell_mask, quota in zip(cell_masks, quotas):
        quota += deficit
        if quota <= 0:
            deficit = 0
            continue
        if np.any(cell_mask):
            chunks.append(uniform_mesh_points(mesh, quota, rng, cell_mask=cell_mask))
            deficit = 0
        else:
            deficit = quota
    if deficit:
        chunks.append(uniform_mesh_points(mesh, deficit, rng))
    points = np.concatenate(chunks, axis=0)
    rng.shuffle(points)
    return points


def evaluate_targets(theta, velocity, points):
    values = np.empty((len(points), 3), dtype=np.float64)
    for row, point in enumerate(points):
        try:
            tval = theta(point)
            uval = velocity(point)
        except RuntimeError as exc:
            raise RuntimeError("FEniCS evaluation failed at star point %r" % point.tolist()) from exc
        values[row] = (float(tval), float(uval[0]), float(uval[1]))
    if not np.all(np.isfinite(values)):
        raise ValueError("Checkpoint fields produced NaN or infinite targets")
    return values


def build_case(case, experiment, registry_dir, samples_per_case, seed, sampling):
    checkpoint = resolve_checkpoint_dir(case, registry_dir)
    mesh, theta, velocity, metadata = load_checkpoint(checkpoint)
    geometry = case_geometry(experiment, case)
    lref = length_reference(experiment, case, geometry)
    features = case_features(experiment, case, geometry)
    rng = np.random.RandomState(int(seed))
    points = regionally_sample(mesh, geometry, lref, samples_per_case, rng, sampling)
    spatial = ml_coordinates(points, geometry, lref)
    inputs = np.column_stack((spatial, np.tile(features, (samples_per_case, 1))))
    targets = evaluate_targets(theta, velocity, points)
    return inputs, targets, checkpoint, metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=os.path.join(os.path.dirname(__file__), "converged_cases.json"))
    parser.add_argument("--output", required=True, help="Output compressed .npz file")
    parser.add_argument("--project-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
    parser.add_argument("--experiments-json")
    parser.add_argument("--schema-json")
    parser.add_argument("--samples-per-case", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.samples_per_case <= 0:
        raise ValueError("--samples-per-case must be positive")
    fenics = _fenics()
    if fenics.MPI.size(fenics.MPI.comm_world) != 1:
        raise RuntimeError("Dataset construction must be run with one MPI process")

    registry_path = os.path.abspath(args.registry)
    registry_dir = os.path.dirname(registry_path)
    registry = load_registry(registry_path)
    experiments_json = args.experiments_json or registry.get("experiments_json")
    schema_json = args.schema_json or registry.get("schema_json")
    if experiments_json and not os.path.isabs(experiments_json):
        experiments_json = os.path.abspath(os.path.join(registry_dir, experiments_json))
    if schema_json and not os.path.isabs(schema_json):
        schema_json = os.path.abspath(os.path.join(registry_dir, schema_json))
    experiments = load_project_experiments(
        args.project_root,
        experiments_json,
        schema_json,
    )
    default_sampling = dict(registry.get("sampling", {}))
    input_blocks, target_blocks, case_blocks, index_blocks = [], [], [], []
    checkpoint_paths, checkpoint_steps = [], []

    for ordinal, case in enumerate(registry["cases"]):
        experiment_index = int(case["experiment_index"])
        if experiment_index < 0 or experiment_index >= len(experiments):
            raise IndexError("experiment_index %d is outside [0, %d)" % (experiment_index, len(experiments)))
        case_id = str(case.get("case_id", "experiment_%d" % experiment_index))
        per_case_count = int(case.get("samples", args.samples_per_case))
        sampling = dict(default_sampling)
        sampling.update(case.get("sampling", {}))
        print("[%d/%d] %s" % (ordinal + 1, len(registry["cases"]), case_id))
        inputs, targets, checkpoint, metadata = build_case(
            case, experiments[experiment_index], registry_dir, per_case_count,
            args.seed + ordinal, sampling,
        )
        input_blocks.append(inputs)
        target_blocks.append(targets)
        case_blocks.append(np.full(per_case_count, case_id, dtype="U%d" % max(1, len(case_id))))
        index_blocks.append(np.full(per_case_count, experiment_index, dtype=np.int64))
        checkpoint_paths.append(checkpoint)
        checkpoint_steps.append(int(metadata.get("step", -1)))

    if not input_blocks:
        raise ValueError("Registry contains no cases")
    output = os.path.abspath(args.output)
    parent = os.path.dirname(output)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    np.savez_compressed(
        output,
        inputs=np.concatenate(input_blocks),
        targets=np.concatenate(target_blocks),
        case_id=np.concatenate(case_blocks),
        experiment_index=np.concatenate(index_blocks),
        input_names=INPUT_NAMES,
        target_names=TARGET_NAMES,
        checkpoint_path=np.asarray(checkpoint_paths, dtype="U"),
        checkpoint_step=np.asarray(checkpoint_steps, dtype=np.int64),
    )
    print("Wrote %d samples from %d cases to %s" % (sum(map(len, input_blocks)), len(input_blocks), output))


if __name__ == "__main__":
    main()
