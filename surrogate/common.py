"""Shared utilities for building the plume surrogate dataset.

This module deliberately keeps NumPy/JSON helpers independent of FEniCS.  FEniCS
is imported lazily by :mod:`surrogate.build_dataset`, so registry inspection and
feature calculations also work in a normal Python environment.
"""
from __future__ import print_function

import json
import math
import os
import sys

import numpy as np


INPUT_NAMES = np.asarray(
    ["X", "Y", "Rwire", "Pr", "log10(Gr_H^q)", "log10(K)", "A=H/W", "d/H", "y_w/H"]
)
TARGET_NAMES = np.asarray(["theta", "u_star", "v_star"])


def _value(obj, path, default=None):
    """Read a dotted path from nested objects and/or dictionaries."""
    cur = obj
    for name in path.split("."):
        if isinstance(cur, dict):
            if name not in cur:
                return default
            cur = cur[name]
        else:
            if not hasattr(cur, name):
                return default
            cur = getattr(cur, name)
    return cur


def _first(obj, paths, default=None):
    for path in paths:
        val = _value(obj, path, None)
        if val is not None:
            return val
    return default


def load_registry(path):
    with open(path, "r") as stream:
        registry = json.load(stream)
    if not isinstance(registry, dict) or not isinstance(registry.get("cases"), list):
        raise ValueError("Registry must be a JSON object containing a 'cases' list")
    return registry


def load_project_experiments(project_root, experiments_json=None, schema_json=None):
    """Load experiments with the same ``utils.parser.parser`` used by main.py."""
    project_root = os.path.abspath(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from utils.parser import parser

    if not experiments_json or not schema_json:
        try:
            from utils.imports import EXPERIMENTS_JSON_PATH, SCHEMA_JSON_PATH
            experiments_json = experiments_json or EXPERIMENTS_JSON_PATH
            schema_json = schema_json or SCHEMA_JSON_PATH
        except (ImportError, AttributeError):
            pass

    if not experiments_json or not schema_json:
        raise ValueError(
            "Could not infer experiment/schema JSON paths; provide them in the "
            "registry or with --experiments-json and --schema-json"
        )
    return parser(
        experiments_json_path=experiments_json,
        schema_json_path=schema_json,
    )


def resolve_checkpoint_dir(case, registry_dir):
    """Resolve an explicit checkpoint or find the steady checkpoint below run_root."""
    explicit = case.get("checkpoint_dir") or case.get("steady_checkpoint")
    run_root = case.get("run_root")

    def absolute(path, base):
        path = os.path.expanduser(str(path))
        return path if os.path.isabs(path) else os.path.abspath(os.path.join(base, path))

    if explicit:
        base = absolute(run_root, registry_dir) if run_root else registry_dir
        candidate = absolute(explicit, base)
        _validate_checkpoint(candidate)
        return candidate
    if not run_root:
        raise ValueError("Case needs either 'checkpoint_dir' or 'run_root'")

    root = absolute(run_root, registry_dir)
    matches = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "state.h5" in filenames and "state.json" in filenames:
            name = os.path.basename(dirpath).lower()
            if "steady" in name or "steady" in dirpath.lower():
                matches.append(os.path.abspath(dirpath))
    if not matches:
        raise IOError("No steady state.h5/state.json checkpoint found below %s" % root)

    preferred = [p for p in matches if os.path.basename(p) == "steady_from_transient_checkpoint"]
    candidates = preferred or matches
    if len(candidates) != 1:
        raise ValueError(
            "Found multiple steady checkpoints below %s; set checkpoint_dir explicitly:\n  %s"
            % (root, "\n  ".join(sorted(candidates)))
        )
    return candidates[0]


def _validate_checkpoint(path):
    missing = [name for name in ("state.h5", "state.json") if not os.path.isfile(os.path.join(path, name))]
    if missing:
        raise IOError("Checkpoint %s is missing %s" % (path, ", ".join(missing)))


def case_geometry(experiment, case):
    """Return physical W, H, wire diameter and wire centre.

    Per-case ``geometry`` values take precedence.  The y-wire fallback reproduces
    the geometry convention visible in the legacy solver (ymax/10 + 11*r).
    """
    override = case.get("geometry", {})
    domain = _first(experiment, ["dimensions.domain"], {})
    wire = _first(experiment, ["dimensions.wire"], {})

    x_max = float(override.get("x_max", _first(domain, ["x_max"])))
    y_max = float(override.get("y_max", _first(domain, ["y_max"])))
    x_min_raw = override.get("x_min", _first(domain, ["x_min"], None))
    y_min_raw = override.get("y_min", _first(domain, ["y_min"], None))
    x_min = float(-x_max if x_min_raw is None else x_min_raw)
    y_min = float(0.0 if y_min_raw is None else y_min_raw)
    width = float(override.get("W", x_max - x_min))
    height = float(override.get("H", y_max - y_min))
    diameter = float(override.get("d", _first(wire, ["diameter"])))

    x_w_default = _first(wire, ["x", "x_w", "x_center", "center_x"], 0.0)
    y_w_default = _first(wire, ["y", "y_w", "y_center", "center_y"], None)
    if y_w_default is None:
        y_w_default = y_max / 10.0 + 11.0 * diameter / 2.0
    x_w = float(override.get("x_w", x_w_default))
    y_w = float(override.get("y_w", y_w_default))

    for name, val in (("W", width), ("H", height), ("d", diameter)):
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError("Geometry value %s must be positive, got %r" % (name, val))
    return {"W": width, "H": height, "d": diameter, "x_w": x_w, "y_w": y_w}


def _line_heat_input(experiment):
    ic = _first(experiment, ["initial_conditions"], {})
    diameter = float(_first(experiment, ["dimensions.wire.diameter"]))
    q_line = _first(ic, ["heat_length"], None)
    if q_line is not None:
        return float(q_line)
    q_surface = _first(ic, ["heat_surface"], None)
    if q_surface is not None:
        return float(q_surface) * math.pi * diameter
    q_volume = _first(ic, ["heat_volume"], None)
    if q_volume is not None:
        return float(q_volume) * math.pi * (0.5 * diameter) ** 2
    raise ValueError("Experiment has no heat_length, heat_surface, or heat_volume")


def case_features(experiment, case, geometry):
    """Compute [Pr, log10(Gr_H^q), log10(K), A, d/H, y_w/H]."""
    override = case.get("features", {})
    props = _first(experiment, ["fluid.properties"], {})
    rho = float(_first(props, ["rho"]))
    mu = float(_first(props, ["mu"]))
    k_air = float(_first(props, ["k"]))
    cp = float(_first(props, ["cp"]))
    beta = float(_first(props, ["beta"]))
    gravity = float(_first(props, ["g"], 9.81))
    nu = mu / rho
    alpha = k_air / (rho * cp)
    pr = float(override.get("Pr", nu / alpha))

    q_line = float(override.get("Q_L", _line_heat_input(experiment)))
    gr_h_q = float(override.get("Gr_H_q", gravity * beta * q_line * geometry["H"] ** 3 / (k_air * nu ** 2)))

    k_wire = override.get("k_wire", _first(experiment, ["wire.properties.k"], None))
    if "K" in override:
        conductivity_ratio = float(override["K"])
    elif k_wire is not None and np.isscalar(k_wire):
        conductivity_ratio = float(k_wire) / k_air
    else:
        raise ValueError("Cannot infer K; set cases[].features.K in the registry")

    if gr_h_q <= 0.0 or conductivity_ratio <= 0.0:
        raise ValueError("Gr_H_q and K must be positive before log10 transformation")
    return np.asarray(
        [
            pr,
            math.log10(gr_h_q),
            math.log10(conductivity_ratio),
            geometry["H"] / geometry["W"],
            geometry["d"] / geometry["H"],
            geometry["y_w"] / geometry["H"],
        ],
        dtype=np.float64,
    )


def ml_coordinates(points_star, geometry, length_reference):
    """Convert solver-star mesh points to X, Y and diameter-scaled Rwire."""
    points_dim = np.asarray(points_star, dtype=np.float64)[:, :2] * float(length_reference)
    dx = points_dim[:, 0] - geometry["x_w"]
    dy = points_dim[:, 1] - geometry["y_w"]
    return np.column_stack(
        (dx / geometry["W"], points_dim[:, 1] / geometry["H"], np.sqrt(dx * dx + dy * dy) / geometry["d"])
    )


def length_reference(experiment, case, geometry):
    """Checkpoint mesh scaling; current legacy solver uses the wire radius."""
    return float(case.get("length_reference", 0.5 * geometry["d"]))
