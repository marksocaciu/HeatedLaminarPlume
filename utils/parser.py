# experiment_loader.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import json


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class Material:
    material_name: str
    properties: Dict[str, Any]
    model: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Domain2D:
    """
    2D domain in (x, y).

    If you interpret this as axisymmetric, you may use:
      x = r (radial), y = z (axial)
    but the parser does not enforce that semantic; it only enforces domain.type.
    """
    type: Literal["2D_symmetric"]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class WireGeometry:
    diameter: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Dimensions:
    domain: Domain2D
    wire: WireGeometry
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Mesh:
    source: str
    tags: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitialConditions:
    # Keep these aligned to whatever you store in JSON; unknown keys go to .extra
    temperature: Optional[float] = None
    wire_length: Optional[float] = None
    heat_volume: Optional[float] = None
    heat_surface: Optional[float] = None
    heat_length: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeasurementPoint:
    x: float
    y: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Experiment:
    # Required
    name: str
    author: str
    fluid: Material
    wire: Material
    dimensions: Dimensions

    # Optional (if present in JSON)
    notes: Optional[str] = None
    mesh: Optional[Mesh] = None
    initial_conditions: Optional[InitialConditions] = None
    measurement_points: List[MeasurementPoint] = field(default_factory=list)

    # Preserve any additional keys
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Defaults:
    units: Dict[str, Any] = field(default_factory=dict)
    symmetry: Dict[str, Any] = field(default_factory=dict)
    gravity: List[float] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Utilities
# -----------------------------
def _split_known_unknown(d: Dict[str, Any], known_keys: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    known, unknown = {}, {}
    for k, v in d.items():
        (known if k in known_keys else unknown)[k] = v
    return known, unknown


def _as_float(d: Dict[str, Any], key: str) -> float:
    if key not in d:
        raise ValueError(f"Missing required key '{key}'")
    return float(d[key])


# -----------------------------
# Optional schema validation
# -----------------------------
def validate_against_schema(
    payload: Dict[str, Any],
    schema: Dict[str, Any],
    *,
    strict: bool = False
) -> List[str]:
    """
    Returns a list of human-readable validation errors (empty if none).

    If jsonschema is not installed, returns [].
    If strict=True and errors exist, raises ValueError.
    """
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return []

    errors_out: List[str] = []

    Validator = getattr(jsonschema, "Draft202012Validator", None)
    if Validator is None:
        try:
            jsonschema.validate(instance=payload, schema=schema)  # type: ignore
        except Exception as e:
            errors_out.append(str(e))
        if errors_out and strict:
            raise ValueError("\n".join(errors_out))
        return errors_out

    validator = Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    for e in errors:
        path = "$" + "".join([f"[{p}]" if isinstance(p, int) else f".{p}" for p in e.path])
        errors_out.append(f"{path}: {e.message}")

    if errors_out and strict:
        raise ValueError("Schema validation failed:\n" + "\n".join(errors_out))
    return errors_out


# -----------------------------
# Parsing functions
# -----------------------------
def parse_material(d: Dict[str, Any]) -> Material:
    known, extra = _split_known_unknown(d, ["material_name", "model", "properties"])
    if "material_name" not in known:
        raise ValueError("Material missing 'material_name'")
    if "properties" not in known or not isinstance(known["properties"], dict):
        raise ValueError("Material missing object 'properties'")
    return Material(
        material_name=str(known["material_name"]),
        model=str(known["model"]) if "model" in known and known["model"] is not None else None,
        properties=dict(known["properties"]),
        extra=extra,
    )


def parse_domain(d: Dict[str, Any]) -> Domain2D:
    """
    Option A: enforce that domain.type == "2D_symmetric".
    """
    known, extra = _split_known_unknown(d, ["type", "coordinates"])

    if "type" not in known:
        raise ValueError("Domain missing 'type'")

    domain_type = str(known["type"])
    if domain_type != "2D_symmetric":
        raise ValueError(f"Unsupported domain.type='{domain_type}'. Expected '2D_symmetric'.")

    coords = known.get("coordinates", None)
    if not isinstance(coords, dict):
        raise ValueError("Domain missing object 'coordinates'")

    x_min = _as_float(coords, "x_min")
    x_max = _as_float(coords, "x_max")
    y_min = _as_float(coords, "y_min")
    y_max = _as_float(coords, "y_max")

    _, coords_extra = _split_known_unknown(coords, ["x_min", "x_max", "y_min", "y_max"])
    extra = {**extra, "coordinates_extra": coords_extra}

    return Domain2D(
        type="2D_symmetric",
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        extra=extra,
    )


def parse_dimensions(d: Dict[str, Any]) -> Dimensions:
    known, extra = _split_known_unknown(d, ["domain", "wire"])
    if "domain" not in known or not isinstance(known["domain"], dict):
        raise ValueError("Dimensions missing object 'domain'")
    if "wire" not in known or not isinstance(known["wire"], dict):
        raise ValueError("Dimensions missing object 'wire'")

    domain = parse_domain(known["domain"])

    wire_known, wire_extra = _split_known_unknown(known["wire"], ["diameter"])
    if "diameter" not in wire_known:
        raise ValueError("Wire geometry missing required key 'diameter'")

    wire = WireGeometry(diameter=float(wire_known["diameter"]), extra=wire_extra)
    return Dimensions(domain=domain, wire=wire, extra=extra)


def parse_mesh(d: Dict[str, Any]) -> Mesh:
    known, extra = _split_known_unknown(d, ["source", "tags"])
    if "source" not in known:
        raise ValueError("Mesh missing 'source'")
    if "tags" not in known or not isinstance(known["tags"], dict):
        raise ValueError("Mesh missing object 'tags'")
    return Mesh(source=str(known["source"]), tags=dict(known["tags"]), extra=extra)


def parse_initial_conditions(d: Dict[str, Any]) -> InitialConditions:
    known_keys = ["temperature", "wire_length", "heat_volume", "heat_surface", "heat_length"]
    known, extra = _split_known_unknown(d, known_keys)
    return InitialConditions(
        temperature=float(known["temperature"]) if "temperature" in known else None,
        wire_length=float(known["wire_length"]) if "wire_length" in known else None,
        heat_volume=float(known["heat_volume"]) if "heat_volume" in known else None,
        heat_surface=float(known["heat_surface"]) if "heat_surface" in known else None,
        heat_length=float(known["heat_length"]) if "heat_length" in known else None,
        extra=extra,
    )


def parse_measurement_points(arr: Any) -> List[MeasurementPoint]:
    if arr is None:
        return []
    if not isinstance(arr, list):
        raise ValueError("'measurement_points' must be an array")
    out: List[MeasurementPoint] = []
    for i, p in enumerate(arr):
        if not isinstance(p, dict):
            raise ValueError(f"measurement_points[{i}] must be an object")
        known, extra = _split_known_unknown(p, ["x", "y"])
        if "x" not in known or "y" not in known:
            raise ValueError(f"measurement_points[{i}] must contain keys 'x' and 'y'")
        out.append(MeasurementPoint(x=float(known["x"]), y=float(known["y"]), extra=extra))
    return out


def parse_experiment(d: Dict[str, Any]) -> Experiment:
    required = ["name", "author", "fluid", "wire", "dimensions"]
    for k in required:
        if k not in d:
            raise ValueError(f"Experiment missing required key '{k}'")

    known_keys = [
        "name",
        "author",
        "notes",
        "fluid",
        "wire",
        "dimensions",
        "mesh",
        "initial_conditions",
        "measurement_points",
    ]
    known, extra = _split_known_unknown(d, known_keys)

    fluid = parse_material(known["fluid"])
    wire = parse_material(known["wire"])
    dims = parse_dimensions(known["dimensions"])

    mesh_obj = None
    if "mesh" in known and known["mesh"] is not None:
        if not isinstance(known["mesh"], dict):
            raise ValueError("'mesh' must be an object")
        mesh_obj = parse_mesh(known["mesh"])

    ic_obj = None
    if "initial_conditions" in known and known["initial_conditions"] is not None:
        if not isinstance(known["initial_conditions"], dict):
            raise ValueError("'initial_conditions' must be an object")
        ic_obj = parse_initial_conditions(known["initial_conditions"])

    mpoints = parse_measurement_points(known.get("measurement_points"))

    return Experiment(
        name=str(known["name"]),
        author=str(known["author"]),
        notes=str(known["notes"]) if "notes" in known and known["notes"] is not None else None,
        fluid=fluid,
        wire=wire,
        dimensions=dims,
        mesh=mesh_obj,
        initial_conditions=ic_obj,
        measurement_points=mpoints,
        extra=extra,
    )


def parse_defaults(d: Any) -> Defaults:
    if not isinstance(d, dict):
        return Defaults()
    known, extra = _split_known_unknown(d, ["units", "symmetry", "gravity"])
    units = dict(known["units"]) if isinstance(known.get("units"), dict) else {}
    symmetry = dict(known["symmetry"]) if isinstance(known.get("symmetry"), dict) else {}
    gravity = list(known["gravity"]) if isinstance(known.get("gravity"), list) else []
    return Defaults(units=units, symmetry=symmetry, gravity=gravity, extra=extra)


# -----------------------------
# Public loader
# -----------------------------
def load_experiment_config(
    experiments_json_path: Union[str, Path],
    schema_json_path: Optional[Union[str, Path]] = None,
    *,
    validate: bool = True,
    strict_schema: bool = False
) -> Tuple[str, Defaults, List[Experiment], List[str]]:
    """
    Returns (config_version, defaults, experiments, schema_errors)

    - If validate=True and schema_json_path is provided, attempts JSON Schema validation.
    - If strict_schema=True, raises on schema errors; otherwise returns errors and continues parsing.
    """
    experiments_json_path = Path(experiments_json_path)
    payload = json.loads(experiments_json_path.read_text(encoding="utf-8"))

    schema_errors: List[str] = []
    if validate and schema_json_path is not None:
        schema_json_path = Path(schema_json_path)
        schema = json.loads(schema_json_path.read_text(encoding="utf-8"))
        schema_errors = validate_against_schema(payload, schema, strict=strict_schema)

    if "config_version" not in payload:
        raise ValueError("Top-level key 'config_version' is required")
    if "experiments" not in payload or not isinstance(payload["experiments"], list):
        raise ValueError("Top-level key 'experiments' must be an array")

    defaults = parse_defaults(payload.get("defaults", {}))
    experiments = [parse_experiment(e) for e in payload["experiments"]]
    return str(payload["config_version"]), defaults, experiments, schema_errors


# -----------------------------
# CLI
# -----------------------------
def parser() -> List[Experiment]:
    import argparse

    parser = argparse.ArgumentParser(description="Load experiments.json into Experiment objects.")
    parser.add_argument("experiments_json", type=str, help="Path to experiments.json")
    parser.add_argument("--schema", type=str, default=None, help="Path to experiments.schema.json (optional)")
    parser.add_argument("--no-validate", action="store_true", help="Disable schema validation")
    parser.add_argument("--strict-schema", action="store_true", help="Fail if schema validation reports errors")
    args = parser.parse_args()

    cfg_version, defaults, exps, schema_errors = load_experiment_config(
        args.experiments_json,
        schema_json_path=args.schema,
        validate=not args.no_validate,
        strict_schema=args.strict_schema,
    )

    print(f"config_version: {cfg_version}")
    print(f"defaults.gravity: {defaults.gravity}")
    print(f"loaded experiments: {len(exps)}")

    if schema_errors:
        print("\nSchema validation issues (parsing continued):")
        for e in schema_errors:
            print(f" - {e}")

    for i, e in enumerate(exps, start=1):
        d = e.dimensions.domain
        w = e.dimensions.wire
        print(
            f"[{i}] {e.name} ({e.author}) | domain={d.type} "
            f"| x:[{d.x_min}, {d.x_max}] y:[{d.y_min}, {d.y_max}] | wire_d={w.diameter}"
        )
        
    return exps


