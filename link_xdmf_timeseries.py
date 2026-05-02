#!/usr/bin/env python3
"""
Create a temporal XDMF collection from a directory of per-step XDMF/HDF5 files.

Typical use case:
    You have many XDMF files such as
        temperature_0000.xdmf
        temperature_0001.xdmf
        temperature_0002.xdmf
    each referencing its own HDF5 heavy-data file, and you want one new XDMF file
    that exposes them as a single time evolution.

This script does *not* rewrite the HDF5 data. It builds a lightweight XDMF wrapper
    <Grid GridType="Collection" CollectionType="Temporal">
that points to the existing per-step XDMF metadata / HDF5 heavy data.

It is intentionally conservative and works by:
  1. parsing each source .xdmf,
  2. locating the first Uniform grid,
  3. copying Topology / Geometry / Attribute nodes,
  4. inserting or overriding a <Time Value="..."/> node,
  5. writing one combined output .xdmf.

Assumptions:
  - all source XDMF files describe the same mesh / topology layout,
  - each source file contains at least one Uniform grid,
  - heavy-data references are stored in standard XDMF DataItem text,
  - the source files are in the same directory tree as the output, or at least the
    referenced HDF5 paths remain valid after relativization.

Examples
--------
Combine all matching files, infer time from filename integer:
    python link_xdmf_timeseries.py ./results 'temperature_*.xdmf' temperature_series.xdmf

Combine with explicit times from filenames like field_t0.005.xdmf:
    python link_xdmf_timeseries.py ./results 'field_t*.xdmf' series.xdmf --time-regex 't([0-9.eE+-]+)'

Use times already stored in the source XDMF files:
    python link_xdmf_timeseries.py ./results 'field_*.xdmf' series.xdmf --prefer-source-time

Use a constant dt when filenames only encode order:
    python link_xdmf_timeseries.py ./results 'u_*.xdmf' u_series.xdmf --dt 0.02
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

XDMF_NS = ""
XML_NS = "http://www.w3.org/XML/1998/namespace"
XI_NS = "http://www.w3.org/2001/XInclude"


def strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def find_first_child(parent: ET.Element, name: str) -> Optional[ET.Element]:
    for child in parent:
        if strip_ns(child.tag) == name:
            return child
    return None


def iter_descendants(parent: ET.Element, name: str) -> Iterable[ET.Element]:
    for elem in parent.iter():
        if strip_ns(elem.tag) == name:
            yield elem


def find_first_uniform_grid(root: ET.Element) -> Optional[ET.Element]:
    for grid in iter_descendants(root, "Grid"):
        if grid.attrib.get("GridType", "Uniform") == "Uniform":
            return grid
    return None


def parse_time_from_grid(grid: ET.Element) -> Optional[float]:
    time_elem = find_first_child(grid, "Time")
    if time_elem is None:
        return None
    value = time_elem.attrib.get("Value")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_time_from_filename(path: Path, regex: Optional[re.Pattern[str]]) -> Optional[float]:
    stem = path.stem
    if regex is not None:
        m = regex.search(stem)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", stem)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def relativize_dataitems(elem: ET.Element, src_xdmf: Path, out_xdmf: Path) -> None:
    """
    Rewrite heavy-data paths inside DataItem text so they stay valid from the new output file.

    Example of XDMF heavy-data text:
        somefile.h5:/VisualisationVector/0
    """
    for data_item in iter_descendants(elem, "DataItem"):
        if data_item.text is None:
            continue
        text = data_item.text.strip()
        if not text or ":" not in text:
            continue

        left, right = text.split(":", 1)
        # Skip likely inline XML or non-file values.
        if not left.endswith((".h5", ".hdf5", ".bin")):
            continue

        src_data_path = (src_xdmf.parent / left).resolve()
        rel = os.path.relpath(src_data_path, start=out_xdmf.parent.resolve())
        data_item.text = f"{rel}:{right}"


def make_temporal_grid_from_source(
    src_xdmf: Path,
    out_xdmf: Path,
    time_value: float,
    grid_name: Optional[str] = None,
) -> ET.Element:
    tree = ET.parse(src_xdmf)
    root = tree.getroot()

    src_grid = find_first_uniform_grid(root)
    if src_grid is None:
        raise ValueError(f"No Uniform Grid found in {src_xdmf}")

    new_grid = ET.Element("Grid", {
        "Name": grid_name or src_grid.attrib.get("Name", src_xdmf.stem),
        "GridType": "Uniform",
    })

    new_grid.append(ET.Element("Time", {"Value": f"{time_value:.16g}"}))

    for child_name in ("Topology", "Geometry"):
        child = find_first_child(src_grid, child_name)
        if child is None:
            raise ValueError(f"Missing {child_name} in {src_xdmf}")
        child_copy = copy.deepcopy(child)
        relativize_dataitems(child_copy, src_xdmf, out_xdmf)
        new_grid.append(child_copy)

    attributes_found = 0
    for child in src_grid:
        if strip_ns(child.tag) != "Attribute":
            continue
        attr_copy = copy.deepcopy(child)
        relativize_dataitems(attr_copy, src_xdmf, out_xdmf)
        new_grid.append(attr_copy)
        attributes_found += 1

    if attributes_found == 0:
        raise ValueError(f"No Attribute nodes found in {src_xdmf}")

    return new_grid


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def build_output_tree(grids: List[ET.Element]) -> ET.ElementTree:
    xdmf = ET.Element("Xdmf", {"Version": "3.0"})
    domain = ET.SubElement(xdmf, "Domain")
    collection = ET.SubElement(domain, "Grid", {
        "Name": "TimeSeries",
        "GridType": "Collection",
        "CollectionType": "Temporal",
    })
    for grid in grids:
        collection.append(grid)
    indent_xml(xdmf)
    return ET.ElementTree(xdmf)


def collect_files(input_dir: Path, pattern: str) -> List[Path]:
    paths = [Path(p) for p in glob.glob(str(input_dir / pattern))]
    paths = [p for p in paths if p.suffix.lower() == ".xdmf"]
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a temporal XDMF collection from per-step XDMF files.")
    parser.add_argument("input_dir", type=Path, help="Directory containing source .xdmf files")
    parser.add_argument("pattern", help="Glob pattern for source files, e.g. 'temperature_*.xdmf'")
    parser.add_argument("output", type=Path, help="Output .xdmf file")
    parser.add_argument(
        "--time-regex",
        default=None,
        help=r"Regex with one capture group for time from filename stem, e.g. 't([0-9.eE+-]+)'",
    )
    parser.add_argument(
        "--prefer-source-time",
        action="store_true",
        help="Use <Time Value='...'> from each source XDMF when available; otherwise fall back to filename parsing.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="If provided, override parsed times and use t_n = n*dt after sorting the matched files.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Starting time used together with --dt (default: 0.0)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output = args.output.resolve()
    time_regex = re.compile(args.time_regex) if args.time_regex else None

    if not input_dir.is_dir():
        print0(f"ERROR: input_dir does not exist or is not a directory: {input_dir}", file=sys.stderr)
        return 2

    files = collect_files(input_dir, args.pattern)
    if not files:
        print0(f"ERROR: no .xdmf files matched pattern {args.pattern!r} in {input_dir}", file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)

    time_file_pairs: List[Tuple[float, Path]] = []

    if args.dt is not None:
        for i, path in enumerate(files):
            time_value = args.start_time + i * args.dt
            time_file_pairs.append((time_value, path))
    else:
        for path in files:
            time_value: Optional[float] = None

            if args.prefer_source_time:
                try:
                    tree = ET.parse(path)
                    root = tree.getroot()
                    grid = find_first_uniform_grid(root)
                    if grid is not None:
                        time_value = parse_time_from_grid(grid)
                except ET.ParseError as exc:
                    print0(f"ERROR: failed to parse {path}: {exc}", file=sys.stderr)
                    return 2

            if time_value is None:
                time_value = parse_time_from_filename(path, time_regex)

            if time_value is None:
                print0(
                    f"ERROR: could not determine time for {path.name}. "
                    f"Use --time-regex, --prefer-source-time, or --dt.",
                    file=sys.stderr,
                )
                return 2

            time_file_pairs.append((time_value, path))

    time_file_pairs.sort(key=lambda pair: pair[0])

    grids: List[ET.Element] = []
    for time_value, src_xdmf in time_file_pairs:
        try:
            grids.append(make_temporal_grid_from_source(src_xdmf, output, time_value))
        except Exception as exc:
            print0(f"ERROR while processing {src_xdmf}: {exc}", file=sys.stderr)
            return 2

    out_tree = build_output_tree(grids)
    out_tree.write(output, encoding="utf-8", xml_declaration=True)

    print0(f"Wrote temporal XDMF collection: {output}")
    print0("Included files:")
    for t, p in time_file_pairs:
        print0(f"  t={t:.16g}  <-  {p.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
