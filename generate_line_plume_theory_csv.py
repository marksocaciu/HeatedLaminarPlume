#!/usr/bin/env python3
"""
Generate a boundary-layer / self-similar line-plume reference CSV for the
steady-plume postprocessor overlay plots.

Purpose
-------
This is meant for your present Brodowicz / air-like case and for the plotting
interface used by postprocess_steady_plume_modified.py:

    --theory-profile-csv line_plume_theory_current_case.csv

Important interpretation
------------------------
The script does NOT extract data from Deschamps & Desrayaud. Deschamps is the
confined finite-domain/cylinder-reference case. The dashed theory curve should
represent the unbounded first-order line-plume boundary-layer behaviour.

This generator creates a practical "Fujii/Gebhart-style" similarity reference:
  DeltaT_c(h) ~ (h - h0_T)^(-3/5)
  uy_c(h)     ~ (h - h0_U)^(+1/5)
  plume width ~ h^(2/5)

For the transverse shapes, it can either
  (A) use a user-supplied normalized similarity-shape CSV with columns
      eta, theta_norm, uy_norm, or
  (B) use smooth built-in surrogate shapes.

For thesis-quality final plots, option (A) is preferred if you digitize the
Pr ~= 0.7 Fujii/Gebhart profile curves. Option (B) is useful immediately for
checking the plotting workflow and showing the expected similarity scaling.

Inputs
------
The script can calibrate amplitudes and widths from your postprocessing output:
  centerline.csv
  virtual_origin_fits.csv
  plane_profiles.csv

If these are absent, it falls back to command-line defaults.

Output columns
--------------
height_m,x_m,DeltaT_K,uy_m_per_s,label
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: Dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        v = row.get(key, "")
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def load_virtual_origin_fits(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in read_csv_dicts(path):
        field = row.get("field", "")
        if not field:
            continue
        out[field] = {k: as_float(row, k) for k in row.keys() if k != "field"}
    return out


def robust_power_fit(h: np.ndarray, a: np.ndarray, exponent: float, hmin: float | None, hmax: float | None) -> Tuple[float, float]:
    """Fit a ~= C h^exponent with fixed exponent. Returns C and RMS log error."""
    h = np.asarray(h, dtype=float)
    a = np.asarray(a, dtype=float)
    m = np.isfinite(h) & np.isfinite(a) & (h > 0.0) & (a > 0.0)
    if hmin is not None:
        m &= h >= hmin
    if hmax is not None:
        m &= h <= hmax
    if np.count_nonzero(m) < 3:
        return np.nan, np.nan
    logC_vals = np.log(a[m]) - exponent * np.log(h[m])
    logC = float(np.median(logC_vals))
    err = float(np.sqrt(np.mean((logC_vals - logC) ** 2)))
    return math.exp(logC), err


def halfmax_width_from_profile(x: np.ndarray, q: np.ndarray) -> float:
    """Estimate positive half-width where q falls to half of centreline/max value."""
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    m = np.isfinite(x) & np.isfinite(q)
    x = x[m]
    q = q[m]
    if x.size < 5:
        return np.nan

    # Use the positive side. If only one side is available, use abs(x).
    if np.any(x >= 0.0):
        side = x >= 0.0
        xs = x[side]
        qs = q[side]
    else:
        xs = np.abs(x)
        qs = q
    order = np.argsort(xs)
    xs = xs[order]
    qs = qs[order]
    if xs.size < 5:
        return np.nan

    q0 = np.nanmax(qs)
    if not np.isfinite(q0) or q0 <= 0.0:
        return np.nan
    target = 0.5 * q0

    # Find first crossing below half maximum.
    above = qs >= target
    if np.all(above):
        return np.nan
    idx_candidates = np.where(~above)[0]
    idx = int(idx_candidates[0])
    if idx == 0:
        return float(xs[0])
    x1, x2 = xs[idx - 1], xs[idx]
    q1, q2 = qs[idx - 1], qs[idx]
    if q2 == q1:
        return float(x2)
    frac = (target - q1) / (q2 - q1)
    return float(x1 + frac * (x2 - x1))


def calibrate_widths_from_plane_profiles(path: Path, quantity_key: str, exponent: float = 2.0 / 5.0) -> Tuple[float, float]:
    rows = read_csv_dicts(path)
    if not rows:
        return np.nan, np.nan
    by_h: Dict[float, List[Tuple[float, float]]] = {}
    for r in rows:
        h = as_float(r, "height_m")
        x = as_float(r, "x_m")
        q = as_float(r, quantity_key)
        if np.isfinite(h) and np.isfinite(x) and np.isfinite(q):
            by_h.setdefault(h, []).append((x, q))
    hs = []
    widths = []
    for h, pairs in by_h.items():
        arr = np.array(pairs, dtype=float)
        w = halfmax_width_from_profile(arr[:, 0], arr[:, 1])
        if np.isfinite(w) and w > 0.0:
            hs.append(h)
            widths.append(w)
    if len(hs) < 2:
        return np.nan, np.nan
    hs = np.array(hs)
    widths = np.array(widths)
    C, err = robust_power_fit(hs, widths, exponent, None, None)
    return C, err


def load_shape(path: Path | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if path is not None and path.exists():
        rows = read_csv_dicts(path)
        eta = []
        th = []
        uy = []
        for r in rows:
            e = as_float(r, "eta")
            t = as_float(r, "theta_norm")
            u = as_float(r, "uy_norm")
            if np.isfinite(e):
                eta.append(e)
                th.append(t)
                uy.append(u)
        eta = np.asarray(eta, dtype=float)
        th = np.asarray(th, dtype=float)
        uy = np.asarray(uy, dtype=float)
        m = np.isfinite(eta) & np.isfinite(th) & np.isfinite(uy)
        if np.count_nonzero(m) >= 8:
            eta = np.abs(eta[m])
            th = th[m]
            uy = uy[m]
            order = np.argsort(eta)
            eta = eta[order]
            th = th[order]
            uy = uy[order]
            # normalize defensively
            if np.nanmax(th) > 0:
                th = th / np.nanmax(th)
            if np.nanmax(uy) > 0:
                uy = uy / np.nanmax(uy)
            return eta, th, uy, path.stem

    # Built-in smooth surrogate shapes. These are not digitized Fujii data.
    # eta is interpreted as x / delta_half, so theta_norm(eta=1) ~= 0.5.
    eta = np.linspace(0.0, 4.0, 401)
    theta = np.exp(-math.log(2.0) * eta**2)
    # velocity is usually a little broader than temperature in air-like Pr.
    uy = np.exp(-math.log(2.0) * (eta / 1.18) ** 2)
    return eta, theta, uy, "similarity_scaling_surrogate"


def interp_shape(eta_grid: np.ndarray, values: np.ndarray, eta_abs: np.ndarray) -> np.ndarray:
    return np.interp(eta_abs, eta_grid, values, left=values[0], right=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--postprocess-dir", default=".", help="Directory containing centerline.csv, plane_profiles.csv, virtual_origin_fits.csv")
    ap.add_argument("--out", default="line_plume_theory_current_case.csv")
    ap.add_argument("--heights", type=float, nargs="+", default=[0.01, 0.02, 0.04, 0.08], help="Profile heights above the wire [m]")
    ap.add_argument("--x-half-width", type=float, default=0.012, help="Half-width of generated profiles [m]")
    ap.add_argument("--nx", type=int, default=401)
    ap.add_argument("--shape-csv", default=None, help="Optional normalized similarity shape CSV: eta,theta_norm,uy_norm")
    ap.add_argument("--label", default=None)

    # Fallback constants if no postprocessing calibration files exist.
    ap.add_argument("--temp-C", type=float, default=None, help="Fallback C_T for DeltaT_c = C_T*(h-h0_T)^(-3/5)")
    ap.add_argument("--vel-C", type=float, default=None, help="Fallback C_U for uy_c = C_U*(h-h0_U)^(+1/5)")
    ap.add_argument("--temp-h0", type=float, default=0.0, help="Fallback temperature virtual origin [m]")
    ap.add_argument("--vel-h0", type=float, default=0.0, help="Fallback velocity virtual origin [m]")
    ap.add_argument("--temp-fit-min", type=float, default=None)
    ap.add_argument("--temp-fit-max", type=float, default=None)
    ap.add_argument("--vel-fit-min", type=float, default=None)
    ap.add_argument("--vel-fit-max", type=float, default=None)
    ap.add_argument("--width-C", type=float, default=None, help="Fallback delta_half = C_delta*h^(2/5)")
    ap.add_argument("--width-reference-height", type=float, default=0.04)
    ap.add_argument("--width-at-reference-height", type=float, default=0.0030, help="Fallback half-width at reference height if no plane_profiles.csv exists [m]")
    args = ap.parse_args()

    pp = Path(args.postprocess_dir)
    center_rows = read_csv_dicts(pp / "centerline.csv")
    fits = load_virtual_origin_fits(pp / "virtual_origin_fits.csv")

    # Determine centreline amplitudes and virtual origins.
    temp_fit = fits.get("temperature_centerline", {})
    vel_fit = fits.get("velocity_centerline", {})

    def first_finite(d: Dict[str, float], keys: List[str]) -> float:
        for k in keys:
            if k in d and np.isfinite(d[k]):
                return float(d[k])
        return np.nan

    def infer_wire_y_abs(*fit_dicts: Dict[str, float]) -> float:
        """Infer absolute wire y-coordinate from fit_y_min_m - fit_height_min_m.

        Some postprocessing versions store the fitted virtual origin as absolute
        y0, while profile generation heights are given above the wire.  In that
        case we need h0_height = y0_abs - y_wire_abs.
        """
        candidates = []
        for d in fit_dicts:
            y_min = first_finite(d, ["fit_y_min_m", "y_fit_min_m"])
            h_min = first_finite(d, ["fit_height_min_m", "height_fit_min_m"])
            if np.isfinite(y_min) and np.isfinite(h_min):
                candidates.append(y_min - h_min)
            y_max = first_finite(d, ["fit_y_max_m", "y_fit_max_m"])
            h_max = first_finite(d, ["fit_height_max_m", "height_fit_max_m"])
            if np.isfinite(y_max) and np.isfinite(h_max):
                candidates.append(y_max - h_max)
        if candidates:
            return float(np.median(candidates))
        return 0.0

    wire_y_abs = infer_wire_y_abs(temp_fit, vel_fit)

    # Accept both the older and newer virtual_origin_fits.csv column names.
    # Newer postprocess output uses C and y0; older drafts used A and h0_m.
    temp_y0_raw = first_finite(temp_fit, ["h0_m", "y0", "y0_m"])
    vel_y0_raw = first_finite(vel_fit, ["h0_m", "y0", "y0_m"])

    def convert_y0_to_height_coordinate(y0_raw: float, fit_dict: Dict[str, float], fallback: float) -> float:
        if not np.isfinite(y0_raw):
            return fallback
        # If the file contains absolute fit coordinates, convert to height above wire.
        # This is indicated by fit_y_min_m and fit_height_min_m both being present.
        has_abs_and_height = (
            np.isfinite(first_finite(fit_dict, ["fit_y_min_m", "y_fit_min_m"]))
            and np.isfinite(first_finite(fit_dict, ["fit_height_min_m", "height_fit_min_m"]))
        )
        if has_abs_and_height:
            return float(y0_raw - wire_y_abs)
        return float(y0_raw)

    temp_h0 = convert_y0_to_height_coordinate(temp_y0_raw, temp_fit, args.temp_h0)
    vel_h0 = convert_y0_to_height_coordinate(vel_y0_raw, vel_fit, args.vel_h0)

    temp_C = first_finite(temp_fit, ["A", "C", "amplitude"])
    vel_C = first_finite(vel_fit, ["A", "C", "amplitude"])

    print(f"Loaded {len(center_rows)} centerline rows and virtual origin fits for fields: {', '.join(fits.keys()) if fits else 'none'}")
    print(f"Inferred wire_y_abs = {wire_y_abs:.8e} m")
    print(f"Using temperature virtual origin h0_T = {temp_h0:.8e} m above wire")
    print(f"Using velocity virtual origin h0_U = {vel_h0:.8e} m above wire")
    print(f"Using temperature amplitude C_T = {temp_C:.8e}" if np.isfinite(temp_C) else "Temperature amplitude will be refitted/fallback")
    print(f"Using velocity amplitude C_U = {vel_C:.8e}" if np.isfinite(vel_C) else "Velocity amplitude will be refitted/fallback")

    # If virtual_origin_fits.csv did not contain usable amplitudes, refit fixed-exponent constants from centerline.csv.
    if (not np.isfinite(temp_C)) or (not np.isfinite(vel_C)):
        h = []
        dT = []
        uy = []
        for r in center_rows:
            # centerline.csv stores heights above wire in height_above_wire_m in some versions, height_m in others.
            hh = as_float(r, "height_above_wire_m")
            if not np.isfinite(hh):
                hh = as_float(r, "height_m")
            dd = as_float(r, "DeltaT_centerline_K")
            if not np.isfinite(dd):
                dd = as_float(r, "DeltaT_K")
            uu = as_float(r, "uy_centerline_m_per_s")
            if not np.isfinite(uu):
                uu = as_float(r, "uy_m_per_s")
            if np.isfinite(hh):
                h.append(hh)
                dT.append(dd)
                uy.append(uu)
        h = np.asarray(h, dtype=float)
        dT = np.asarray(dT, dtype=float)
        uy = np.asarray(uy, dtype=float)
        if not np.isfinite(temp_C):
            ht = h - temp_h0
            temp_C, _ = robust_power_fit(ht, dT, -3.0 / 5.0, args.temp_fit_min, args.temp_fit_max)
        if not np.isfinite(vel_C):
            hv = h - vel_h0
            vel_C, _ = robust_power_fit(hv, uy, +1.0 / 5.0, args.vel_fit_min, args.vel_fit_max)

    if not np.isfinite(temp_C):
        if args.temp_C is None:
            raise SystemExit("Could not infer temperature amplitude. Provide --temp-C or run in a postprocess directory containing centerline.csv / virtual_origin_fits.csv.")
        temp_C = args.temp_C
    if not np.isfinite(vel_C):
        if args.vel_C is None:
            raise SystemExit("Could not infer velocity amplitude. Provide --vel-C or run in a postprocess directory containing centerline.csv / virtual_origin_fits.csv.")
        vel_C = args.vel_C

    # Calibrate plume half-width. Prefer temperature half-width from plane profiles.
    width_C, width_err = calibrate_widths_from_plane_profiles(pp / "plane_profiles.csv", "DeltaT_K", exponent=2.0 / 5.0)
    if not np.isfinite(width_C):
        width_C = args.width_C
    if width_C is None or not np.isfinite(width_C):
        width_C = args.width_at_reference_height / (args.width_reference_height ** (2.0 / 5.0))

    shape_path = Path(args.shape_csv) if args.shape_csv else None
    eta_grid, theta_shape, uy_shape, shape_label = load_shape(shape_path)
    label = args.label or f"BL line plume ({shape_label})"

    xs = np.linspace(-args.x_half_width, args.x_half_width, args.nx)
    out_rows: List[Dict[str, object]] = []
    for H in args.heights:
        ht = H - temp_h0
        hv = H - vel_h0
        delta = width_C * max(H, 1.0e-12) ** (2.0 / 5.0)
        if delta <= 0.0 or not np.isfinite(delta):
            continue
        eta_abs = np.abs(xs) / delta

        if ht > 0.0:
            dT_c = temp_C * ht ** (-3.0 / 5.0)
            dT_vals = dT_c * interp_shape(eta_grid, theta_shape, eta_abs)
        else:
            dT_vals = np.full_like(xs, np.nan, dtype=float)

        if hv > 0.0:
            uy_c = vel_C * hv ** (+1.0 / 5.0)
            uy_vals = uy_c * interp_shape(eta_grid, uy_shape, eta_abs)
        else:
            uy_vals = np.full_like(xs, np.nan, dtype=float)
        for x, dT, uyv in zip(xs, dT_vals, uy_vals):
            out_rows.append({
                "height_m": f"{H:.12g}",
                "x_m": f"{x:.12g}",
                "DeltaT_K": "" if not np.isfinite(dT) else f"{dT:.12g}",
                "uy_m_per_s": "" if not np.isfinite(uyv) else f"{uyv:.12g}",
                "label": label,
            })

    out = Path(args.out)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["height_m", "x_m", "DeltaT_K", "uy_m_per_s", "label"])
        writer.writeheader()
        writer.writerows(out_rows)

    meta = out.with_suffix(".meta.txt")
    meta.write_text(
        "Generated line-plume theory overlay CSV\n"
        f"output = {out}\n"
        f"label = {label}\n"
        f"temperature law: DeltaT_c = {temp_C:.8e} * (h - {temp_h0:.8e})^(-3/5)\n"
        f"velocity law:    uy_c     = {vel_C:.8e} * (h - {vel_h0:.8e})^(+1/5)\n"
        f"width law:       delta    = {width_C:.8e} * h^(2/5)\n"
        f"shape source = {shape_label}\n"
        "NOTE: if shape source is similarity_scaling_surrogate, this is a scaling-consistent surrogate, not digitized Fujii/Gebhart data.\n"
    )
    print(f"Wrote {out} with {len(out_rows)} rows")
    print(f"Wrote {meta}")


if __name__ == "__main__":
    main()
