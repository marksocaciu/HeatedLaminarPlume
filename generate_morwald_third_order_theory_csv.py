#!/usr/bin/env python3
"""
Generate a Mörwald--Mitsotakis--Schneider-style higher-order laminar
plane-plume theory overlay CSV for postprocess_steady_plume_modified.py.

Output columns are exactly the same as the Fujii/Gebhart generator:

    height_m,x_m,DeltaT_K,uy_m_per_s,label

Important scientific interpretation
-----------------------------------
This script is a practical plotting generator.  Mörwald et al. (1986) do not
provide a simple ready-to-use table of full transverse third-order profiles in
the short conference paper.  Their theory is a higher-order asymptotic expansion:

    stream function:   psi = eps^-2 f1(eta) + f2(eta) + eps^2 f3(eta)   [no wall]
    temperature:      theta = eps^2 t1(eta) + eps^4 t2(eta) + eps^6 t3(eta)

with eps = Gr_x^(-1/10), for the unbounded plane plume.  In the wall-bounded
case the third-order structure differs because of a wall-boundary-layer
displacement effect.

For immediate thesis plots this script applies the higher-order correction to
the centreline amplitudes and uses either

  (A) a supplied normalized transverse shape CSV with columns
      eta, theta_norm, uy_norm, optionally theta2_norm, uy2_norm,
      theta3_norm, uy3_norm, or
  (B) the same smooth surrogate shape used in the Fujii/Gebhart generator.

If only eta,theta_norm,uy_norm are supplied, the correction changes the
centreline level but not the transverse shape.  This is still useful for showing
how a Mörwald-type corrected BL reference differs in magnitude from the
first-order theory, but it should not be presented as a fully resolved exact
third-order transverse profile unless you supply digitized/solved higher-order
shape functions.

Units/coordinates
-----------------
The profile heights are heights above the wire [m].  If your postprocessing
output stores the virtual origin in absolute y, the script detects the wire
coordinate from fit_y_min_m - fit_height_min_m and converts it.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

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


def first_finite(d: Dict[str, float], keys: List[str]) -> float:
    for k in keys:
        if k in d and np.isfinite(d[k]):
            return float(d[k])
    return np.nan


def infer_wire_y_abs(*fit_dicts: Dict[str, float]) -> float:
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


def convert_y0_to_height_coordinate(y0_raw: float, fit_dict: Dict[str, float], wire_y_abs: float, fallback: float) -> float:
    if not np.isfinite(y0_raw):
        return fallback
    has_abs_and_height = (
        np.isfinite(first_finite(fit_dict, ["fit_y_min_m", "y_fit_min_m"]))
        and np.isfinite(first_finite(fit_dict, ["fit_height_min_m", "height_fit_min_m"]))
    )
    if has_abs_and_height:
        return float(y0_raw - wire_y_abs)
    return float(y0_raw)


def robust_power_fit(h: np.ndarray, a: np.ndarray, exponent: float, hmin: float | None, hmax: float | None) -> Tuple[float, float]:
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
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    m = np.isfinite(x) & np.isfinite(q)
    x = x[m]
    q = q[m]
    if x.size < 5:
        return np.nan
    side = x >= 0.0 if np.any(x >= 0.0) else np.ones_like(x, dtype=bool)
    xs = np.abs(x[side])
    qs = q[side]
    order = np.argsort(xs)
    xs = xs[order]
    qs = qs[order]
    if xs.size < 5:
        return np.nan
    q0 = np.nanmax(qs)
    if not np.isfinite(q0) or q0 <= 0.0:
        return np.nan
    target = 0.5 * q0
    above = qs >= target
    if np.all(above):
        return np.nan
    idx = int(np.where(~above)[0][0])
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
    hs, widths = [], []
    for h, pairs in by_h.items():
        arr = np.array(pairs, dtype=float)
        w = halfmax_width_from_profile(arr[:, 0], arr[:, 1])
        if np.isfinite(w) and w > 0.0:
            hs.append(h)
            widths.append(w)
    if len(hs) < 2:
        return np.nan, np.nan
    return robust_power_fit(np.array(hs), np.array(widths), exponent, None, None)


def load_shape(path: Path | None) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
    """Load normalized shape functions.

    Required if user supplies a file: eta, theta_norm, uy_norm.
    Optional: theta2_norm, uy2_norm, theta3_norm, uy3_norm.  If absent, zeros
    are used for those shape corrections, so only centreline correction is applied.
    """
    if path is not None and path.exists():
        rows = read_csv_dicts(path)
        cols: Dict[str, List[float]] = {
            "eta": [], "theta_norm": [], "uy_norm": [],
            "theta2_norm": [], "uy2_norm": [], "theta3_norm": [], "uy3_norm": [],
        }
        for r in rows:
            e = as_float(r, "eta")
            t = as_float(r, "theta_norm")
            u = as_float(r, "uy_norm")
            if np.isfinite(e) and np.isfinite(t) and np.isfinite(u):
                cols["eta"].append(abs(e))
                cols["theta_norm"].append(t)
                cols["uy_norm"].append(u)
                cols["theta2_norm"].append(as_float(r, "theta2_norm", 0.0))
                cols["uy2_norm"].append(as_float(r, "uy2_norm", 0.0))
                cols["theta3_norm"].append(as_float(r, "theta3_norm", 0.0))
                cols["uy3_norm"].append(as_float(r, "uy3_norm", 0.0))
        eta = np.asarray(cols["eta"], dtype=float)
        if eta.size >= 8:
            order = np.argsort(eta)
            eta = eta[order]
            out = {}
            for key in cols:
                if key == "eta":
                    continue
                arr = np.asarray(cols[key], dtype=float)[order]
                if key in ("theta_norm", "uy_norm") and np.nanmax(np.abs(arr)) > 0.0:
                    arr = arr / np.nanmax(arr)
                out[key] = arr
            return eta, out, path.stem

    eta = np.linspace(0.0, 4.0, 401)
    theta = np.exp(-math.log(2.0) * eta**2)
    uy = np.exp(-math.log(2.0) * (eta / 1.18) ** 2)
    z = np.zeros_like(eta)
    return eta, {
        "theta_norm": theta,
        "uy_norm": uy,
        "theta2_norm": z,
        "uy2_norm": z,
        "theta3_norm": z,
        "uy3_norm": z,
    }, "morwald_centerline_corrected_surrogate"


def interp_shape(eta_grid: np.ndarray, values: np.ndarray, eta_abs: np.ndarray) -> np.ndarray:
    return np.interp(eta_abs, eta_grid, values, left=values[0], right=0.0)


def get_centerline_data(center_rows: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, dT, uy = [], [], []
    for r in center_rows:
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
    return np.asarray(h), np.asarray(dT), np.asarray(uy)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--postprocess-dir", default=".")
    ap.add_argument("--out", default="morwald_third_order_theory_current_case.csv")
    ap.add_argument("--heights", type=float, nargs="+", default=[0.01, 0.02, 0.04, 0.08])
    ap.add_argument("--x-half-width", type=float, default=0.012)
    ap.add_argument("--nx", type=int, default=401)
    ap.add_argument("--shape-csv", default=None, help="Optional normalized BL shape CSV. Columns: eta,theta_norm,uy_norm[,theta2_norm,uy2_norm,theta3_norm,uy3_norm]")
    ap.add_argument("--label", default="Mörwald 3rd-order BL reference")

    # Fit/calibration fallback values.
    ap.add_argument("--temp-C", type=float, default=None)
    ap.add_argument("--vel-C", type=float, default=None)
    ap.add_argument("--temp-h0", type=float, default=0.0)
    ap.add_argument("--vel-h0", type=float, default=0.0)
    ap.add_argument("--temp-fit-min", type=float, default=None)
    ap.add_argument("--temp-fit-max", type=float, default=None)
    ap.add_argument("--vel-fit-min", type=float, default=None)
    ap.add_argument("--vel-fit-max", type=float, default=None)
    ap.add_argument("--width-C", type=float, default=None)
    ap.add_argument("--width-reference-height", type=float, default=0.04)
    ap.add_argument("--width-at-reference-height", type=float, default=0.0030)

    # Mörwald-type correction controls.  eps = Gr_h^(-1/10).
    ap.add_argument("--Gr-ref", type=float, default=None, help="Use a constant local Grashof number for the correction. If omitted, compute from fluid properties and q.")
    ap.add_argument("--q-input-per-length", type=float, default=None, help="Heat input per unit length Q [W/m], used to compute local Gr_h if --Gr-ref is omitted")
    ap.add_argument("--rho", type=float, default=1.1614)
    ap.add_argument("--cp", type=float, default=1007.0)
    ap.add_argument("--k", type=float, default=0.0257)
    ap.add_argument("--mu", type=float, default=1.85e-5)
    ap.add_argument("--beta", type=float, default=0.0034)
    ap.add_argument("--g", type=float, default=9.81)
    ap.add_argument("--Pr", type=float, default=None, help="Override Pr. Otherwise computed as mu*cp/k.")

    # Dimensionless centreline coefficients.  Defaults are approximate air/no-wall
    # values read from Mörwald et al. Table 2 snippet; use your own values if you
    # digitize/solve the table more accurately.
    ap.add_argument("--t1-0", type=float, default=0.373, help="First-order centreline temperature coefficient t1(0)")
    ap.add_argument("--t2-0", type=float, default=-0.168, help="Second-order centreline temperature coefficient t2(0)")
    ap.add_argument("--t3-0", type=float, default=-1.974, help="Third-order centreline temperature coefficient t3(0), no-wall air default approximate")
    ap.add_argument("--u1-0", type=float, default=0.809, help="First-order centreline velocity-shape coefficient f1''(0) or equivalent")
    ap.add_argument("--u2-0", type=float, default=0.139, help="Second-order centreline velocity correction coefficient")
    ap.add_argument("--u3-0", type=float, default=-2.865, help="Third-order centreline velocity correction coefficient, no-wall air default approximate")
    ap.add_argument("--disable-velocity-correction", action="store_true", help="Apply Mörwald correction only to temperature")
    ap.add_argument("--correction-clip", type=float, default=3.0, help="Clip multiplicative corrections to [1/clip, clip] to avoid nonsense at low Gr")
    args = ap.parse_args()

    pp = Path(args.postprocess_dir)
    center_rows = read_csv_dicts(pp / "centerline.csv")
    fits = load_virtual_origin_fits(pp / "virtual_origin_fits.csv")
    temp_fit = fits.get("temperature_centerline", {})
    vel_fit = fits.get("velocity_centerline", {})
    wire_y_abs = infer_wire_y_abs(temp_fit, vel_fit)

    temp_y0_raw = first_finite(temp_fit, ["h0_m", "y0", "y0_m"])
    vel_y0_raw = first_finite(vel_fit, ["h0_m", "y0", "y0_m"])
    temp_h0 = convert_y0_to_height_coordinate(temp_y0_raw, temp_fit, wire_y_abs, args.temp_h0)
    vel_h0 = convert_y0_to_height_coordinate(vel_y0_raw, vel_fit, wire_y_abs, args.vel_h0)

    temp_C = first_finite(temp_fit, ["A", "C", "amplitude"])
    vel_C = first_finite(vel_fit, ["A", "C", "amplitude"])
    hdata, dTdata, uydata = get_centerline_data(center_rows)
    if not np.isfinite(temp_C):
        temp_C, _ = robust_power_fit(hdata - temp_h0, dTdata, -3.0 / 5.0, args.temp_fit_min, args.temp_fit_max)
    if not np.isfinite(vel_C):
        vel_C, _ = robust_power_fit(hdata - vel_h0, uydata, +1.0 / 5.0, args.vel_fit_min, args.vel_fit_max)
    if not np.isfinite(temp_C):
        if args.temp_C is None:
            raise SystemExit("Could not infer temperature amplitude. Provide --temp-C or postprocess files.")
        temp_C = args.temp_C
    if not np.isfinite(vel_C):
        if args.vel_C is None:
            raise SystemExit("Could not infer velocity amplitude. Provide --vel-C or postprocess files.")
        vel_C = args.vel_C

    width_C, _ = calibrate_widths_from_plane_profiles(pp / "plane_profiles.csv", "DeltaT_K")
    if not np.isfinite(width_C):
        width_C = args.width_C
    if width_C is None or not np.isfinite(width_C):
        width_C = args.width_at_reference_height / (args.width_reference_height ** (2.0 / 5.0))

    Pr = args.Pr if args.Pr is not None else args.mu * args.cp / args.k
    nu = args.mu / args.rho
    alpha = args.k / (args.rho * args.cp)

    eta_grid, shapes, shape_label = load_shape(Path(args.shape_csv) if args.shape_csv else None)
    xs = np.linspace(-args.x_half_width, args.x_half_width, args.nx)
    rows_out: List[Dict[str, object]] = []

    def local_Gr(H: float) -> float:
        if args.Gr_ref is not None:
            return float(args.Gr_ref)
        if args.q_input_per_length is None:
            # If Q is unavailable, use a large-Gr correction that is very small.
            return 1.0e8
        # Line-source heat-flux Grashof number used in plume literature up to
        # convention-dependent constants: Gr_h = g beta Q h^3 / (rho cp alpha nu^2).
        # Since alpha=k/(rho cp), this is equivalent to g beta Q h^3 /(k nu^2).
        return args.g * args.beta * args.q_input_per_length * max(H, 1e-15) ** 3 / (args.k * nu**2)

    def correction_factor(eps: float, a1: float, a2: float, a3: float) -> float:
        # centreline ratio (a1 + eps^2 a2 + eps^4 a3) / a1 for the no-wall algebraic expansion.
        if a1 == 0.0 or not np.isfinite(eps):
            return 1.0
        fac = (a1 + eps**2 * a2 + eps**4 * a3) / a1
        if not np.isfinite(fac) or fac <= 0.0:
            fac = 1.0
        lo = 1.0 / max(args.correction_clip, 1.0)
        hi = max(args.correction_clip, 1.0)
        return float(np.clip(fac, lo, hi))

    for H in args.heights:
        ht = H - temp_h0
        hv = H - vel_h0
        delta = width_C * max(H, 1.0e-12) ** (2.0 / 5.0)
        if delta <= 0.0 or not np.isfinite(delta):
            continue
        GrH = max(local_Gr(H), 1.0e-30)
        eps = GrH ** (-1.0 / 10.0)
        theta_fac = correction_factor(eps, args.t1_0, args.t2_0, args.t3_0)
        if args.disable_velocity_correction:
            uy_fac = 1.0
        else:
            uy_fac = correction_factor(eps, args.u1_0, args.u2_0, args.u3_0)

        eta_abs = np.abs(xs) / delta
        th1 = interp_shape(eta_grid, shapes["theta_norm"], eta_abs)
        uy1 = interp_shape(eta_grid, shapes["uy_norm"], eta_abs)
        th2 = interp_shape(eta_grid, shapes["theta2_norm"], eta_abs)
        uy2 = interp_shape(eta_grid, shapes["uy2_norm"], eta_abs)
        th3 = interp_shape(eta_grid, shapes["theta3_norm"], eta_abs)
        uy3 = interp_shape(eta_grid, shapes["uy3_norm"], eta_abs)

        # If higher-order shapes are supplied, use them additively and normalize by
        # centreline coefficients. Otherwise th2/th3/uy2/uy3 are zero and the
        # centreline correction factors control the difference from first order.
        if ht > 0.0:
            dT_first = temp_C * ht ** (-3.0 / 5.0)
            if np.any(th2) or np.any(th3):
                dT_shape = (args.t1_0 * th1 + eps**2 * args.t2_0 * th2 + eps**4 * args.t3_0 * th3) / args.t1_0
                dT_vals = dT_first * np.maximum(dT_shape, 0.0)
            else:
                dT_vals = dT_first * theta_fac * th1
        else:
            dT_vals = np.full_like(xs, np.nan)

        if hv > 0.0:
            uy_first = vel_C * hv ** (+1.0 / 5.0)
            if np.any(uy2) or np.any(uy3):
                uy_shape = (args.u1_0 * uy1 + eps**2 * args.u2_0 * uy2 + eps**4 * args.u3_0 * uy3) / args.u1_0
                uy_vals = uy_first * np.maximum(uy_shape, 0.0)
            else:
                uy_vals = uy_first * uy_fac * uy1
        else:
            uy_vals = np.full_like(xs, np.nan)

        for x, dT, uyv in zip(xs, dT_vals, uy_vals):
            rows_out.append({
                "height_m": f"{H:.12g}",
                "x_m": f"{x:.12g}",
                "DeltaT_K": "" if not np.isfinite(dT) else f"{dT:.12g}",
                "uy_m_per_s": "" if not np.isfinite(uyv) else f"{uyv:.12g}",
                "label": args.label,
            })

    out = Path(args.out)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["height_m", "x_m", "DeltaT_K", "uy_m_per_s", "label"])
        writer.writeheader()
        writer.writerows(rows_out)

    meta = out.with_suffix(".meta.txt")
    meta.write_text(
        "Generated Mörwald-style third-order BL overlay CSV\n"
        f"output = {out}\n"
        f"label = {args.label}\n"
        f"shape source = {shape_label}\n"
        f"Pr = {Pr:.8g}, nu = {nu:.8e}, alpha = {alpha:.8e}\n"
        f"temperature first-order law: DeltaT_c = {temp_C:.8e} * (h - {temp_h0:.8e})^(-3/5)\n"
        f"velocity first-order law:    uy_c     = {vel_C:.8e} * (h - {vel_h0:.8e})^(+1/5)\n"
        f"width law:                   delta    = {width_C:.8e} * h^(2/5)\n"
        f"Mörwald coefficients: t1(0)={args.t1_0}, t2(0)={args.t2_0}, t3(0)={args.t3_0}, u1(0)={args.u1_0}, u2(0)={args.u2_0}, u3(0)={args.u3_0}\n"
        "WARNING: Unless a --shape-csv with higher-order shape columns was supplied, this is a centreline-corrected higher-order reference, not an exact resolved third-order transverse profile.\n"
    )
    print(f"Loaded {len(center_rows)} centerline rows; fits: {', '.join(fits.keys()) if fits else 'none'}")
    print(f"Inferred wire_y_abs = {wire_y_abs:.8e} m")
    print(f"Using h0_T = {temp_h0:.8e} m, C_T = {temp_C:.8e}")
    print(f"Using h0_U = {vel_h0:.8e} m, C_U = {vel_C:.8e}")
    print(f"Wrote {out} with {len(rows_out)} rows")
    print(f"Wrote {meta}")


if __name__ == "__main__":
    main()
