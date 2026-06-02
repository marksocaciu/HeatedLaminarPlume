# Post-processing README

This document describes the three post-processing scripts currently used with the heated laminar plume project:

- `postprocess_steady_plume.py` — detailed steady-state plume analysis from exported XDMF/HDF5 fields.
- `postprocessing.py` — transient convergence and enthalpy-flux monitoring from saved `.h5` snapshots.
- `plot_plume_recirculation.py` — transient recirculation visualisation from saved `.h5` snapshots.

All scripts are written for the legacy FEniCS/DOLFIN output layout used in this project. They do **not** require FEniCS at post-processing time; they read XDMF/HDF5 or HDF5 files directly and use `h5py`, `numpy`, and `matplotlib`.

---

## 1. Typical workflow

A normal workflow is:

1. Run the plume solver from `main.py` until either transient or steady fields are written.
2. Use `postprocessing.py` during or after a transient run to monitor convergence and convective enthalpy fluxes.
3. Use `plot_plume_recirculation.py` to make streamline, temperature, speed, and vorticity plots for selected transient snapshots.
4. Use `postprocess_steady_plume.py` once a steady or quasi-steady solution exists, especially for thesis-level diagnostics: plane integrals, centreline fits, similarity profiles, control-volume energy and momentum balances, and final plots.

Recommended environment variables for heavy HDF5/interpolation runs:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python <script_name>.py <arguments>
```

This avoids oversubscription when using multiprocessing or BLAS-backed operations.

---

## 2. `postprocess_steady_plume.py`

### Purpose

`postprocess_steady_plume.py` is the main steady-state diagnostic script. It reads nodal temperature and velocity fields from XDMF/HDF5 files, and optionally pressure and cell-centred heat-flux fields. It then computes plume profiles, energy transport, boundary heat escape, centreline power-law/virtual-origin fits, near-wire thermal boundary-layer thickness, and momentum-balance diagnostics.

Use this script for the final steady results and thesis figures/tables.

### Required inputs

At minimum:

- `--temperature-xdmf`: XDMF file for the temperature field.
- `--velocity-xdmf`: XDMF file for the velocity field.
- `--T-inf`: ambient/reference temperature in K.
- `--rho`: density in kg/m³.
- `--cp`: specific heat capacity in J/(kg K).
- `--k`: thermal conductivity in W/(m K).

Usually also provide:

- `--coords-are-dimensionless --lref <wire_radius_m>` if the solver mesh coordinates are nondimensional.
- `--heatflux-xdmf` for conductive heat flux and total heat-flow diagnostics.
- `--pressure-xdmf` for full vertical momentum control-volume diagnostics.
- `--mu` and `--beta` for Reynolds-like and buoyancy diagnostics.
- `--q-input-per-length` for energy-balance comparison against the imposed line heat input.

### Coordinate convention

The script converts mesh coordinates to physical metres using one of these modes:

```bash
--coords-are-dimensionless --lref 0.001
```

or

```bash
--coords-are-dimensional
```

or

```bash
--coordinate-scale <factor_to_metres>
```

For this project, the wire/source centre is inferred internally as

```text
y_wire = y_min + H/10 + 11*r
```

where `r = --lref`. The heights supplied through `--planes`, `--fit-y-min`, `--fit-y-max`, and control-volume options are physical heights above the wire centre, in metres.

### Basic example: steady plume post-processing

```bash
python postprocess_steady_plume.py \
  --temperature-xdmf PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/abe/base/air_temperature_steady_from_transient_step_02700.xdmf \
  --velocity-xdmf PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/abe/base/air_velocity_steady_from_transient_step_02700.xdmf \
  --heatflux-xdmf PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/abe/base/air_temperature_heatflux_final_steady_02700.xdmf \
  --outdir PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/abe/base/postprocess_thesis \
  --coords-are-dimensionless --lref 0.0007142857 \
  --T-inf 292.95 \
  --rho 1.1614 --cp 1007.0 --k 0.0257 --mu 1.85e-5 --beta 0.0034 \
  --q-input-per-length 1.0 \
  --planes 0.01 0.02 0.04 0.08
```

### Example with pressure and full momentum diagnostics

```bash
python postprocess_steady_plume.py \
  --temperature-xdmf runs/abe/base/air_temperature_steady_from_transient_step_02700.xdmf \
  --velocity-xdmf runs/abe/base/air_velocity_steady_from_transient_step_02700.xdmf \
  --pressure-xdmf runs/abe/base/air_pressure_steady_from_transient_step_02700.xdmf \
  --heatflux-xdmf runs/abe/base/air_temperature_heatflux_final_steady_02700.xdmf \
  --outdir runs/abe/base/postprocess_momentum \
  --coords-are-dimensionless --lref 0.0007142857 \
  --T-inf 292.95 \
  --rho 1.1614 --cp 1007.0 --k 0.0257 --mu 1.85e-5 --beta 0.0034 \
  --q-input-per-length 1.0 \
  --profile-half-width 0.05 \
  --momentum-cv-half-width 0.015 \
  --balance-y-min 0.001 \
  --balance-y-max 0.0821
```

Use `--pressure-scale` if the saved pressure is nondimensional or otherwise needs rescaling before traction integration.

### Example with energy control-volume budget

```bash
python postprocess_steady_plume.py \
  --temperature-xdmf runs/base/air_temperature_steady_from_transient_step_02700.xdmf \
  --velocity-xdmf runs/base/air_velocity_steady_from_transient_step_02700.xdmf \
  --heatflux-xdmf runs/base/air_temperature_heatflux_final_steady_02700.xdmf \
  --outdir runs/base/postprocess_energy_cv \
  --coords-are-dimensionless --lref 0.0007142857 \
  --T-inf 292.95 \
  --rho 1.1614 --cp 1007.0 --k 0.0257 --mu 1.85e-5 --beta 0.0034 \
  --q-input-per-length 1.0 \
  --energy-cv \
  --energy-cv-width-mode eta \
  --energy-cv-eta-half-width 9 \
  --energy-cv-min-half-width-m 1.5913e-2 \
  --energy-cv-y-bottom -7.14e-4 \
  --energy-cv-y-top 8.21e-2
```

The energy CV sidewalls follow the plume similarity width only once the `|eta|` half-width becomes larger than the minimum half-width. Below that height, the control volume remains vertical at the minimum half-width.

### Useful options

| Option | Meaning |
|---|---|
| `--planes 0.01 0.02 0.04 0.08` | Heights above the wire centre where horizontal profiles and plane integrals are sampled. |
| `--nx` | Number of samples along each horizontal profile. Increase for smoother integrals. |
| `--ny-balance` | Number of vertical levels for centreline and cumulative balance curves. |
| `--profile-half-width` | Physical half-width used for sampled profiles and some integrals. |
| `--velocity-scale-factor` | Multiplier applied to saved velocity before all diagnostics. Useful if exported velocities need dimensional rescaling. |
| `--heatflux-scale` | Multiplier applied to the exported heat-flux field. Defaults to `1/lref` for nondimensional coordinates. |
| `--eta-origin` | Uses `wire`, `temperature-virtual-origin`, or `velocity-virtual-origin` for eta plots. |
| `--eta-origin-height` | Explicit eta-origin height above wire centre; overrides `--eta-origin`. |
| `--fit-y-min`, `--fit-y-max` | Manual window for virtual-origin fits. If omitted, the script chooses an automatic window. |
| `--experiment-profile-csv` | Add experimental comparison CSV. Can be repeated. |
| `--theory-profile-csv` | Add theoretical/similarity comparison CSV. Can be repeated. |
| `--plot-width-inch`, `--plot-height-inch`, `--plot-font-size` | Thesis-style figure sizing. |
| `--plot-titles` | Adds titles inside figures; default omits titles for cleaner thesis captions. |

### Main outputs

The script writes a text summary plus many CSV and PNG files. The most important are:

- `summary.txt`
- `plane_integrals.csv`
- `plane_profiles.csv`
- `plane_profiles_eta.csv`
- `centerline.csv`
- `virtual_origin_fits.csv`
- `centerline_loglog_powerlaw_fits.csv`
- `near_wire_boundary_layer.csv`
- `near_wire_boundary_layer_by_angle.csv`
- `boundary_heat_escape.csv`
- `plume_enthalpy_balance.csv`
- `fixed_eta_mass_momentum_fluxes.csv`
- `selected_cv_momentum_cumulative.csv`
- `balance_curves.csv`
- `momentum_balance_proxy.csv`
- `momentum_balance_full.csv`, if pressure is supplied
- `momentum_control_volume_budget.csv`, if pressure and selected CV settings are available
- `energy_control_volume_budget.csv`, if `--energy-cv` is supplied

Important plot outputs include:

- `energy_flux_vs_height.png`
- `energy_budget_summary.png`
- `mass_momentum_vs_height.png`
- `fixed_eta_mass_momentum_fluxes.png`
- `selected_cv_momentum_cumulative_terms.png`
- `centerline_temperature_virtual_origin.png`
- `centerline_velocity_virtual_origin.png`
- `centerline_temperature_loglog_powerlaw.png`
- `centerline_velocity_loglog_powerlaw.png`
- `profiles_temperature_all_heights.png`
- `profiles_uy_all_heights.png`
- `profiles_ux_all_heights.png`
- `profiles_qtotal_all_heights.png`
- `profiles_temperature_eta_similarity_all_heights.png`
- `profiles_uy_eta_similarity_all_heights.png`

### Notes and pitfalls

- The script assumes temperature and velocity are on the same mesh.
- Pressure must also be nodal and on the same mesh if supplied.
- The heat-flux field is assumed to represent `q = -k grad(T)`. If it was computed on nondimensional mesh coordinates, check `--heatflux-scale`.
- The momentum balance is sensitive to interpolation and to whether the control-volume boundaries align with element edges. Large residuals can result from post-processing integration error, especially for curved/plume-following boundaries sampled by interpolation.
- All integrated quantities are per unit out-of-plane depth: W/m, kg/(s m), N/m, etc.

---

## 3. `postprocessing.py`

### Purpose

`postprocessing.py` is the transient-run monitoring script. It reads all saved temperature and velocity `.h5` snapshots in a results folder and computes convergence metrics, temperature evolution, horizontal-plane convective enthalpy fluxes, and outer-boundary enthalpy-flow integrals.

Use this script when deciding whether a transient run has settled enough to use as a steady restart or steady final state.

### Expected files

The script expects files named like:

```text
air_temperature_transient_*.h5
air_velocity_transient_*.h5
```

with the standard DOLFIN HDF5 structure:

```text
Mesh/mesh/geometry
Mesh/mesh/topology
VisualisationVector/0
```

Temperature is expected in K. Velocity is expected in m/s after applying `--velocity-scale-factor`.

### Basic example

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python postprocessing.py PlumeCase_Brodowicz_Air_reduced/runs/base \
  --workers 8 \
  --velocity-scale-factor 0.153657 \
  --T-inf 292.96 \
  --rho 1.1614 --cp 1007.0 \
  --input-line-power 9.75
```

### Example with custom planes and flux windows

```bash
python postprocessing.py runs/base \
  --workers 4 \
  --lref 3.75e-5 \
  --radius 3.75e-5 \
  --domain-height 1.0 \
  --T-inf 292.96 \
  --rho 1.1614 --cp 1007.0 \
  --velocity-scale-factor 0.153657 \
  --plane-offsets 0.01 0.02 0.04 0.08 \
  --flux-half-widths 0.01 0.02 0.04 0.08 0.20 \
  --box-half-width 0.20 \
  --box-below-wire 0.02 \
  --box-height 0.20 \
  --output-dir runs/base/postprocess_temperature
```

### Useful options

| Option | Meaning |
|---|---|
| `results_dir` | Folder containing transient `.h5` files. This is a positional argument. |
| `--workers` | Number of worker processes. Use more for many snapshots, but avoid BLAS oversubscription. |
| `--lref` | Length scale used when coordinates are nondimensional. Default is Brodowicz wire radius. |
| `--radius` | Wire radius in m. |
| `--domain-height` | Physical domain height used to infer the wire centre if `--wire-y` is not given. |
| `--wire-y` | Explicit wire centre height in m. If omitted, uses `h/10 + 11R`. |
| `--box-half-width`, `--box-below-wire`, `--box-height` | Defines the convergence-monitoring box around the plume/wire. |
| `--line-half-width` | Half-width of extracted horizontal profiles; defaults to `--box-half-width`. |
| `--num-line-points` | Number of sample points on each profile line. |
| `--plane-offsets` | Heights above the wire centre where profiles/fluxes are evaluated. |
| `--flux-half-widths` | Integration half-widths for convective enthalpy flux on each plane. |
| `--rho`, `--cp`, `--T-inf` | Material/reference properties for enthalpy flux. |
| `--velocity-scale-factor` | Multiplier applied to velocity before diagnostics. |
| `--input-line-power` | Reference line power in W/m, shown in flux plots. |
| `--boundary-n-points` | Number of sample points per outer-boundary side. |
| `--boundary-inset` | Inset used when sampling the outer boundary; `0` means automatic small inset. |
| `--output-dir` | Output folder. Defaults to `results_dir/postprocess_temperature`. |
| `--plot-*` | Controls plot font sizes, figure size, and DPI. |

### Main outputs

The output folder contains:

- `temperature_convergence_box.csv`
- `temperature_plane_peaks.csv`
- `enthalpy_flux_planes.csv`
- `enthalpy_flux_outer_boundary.csv`
- `temperature_box_relative_l2_update.png`
- `temperature_box_linf_update.png`
- `temperature_excess_box_peak.png`
- `temperature_excess_plane_peak_evolution.png`
- `enthalpy_flux_net_window_evolution_y_plus_*.png`
- `enthalpy_flux_up_down_window_evolution_y_plus_*.png`
- `uy_net_fraction_window_evolution_y_plus_*.png`
- `enthalpy_flux_outer_boundary_evolution.png`
- `enthalpy_flux_outer_boundary_positive_negative_evolution.png`
- `temperature_profiles_step_<final_step>.csv`
- `final_profiles_T_theta_ux_uy_hflux_step_<final_step>.csv`
- `temperature_excess_profiles_step_<final_step>.png`
- `velocity_profiles_uy_and_10ux_step_<final_step>.png`
- `enthalpy_flux_density_profiles_step_<final_step>.png`

### Notes and pitfalls

- This script is snapshot-based. It does not solve anything and does not need FEniCS.
- It assumes standard transient file names. If the solver output names change, the file discovery logic must be adjusted.
- For nondimensional coordinates, keep `--lref`, `--radius`, `--domain-height`, and `--wire-y` consistent with the case setup.
- `--velocity-scale-factor` is applied before all flux and profile diagnostics. If the saved field is already dimensional in the desired units, leave it at `1.0`.

---

## 4. `plot_plume_recirculation.py`

### Purpose

`plot_plume_recirculation.py` creates visual diagnostics from selected transient snapshots:

- temperature contours with streamlines,
- speed magnitude with streamlines,
- vorticity with streamlines,
- optional multi-panel sequence plot,
- summary CSV of snapshot extrema/statistics.

Use this script for visualising plume-head formation, recirculation cells, corner vortices, and qualitative transient evolution.

### Expected files

For each requested step, the input directory should contain:

```text
air_velocity_transient_<step>.h5
air_velocity_transient_<step>.xdmf
air_temperature_transient_<step>.h5
air_temperature_transient_<step>.xdmf
```

The script reads the `.h5` files directly and assumes the standard DOLFIN layout:

```text
Mesh/mesh/geometry
Mesh/mesh/topology
VisualisationVector/0
```

### Basic example

```bash
python plot_plume_recirculation.py \
  --input-dir PlumeCase_Brodowicz_Air_reduced/runs/base \
  --steps 05000 10000 14500 \
  --out-dir PlumeCase_Brodowicz_Air_reduced/runs/base/recirculation_figures \
  --make-sequence
```

### Example for large-domain plots without speed figures

```bash
python plot_plume_recirculation.py \
  --input-dir runs/base \
  --steps 85000 105000 \
  --out-dir runs/base/large_domain_figures \
  --nx 420 --ny 420 \
  --dpi 250 \
  --no-speed
```

### Example for only vorticity plots

```bash
python plot_plume_recirculation.py \
  --input-dir runs/base \
  --steps 00500 01000 02000 \
  --out-dir runs/base/vorticity_figures \
  --no-temperature \
  --no-speed
```

### Useful options

| Option | Meaning |
|---|---|
| `--input-dir` | Directory containing the transient `.h5` files. |
| `--out-dir` | Directory where PNG/CSV outputs are written. |
| `--steps` | Required list of step IDs, including leading zeros if present in the filenames. |
| `--nx`, `--ny` | Interpolation grid resolution. Increase for smoother plots; decrease for speed. |
| `--dpi` | PNG resolution. |
| `--make-sequence` | Creates a multi-panel sequence plot across the selected steps. |
| `--no-temperature` | Disables temperature + streamline plots. |
| `--no-speed` | Disables speed + streamline plots. |
| `--no-vorticity` | Disables vorticity + streamline plots. |
| `--titles` | Adds plot titles. Default omits titles for thesis-style figures. |

### Main outputs

Depending on selected options, the output folder contains:

- `small_domain_streamlines_<step>.png`
- `small_domain_speed_streamlines_<step>.png`
- `reduced_domain_vorticity_<step>.png`
- `reduced_domain_recirculation_sequence.png`, if `--make-sequence` is used
- `recirculation_summary.csv`

### Notes and pitfalls

- Vorticity is computed after interpolation to a regular grid as `d(uy)/dx - d(ux)/dy`, so it is a visual diagnostic rather than a high-order finite-element derivative.
- The script assumes the coordinates in the saved files are already in the desired plotting units. It does not rescale coordinates.
- The `--steps` values must match the filenames exactly. For example, use `05000`, not `5000`, if the files are named with leading zeros.

---

## 5. Quick command templates

### Transient convergence check

```bash
python postprocessing.py <results_dir> \
  --workers 8 \
  --velocity-scale-factor <scale> \
  --T-inf <ambient_K> \
  --rho <rho> --cp <cp> \
  --input-line-power <Q_L_W_per_m>
```

### Transient recirculation plots

```bash
python plot_plume_recirculation.py \
  --input-dir <results_dir> \
  --steps <step1> <step2> <step3> \
  --out-dir <results_dir>/recirculation_figures \
  --make-sequence
```

### Final steady post-processing

```bash
python postprocess_steady_plume.py \
  --temperature-xdmf <temperature_steady.xdmf> \
  --velocity-xdmf <velocity_steady.xdmf> \
  --heatflux-xdmf <heatflux_steady.xdmf> \
  --pressure-xdmf <pressure_steady.xdmf> \
  --outdir <output_dir> \
  --coords-are-dimensionless --lref <wire_radius_m> \
  --T-inf <ambient_K> \
  --rho <rho> --cp <cp> --k <k> --mu <mu> --beta <beta> \
  --q-input-per-length <Q_L_W_per_m> \
  --planes 0.01 0.02 0.04 0.08
```

---

## 6. Recommended figure settings for thesis use

For figures intended to be placed two per row on an A4 page, use widths around `3.0--3.4 in` and font sizes around `11--13 pt`.

For `postprocess_steady_plume.py`:

```bash
--plot-width-inch 3.2 --plot-font-size 12
```

For `postprocessing.py`:

```bash
--plot-figure-width 3.3 \
--plot-figure-height 2.45 \
--plot-label-font-size 12 \
--plot-tick-font-size 11 \
--plot-legend-font-size 8.5 \
--plot-dpi 300
```

For `plot_plume_recirculation.py`:

```bash
--nx 360 --ny 360 --dpi 220
```

Increase `--nx` and `--ny` if the streamlines or vorticity contours look too coarse.
