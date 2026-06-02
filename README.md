# Heated Laminar Plume Solver

This repository contains a legacy FEniCS project for simulating laminar natural convection above a heated horizontal wire/cylinder. The code is driven from `main.py`; simulations are selected through an experiment entry in `experiments.json` and controlled mainly through command-line arguments.

The current default execution path in `main.py` is the **Standard Boussinesq formulation**. The **Asymptotic Boussinesq formulation** and **Temperature-dependent** branches are implemented in the codebase.

## 1. What the project does

The solver workflow is:

1. Read an experiment definition from `experiments.json`.
2. Generate a Gmsh geometry and mesh for the selected wire/domain configuration.
3. Convert the mesh to XDMF files readable by FEniCS.
4. Solve a thermal conduction problem to build an initial temperature field.
5. Build the nondimensional air-domain Navier–Stokes/energy problem.
6. Compute startup guesses and sign checks.
7. Run pseudo-transient continuation and then a transient calculation, or restart from a checkpoint/transient snapshot.
8. Save dimensional pressure, velocity, temperature, heat-flux fields, restart checkpoints, logs, and plane-integral diagnostics.

The physical model is intended for 2D laminar natural convection in a finite enclosure with a heated horizontal wire represented by a finite cylinder with prescribed heat input per unit length.

## 2. Repository layout

The uploaded files correspond to the following project modules:

```text
main.py                 Main entry point and CLI dispatcher
imports.py              Shared imports, MPI helpers, global constants, tag IDs
geometry.py             Gmsh geometry generation, mesh conversion, XDMF loading
initial.py              Initial thermal/conduction solve and heat-flux setup
params_bcs.py           Boundary conditions, parameters, volume heat source
scales.py               Nondimensional scales and dimensionalization helpers
solver.py               Function spaces, weak forms, startup problems, generic solvers
base_solver.py          Standard Boussinesq PTC, transient, steady-restart solvers
abe_solver.py           Asymptotic Boussinesq Equation solver path
amr.py                  Offline AMR, remeshing, checkpoint transfer utilities
results.py              Plane fluxes and diagnostic output helpers
transfer.py             Mesh/function transfer and coordinate scaling utilities
```

In the original repository these modules are imported as packages such as `utils.geometry` and `solver.base_solver`. Therefore the repository should preserve that package structure, for example:

```text
project-root/
├── main.py
├── experiments.json
├── experiments.schema.json
├── utils/
│   ├── imports.py
│   ├── geometry.py
│   ├── initial.py
│   ├── params_bcs.py
│   ├── scales.py
│   ├── results.py
│   └── transfer.py
└── solver/
    ├── solver.py
    ├── base_solver.py
    ├── abe_solver.py
    └── amr.py
```

If the files are kept flat in one directory, the imports in `main.py` will not resolve unless they are adjusted.

## 3. Requirements

The project is written for **legacy FEniCS/dolfin**, not dolfinx. A typical environment needs:

- Python 3
- legacy `fenics` / `dolfin`
- `mpi4py`
- `meshio`
- `gmsh`
- `numpy`
- `matplotlib`
- PETSc/MUMPS support for robust nonlinear/linear solves

A typical run is MPI-parallel:

```bash
mpirun -np 4 python main.py --experiment-index 0
```

For larger meshes, use more ranks if memory allows:

```bash
mpirun -np 48 --use-hwthread-cpus python main.py --experiment-index 1
```

## 4. Input files

### `experiments.json`

`main.py` reads all cases from:

```text
experiments.json
```

and validates/parses them using:

```text
experiments.schema.json
```

The selected case is controlled by:

```bash
--experiment-index <N>
```

where `N` is zero-based. If no index is supplied, the code defaults to `--experiment-index 1`.

Each experiment is expected to contain at least:

- experiment name
- domain dimensions
- wire diameter and material properties
- fluid properties, such as `rho`, `mu`, `k`, `cp`, `beta`
- initial ambient temperature
- heat input per unit length

The exact schema is defined in `experiments.schema.json`.

## 5. Basic usage

Run the default base Boussinesq simulation for experiment 1:

```bash
python main.py --experiment-index 1
```

Run the same case with MPI:

```bash
mpirun -np 8 python main.py --experiment-index 1
```

Use hardware threads on a cluster node:

```bash
mpirun -np 48 --use-hwthread-cpus python main.py --experiment-index 1
```

Show the available command-line options:

```bash
python main.py --help
```

## 6. Output directory structure

For a fresh run, `main.py` creates a timestamped run directory:

```text
<experiment-name>/runs/base_<YYYYMMDD_HHMMSS>_pid<PID>/
```

Inside that run directory, the base solver writes files such as:

```text
geom.geo
plume.msh
full_cells.xdmf
full_facets.xdmf
air_cells.xdmf
air_facets.xdmf
transient_history.csv
base/
    air_pressure.xdmf
    air_velocity.xdmf
    air_temperature.xdmf
    air_temperature_heatflux.xdmf
    air_temperature_heatflux_mag.xdmf
    plane_fluxes.csv
    restart_checkpoint/
        state.h5
        state.json
```

The transient solver also writes numbered transient snapshots depending on the save interval, for example:

```text
base/air_temperature_transient_00100.xdmf
base/air_velocity_transient_00100.xdmf
base/air_pressure_transient_00100.xdmf
```

## 7. Restarting a transient run

To continue a run from its existing run directory, use:

```bash
python main.py \
  --experiment-index 1 \
  --restart-from-last-transient \
  --existing-run-root PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866
```

With MPI:

```bash
mpirun -np 48 --use-hwthread-cpus python main.py \
  --experiment-index 1 \
  --restart-from-last-transient \
  --existing-run-root PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866
```

The restart logic first tries to load a true checkpoint from:

```text
<run-root>/base/restart_checkpoint/state.h5
<run-root>/base/restart_checkpoint/state.json
```

If no true checkpoint is found, it attempts to reconstruct the state from the latest saved transient XDMF snapshots and `transient_history.csv`.

## 8. Running a steady solve from a transient checkpoint

After a transient has reached a useful state, you can use it as the initial condition for a steady Newton solve:

```bash
python main.py \
  --experiment-index 1 \
  --steady-from-last-transient \
  --existing-run-root PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866
```

This uses the same restart-loading mechanism as the transient restart, then calls the steady-from-checkpoint branch.

## 9. Restarting from a checkpoint-owned mesh

If the checkpoint contains its own mesh, for example after AMR, restart with:

```bash
python main.py \
  --experiment-index 1 \
  --existing-run-root PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866 \
  --restart-from-checkpoint-mesh PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866/base/restart_checkpoint_amr
```

With MPI:

```bash
mpirun -np 48 --use-hwthread-cpus python main.py \
  --experiment-index 1 \
  --existing-run-root PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866 \
  --restart-from-checkpoint-mesh PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866/base/restart_checkpoint_amr
```

Use this mode when the mesh stored in the checkpoint should be the authoritative mesh for the continued run.

## 10. Offline AMR checkpoint refinement

To refine an existing restart checkpoint offline:

```bash
python main.py \
  --experiment-index 1 \
  --refine-restart-checkpoint PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866/base/restart_checkpoint \
  --refined-checkpoint-out PlumeCase_Brodowicz_Air_reduced/runs/base_20260504_142234_pid1546866/base/restart_checkpoint_amr \
  --amr-top-fraction 0.10 \
  --amr-levels 2 \
  --amr-dt-factor 0.25
```

Relevant options:

```text
--refine-restart-checkpoint   Input checkpoint directory containing state.h5/state.json
--refined-checkpoint-out      Output directory for the refined checkpoint
--amr-top-fraction            Fraction of cells marked for refinement
--amr-levels                  Number of refinement levels
--amr-dt-factor               Multiplier applied to the saved checkpoint time step
```

After creating the refined checkpoint, continue from it using `--restart-from-checkpoint-mesh`.

## 11. Offline remeshing from a checkpoint

To transfer an old checkpoint onto a newly generated coarse mesh and then refine it:

```bash
python main.py \
  --experiment-index 1 \
  --remesh-restart-checkpoint OLD_RUN/base/restart_checkpoint \
  --coarse-remesh-run-root REMESH_WORKDIR \
  --remeshed-checkpoint-out NEW_RUN/base/restart_checkpoint_remeshed \
  --amr-top-fraction 0.05 \
  --amr-levels 1 \
  --amr-dt-factor 0.25 \
  --remesh-wire-ring-factor 8.0
```

Relevant options:

```text
--remesh-restart-checkpoint   Input checkpoint directory containing state.h5/state.json
--coarse-remesh-run-root      Directory where the new coarse Gmsh/XDMF files are generated
--remeshed-checkpoint-out     Output checkpoint directory after solution transfer/refinement
--remesh-wire-ring-factor     Forced near-wire refinement radius in multiples of the wire radius
```

## 12. Projecting a foreign checkpoint to another experiment

The code can project a checkpoint from one experiment definition to another target experiment:

```bash
python main.py \
  --foreign-restart-checkpoint SOURCE_RUN/base/restart_checkpoint \
  --foreign-source-experiment-index 0 \
  --foreign-target-experiment-index 1 \
  --coarse-remesh-run-root FOREIGN_TRANSFER_WORKDIR \
  --foreign-checkpoint-out TARGET_RUN/base/restart_checkpoint_projected \
  --amr-top-fraction 0.05 \
  --amr-levels 1 \
  --amr-dt-factor 0.25
```

This branch requires all of the following:

```text
--foreign-restart-checkpoint
--foreign-source-experiment-index
--foreign-target-experiment-index
--coarse-remesh-run-root
--foreign-checkpoint-out
```

## 13. Command-line reference

| Option | Default | Meaning |
|---|---:|---|
| `--experiment-index` | `1` | Zero-based index of the experiment in `experiments.json`. |
| `--formulation` | `abs` | Formulation selection for the solver. |
| `--restart-from-last-transient` | off | Continue from the latest true checkpoint or latest saved transient snapshot in an existing run. |
| `--existing-run-root` | empty | Reuse a previous run directory instead of creating a new timestamped one. |
| `--steady-from-last-transient` | off | Load the latest transient/checkpoint state and attempt a steady solve. |
| `--refine-restart-checkpoint` | empty | Offline AMR: input checkpoint directory. |
| `--refined-checkpoint-out` | empty | Offline AMR: output checkpoint directory. |
| `--amr-top-fraction` | `0.05` | Fraction of cells selected for AMR refinement. |
| `--amr-levels` | `1` | Number of AMR refinement levels. |
| `--amr-dt-factor` | `0.25` | Factor applied to the restart time step after interpolation/refinement. |
| `--restart-from-checkpoint-mesh` | empty | Restart using the mesh stored in a checkpoint directory. |
| `--remesh-restart-checkpoint` | empty | Offline remeshing: input old checkpoint directory. |
| `--coarse-remesh-run-root` | empty | Working directory for generated coarse remesh files. |
| `--remeshed-checkpoint-out` | empty | Output checkpoint directory for remeshed restart. |
| `--remesh-wire-ring-factor` | `8.0` | Near-wire forced refinement radius, measured in wire radii. |
| `--foreign-restart-checkpoint` | empty | Input checkpoint for source-to-target experiment projection. |
| `--foreign-source-experiment-index` | `-1` | Source experiment index for foreign checkpoint projection. |
| `--foreign-target-experiment-index` | `-1` | Target experiment index for foreign checkpoint projection. |
| `--foreign-checkpoint-out` | empty | Output checkpoint directory for foreign checkpoint projection. |

## 14. Typical workflow

A practical workflow is:

```bash
# 1. Start a fresh abs simulation
mpirun -np 48 --use-hwthread-cpus python main.py --experiment-index 1

# 2. Continue the transient later
mpirun -np 48 --use-hwthread-cpus python main.py \
  --experiment-index 1 \
  --restart-from-last-transient \
  --existing-run-root PlumeCase_Brodowicz_Air_reduced/runs/base_YYYYMMDD_HHMMSS_pidXXXXX

# 3. Refine the latest checkpoint offline
python main.py \
  --experiment-index 1 \
  --refine-restart-checkpoint PlumeCase_Brodowicz_Air_reduced/runs/base_YYYYMMDD_HHMMSS_pidXXXXX/base/restart_checkpoint \
  --refined-checkpoint-out PlumeCase_Brodowicz_Air_reduced/runs/base_YYYYMMDD_HHMMSS_pidXXXXX/base/restart_checkpoint_amr \
  --amr-top-fraction 0.10 \
  --amr-levels 2 \
  --amr-dt-factor 0.25

# 4. Continue from the AMR checkpoint-owned mesh
mpirun -np 48 --use-hwthread-cpus python main.py \
  --experiment-index 1 \
  --existing-run-root PlumeCase_Brodowicz_Air_reduced/runs/base_YYYYMMDD_HHMMSS_pidXXXXX \
  --restart-from-checkpoint-mesh PlumeCase_Brodowicz_Air_reduced/runs/base_YYYYMMDD_HHMMSS_pidXXXXX/base/restart_checkpoint_amr

# 5. Optionally compute a steady solution from the latest transient state
mpirun -np 48 --use-hwthread-cpus python main.py \
  --experiment-index 1 \
  --steady-from-last-transient \
  --existing-run-root PlumeCase_Brodowicz_Air_reduced/runs/base_YYYYMMDD_HHMMSS_pidXXXXX
```

## 15. Notes and caveats

- Fresh runs regenerate the mesh unless an existing run root with the required mesh files is supplied.
- Restart runs are safest when `state.h5` and `state.json` exist in the checkpoint directory.
- The code assumes legacy FEniCS APIs such as `fenics.XDMFFile`, `fenics.HDF5File`, `SubMesh`, and mixed `FunctionAssigner` workflows.
- The output mesh may be scaled internally for nondimensional solving and then dimensionalized again for saved fields.
- MPI runs should be launched consistently; avoid mixing serial and MPI-generated checkpoint data unless you have verified compatibility.

## 16. Inspecting results

The main field outputs are XDMF/HDF5 files. They can be opened in ParaView:

```text
base/air_temperature.xdmf
base/air_velocity.xdmf
base/air_pressure.xdmf
base/air_temperature_heatflux.xdmf
base/air_temperature_heatflux_mag.xdmf
```

The main CSV diagnostics are:

```text
transient_history.csv
base/plane_fluxes.csv
```

`transient_history.csv` is also used by the approximate restart path if a true checkpoint is not available.
