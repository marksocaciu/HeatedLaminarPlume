# Chunked Plume Run + Checkpoint Archive + Periodic Remesh

This README explains how to use `run_plume_chunked_remesh.sh` for long legacy FEniCS plume simulations.

The script is designed for the modified `main.py` that supports:

```bash
--stop-at-step <GLOBAL_STEP>
```

The purpose is to run the transient simulation in safe chunks, archive restart checkpoints after each chunk, and optionally perform offline remeshing at regular global timestep intervals.

---

## 1. What the script does

The script repeatedly performs the following sequence:

```text
1. Run solver until a clean global stop step.
2. Let the solver write RUN_ROOT/base/restart_checkpoint.
3. After the solver exits, copy that checkpoint to RUN_ROOT/base/restart_checkpoint_<step>.
4. Every REMESH_EVERY steps, run offline AMR/remeshing.
5. Continue the next solver segment from the remeshed checkpoint if one was created.
```

A typical sequence with `ARCHIVE_EVERY=5000` and `REMESH_EVERY=10000` is:

```text
run 0      -> 5000
archive    restart_checkpoint_5000

run 5000   -> 10000
archive    restart_checkpoint_10000
remesh     restart_checkpoint_10000 -> restart_checkpoint_remesh_10000

run 10000  -> 15000 from restart_checkpoint_remesh_10000
archive    restart_checkpoint_15000

run 15000  -> 20000
archive    restart_checkpoint_20000
remesh     restart_checkpoint_20000 -> restart_checkpoint_remesh_20000
```

The script does **not** copy checkpoints while the solver is running. This is important because copying HDF5 checkpoint files during writing can corrupt or partially copy the checkpoint.

---

## 2. Required patched solver behavior

This script assumes that `main.py`, `solver/base_solver.py`, and `solver/abe_solver.py` have been patched so that:

```bash
--stop-at-step 15000
```

means:

```text
stop cleanly after accepting global timestep 15000
```

The start step is **not** passed manually to `main.py`. It is read from the checkpoint metadata:

```text
restart_checkpoint/state.json
```

For example, if `state.json` contains:

```json
{
  "step": 10000,
  "time": 12.345,
  "dt": 0.001
}
```

and the command uses:

```bash
--stop-at-step 15000
```

then the solver advances from global step `10000` to global step `15000`.

---

## 3. Important path convention

Use `RUN_ROOT` as the parent run folder, not the internal `base` folder.

Preferred:

```bash
RUN_ROOT="PlumeCase_Brodowicz_Air/runs/base_20260601_121043_pid16609"
```

The solver writes checkpoints under:

```text
RUN_ROOT/base/restart_checkpoint
```

So the full live checkpoint path becomes:

```text
PlumeCase_Brodowicz_Air/runs/base_20260601_121043_pid16609/base/restart_checkpoint
```

Do **not** normally set:

```bash
RUN_ROOT="PlumeCase_Brodowicz_Air/runs/base_20260601_121043_pid16609/base"
```

because older scripts would then look for:

```text
.../base/base/restart_checkpoint
```

This script includes normalization logic: if `RUN_ROOT` accidentally ends in `/base`, it strips that last folder and uses the parent run folder. Still, the cleaner habit is to provide the parent run folder directly.

---

## 4. Main variables

The script is configured through environment variables.

| Variable | Meaning | Default |
|---|---|---:|
| `PYTHON_BIN` | Python executable | `python3` |
| `MPIRUN_BIN` | MPI launcher | `mpirun` |
| `NP` | Number of MPI ranks | `8` |
| `MAIN_PY` | Path to `main.py` | `main.py` |
| `FORMULATION` | Solver formulation | `base` |
| `EXPERIMENT_INDEX` | Experiment index passed to `main.py` | `1` |
| `RUN_ROOT` | Parent run folder | empty |
| `OUTPUT_SUBDIR` | Subfolder containing checkpoints | `base` |
| `START_STEP` | Starting global step used only for non-auto-resume logic | `0` |
| `FINAL_STEP` | Final global timestep target | `50000` |
| `ARCHIVE_EVERY` | Checkpoint archive cadence | `5000` |
| `REMESH_EVERY` | Remesh cadence | `10000` |
| `AUTO_RESUME` | Resume from newest checkpoint automatically | `0` |
| `FIRST_RESTART_CHECKPOINT` | Manual restart checkpoint for first segment | empty |
| `AMR_TOP_FRACTION` | Fraction of cells marked for AMR | `0.05` |
| `AMR_LEVELS` | Number of AMR refinement levels | `1` |
| `AMR_DT_FACTOR` | Restart timestep reduction factor after AMR | `0.25` |
| `REMESH_WIRE_RING_FACTOR` | Wire ring refinement factor | `8.0` |

---

## 5. Fresh run example

For a fresh base-formulation run:

```bash
chmod +x run_plume_chunked_remesh.sh

FORMULATION=base \
EXPERIMENT_INDEX=1 \
NP=8 \
FINAL_STEP=50000 \
ARCHIVE_EVERY=5000 \
REMESH_EVERY=10000 \
./run_plume_chunked_remesh.sh
```

For ABE:

```bash
chmod +x run_plume_chunked_remesh.sh

FORMULATION=abe \
EXPERIMENT_INDEX=1 \
NP=8 \
FINAL_STEP=50000 \
ARCHIVE_EVERY=5000 \
REMESH_EVERY=10000 \
./run_plume_chunked_remesh.sh
```

For a fresh run, `RUN_ROOT` may be left empty. After the first segment exits, the script tries to infer the newly created run folder by searching for:

```text
*/runs/base_*/base/restart_checkpoint/state.json
```

If this inference fails, rerun with an explicit `RUN_ROOT` after the first chunk has been created.

---

## 6. Continue an existing run manually

Use this when you know exactly which checkpoint should be used first.

Example: continue from a remeshed checkpoint at global step `10000`:

```bash
FORMULATION=base \
EXPERIMENT_INDEX=1 \
NP=8 \
RUN_ROOT="PlumeCase_Brodowicz_Air/runs/base_20260601_121043_pid16609" \
START_STEP=10000 \
FINAL_STEP=50000 \
FIRST_RESTART_CHECKPOINT="PlumeCase_Brodowicz_Air/runs/base_20260601_121043_pid16609/base/restart_checkpoint_remesh_10000" \
ARCHIVE_EVERY=5000 \
REMESH_EVERY=10000 \
./run_plume_chunked_remesh.sh
```

In this case:

```text
START_STEP=10000
FIRST_RESTART_CHECKPOINT=.../restart_checkpoint_remesh_10000
```

must be consistent with the `step` value inside:

```text
restart_checkpoint_remesh_10000/state.json
```

---

## 7. Auto-resume after a crash or failed run

The safest way to resume after a crash is:

```bash
AUTO_RESUME=1 \
RUN_ROOT="PlumeCase_Brodowicz_Air/runs/base_20260601_121043_pid16609" \
FORMULATION=base \
EXPERIMENT_INDEX=1 \
NP=8 \
FINAL_STEP=50000 \
ARCHIVE_EVERY=5000 \
REMESH_EVERY=10000 \
./run_plume_chunked_remesh.sh
```

With `AUTO_RESUME=1`, the script scans:

```text
RUN_ROOT/base
```

for valid checkpoint folders:

```text
restart_checkpoint
restart_checkpoint_<step>
restart_checkpoint_remesh_<step>
```

A checkpoint is considered valid only if it contains both:

```text
state.h5
state.json
```

The script reads `step` from each `state.json`, selects the checkpoint with the largest global step, and resumes from it.

If both of these exist at the same step:

```text
restart_checkpoint_10000
restart_checkpoint_remesh_10000
```

then the script prefers:

```text
restart_checkpoint_remesh_10000
```

because after a remesh milestone, the remeshed checkpoint is the better restart source.

---

## 8. How the next stop step is chosen

The script uses absolute/global steps.

With:

```bash
ARCHIVE_EVERY=5000
```

if the current checkpoint is at:

```text
current_step = 13400
```

then the next stop step is:

```text
15000
```

not:

```text
18400
```

This keeps all archives aligned to clean milestone steps:

```text
5000, 10000, 15000, 20000, ...
```

---

## 9. Output files and folders

The live checkpoint is always:

```text
RUN_ROOT/base/restart_checkpoint
```

Archived checkpoints are:

```text
RUN_ROOT/base/restart_checkpoint_5000
RUN_ROOT/base/restart_checkpoint_10000
RUN_ROOT/base/restart_checkpoint_15000
...
```

Remeshed checkpoints are:

```text
RUN_ROOT/base/restart_checkpoint_remesh_10000
RUN_ROOT/base/restart_checkpoint_remesh_20000
...
```

Coarse remesh files are written under:

```text
RUN_ROOT/coarse
```

The transient history remains:

```text
RUN_ROOT/transient_history.csv
```

As long as you continue with the same `RUN_ROOT`, the transient CSV should continue appending from one segment to the next.

---

## 10. Commands executed internally

### Solver segment

A solver segment command has the form:

```bash
mpirun -np "$NP" --use-hwthread-cpus python3 main.py \
  --formulation "$FORMULATION" \
  --experiment-index "$EXPERIMENT_INDEX" \
  --existing-run-root "$RUN_ROOT" \
  --restart-from-checkpoint-mesh "$RESTART_CHECKPOINT" \
  --stop-at-step "$NEXT_STEP"
```

For the first fresh segment, `--existing-run-root` and `--restart-from-checkpoint-mesh` may be omitted.

### Offline remesh segment

A remesh command has the form:

```bash
mpirun -np 1 python3 main.py \
  --experiment-index "$EXPERIMENT_INDEX" \
  --remesh-restart-checkpoint "$RUN_ROOT/base/restart_checkpoint_$STEP" \
  --coarse-remesh-run-root "$RUN_ROOT/coarse" \
  --remeshed-checkpoint-out "$RUN_ROOT/base/restart_checkpoint_remesh_$STEP" \
  --amr-top-fraction "$AMR_TOP_FRACTION" \
  --amr-levels "$AMR_LEVELS" \
  --amr-dt-factor "$AMR_DT_FACTOR" \
  --remesh-wire-ring-factor "$REMESH_WIRE_RING_FACTOR"
```

The remesh command is run with one MPI rank by default:

```bash
mpirun -np 1
```

---

## 11. Common usage patterns

### Run until 100000, archive every 5000, remesh every 10000

```bash
FORMULATION=base \
EXPERIMENT_INDEX=1 \
NP=8 \
FINAL_STEP=100000 \
ARCHIVE_EVERY=5000 \
REMESH_EVERY=10000 \
./run_plume_chunked_remesh.sh
```

### Resume after crash

```bash
AUTO_RESUME=1 \
RUN_ROOT="PlumeCase_Brodowicz_Air/runs/base_20260601_121043_pid16609" \
FORMULATION=base \
EXPERIMENT_INDEX=1 \
NP=8 \
FINAL_STEP=100000 \
ARCHIVE_EVERY=5000 \
REMESH_EVERY=10000 \
./run_plume_chunked_remesh.sh
```

### Use fewer MPI ranks

```bash
NP=4 \
AUTO_RESUME=1 \
RUN_ROOT="PlumeCase_Brodowicz_Air/runs/base_20260601_121043_pid16609" \
./run_plume_chunked_remesh.sh
```

### Use a different Python executable

```bash
PYTHON_BIN="/opt/miniconda3/envs/fenicsproject/bin/python" \
AUTO_RESUME=1 \
RUN_ROOT="PlumeCase_Brodowicz_Air/runs/base_20260601_121043_pid16609" \
./run_plume_chunked_remesh.sh
```

---

## 12. Safety checks included in the script

The script checks that:

```text
ARCHIVE_EVERY > 0
REMESH_EVERY > 0
FINAL_STEP > START_STEP, unless AUTO_RESUME=1
live checkpoint exists before archiving
live checkpoint step matches the archive label
remeshed checkpoint exists after remeshing
AUTO_RESUME finds a valid checkpoint before continuing
```

The most important safety check is this one:

```text
Live checkpoint step must equal the archive step.
```

For example, before creating:

```text
restart_checkpoint_15000
```

the script reads:

```text
restart_checkpoint/state.json
```

and verifies that it contains:

```json
{"step": 15000}
```

If the step does not match, the script refuses to archive the checkpoint to avoid mislabeling.

---

## 13. Troubleshooting

### Problem: script looks for `base/base/restart_checkpoint`

Use the parent run folder:

```bash
RUN_ROOT="PlumeCase_Brodowicz_Air/runs/base_20260601_121043_pid16609"
```

not:

```bash
RUN_ROOT="PlumeCase_Brodowicz_Air/runs/base_20260601_121043_pid16609/base"
```

The script should normalize this automatically, but using the parent folder is cleaner.

### Problem: auto-resume says no valid checkpoint found

Check that the folder contains:

```text
RUN_ROOT/base/restart_checkpoint/state.h5
RUN_ROOT/base/restart_checkpoint/state.json
```

or archived versions such as:

```text
RUN_ROOT/base/restart_checkpoint_10000/state.h5
RUN_ROOT/base/restart_checkpoint_10000/state.json
```

### Problem: transient CSV has duplicate rows

This can happen if you resume from an older checkpoint while `transient_history.csv` already contains later rows.

The normal forward-only workflow avoids this. If you intentionally roll back to an older checkpoint, manually backup or trim the CSV first.

### Problem: first fresh run cannot infer `RUN_ROOT`

Set `RUN_ROOT` manually after the first segment or run the first segment directly with `main.py`, then use this script for the remaining chunks.

### Problem: remesh fails under MPI

The script uses:

```bash
mpirun -np 1
```

for remeshing. If your local installation prefers serial execution for remeshing, edit `run_remesh()` and replace:

```bash
local cmd=("${MPIRUN_BIN}" -np 1 "${PYTHON_BIN}" "${MAIN_PY}"
```

with:

```bash
local cmd=("${PYTHON_BIN}" "${MAIN_PY}"
```

---

## 14. Recommended workflow

For long production runs, use this pattern:

```bash
FORMULATION=base \
EXPERIMENT_INDEX=1 \
NP=8 \
FINAL_STEP=100000 \
ARCHIVE_EVERY=5000 \
REMESH_EVERY=10000 \
./run_plume_chunked_remesh.sh
```

After any failure, resume with:

```bash
AUTO_RESUME=1 \
RUN_ROOT="PlumeCase_Brodowicz_Air/runs/base_YYYYMMDD_HHMMSS_pidXXXXX" \
FORMULATION=base \
EXPERIMENT_INDEX=1 \
NP=8 \
FINAL_STEP=100000 \
ARCHIVE_EVERY=5000 \
REMESH_EVERY=10000 \
./run_plume_chunked_remesh.sh
```

This keeps the workflow deterministic, avoids unsafe checkpoint copying, and keeps remeshing aligned with clean global timestep milestones.
