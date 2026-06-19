#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Chunked plume runner with safe checkpoint archiving and periodic offline remesh
# -----------------------------------------------------------------------------
# Assumes main.py has the --stop-at-step patch.
# Workflow:
#   1) run solver until an absolute/global stop step
#   2) after solver exits, archive base/restart_checkpoint to base/restart_checkpoint_<step>
#   3) every REMESH_EVERY steps, perform offline remesh from that archived checkpoint
#   4) next segment restarts from the remeshed checkpoint; otherwise from the normal checkpoint
# -----------------------------------------------------------------------------

# ----------------------------- user settings ---------------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
NP="${NP:-12}"

MAIN_PY="${MAIN_PY:-main.py}"
FORMULATION="${FORMULATION:-abe}"        # base or abe
EXPERIMENT_INDEX="${EXPERIMENT_INDEX:-1}"

# Existing run root. Leave empty for a fresh first segment.
# Example: PlumeCase_Brodowicz_Air/runs/base_20260601_121043_pid16609
RUN_ROOT="${RUN_ROOT:-}"

# Absolute/global accepted-step controls.
START_STEP="${START_STEP:-0}"
FINAL_STEP="${FINAL_STEP:-5000000}"
ARCHIVE_EVERY="${ARCHIVE_EVERY:-5000}"
REMESH_EVERY="${REMESH_EVERY:-10000}"

# Remeshing controls.
AMR_TOP_FRACTION="${AMR_TOP_FRACTION:-0.05}"
AMR_LEVELS="${AMR_LEVELS:-1}"
AMR_DT_FACTOR="${AMR_DT_FACTOR:-0.25}"
REMESH_WIRE_RING_FACTOR="${REMESH_WIRE_RING_FACTOR:-8.0}"

# Set to 1 if the first segment should restart from an existing checkpoint.
# Then FIRST_RESTART_CHECKPOINT must point to a valid checkpoint directory.
FIRST_RESTART_CHECKPOINT="${FIRST_RESTART_CHECKPOINT:-}"
# -----------------------------------------------------------------------------

if (( ARCHIVE_EVERY <= 0 )); then
    echo "ARCHIVE_EVERY must be > 0" >&2
    exit 1
fi

if (( REMESH_EVERY <= 0 )); then
    echo "REMESH_EVERY must be > 0" >&2
    exit 1
fi

if (( FINAL_STEP <= START_STEP )); then
    echo "FINAL_STEP must be greater than START_STEP" >&2
    exit 1
fi

current_step="${START_STEP}"
restart_checkpoint="${FIRST_RESTART_CHECKPOINT}"

run_solver_segment() {
    local stop_step="$1"
    local restart_dir="$2"

    local cmd=("${MPIRUN_BIN}" -np "${NP}" --use-hwthread-cpus "${PYTHON_BIN}" "${MAIN_PY}"
        --formulation "${FORMULATION}"
        --experiment-index "${EXPERIMENT_INDEX}"
        --stop-at-step "${stop_step}"
    )

    if [[ -n "${RUN_ROOT}" ]]; then
        cmd+=(--existing-run-root "${RUN_ROOT}")
    fi

    if [[ -n "${restart_dir}" ]]; then
        cmd+=(--restart-from-checkpoint-mesh "${restart_dir}")
    elif [[ -n "${RUN_ROOT}" ]]; then
        # Continue from the run_root/base/restart_checkpoint written by the previous segment.
        cmd+=(--restart-from-checkpoint-mesh "${RUN_ROOT}/base/restart_checkpoint")
    fi

    echo
    echo "======================================================================"
    echo "Running solver segment to global step ${stop_step}"
    echo "Command: ${cmd[*]}"
    echo "======================================================================"
    "${cmd[@]}"
}

infer_run_root_after_fresh_start() {
    # If RUN_ROOT was empty, infer the newest run directory for this experiment.
    # This assumes the experiment directory name starts with PlumeCase and contains runs/base_*.
    # For maximum robustness, set RUN_ROOT explicitly after the first run.
    local newest
    newest=$(find . -path "*/runs/base_*" -type d -print0 | xargs -0 ls -dt 2>/dev/null | head -n 1 || true)
    if [[ -z "${newest}" ]]; then
        echo "Could not infer RUN_ROOT after fresh run. Set RUN_ROOT manually." >&2
        exit 1
    fi
    RUN_ROOT="${newest#./}"
    echo "Inferred RUN_ROOT=${RUN_ROOT}"
}

archive_checkpoint() {
    local step="$1"
    local live="${RUN_ROOT}/base/restart_checkpoint"
    local archive="${RUN_ROOT}/base/restart_checkpoint_${step}"

    if [[ ! -f "${live}/state.h5" || ! -f "${live}/state.json" ]]; then
        echo "Missing live restart checkpoint at ${live}" >&2
        exit 1
    fi

    rm -rf "${archive}"
    cp -a "${live}" "${archive}"
    echo "Archived checkpoint: ${archive}"
}

run_remesh() {
    local step="$1"
    local input="${RUN_ROOT}/base/restart_checkpoint_${step}"
    local coarse="${RUN_ROOT}/coarse"
    local output="${RUN_ROOT}/base/restart_checkpoint_remesh_${step}"

    mkdir -p "${coarse}"
    rm -rf "${output}"

    local cmd=("${PYTHON_BIN}" "${MAIN_PY}"
        --experiment-index "${EXPERIMENT_INDEX}"
        --remesh-restart-checkpoint "${input}"
        --coarse-remesh-run-root "${coarse}"
        --remeshed-checkpoint-out "${output}"
        --amr-top-fraction "${AMR_TOP_FRACTION}"
        --amr-levels "${AMR_LEVELS}"
        --amr-dt-factor "${AMR_DT_FACTOR}"
        --remesh-wire-ring-factor "${REMESH_WIRE_RING_FACTOR}"
    )

    echo
    echo "======================================================================"
    echo "Running offline remesh at global step ${step}"
    echo "Command: ${cmd[*]}"
    echo "======================================================================"
    "${cmd[@]}"

    if [[ ! -f "${output}/state.h5" || ! -f "${output}/state.json" ]]; then
        echo "Remeshed checkpoint was not created correctly: ${output}" >&2
        exit 1
    fi

    echo "Remeshed checkpoint: ${output}"
    restart_checkpoint="${output}"
}

while (( current_step < FINAL_STEP )); do
    next_step=$(( current_step + ARCHIVE_EVERY ))
    if (( next_step > FINAL_STEP )); then
        next_step="${FINAL_STEP}"
    fi

    run_solver_segment "${next_step}" "${restart_checkpoint}"

    if [[ -z "${RUN_ROOT}" ]]; then
        infer_run_root_after_fresh_start
    fi

    archive_checkpoint "${next_step}"

    restart_checkpoint="${RUN_ROOT}/base/restart_checkpoint_${next_step}"

    if (( next_step % REMESH_EVERY == 0 )); then
        run_remesh "${next_step}"
    fi

    current_step="${next_step}"
done

echo
echo "Completed chunked run up to global step ${FINAL_STEP}."
echo "Final restart source: ${restart_checkpoint}"
echo "Run root: ${RUN_ROOT}"
