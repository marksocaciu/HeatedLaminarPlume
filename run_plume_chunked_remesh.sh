#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Chunked plume runner with safe checkpoint archiving, periodic offline remesh,
# and automatic resume from the newest valid checkpoint.
# -----------------------------------------------------------------------------
# Requires main.py with --stop-at-step.
#
# Normal workflow:
#   1) run solver until an absolute/global stop step
#   2) after solver exits, archive ${OUTPUT_SUBDIR}/restart_checkpoint to ${OUTPUT_SUBDIR}/restart_checkpoint_<step>
#   3) every REMESH_EVERY steps, perform offline remesh from that archived checkpoint
#   4) next segment restarts from the remeshed checkpoint; otherwise from archive/live checkpoint
#
# Crash recovery:
#   AUTO_RESUME=1 RUN_ROOT=<existing run root> ./run_plume_chunked_remesh_autoresume.sh
# scans RUN_ROOT/${OUTPUT_SUBDIR} for:
#   - restart_checkpoint
#   - restart_checkpoint_<step>
#   - restart_checkpoint_remesh_<step>
# and resumes from the checkpoint with the largest metadata step. If remeshed and
# non-remeshed checkpoints exist for the same step, the remeshed one is preferred.
# -----------------------------------------------------------------------------

# ----------------------------- user settings ---------------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
NP="${NP:-8}"

MAIN_PY="${MAIN_PY:-main.py}"
FORMULATION="${FORMULATION:-base}"        # base or abe
EXPERIMENT_INDEX="${EXPERIMENT_INDEX:-1}"

# Existing run root. Leave empty only for a fresh first segment.
# Preferred value: the parent run folder, e.g.
#   PlumeCase_Brodowicz_Air/runs/base_20260601_121043_pid16609
#
# For convenience, this script also accepts the output folder itself, e.g.
#   PlumeCase_Brodowicz_Air/runs/base_20260601_121043_pid16609/base
# and normalizes it back to the parent run folder to avoid base/base paths.
RUN_ROOT="${RUN_ROOT:-}"

# Folder inside RUN_ROOT where the solver writes live checkpoints.
# Your current base and ABE workflows both write to RUN_ROOT/base.
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-base}"

# Absolute/global accepted-step controls.
START_STEP="${START_STEP:-0}"
FINAL_STEP="${FINAL_STEP:-50000}"
ARCHIVE_EVERY="${ARCHIVE_EVERY:-5000}"
REMESH_EVERY="${REMESH_EVERY:-10000}"

# Set AUTO_RESUME=1 to ignore START_STEP/FIRST_RESTART_CHECKPOINT and continue
# from the newest valid checkpoint found under RUN_ROOT/${OUTPUT_SUBDIR}.
AUTO_RESUME="${AUTO_RESUME:-0}"

# Remeshing controls.
AMR_TOP_FRACTION="${AMR_TOP_FRACTION:-0.05}"
AMR_LEVELS="${AMR_LEVELS:-1}"
AMR_DT_FACTOR="${AMR_DT_FACTOR:-0.25}"
REMESH_WIRE_RING_FACTOR="${REMESH_WIRE_RING_FACTOR:-8.0}"

# Manual first restart source. Usually empty; AUTO_RESUME is preferred after crashes.
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

if (( FINAL_STEP <= START_STEP )) && [[ "${AUTO_RESUME}" != "1" ]]; then
    echo "FINAL_STEP must be greater than START_STEP" >&2
    exit 1
fi

normalize_run_root() {
    # Accept either:
    #   RUN_ROOT=/.../runs/base_YYYY...
    # or accidentally:
    #   RUN_ROOT=/.../runs/base_YYYY.../base
    # and store the parent in RUN_ROOT.
    if [[ -n "${RUN_ROOT}" ]]; then
        # Remove a trailing slash first.
        RUN_ROOT="${RUN_ROOT%/}"

        local last_component
        last_component="$(basename "${RUN_ROOT}")"

        if [[ "${last_component}" == "${OUTPUT_SUBDIR}" ]]; then
            echo "RUN_ROOT points to the output folder (${RUN_ROOT})."
            RUN_ROOT="$(dirname "${RUN_ROOT}")"
            echo "Normalized RUN_ROOT to parent run folder: ${RUN_ROOT}"
        fi
    fi
}

mode_dir() {
    echo "${RUN_ROOT}/${OUTPUT_SUBDIR}"
}

normalize_run_root

checkpoint_step() {
    local checkpoint_dir="$1"
    if [[ ! -f "${checkpoint_dir}/state.h5" || ! -f "${checkpoint_dir}/state.json" ]]; then
        return 1
    fi

    "${PYTHON_BIN}" - "${checkpoint_dir}/state.json" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r") as f:
    meta = json.load(f)
print(int(meta["step"]))
PY
}

checkpoint_priority() {
    local checkpoint_dir="$1"
    local name
    name="$(basename "${checkpoint_dir}")"
    if [[ "${name}" =~ ^restart_checkpoint_remesh_[0-9]+$ ]]; then
        echo 2
    elif [[ "${name}" =~ ^restart_checkpoint_[0-9]+$ ]]; then
        echo 1
    elif [[ "${name}" == "restart_checkpoint" ]]; then
        echo 0
    else
        echo -1
    fi
}

find_latest_checkpoint() {
    local base_dir="$(mode_dir)"
    local best_dir=""
    local best_step=-1
    local best_priority=-1

    if [[ -z "${RUN_ROOT}" ]]; then
        echo "AUTO_RESUME requires RUN_ROOT to be set." >&2
        exit 1
    fi

    if [[ ! -d "${base_dir}" ]]; then
        echo "Cannot auto-resume: missing base directory ${base_dir}" >&2
        exit 1
    fi

    while IFS= read -r -d '' candidate; do
        local step priority
        if ! step="$(checkpoint_step "${candidate}" 2>/dev/null)"; then
            continue
        fi
        priority="$(checkpoint_priority "${candidate}")"

        if (( step > best_step )) || { (( step == best_step )) && (( priority > best_priority )); }; then
            best_step="${step}"
            best_priority="${priority}"
            best_dir="${candidate}"
        fi
    done < <(
        find "${base_dir}" -maxdepth 1 -type d \
            \( -name 'restart_checkpoint' \
            -o -name 'restart_checkpoint_[0-9]*' \
            -o -name 'restart_checkpoint_remesh_[0-9]*' \) \
            -print0
    )

    if [[ -z "${best_dir}" ]]; then
        echo "No valid checkpoint found under ${base_dir}" >&2
        exit 1
    fi

    echo "${best_step}|${best_dir}"
}

next_stop_step_after() {
    local current="$1"
    local next

    if (( current % ARCHIVE_EVERY == 0 )); then
        next=$(( current + ARCHIVE_EVERY ))
    else
        next=$(( ((current / ARCHIVE_EVERY) + 1) * ARCHIVE_EVERY ))
    fi

    if (( next > FINAL_STEP )); then
        next="${FINAL_STEP}"
    fi

    echo "${next}"
}

infer_run_root_after_fresh_start() {
    local newest_mode_dir

    # Find the newest solver output directory that already has a live checkpoint,
    # then strip OUTPUT_SUBDIR to recover the parent run root expected by main.py.
    newest_mode_dir=$(
        find . -type f -path "*/runs/base_*/${OUTPUT_SUBDIR}/restart_checkpoint/state.json" -print0 \
        | xargs -0 -r ls -t 2>/dev/null \
        | head -n 1 \
        | sed "s#/${OUTPUT_SUBDIR}/restart_checkpoint/state.json##" \
        || true
    )

    if [[ -z "${newest_mode_dir}" ]]; then
        echo "Could not infer RUN_ROOT after fresh run. Set RUN_ROOT manually." >&2
        exit 1
    fi

    RUN_ROOT="${newest_mode_dir#./}"
    normalize_run_root
    echo "Inferred RUN_ROOT=${RUN_ROOT}"
}

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
        cmd+=(--restart-from-checkpoint-mesh "${RUN_ROOT}/${OUTPUT_SUBDIR}/restart_checkpoint")
    fi

    echo
    echo "======================================================================"
    echo "Running solver segment to global step ${stop_step}"
    if [[ -n "${restart_dir}" ]]; then
        echo "Restart source: ${restart_dir}"
    fi
    echo "Command: ${cmd[*]}"
    echo "======================================================================"
    "${cmd[@]}"
}

archive_checkpoint() {
    local step="$1"
    local live="${RUN_ROOT}/${OUTPUT_SUBDIR}/restart_checkpoint"
    local archive="${RUN_ROOT}/${OUTPUT_SUBDIR}/restart_checkpoint_${step}"
    local live_step

    if [[ ! -f "${live}/state.h5" || ! -f "${live}/state.json" ]]; then
        echo "Missing live restart checkpoint at ${live}" >&2
        exit 1
    fi

    live_step="$(checkpoint_step "${live}")"
    if (( live_step != step )); then
        echo "Live checkpoint step (${live_step}) does not match expected archive step (${step})." >&2
        echo "Not archiving to avoid mislabeling a checkpoint." >&2
        exit 1
    fi

    rm -rf "${archive}"
    cp -a "${live}" "${archive}"
    echo "Archived checkpoint: ${archive}"
}

run_remesh() {
    local step="$1"
    local input="${RUN_ROOT}/${OUTPUT_SUBDIR}/restart_checkpoint_${step}"
    local coarse="${RUN_ROOT}/coarse"
    local output="${RUN_ROOT}/${OUTPUT_SUBDIR}/restart_checkpoint_remesh_${step}"

    mkdir -p "${coarse}"
    rm -rf "${output}"

    local cmd=("${MPIRUN_BIN}" -np 1 ${PYTHON_BIN}" "${MAIN_PY}"
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

current_step="${START_STEP}"
restart_checkpoint="${FIRST_RESTART_CHECKPOINT}"

if [[ "${AUTO_RESUME}" == "1" ]]; then
    latest="$(find_latest_checkpoint)"
    current_step="${latest%%|*}"
    restart_checkpoint="${latest#*|}"
    echo "AUTO_RESUME selected checkpoint: ${restart_checkpoint}"
    echo "AUTO_RESUME current global step: ${current_step}"

    if (( current_step >= FINAL_STEP )); then
        echo "Latest checkpoint step ${current_step} is already >= FINAL_STEP ${FINAL_STEP}. Nothing to do."
        exit 0
    fi
fi

while (( current_step < FINAL_STEP )); do
    next_step="$(next_stop_step_after "${current_step}")"

    run_solver_segment "${next_step}" "${restart_checkpoint}"

    if [[ -z "${RUN_ROOT}" ]]; then
        infer_run_root_after_fresh_start
    fi

    archive_checkpoint "${next_step}"

    restart_checkpoint="${RUN_ROOT}/${OUTPUT_SUBDIR}/restart_checkpoint_${next_step}"

    if (( next_step % REMESH_EVERY == 0 )); then
        run_remesh "${next_step}"
    fi

    current_step="${next_step}"
done

echo
echo "Completed chunked run up to global step ${FINAL_STEP}."
echo "Final restart source: ${restart_checkpoint}"
echo "Run root: ${RUN_ROOT}"
