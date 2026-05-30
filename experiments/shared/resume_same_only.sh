#!/usr/bin/env bash
set -euo pipefail

WS_ROOT="${WS_ROOT:-/home/pt/SWE-bench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE7_SCRIPT_DIR="${SCRIPT_DIR}"

RUNS_ROOT="${RUNS_ROOT:-${WS_ROOT}/PDDL_work_mem/06_artificial_intelligence/experiments/final_validation/phase8}"
RUN_TAG="${RUN_TAG:-}"
RUN_ROOT="${RUN_ROOT:-}"
if [[ -z "${RUN_ROOT}" ]]; then
  if [[ -n "${RUN_TAG}" ]]; then
    RUN_ROOT="${RUNS_ROOT}/${RUN_TAG}"
  else
    RUN_ROOT="$(ls -1dt "${RUNS_ROOT}"/phase8_same_only_* 2>/dev/null | head -n 1 || true)"
  fi
fi
if [[ -z "${RUN_ROOT}" || ! -d "${RUN_ROOT}" ]]; then
  echo "unable to resolve RUN_ROOT under ${RUNS_ROOT}" >&2
  exit 2
fi

SAME_JSON="${SAME_JSON:-${RUN_ROOT}/same_instances.json}"
if [[ ! -f "${SAME_JSON}" ]]; then
  echo "same instance list not found: ${SAME_JSON}" >&2
  exit 2
fi

LOCK_FILE="${RUN_ROOT}/same_only_resume.lock"
mkdir -p "${RUN_ROOT}/orchestrator_logs" "${RUN_ROOT}/orchestrator_state"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "same-only resume is already running for ${RUN_ROOT}" >&2
  exit 3
fi

REPEATS="${REPEATS:-10}"
PROMPT_PROFILE="${PROMPT_PROFILE:-prompt_base}"
MODEL_CONFIG="${MODEL_CONFIG:-${WS_ROOT}/SWE-agent/config/kimi25_siliconflow.yaml}"
PER_INSTANCE_CALL_LIMIT="${PER_INSTANCE_CALL_LIMIT:-0}"
NOMEM_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${NOMEM_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC:-1200}"
WITHMEM_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${WITHMEM_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC:-1200}"
MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-1}"
EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-1200}"
ENV_ERROR_RETRIES="${ENV_ERROR_RETRIES:-1}"
ENV_FATAL_EXIT_CODE="${ENV_FATAL_EXIT_CODE:-86}"
IMAGE_PULL_TIMEOUT_SEC="${IMAGE_PULL_TIMEOUT_SEC:-1800}"
PHASE7_PREPARE_SLOTS="${PHASE7_PREPARE_SLOTS:-2}"
PHASE7_PREPARE_SLOT_POLL_SEC="${PHASE7_PREPARE_SLOT_POLL_SEC:-10}"
NOMEM_RUNTIME_PREWARM="${NOMEM_RUNTIME_PREWARM:-0}"
WITHMEM_RUNTIME_PREWARM="${WITHMEM_RUNTIME_PREWARM:-1}"
NOMEM_RUNTIME_WARMUP_TIMEOUT_SEC="${NOMEM_RUNTIME_WARMUP_TIMEOUT_SEC:-1800}"
WITHMEM_RUNTIME_WARMUP_TIMEOUT_SEC="${WITHMEM_RUNTIME_WARMUP_TIMEOUT_SEC:-1800}"
NOMEM_PYTHON_STANDALONE_DIR="${NOMEM_PYTHON_STANDALONE_DIR:-/root}"
WITHMEM_PYTHON_STANDALONE_DIR="${WITHMEM_PYTHON_STANDALONE_DIR:-/root}"
SKIP_INSTANCE_PREPARE="${SKIP_INSTANCE_PREPARE:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

GLOBAL_HEAVY_SLOTS="${GLOBAL_HEAVY_SLOTS:-2}"
GLOBAL_HEAVY_SLOT_DIR="${GLOBAL_HEAVY_SLOT_DIR:-${RUN_ROOT}/resource_slots/same_global}"
GLOBAL_HEAVY_SLOT_POLL_SEC="${GLOBAL_HEAVY_SLOT_POLL_SEC:-10}"

NOMEM_HEAVY_SLOTS="${NOMEM_HEAVY_SLOTS:-1}"
WITHMEM_HEAVY_SLOTS="${WITHMEM_HEAVY_SLOTS:-1}"
NOMEM_HEAVY_SLOT_DIR="${NOMEM_HEAVY_SLOT_DIR:-${RUN_ROOT}/resource_slots/same_nomem}"
WITHMEM_HEAVY_SLOT_DIR="${WITHMEM_HEAVY_SLOT_DIR:-${RUN_ROOT}/resource_slots/same_with_mem}"
LANE_HEAVY_SLOT_POLL_SEC="${LANE_HEAVY_SLOT_POLL_SEC:-10}"

RESUME_MODE="${RESUME_MODE:-1}"
REDO_EXISTING="${REDO_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

NOMEM_SCRIPT="${PHASE7_SCRIPT_DIR}/run_same_problem_nomem.sh"
# Allow cost-saving override: SKIP_NOMEM=1 replaces nomem with a no-op.
# Use when historical nomem data already exists and re-running adds no new information.
if [[ "${SKIP_NOMEM:-0}" == "1" ]]; then
  NOMEM_SCRIPT="/bin/true"
fi
WITHMEM_SCRIPT="${PHASE7_SCRIPT_DIR}/run_same_problem_withmem.sh"

echo "run_root=${RUN_ROOT}"
echo "same_json=${SAME_JSON}"
echo "repeats=${REPEATS}"
echo "global_heavy_slots=${GLOBAL_HEAVY_SLOTS}"
echo "nomem_heavy_slots=${NOMEM_HEAVY_SLOTS}"
echo "withmem_heavy_slots=${WITHMEM_HEAVY_SLOTS}"
echo "image_pull_timeout_sec=${IMAGE_PULL_TIMEOUT_SEC}"
echo "prepare_slots=${PHASE7_PREPARE_SLOTS}"
echo "nomem_runtime_prewarm=${NOMEM_RUNTIME_PREWARM}"
echo "withmem_runtime_prewarm=${WITHMEM_RUNTIME_PREWARM}"
echo "nomem_python_standalone_dir=${NOMEM_PYTHON_STANDALONE_DIR}"
echo "withmem_python_standalone_dir=${WITHMEM_PYTHON_STANDALONE_DIR}"

PID_NOMEM=""
PID_WITHMEM=""
cleanup_children() {
  local rc=$?
  trap - EXIT INT TERM
  for pid in "${PID_NOMEM:-}" "${PID_WITHMEM:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  [[ -n "${PID_NOMEM:-}" ]] && wait "${PID_NOMEM}" 2>/dev/null || true
  [[ -n "${PID_WITHMEM:-}" ]] && wait "${PID_WITHMEM}" 2>/dev/null || true
  exit "${rc}"
}
trap cleanup_children EXIT INT TERM

run_nomem() {
  env \
    RUN_ROOT="${RUN_ROOT}" \
    INSTANCE_LIST_JSON="${SAME_JSON}" \
    REPEATS="${REPEATS}" \
    PROMPT_PROFILE="${PROMPT_PROFILE}" \
    MODEL_CONFIG="${MODEL_CONFIG}" \
    PER_INSTANCE_CALL_LIMIT="${PER_INSTANCE_CALL_LIMIT}" \
    INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${NOMEM_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC}" \
    MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL}" \
    EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC}" \
    ENV_ERROR_RETRIES="${ENV_ERROR_RETRIES}" \
    ENV_FATAL_EXIT_CODE="${ENV_FATAL_EXIT_CODE}" \
    IMAGE_PULL_TIMEOUT_SEC="${IMAGE_PULL_TIMEOUT_SEC}" \
    PHASE7_PREPARE_SLOTS="${PHASE7_PREPARE_SLOTS}" \
    PHASE7_PREPARE_SLOT_POLL_SEC="${PHASE7_PREPARE_SLOT_POLL_SEC}" \
    NOMEM_RUNTIME_PREWARM="${NOMEM_RUNTIME_PREWARM}" \
    NOMEM_RUNTIME_WARMUP_TIMEOUT_SEC="${NOMEM_RUNTIME_WARMUP_TIMEOUT_SEC}" \
    NOMEM_PYTHON_STANDALONE_DIR="${NOMEM_PYTHON_STANDALONE_DIR}" \
    SKIP_INSTANCE_PREPARE="${SKIP_INSTANCE_PREPARE}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE}" \
    RESUME_MODE="${RESUME_MODE}" \
    REDO_EXISTING="${REDO_EXISTING}" \
    DRY_RUN="${DRY_RUN}" \
    PHASE7_GLOBAL_HEAVY_SLOT_DIR="${GLOBAL_HEAVY_SLOT_DIR}" \
    PHASE7_GLOBAL_HEAVY_SLOTS="${GLOBAL_HEAVY_SLOTS}" \
    PHASE7_GLOBAL_HEAVY_SLOT_POLL_SEC="${GLOBAL_HEAVY_SLOT_POLL_SEC}" \
    PHASE7_HEAVY_SLOT_DIR="${NOMEM_HEAVY_SLOT_DIR}" \
    PHASE7_HEAVY_SLOTS="${NOMEM_HEAVY_SLOTS}" \
    PHASE7_HEAVY_SLOT_POLL_SEC="${LANE_HEAVY_SLOT_POLL_SEC}" \
    "${NOMEM_SCRIPT}"
}

run_withmem() {
  env \
    RUN_ROOT="${RUN_ROOT}" \
    INSTANCE_LIST_JSON="${SAME_JSON}" \
    REPEATS="${REPEATS}" \
    PROMPT_PROFILE="${PROMPT_PROFILE}" \
    MODEL_CONFIG="${MODEL_CONFIG}" \
    PER_INSTANCE_CALL_LIMIT="${PER_INSTANCE_CALL_LIMIT}" \
    INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${WITHMEM_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC}" \
    MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL}" \
    EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC}" \
    ENV_ERROR_RETRIES="${ENV_ERROR_RETRIES}" \
    ENV_FATAL_EXIT_CODE="${ENV_FATAL_EXIT_CODE}" \
    IMAGE_PULL_TIMEOUT_SEC="${IMAGE_PULL_TIMEOUT_SEC}" \
    PHASE7_PREPARE_SLOTS="${PHASE7_PREPARE_SLOTS}" \
    PHASE7_PREPARE_SLOT_POLL_SEC="${PHASE7_PREPARE_SLOT_POLL_SEC}" \
    WITHMEM_RUNTIME_PREWARM="${WITHMEM_RUNTIME_PREWARM}" \
    WITHMEM_RUNTIME_WARMUP_TIMEOUT_SEC="${WITHMEM_RUNTIME_WARMUP_TIMEOUT_SEC}" \
    WITHMEM_PYTHON_STANDALONE_DIR="${WITHMEM_PYTHON_STANDALONE_DIR}" \
    SKIP_INSTANCE_PREPARE="${SKIP_INSTANCE_PREPARE}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE}" \
    RESUME_MODE="${RESUME_MODE}" \
    REDO_EXISTING="${REDO_EXISTING}" \
    DRY_RUN="${DRY_RUN}" \
    PHASE7_GLOBAL_HEAVY_SLOT_DIR="${GLOBAL_HEAVY_SLOT_DIR}" \
    PHASE7_GLOBAL_HEAVY_SLOTS="${GLOBAL_HEAVY_SLOTS}" \
    PHASE7_GLOBAL_HEAVY_SLOT_POLL_SEC="${GLOBAL_HEAVY_SLOT_POLL_SEC}" \
    PHASE7_HEAVY_SLOT_DIR="${WITHMEM_HEAVY_SLOT_DIR}" \
    PHASE7_HEAVY_SLOTS="${WITHMEM_HEAVY_SLOTS}" \
    PHASE7_HEAVY_SLOT_POLL_SEC="${LANE_HEAVY_SLOT_POLL_SEC}" \
    "${WITHMEM_SCRIPT}"
}

run_nomem | tee "${RUN_ROOT}/orchestrator_logs/same_only_nomem.log" &
PID_NOMEM=$!
run_withmem | tee "${RUN_ROOT}/orchestrator_logs/same_only_withmem.log" &
PID_WITHMEM=$!

RC_NOMEM=0
RC_WITHMEM=0
DONE_PID=""

set +e
wait -n -p DONE_PID "${PID_NOMEM}" "${PID_WITHMEM}"
FIRST_RC=$?
set -e

if [[ "${DONE_PID}" == "${PID_NOMEM}" ]]; then
  RC_NOMEM="${FIRST_RC}"
else
  RC_WITHMEM="${FIRST_RC}"
fi

if [[ "${FIRST_RC}" -ne 0 ]]; then
  if [[ "${DONE_PID}" == "${PID_NOMEM}" ]]; then
    echo "nomem branch failed rc=${FIRST_RC}; terminating withmem sibling" >&2
    kill "${PID_WITHMEM}" 2>/dev/null || true
  else
    echo "withmem branch failed rc=${FIRST_RC}; terminating nomem sibling" >&2
    kill "${PID_NOMEM}" 2>/dev/null || true
  fi
fi

if [[ "${DONE_PID}" != "${PID_NOMEM}" ]]; then
  set +e
  wait "${PID_NOMEM}"
  RC_NOMEM=$?
  set -e
fi
if [[ "${DONE_PID}" != "${PID_WITHMEM}" ]]; then
  set +e
  wait "${PID_WITHMEM}"
  RC_WITHMEM=$?
  set -e
fi

echo "${RC_NOMEM}" > "${RUN_ROOT}/orchestrator_state/same_only_nomem.exit"
echo "${RC_WITHMEM}" > "${RUN_ROOT}/orchestrator_state/same_only_withmem.exit"

if [[ "${RC_NOMEM}" -ne 0 || "${RC_WITHMEM}" -ne 0 ]]; then
  echo "same_only_nomem_rc=${RC_NOMEM} same_only_withmem_rc=${RC_WITHMEM}" >&2
  exit 1
fi

trap - EXIT INT TERM
echo "same-only phase8 resume completed"
