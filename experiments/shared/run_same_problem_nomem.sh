#!/usr/bin/env bash
set -euo pipefail

WS_ROOT="${WS_ROOT:-/home/pt/SWE-bench}"
SWE_AGENT_ROOT="${SWE_AGENT_ROOT:-${WS_ROOT}/SWE-agent}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SWE_AGENT_ROOT}/.venv/bin/python}"
NO_MEM_SCRIPT="${NO_MEM_SCRIPT:-${SCRIPT_DIR}/run_nomem_attempts.sh}"
EVAL_STEP="${EVAL_STEP:-${SCRIPT_DIR}/official_eval_step.py}"
RESOURCE_SLOT_SH="${RESOURCE_SLOT_SH:-${SCRIPT_DIR}/with_resource_slot.sh}"
PREPARE_SCRIPT="${PREPARE_SCRIPT:-${SCRIPT_DIR}/prepare_instance_environment.sh}"

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
INSTANCE_ID="${INSTANCE_ID:-astropy__astropy-14182}"
INSTANCE_LIST_JSON="${INSTANCE_LIST_JSON:-}"
SOURCE_INSTANCE_LIST_JSON="${INSTANCE_LIST_JSON:-}"
REPEATS="${REPEATS:-10}"
MODEL_CONFIG="${MODEL_CONFIG:-${WS_ROOT}/SWE-agent/config/kimi25_siliconflow.yaml}"
PROMPT_PROFILE="${PROMPT_PROFILE:-prompt_base}"
SWEBENCH_SUBSET="${SWEBENCH_SUBSET:-lite}"
SWEBENCH_SPLIT="${SWEBENCH_SPLIT:-test}"
ENV_VAR_PATH="${ENV_VAR_PATH:-${SWE_AGENT_ROOT}/.env}"
PER_INSTANCE_CALL_LIMIT="${PER_INSTANCE_CALL_LIMIT:-0}"
INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${INSTANCE_TOTAL_EXEC_TIMEOUT_SEC:-1200}"
MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-1}"
EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-1200}"
RESUME_MODE="${RESUME_MODE:-1}"
REDO_EXISTING="${REDO_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"
ENV_ERROR_RETRIES="${ENV_ERROR_RETRIES:-1}"
ENV_FATAL_EXIT_CODE="${ENV_FATAL_EXIT_CODE:-86}"
IMAGE_PULL_TIMEOUT_SEC="${IMAGE_PULL_TIMEOUT_SEC:-1800}"
NOMEM_RUNTIME_PREWARM="${NOMEM_RUNTIME_PREWARM:-0}"
NOMEM_RUNTIME_WARMUP_TIMEOUT_SEC="${NOMEM_RUNTIME_WARMUP_TIMEOUT_SEC:-1800}"
NOMEM_PYTHON_STANDALONE_DIR="${NOMEM_PYTHON_STANDALONE_DIR:-__AUTO__}"
SKIP_INSTANCE_PREPARE="${SKIP_INSTANCE_PREPARE:-0}"

if [[ "${NOMEM_PYTHON_STANDALONE_DIR}" == "__AUTO__" ]]; then
  if [[ "${SKIP_INSTANCE_PREPARE}" == "1" ]]; then
    NOMEM_PYTHON_STANDALONE_DIR="/root"
  else
    NOMEM_PYTHON_STANDALONE_DIR="__NONE__"
  fi
fi

EXP_DIR="${RUN_ROOT}/same_problem/nomem"
OUTDIR="${EXP_DIR}/outputs"
LOG_DIR="${EXP_DIR}/logs"
METRICS_DIR="${EXP_DIR}/metrics"
CACHE_DIR="${EXP_DIR}/cache"
PREPARED_STATE_DIR="${RUN_ROOT}/orchestrator_state/prepared_env/nomem"
mkdir -p "${OUTDIR}" "${LOG_DIR}" "${METRICS_DIR}/attempt_reports" "${METRICS_DIR}/attempt_summaries" "${METRICS_DIR}/raw_attempt_runs" "${CACHE_DIR}"
mkdir -p "${PREPARED_STATE_DIR}"

prepared_marker_path() {
  local iid="$1"
  local safe_iid="${iid//\//_}"
  printf '%s/%s.ok\n' "${PREPARED_STATE_DIR}" "${safe_iid}"
}

write_prepared_marker() {
  local iid="$1"
  local image="$2"
  local marker
  marker="$(prepared_marker_path "${iid}")"
  cat >"${marker}" <<EOF
instance_id=${iid}
branch=nomem
runtime_prewarm=${NOMEM_RUNTIME_PREWARM}
image=${image}
prepared_at=$(date -Is)
EOF
}

require_prepared_marker() {
  local iid="$1"
  local marker
  marker="$(prepared_marker_path "${iid}")"
  if [[ ! -f "${marker}" ]]; then
    echo "[fatal] same-problem nomem instance=${iid} missing prepared marker: ${marker}" >&2
    exit "${ENV_FATAL_EXIT_CODE}"
  fi
}

run_with_resource_limits() {
  local label="$1"
  shift
  local local_slot_dir="${PHASE7_HEAVY_SLOT_DIR:-${RUN_ROOT}/resource_slots/heavy}"
  local local_slots="${PHASE7_HEAVY_SLOTS:-1}"
  local local_poll="${PHASE7_HEAVY_SLOT_POLL_SEC:-10}"
  if [[ -n "${PHASE7_GLOBAL_HEAVY_SLOT_DIR:-}" ]]; then
    local global_slots="${PHASE7_GLOBAL_HEAVY_SLOTS:-1}"
    local global_poll="${PHASE7_GLOBAL_HEAVY_SLOT_POLL_SEC:-${local_poll}}"
    "${RESOURCE_SLOT_SH}" \
      --slot-dir "${PHASE7_GLOBAL_HEAVY_SLOT_DIR}" \
      --slots "${global_slots}" \
      --poll-sec "${global_poll}" \
      --label "global:${label}" \
      -- \
      "${RESOURCE_SLOT_SH}" \
        --slot-dir "${local_slot_dir}" \
        --slots "${local_slots}" \
        --poll-sec "${local_poll}" \
        --label "${label}" \
        -- \
        "$@"
    return $?
  fi
  "${RESOURCE_SLOT_SH}" \
    --slot-dir "${local_slot_dir}" \
    --slots "${local_slots}" \
    --poll-sec "${local_poll}" \
    --label "${label}" \
    -- \
    "$@"
}

run_prepare_with_resource_limits() {
  local label="$1"
  shift
  local prepare_slot_dir="${PHASE7_PREPARE_SLOT_DIR:-${RUN_ROOT}/resource_slots/prepare_global}"
  local prepare_slots="${PHASE7_PREPARE_SLOTS:-2}"
  local prepare_poll="${PHASE7_PREPARE_SLOT_POLL_SEC:-10}"
  "${RESOURCE_SLOT_SH}" \
    --slot-dir "${prepare_slot_dir}" \
    --slots "${prepare_slots}" \
    --poll-sec "${prepare_poll}" \
    --label "${label}" \
    -- \
    "$@"
}

load_instance_ids() {
  if [[ -n "${INSTANCE_LIST_JSON}" && -f "${INSTANCE_LIST_JSON}" ]]; then
    "${PYTHON_BIN}" - <<'PY' "${INSTANCE_LIST_JSON}"
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
for iid in data.get("instance_ids", []):
    print(iid)
PY
    return 0
  fi
  printf '%s\n' "${INSTANCE_ID}"
}

if [[ "${DRY_RUN}" == "1" ]]; then
  mapfile -t DRY_IDS < <(load_instance_ids)
  echo "[dry-run] same-problem nomem"
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "INSTANCE_COUNT=${#DRY_IDS[@]}"
  printf '%s\n' "${DRY_IDS[@]}" | sed -n '1,20p'
  echo "REPEATS=${REPEATS}"
  exit 0
fi

INSTANCE_LIST_JSON="${CACHE_DIR}/instance_ids.json"
if [[ -n "${SOURCE_INSTANCE_LIST_JSON}" ]]; then
  INSTANCE_LIST_JSON="${SOURCE_INSTANCE_LIST_JSON}"
fi
mapfile -t SELECTED_IDS < <(load_instance_ids)
if [[ "${#SELECTED_IDS[@]}" -eq 0 ]]; then
  echo "No same-problem instance ids selected. Abort." >&2
  exit 1
fi
INSTANCE_IDS_CSV="$(IFS=,; echo "${SELECTED_IDS[*]}")"
INSTANCE_LIST_JSON="${CACHE_DIR}/instance_ids.json"

for iid in "${SELECTED_IDS[@]}"; do
  safe_iid="${iid//\//_}"
  prepare_log="${LOG_DIR}/${safe_iid}.prepare.log"
  if [[ "${SKIP_INSTANCE_PREPARE}" == "1" ]]; then
    require_prepared_marker "${iid}"
  else
    set +e
    run_prepare_with_resource_limits \
      "same_problem_nomem_prepare_${iid}" \
      env \
        INSTANCE_ID="${iid}" \
        LOG_FILE="${prepare_log}" \
        SWEBENCH_SUBSET="${SWEBENCH_SUBSET}" \
        SWEBENCH_SPLIT="${SWEBENCH_SPLIT}" \
        IMAGE_PULL_TIMEOUT_SEC="${IMAGE_PULL_TIMEOUT_SEC}" \
        RUNTIME_PREWARM="${NOMEM_RUNTIME_PREWARM}" \
        RUNTIME_WARMUP_TIMEOUT_SEC="${NOMEM_RUNTIME_WARMUP_TIMEOUT_SEC}" \
        PYTHON_STANDALONE_DIR="${NOMEM_PYTHON_STANDALONE_DIR}" \
        ENV_FATAL_EXIT_CODE="${ENV_FATAL_EXIT_CODE}" \
        bash "${PREPARE_SCRIPT}"
    prepare_rc=$?
    set -e
    if [[ "${prepare_rc}" -ne 0 ]]; then
      echo "[fatal] same-problem nomem instance=${iid} environment prepare failed rc=${prepare_rc}" >&2
      exit "${prepare_rc}"
    fi
    prepared_image="$(awk -F= '/^instance_image=/{print $2}' "${prepare_log}" | tail -n 1)"
    write_prepared_marker "${iid}" "${prepared_image}"
  fi

  for attempt in $(seq 1 "${REPEATS}"); do
    tag="$(printf "%02d" "${attempt}")"
    attempt_dir="${OUTDIR}/${iid}/attempt_${tag}"
    pred_json="${METRICS_DIR}/attempt_reports/${iid}.attempt_${tag}.predictions.json"
    report_json="${METRICS_DIR}/attempt_reports/${iid}.attempt_${tag}.official_eval.json"
    summary_json="${METRICS_DIR}/attempt_summaries/${iid}.attempt_${tag}.summary.json"
    if [[ "${RESUME_MODE}" == "1" && -f "${summary_json}" ]]; then
      echo "[resume] same-problem nomem instance=${iid} attempt=${tag} summary exists"
      continue
    fi
    set +e
    run_with_resource_limits \
      "same_problem_nomem_run_${iid}_attempt_${tag}" \
      env \
        EXP_DIR="${EXP_DIR}" \
        OUTDIR="${OUTDIR}" \
        LOG_DIR="${LOG_DIR}" \
        CACHE_DIR="${CACHE_DIR}" \
        METRICS_DIR="${METRICS_DIR}" \
        INSTANCE_LIST_FILE="${INSTANCE_LIST_JSON}" \
        SUMMARY_OUT="${METRICS_DIR}/raw_attempt_runs/${iid}.attempt_${tag}.json" \
        SWEBENCH_SUBSET="${SWEBENCH_SUBSET}" \
        SWEBENCH_SPLIT="${SWEBENCH_SPLIT}" \
        REPEATS=1 \
        ATTEMPT_START="${attempt}" \
        MODEL_CONFIG="${MODEL_CONFIG}" \
        NUM_WORKERS=1 \
        PER_INSTANCE_CALL_LIMIT="${PER_INSTANCE_CALL_LIMIT}" \
        INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${INSTANCE_TOTAL_EXEC_TIMEOUT_SEC}" \
        ENV_VAR_PATH="${ENV_VAR_PATH}" \
        REDO_EXISTING="${REDO_EXISTING}" \
        RESUME_MODE="${RESUME_MODE}" \
        ENV_ERROR_RETRIES="${ENV_ERROR_RETRIES}" \
        ENV_FATAL_EXIT_CODE="${ENV_FATAL_EXIT_CODE}" \
        PYTHON_STANDALONE_DIR="${NOMEM_PYTHON_STANDALONE_DIR}" \
        INSTANCE_IDS_CSV="${iid}" \
        "${NO_MEM_SCRIPT}"
    run_rc=$?
    set -e
    if [[ "${run_rc}" -eq "${ENV_FATAL_EXIT_CODE}" ]]; then
      echo "[fatal] same-problem nomem instance=${iid} attempt=${tag} environment runtime unavailable" >&2
      exit "${run_rc}"
    fi
    if [[ "${run_rc}" -ne 0 ]]; then
      echo "[warn] same-problem nomem instance=${iid} attempt=${tag} exited rc=${run_rc}; continue to immediate eval" >&2
    fi

    run_with_resource_limits \
      "same_problem_nomem_eval_${iid}_attempt_${tag}" \
      "${PYTHON_BIN}" "${EVAL_STEP}" \
        --workspace-root "${WS_ROOT}" \
        --python-bin "${PYTHON_BIN}" \
        --instance-id "${iid}" \
        --attempt-dir "${attempt_dir}" \
        --predictions-json "${pred_json}" \
        --report-json "${report_json}" \
        --summary-json "${summary_json}" \
        --run-id "phase7_same_nomem_${iid}_attempt_${tag}" \
        --scope "Phase7 same-problem no-mem ${iid} attempt ${tag}" \
        --dataset "SWE-bench/SWE-bench ${SWEBENCH_SPLIT}" \
        --split "${SWEBENCH_SPLIT}" \
        --eval-timeout-sec "${EVAL_TIMEOUT_SEC}" \
        --max-workers-eval "${MAX_WORKERS_EVAL}"
  done
done

"${PYTHON_BIN}" - <<'PY' "${METRICS_DIR}" "${INSTANCE_LIST_JSON}" "${REPEATS}" "${PROMPT_PROFILE}"
import json, pathlib, sys
metrics_dir = pathlib.Path(sys.argv[1]); instance_list_path = pathlib.Path(sys.argv[2]); repeats = int(sys.argv[3]); prompt = sys.argv[4]
instance_ids = json.loads(instance_list_path.read_text(encoding="utf-8")).get("instance_ids", [])
rows = []
for instance_id in instance_ids:
    attempts = []
    for attempt in range(1, repeats + 1):
        tag = f"{attempt:02d}"
        summary_path = metrics_dir / "attempt_summaries" / f"{instance_id}.attempt_{tag}.summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        attempts.append({
            "attempt": attempt,
            "summary_path": str(summary_path),
            "submitted_instances": summary.get("submitted_instances", 0),
            "resolved_instances": summary.get("resolved_instances", 0),
            "incomplete_instances": summary.get("incomplete_instances", 0),
            "solved_rate_on_planned": summary.get("solved_rate_on_planned", 0.0),
            "prompt_profile": prompt,
            "memory_mode": "no_mem",
        })
    rows.append({"instance_id": instance_id, "attempts": attempts})
out = metrics_dir / "attempt_summaries.json"
out.write_text(json.dumps({"instance_ids": instance_ids, "repeats": repeats, "instances": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
PY
