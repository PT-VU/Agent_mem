#!/usr/bin/env bash
set -euo pipefail

WS_ROOT="${WS_ROOT:-/home/pt/SWE-bench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_MEM_SCRIPT="${WITH_MEM_SCRIPT:-${SCRIPT_DIR}/run_withmem_closed_loop.sh}"
RESOURCE_SLOT_SH="${RESOURCE_SLOT_SH:-${SCRIPT_DIR}/with_resource_slot.sh}"
PREPARE_SCRIPT="${PREPARE_SCRIPT:-${SCRIPT_DIR}/prepare_instance_environment.sh}"

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
INSTANCE_ID="${INSTANCE_ID:-astropy__astropy-14182}"
INSTANCE_LIST_JSON="${INSTANCE_LIST_JSON:-}"
REPEATS="${REPEATS:-10}"
MODEL_CONFIG="${MODEL_CONFIG:-${WS_ROOT}/SWE-agent/config/kimi25_siliconflow.yaml}"
PROMPT_PROFILE="${PROMPT_PROFILE:-prompt_base}"
SWEBENCH_SUBSET="${SWEBENCH_SUBSET:-lite}"
SWEBENCH_SPLIT="${SWEBENCH_SPLIT:-test}"
ENV_VAR_PATH="${ENV_VAR_PATH:-${WS_ROOT}/SWE-agent/.env}"
PER_INSTANCE_CALL_LIMIT="${PER_INSTANCE_CALL_LIMIT:-0}"
INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${INSTANCE_TOTAL_EXEC_TIMEOUT_SEC:-1200}"
MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-1}"
EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-1200}"
RESUME_MODE="${RESUME_MODE:-1}"
REDO_EXISTING="${REDO_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"
MAX_ACTIVE_RUN_BATCHES="${MAX_ACTIVE_RUN_BATCHES:-2}"
ENV_FATAL_EXIT_CODE="${ENV_FATAL_EXIT_CODE:-86}"
IMAGE_PULL_TIMEOUT_SEC="${IMAGE_PULL_TIMEOUT_SEC:-1800}"
WITHMEM_RUNTIME_PREWARM="${WITHMEM_RUNTIME_PREWARM:-1}"
WITHMEM_RUNTIME_WARMUP_TIMEOUT_SEC="${WITHMEM_RUNTIME_WARMUP_TIMEOUT_SEC:-1800}"
WITHMEM_PYTHON_STANDALONE_DIR="${WITHMEM_PYTHON_STANDALONE_DIR:-__AUTO__}"
SKIP_INSTANCE_PREPARE="${SKIP_INSTANCE_PREPARE:-0}"

if [[ "${WITHMEM_PYTHON_STANDALONE_DIR}" == "__AUTO__" ]]; then
  if [[ "${SKIP_INSTANCE_PREPARE}" == "1" ]]; then
    WITHMEM_PYTHON_STANDALONE_DIR="/root"
  else
    WITHMEM_PYTHON_STANDALONE_DIR="/root"
  fi
fi

RUN_TAG="${RUN_TAG:-phase7_same_v21}"
PREPARED_STATE_DIR="${RUN_ROOT}/orchestrator_state/prepared_env/with_mem"
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
branch=with_mem
runtime_prewarm=${WITHMEM_RUNTIME_PREWARM}
image=${image}
prepared_at=$(date -Is)
EOF
}

require_prepared_marker() {
  local iid="$1"
  local marker
  marker="$(prepared_marker_path "${iid}")"
  if [[ ! -f "${marker}" ]]; then
    echo "[fatal] same_problem_v21 instance=${iid} missing prepared marker: ${marker}" >&2
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
    "${WS_ROOT}/SWE-agent/.venv/bin/python" - <<'PY' "${INSTANCE_LIST_JSON}"
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
  echo "[dry-run] same-problem v21"
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "INSTANCE_COUNT=${#DRY_IDS[@]}"
  printf '%s\n' "${DRY_IDS[@]}" | sed -n '1,20p'
  echo "REPEATS=${REPEATS}"
  exit 0
fi

mapfile -t SELECTED_IDS < <(load_instance_ids)
if [[ "${#SELECTED_IDS[@]}" -eq 0 ]]; then
  echo "No same-problem instance ids selected. Abort." >&2
  exit 1
fi

SAME_ROOT="${RUN_ROOT}/same_problem/with_mem"
mkdir -p "${SAME_ROOT}"
printf '{\n  "instance_ids": [\n' > "${SAME_ROOT}/instance_ids.json"
for idx in "${!SELECTED_IDS[@]}"; do
  iid="${SELECTED_IDS[$idx]}"
  comma=","
  if [[ "${idx}" -eq "$(( ${#SELECTED_IDS[@]} - 1 ))" ]]; then
    comma=""
  fi
  printf '    "%s"%s\n' "${iid}" "${comma}" >> "${SAME_ROOT}/instance_ids.json"
done
printf '  ]\n}\n' >> "${SAME_ROOT}/instance_ids.json"

for iid in "${SELECTED_IDS[@]}"; do
  safe_iid="${iid//\//_}"
  instance_run_root="${SAME_ROOT}/${safe_iid}"
  instance_run_tag="same_v21_${safe_iid}"
  prepare_log="${SAME_ROOT}/${safe_iid}.prepare.log"
  if [[ "${SKIP_INSTANCE_PREPARE}" == "1" ]]; then
    require_prepared_marker "${iid}"
  else
    set +e
    run_prepare_with_resource_limits \
      "same_problem_v21_prepare_${safe_iid}" \
      env \
        INSTANCE_ID="${iid}" \
        LOG_FILE="${prepare_log}" \
        SWEBENCH_SUBSET="${SWEBENCH_SUBSET}" \
        SWEBENCH_SPLIT="${SWEBENCH_SPLIT}" \
        IMAGE_PULL_TIMEOUT_SEC="${IMAGE_PULL_TIMEOUT_SEC}" \
        RUNTIME_PREWARM="${WITHMEM_RUNTIME_PREWARM}" \
        RUNTIME_WARMUP_TIMEOUT_SEC="${WITHMEM_RUNTIME_WARMUP_TIMEOUT_SEC}" \
        PYTHON_STANDALONE_DIR="${WITHMEM_PYTHON_STANDALONE_DIR}" \
        ENV_FATAL_EXIT_CODE="${ENV_FATAL_EXIT_CODE}" \
        bash "${PREPARE_SCRIPT}"
    prepare_rc=$?
    set -e
    if [[ "${prepare_rc}" -ne 0 ]]; then
      echo "[fatal] same_problem_v21 instance=${iid} environment prepare failed rc=${prepare_rc}" >&2
      exit "${prepare_rc}"
    fi
    prepared_image="$(awk -F= '/^instance_image=/{print $2}' "${prepare_log}" | tail -n 1)"
    write_prepared_marker "${iid}" "${prepared_image}"
  fi

  set +e
  run_with_resource_limits \
    "same_problem_v21_${safe_iid}" \
    env \
      RUN_TAG="${instance_run_tag}" \
      BASE_EXP_DIR="${SAME_ROOT}" \
      RUN_ROOT="${instance_run_root}" \
      INSTANCE_ID="${iid}" \
      CANDIDATE_INSTANCE_IDS_CSV="${iid}" \
      REPEATS="${REPEATS}" \
      MODEL_CONFIG="${MODEL_CONFIG}" \
      PROMPT_PROFILE="${PROMPT_PROFILE}" \
      SWEBENCH_SUBSET="${SWEBENCH_SUBSET}" \
      SWEBENCH_SPLIT="${SWEBENCH_SPLIT}" \
      ENV_VAR_PATH="${ENV_VAR_PATH}" \
      PER_INSTANCE_CALL_LIMIT="${PER_INSTANCE_CALL_LIMIT}" \
      INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${INSTANCE_TOTAL_EXEC_TIMEOUT_SEC}" \
      MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL}" \
      EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC}" \
      REDO_EXISTING="${REDO_EXISTING}" \
      MAX_ACTIVE_RUN_BATCHES="${MAX_ACTIVE_RUN_BATCHES}" \
      PYTHON_STANDALONE_DIR="${WITHMEM_PYTHON_STANDALONE_DIR}" \
      RUNTIME_PREWARM=0 \
      ENV_FATAL_EXIT_CODE="${ENV_FATAL_EXIT_CODE}" \
      "${WITH_MEM_SCRIPT}"
  instance_rc=$?
  set -e
  if [[ "${instance_rc}" -eq "${ENV_FATAL_EXIT_CODE}" ]]; then
    echo "[fatal] same_problem_v21 instance=${iid} environment runtime unavailable" >&2
    exit "${instance_rc}"
  fi
  if [[ "${instance_rc}" -ne 0 ]]; then
    echo "[warn] same_problem_v21 instance=${iid} exited rc=${instance_rc}; keep progress and continue to next instance" >&2
  fi
done

"${WS_ROOT}/SWE-agent/.venv/bin/python" - <<'PY' "${SAME_ROOT}" "${REPEATS}"
import json, pathlib, sys
same_root = pathlib.Path(sys.argv[1]); repeats = int(sys.argv[2])
instance_ids = json.loads((same_root / "instance_ids.json").read_text(encoding="utf-8")).get("instance_ids", [])
rows = []
for iid in instance_ids:
    root = same_root / iid.replace("/", "_") / "candidates" / iid / "metrics" / "attempt_summaries.json"
    payload = {}
    if root.exists():
        try:
            payload = json.loads(root.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    rows.append({"instance_id": iid, "summary_rollup_path": str(root), "payload": payload, "repeats": repeats})
out = same_root / "attempt_summaries.json"
out.write_text(json.dumps({"instance_ids": instance_ids, "repeats": repeats, "instances": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(str(out))
PY
