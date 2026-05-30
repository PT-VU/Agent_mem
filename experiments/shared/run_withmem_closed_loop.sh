#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Run a closed-loop with-memory sequence for one selected instance.
#
# Contract:
# 1) Select one benchmark instance.
# 2) Preserve Agent-mem state across attempt_01 through attempt_10.
# 3) Evaluate each attempt and feed the official outcome back into memory.
# 4) Allow later attempts to reuse promoted cards and avoid anti-patterns.
#
# Set PER_INSTANCE_CALL_LIMIT=0 for no step limit. Set
# INSTANCE_TOTAL_EXEC_TIMEOUT_SEC=0 only when an unlimited execution budget is
# intentionally required.
# -----------------------------------------------------------------------------

WS_ROOT="${WS_ROOT:-/home/pt/SWE-bench}"
ARTIFACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWE_AGENT_ROOT="${SWE_AGENT_ROOT:-${WS_ROOT}/SWE-agent}"
SWEAGENT_BIN="${SWEAGENT_BIN:-${SWE_AGENT_ROOT}/.venv/bin/sweagent}"
PYTHON_BIN="${PYTHON_BIN:-${SWE_AGENT_ROOT}/.venv/bin/python}"
START_WITH_TOOLS="${START_WITH_TOOLS:-${ARTIFACT_ROOT}/framework/sweagent_external_tools_v2/bin/start_with_external_tools.sh}"
COMMON_DIR="${ARTIFACT_ROOT}/experiments/shared/common"
EVAL_FEEDBACK_APPLIER="${COMMON_DIR}/apply_official_eval_feedback.py"
LOCAL_EVAL_STUB_WRITER="${COMMON_DIR}/write_local_eval_stub.py"
RUNTIME_GUARD_SH="${COMMON_DIR}/docker_runtime_guard.sh"
# Watchdog: budget-aware timeout that excludes bridge-hook overhead.
# Defaults to the v2 framework bin dir; can be overridden by caller.
WATCHDOG_PY="${WATCHDOG_PY:-${ARTIFACT_ROOT}/framework/sweagent_external_tools_v2/bin/watchdog.py}"

RUN_TAG="${RUN_TAG:-lite_failedcase_repeat10_withmem_$(date +%Y%m%d_%H%M%S)}"
BASE_EXP_DIR="${BASE_EXP_DIR:-${WS_ROOT}/PDDL_work_mem/06_artificial_intelligence/experiments/multi-try_experiment/with_mem/auto_selected_failedcase_runs}"
RUN_ROOT="${RUN_ROOT:-${BASE_EXP_DIR}/${RUN_TAG}}"
CANDIDATES_ROOT="${RUN_ROOT}/candidates"
NOTES_DIR="${RUN_ROOT}/notes"
CANDIDATE_LIST_JSON="${RUN_ROOT}/historical_candidate_ids.json"
SELECTION_HISTORY_JSON="${RUN_ROOT}/selection_history.json"
SELECTED_CASE_JSON="${RUN_ROOT}/selected_case.json"

DEFAULT_REPEAT_INSTANCE_ID="${DEFAULT_REPEAT_INSTANCE_ID:-astropy__astropy-14182}"
INSTANCE_ID="${INSTANCE_ID:-${DEFAULT_REPEAT_INSTANCE_ID}}"
CANDIDATE_INSTANCE_IDS_CSV="${CANDIDATE_INSTANCE_IDS_CSV:-}"
REPEATS="${REPEATS:-10}"
MODEL_CONFIG="${MODEL_CONFIG:-${WS_ROOT}/SWE-agent/config/kimi25_siliconflow.yaml}"
PROMPT_PROFILE="${PROMPT_PROFILE:-$(basename "${MODEL_CONFIG%.*}")}"
ENV_VAR_PATH="${ENV_VAR_PATH:-${SWE_AGENT_ROOT}/.env}"
PER_INSTANCE_CALL_LIMIT="${PER_INSTANCE_CALL_LIMIT:-0}"
INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${INSTANCE_TOTAL_EXEC_TIMEOUT_SEC:-0}"
AGENT_UNLIMITED_TIMEOUT_SEC="${AGENT_UNLIMITED_TIMEOUT_SEC:-315360000}"
EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-1800}"
MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-1}"
REDO_EXISTING="${REDO_EXISTING:-0}"
SWEBENCH_SUBSET="${SWEBENCH_SUBSET:-lite}"
SWEBENCH_SPLIT="${SWEBENCH_SPLIT:-test}"
PYTHON_STANDALONE_DIR="${PYTHON_STANDALONE_DIR-/root}"
DEPLOYMENT_STARTUP_TIMEOUT_SEC="${DEPLOYMENT_STARTUP_TIMEOUT_SEC:-600}"
STARTUP_RETRIES="${STARTUP_RETRIES:-1}"
RUNTIME_PREWARM="${RUNTIME_PREWARM:-1}"
MAX_ACTIVE_RUN_BATCHES="${MAX_ACTIVE_RUN_BATCHES:-2}"
DRY_RUN="${DRY_RUN:-0}"
ENV_ERROR_RETRIES="${ENV_ERROR_RETRIES:-5}"
ENV_FATAL_EXIT_CODE="${ENV_FATAL_EXIT_CODE:-86}"
# Seconds to wait between environment-error retries (avoids hammering a flapping daemon)
ENV_RETRY_WAIT_SEC="${ENV_RETRY_WAIT_SEC:-30}"

HISTORICAL_SUMMARY_JSON="${HISTORICAL_SUMMARY_JSON:-${WS_ROOT}/PDDL_work_mem/06_artificial_intelligence/experiments/02_first50_unlimited/with_mem/metrics/official_eval_summary_exp02_first50_withmem_unlimited.json}"
HISTORICAL_REPORT_JSON="${HISTORICAL_REPORT_JSON:-}"

export AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS="${AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS:-1}"
export AGENT_MEM_ENABLE_LLM_EXTRACTION="${AGENT_MEM_ENABLE_LLM_EXTRACTION:-1}"
export AGENT_MEM_LLM_EXTRACT_MODE="${AGENT_MEM_LLM_EXTRACT_MODE:-hybrid}"
export AGENT_MEM_EXTRACT_MAX_SIGNALS="${AGENT_MEM_EXTRACT_MAX_SIGNALS:-0}"
export AGENT_MEM_EXTRACT_MAX_ABSTRACTS="${AGENT_MEM_EXTRACT_MAX_ABSTRACTS:-0}"
export AGENT_MEM_EXTRACT_MAX_REPAIR_PATTERNS="${AGENT_MEM_EXTRACT_MAX_REPAIR_PATTERNS:-0}"
export AGENT_MEM_MAX_HINTS="${AGENT_MEM_MAX_HINTS:-0}"
export AGENT_MEM_V21_ENABLE_SUCCESS_FACT_HOTPATH="${AGENT_MEM_V21_ENABLE_SUCCESS_FACT_HOTPATH:-1}"
export AGENT_MEM_V21_ENABLE_SIDECAR="${AGENT_MEM_V21_ENABLE_SIDECAR:-1}"
export AGENT_MEM_V21_ENABLE_SUBTASK_PROJECTION="${AGENT_MEM_V21_ENABLE_SUBTASK_PROJECTION:-1}"
export AGENT_MEM_V21_ENABLE_CARD_COMPILER="${AGENT_MEM_V21_ENABLE_CARD_COMPILER:-1}"
export AGENT_MEM_V21_ENABLE_GOVERNANCE="${AGENT_MEM_V21_ENABLE_GOVERNANCE:-1}"
export AGENT_MEM_V21_HOTPATH_TIMEOUT_MS="${AGENT_MEM_V21_HOTPATH_TIMEOUT_MS:-50}"
export AGENT_MEM_V21_COLDPATH_TIMEOUT_MS="${AGENT_MEM_V21_COLDPATH_TIMEOUT_MS:-5000}"
export AGENT_MEM_V21_MAX_CARDS_PER_QUERY="${AGENT_MEM_V21_MAX_CARDS_PER_QUERY:-4}"
export SWE_AGENT_EXT_TOOL_A_TIMEOUT_SEC="${SWE_AGENT_EXT_TOOL_A_TIMEOUT_SEC:-90}"
export SWE_AGENT_EXT_TOOL_B_TIMEOUT_SEC="${SWE_AGENT_EXT_TOOL_B_TIMEOUT_SEC:-120}"
export SWE_AGENT_EXT_TOOL_RETRY_TIMEOUT_SEC="${SWE_AGENT_EXT_TOOL_RETRY_TIMEOUT_SEC:-150}"
export SWE_AGENT_EXT_TOOL_MAX_RETRIES="${SWE_AGENT_EXT_TOOL_MAX_RETRIES:-1}"

CURRENT_INSTANCE_ID=""
SAFE_INSTANCE_ID=""
CURRENT_CANDIDATE_DIR=""
CURRENT_OUTDIR=""
CURRENT_LOG_DIR=""
CURRENT_METRICS_DIR=""
CURRENT_PERSIST_DIR=""
CURRENT_GRAPH_DIR=""
CURRENT_EVIDENCE_DIR=""
CURRENT_SIDECAR_DIR=""

is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }
to_bool_token() {
  local raw
  raw="$(echo "${1}" | tr '[:upper:]' '[:lower:]')"
  case "${raw}" in
    1|true|yes|on) echo "True" ;;
    *) echo "False" ;;
  esac
}

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "required file not found: ${path}" >&2
    exit 2
  fi
}

for required in "${PYTHON_BIN}" "${SWEAGENT_BIN}" "${START_WITH_TOOLS}" "${EVAL_FEEDBACK_APPLIER}" "${LOCAL_EVAL_STUB_WRITER}" "${RUNTIME_GUARD_SH}"; do
  require_file "${required}"
done

if ! is_int "${REPEATS}" || (( REPEATS < 1 )); then
  echo "REPEATS must be integer >= 1" >&2
  exit 2
fi
if ! is_int "${MAX_WORKERS_EVAL}" || ! is_int "${EVAL_TIMEOUT_SEC}" || ! is_int "${STARTUP_RETRIES}" || ! is_int "${MAX_ACTIVE_RUN_BATCHES}" || ! is_int "${DEPLOYMENT_STARTUP_TIMEOUT_SEC}" || ! is_int "${AGENT_UNLIMITED_TIMEOUT_SEC}" || ! is_int "${ENV_ERROR_RETRIES}" || ! is_int "${ENV_FATAL_EXIT_CODE}"; then
  echo "MAX_WORKERS_EVAL/EVAL_TIMEOUT_SEC/STARTUP_RETRIES/MAX_ACTIVE_RUN_BATCHES/DEPLOYMENT_STARTUP_TIMEOUT_SEC/AGENT_UNLIMITED_TIMEOUT_SEC/ENV_ERROR_RETRIES/ENV_FATAL_EXIT_CODE must be integers" >&2
  exit 2
fi
if ! [[ "${INSTANCE_TOTAL_EXEC_TIMEOUT_SEC}" =~ ^[0-9]+$ ]]; then
  echo "INSTANCE_TOTAL_EXEC_TIMEOUT_SEC must be integer >= 0" >&2
  exit 2
fi
if (( AGENT_UNLIMITED_TIMEOUT_SEC < 1 )); then
  echo "AGENT_UNLIMITED_TIMEOUT_SEC must be >= 1" >&2
  exit 2
fi

if (( INSTANCE_TOTAL_EXEC_TIMEOUT_SEC == 0 )); then
  EFFECTIVE_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${AGENT_UNLIMITED_TIMEOUT_SEC}"
else
  EFFECTIVE_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${INSTANCE_TOTAL_EXEC_TIMEOUT_SEC}"
fi

mkdir -p "${RUN_ROOT}" "${CANDIDATES_ROOT}" "${NOTES_DIR}"
source "${RUNTIME_GUARD_SH}"

build_candidate_list() {
  "${PYTHON_BIN}" - <<'PY' \
    "${INSTANCE_ID}" \
    "${CANDIDATE_INSTANCE_IDS_CSV}" \
    "${HISTORICAL_SUMMARY_JSON}" \
    "${HISTORICAL_REPORT_JSON}" \
    "${CANDIDATE_LIST_JSON}"
import json
import pathlib
import sys
from datetime import datetime, timezone

manual_instance = sys.argv[1].strip()
manual_csv = sys.argv[2].strip()
summary_path = pathlib.Path(sys.argv[3])
report_hint = sys.argv[4].strip()
out_path = pathlib.Path(sys.argv[5])

source = ""
ids: list[str] = []

if manual_instance:
    ids = [manual_instance]
    source = "manual_instance_id"
elif manual_csv:
    ids = [x.strip() for x in manual_csv.split(",") if x.strip()]
    source = "manual_candidate_csv"
else:
    if not summary_path.exists():
        raise SystemExit(f"historical summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_path = pathlib.Path(report_hint) if report_hint else pathlib.Path(str(summary.get("report_json", "")).strip())
    if not report_path.exists():
        raise SystemExit(f"historical raw report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    unresolved = report.get("unresolved_ids")
    if isinstance(unresolved, list):
        ids = [str(x) for x in unresolved if str(x).strip()]
    else:
        completed = [str(x) for x in report.get("completed_ids", []) if str(x).strip()]
        resolved = {str(x) for x in report.get("resolved_ids", []) if str(x).strip()}
        ids = [iid for iid in completed if iid not in resolved]
    source = str(report_path)

seen = set()
ordered_ids = []
for iid in ids:
    if iid in seen:
        continue
    seen.add(iid)
    ordered_ids.append(iid)

payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "source": source,
    "candidate_count": len(ordered_ids),
    "candidate_ids": ordered_ids,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
for iid in ordered_ids:
    print(iid)
PY
}

set_candidate_context() {
  CURRENT_INSTANCE_ID="$1"
  SAFE_INSTANCE_ID="${CURRENT_INSTANCE_ID//\//_}"
  CURRENT_CANDIDATE_DIR="${CANDIDATES_ROOT}/${SAFE_INSTANCE_ID}"
  CURRENT_OUTDIR="${CURRENT_CANDIDATE_DIR}/outputs"
  CURRENT_LOG_DIR="${CURRENT_CANDIDATE_DIR}/logs"
  CURRENT_METRICS_DIR="${CURRENT_CANDIDATE_DIR}/metrics"
  CURRENT_PERSIST_DIR="${CURRENT_CANDIDATE_DIR}/persistence"
  CURRENT_GRAPH_DIR="${CURRENT_PERSIST_DIR}/graph_store"
  CURRENT_EVIDENCE_DIR="${CURRENT_PERSIST_DIR}/evidence_store"
  CURRENT_SIDECAR_DIR="${CURRENT_PERSIST_DIR}/sidecar_store"
  mkdir -p "${CURRENT_OUTDIR}" "${CURRENT_LOG_DIR}" "${CURRENT_METRICS_DIR}" "${CURRENT_GRAPH_DIR}" "${CURRENT_EVIDENCE_DIR}" "${CURRENT_SIDECAR_DIR}" \
    "${CURRENT_METRICS_DIR}/attempt_reports" "${CURRENT_METRICS_DIR}/attempt_summaries" "${CURRENT_METRICS_DIR}/attempt_feedback" "${CURRENT_METRICS_DIR}/attempt_context"
  export AGENT_MEM_STORAGE_DIR="${CURRENT_GRAPH_DIR}"
  export AGENT_MEM_EVIDENCE_DIR="${CURRENT_EVIDENCE_DIR}"
  export AGENT_MEM_V21_SIDECAR_DIR="${CURRENT_SIDECAR_DIR}"
}

attempt_outdir() {
  local tag
  tag="$(printf "%02d" "$1")"
  printf "%s/%s/attempt_%s" "${CURRENT_OUTDIR}" "${CURRENT_INSTANCE_ID}" "${tag}"
}

attempt_log() {
  local tag
  tag="$(printf "%02d" "$1")"
  printf "%s/%s.attempt_%s.log" "${CURRENT_LOG_DIR}" "${CURRENT_INSTANCE_ID}" "${tag}"
}

attempt_event_log() {
  local tag
  tag="$(printf "%02d" "$1")"
  printf "%s/%s.attempt_%s.jsonl" "${CURRENT_LOG_DIR}" "${CURRENT_INSTANCE_ID}" "${tag}"
}

attempt_pred_json() {
  local tag
  tag="$(printf "%02d" "$1")"
  printf "%s/attempt_reports/%s.attempt_%s.predictions.json" "${CURRENT_METRICS_DIR}" "${CURRENT_INSTANCE_ID}" "${tag}"
}

attempt_raw_report() {
  local tag
  tag="$(printf "%02d" "$1")"
  printf "%s/attempt_reports/%s.attempt_%s.official_eval.json" "${CURRENT_METRICS_DIR}" "${CURRENT_INSTANCE_ID}" "${tag}"
}

attempt_summary_json() {
  local tag
  tag="$(printf "%02d" "$1")"
  printf "%s/attempt_summaries/%s.attempt_%s.summary.json" "${CURRENT_METRICS_DIR}" "${CURRENT_INSTANCE_ID}" "${tag}"
}

attempt_feedback_json() {
  local tag
  tag="$(printf "%02d" "$1")"
  printf "%s/attempt_feedback/%s.attempt_%s.feedback.json" "${CURRENT_METRICS_DIR}" "${CURRENT_INSTANCE_ID}" "${tag}"
}

attempt_context_json() {
  local tag
  tag="$(printf "%02d" "$1")"
  printf "%s/attempt_context/%s.attempt_%s.context.json" "${CURRENT_METRICS_DIR}" "${CURRENT_INSTANCE_ID}" "${tag}"
}

resolve_official_eval_report() {
  local eval_run_id="$1"
  local attempt_tag="$2"
  local attempt_dir="$3"
  local candidate
  local -a candidates=(
    "${attempt_dir}/attempt_${attempt_tag}.${eval_run_id}.json"
    "${attempt_dir}/preds.${eval_run_id}.json"
    "${PWD}/attempt_${attempt_tag}.${eval_run_id}.json"
    "${PWD}/preds.${eval_run_id}.json"
    "${WS_ROOT}/attempt_${attempt_tag}.${eval_run_id}.json"
    "${WS_ROOT}/preds.${eval_run_id}.json"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done

  candidate="$(
    find "${attempt_dir}" "${PWD}" "${WS_ROOT}" -maxdepth 2 -type f \
      \( -name "attempt_${attempt_tag}.${eval_run_id}.json" -o -name "preds.${eval_run_id}.json" -o -name "*${eval_run_id}*.json" \) \
      2>/dev/null | head -n 1 || true
  )"
  if [[ -n "${candidate}" ]]; then
    printf "%s\n" "${candidate}"
    return 0
  fi
  return 1
}

attempt_log_indicates_environment_error() {
  local log_path="$1"
  [[ -f "${log_path}" ]] || return 1
  rg -q \
    -e 'ClientConnectorError' \
    -e 'ClientOSError' \
    -e 'ConnectionRefusedError' \
    -e 'Cannot connect to host' \
    -e 'Connection reset by peer' \
    -e 'Runtime did not start within timeout' \
    -e "docker' could not be found" \
    -e 'docker not found' \
    -e 'DockerException' \
    -e 'docker_preflight_failed' \
    -e 'DockerPullError' \
    -e 'Segmentation fault' \
    -e 'SIGSEGV' \
    "${log_path}"
}

attempt_log_indicates_fatal_environment_error() {
  local log_path="$1"
  [[ -f "${log_path}" ]] || return 1
  rg -q \
    -e 'Segmentation fault' \
    -e 'SIGSEGV' \
    -e 'DockerPullError' \
    -e 'docker_preflight_failed' \
    -e 'docker info failed rc=' \
    -e 'docker ps -a failed rc=' \
    "${log_path}"
}

attempt_log_indicates_account_balance_error() {
  local log_path="$1"
  [[ -f "${log_path}" ]] || return 1
  rg -q \
    -e 'account balance is insufficient' \
    -e 'Sorry, your account balance is insufficient' \
    -e 'insufficient balance' \
    "${log_path}"
}

archive_attempt_retry_artifacts() {
  local attempt_dir="$1"
  local run_log="$2"
  local event_log="$3"
  local pred_json="$4"
  local raw_report_json="$5"
  local summary_json="$6"
  local feedback_json="$7"
  local context_json="$8"
  local retry_index="$9"

  [[ -d "${attempt_dir}" ]] && mv "${attempt_dir}" "${attempt_dir}.env_retry_${retry_index}"
  [[ -f "${run_log}" ]] && mv "${run_log}" "${run_log%.log}.env_retry_${retry_index}.log"
  [[ -f "${event_log}" ]] && mv "${event_log}" "${event_log%.jsonl}.env_retry_${retry_index}.jsonl"
  [[ -f "${pred_json}" ]] && mv "${pred_json}" "${pred_json%.json}.env_retry_${retry_index}.json"
  [[ -f "${raw_report_json}" ]] && mv "${raw_report_json}" "${raw_report_json%.json}.env_retry_${retry_index}.json"
  [[ -f "${summary_json}" ]] && mv "${summary_json}" "${summary_json%.json}.env_retry_${retry_index}.json"
  [[ -f "${feedback_json}" ]] && mv "${feedback_json}" "${feedback_json%.json}.env_retry_${retry_index}.json"
  [[ -f "${context_json}" ]] && mv "${context_json}" "${context_json%.json}.env_retry_${retry_index}.json"
}

attempt_summary_is_resolved() {
  local summary_json="$1"
  "${PYTHON_BIN}" - <<'PY' "${summary_json}"
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    raise SystemExit(2)
data = json.loads(path.read_text(encoding="utf-8"))
print("1" if int(data.get("resolved_instances", 0) or 0) > 0 else "0")
PY
}

record_selection_decision() {
  local candidate_index="$1"
  local decision="$2"
  local first_attempt_resolved="$3"
  "${PYTHON_BIN}" - <<'PY' \
    "${SELECTION_HISTORY_JSON}" \
    "${SELECTED_CASE_JSON}" \
    "${CURRENT_INSTANCE_ID}" \
    "${CURRENT_CANDIDATE_DIR}" \
    "${CURRENT_METRICS_DIR}" \
    "${candidate_index}" \
    "${decision}" \
    "${first_attempt_resolved}" \
    "${RUN_TAG}" \
    "${MODEL_CONFIG}" \
    "${PROMPT_PROFILE}" \
    "${AGENT_MEM_V21_ENABLE_SUCCESS_FACT_HOTPATH}" \
    "${AGENT_MEM_V21_ENABLE_SIDECAR}" \
    "${AGENT_MEM_V21_ENABLE_SUBTASK_PROJECTION}" \
    "${AGENT_MEM_V21_ENABLE_CARD_COMPILER}" \
    "${AGENT_MEM_V21_ENABLE_GOVERNANCE}"
import json
import pathlib
import sys
from datetime import datetime, timezone

history_path = pathlib.Path(sys.argv[1])
selected_path = pathlib.Path(sys.argv[2])
instance_id = sys.argv[3]
candidate_dir = sys.argv[4]
metrics_dir = sys.argv[5]
candidate_index = int(sys.argv[6])
decision = sys.argv[7]
first_attempt_resolved = sys.argv[8] == "1"
run_tag = sys.argv[9]
model_config = sys.argv[10]
prompt_profile = sys.argv[11]
v21_flags = {
    "enable_success_fact_hotpath": sys.argv[12],
    "enable_sidecar": sys.argv[13],
    "enable_subtask_projection": sys.argv[14],
    "enable_card_compiler": sys.argv[15],
    "enable_governance": sys.argv[16],
}

rows = []
if history_path.exists():
    try:
        loaded = json.loads(history_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            rows = loaded
    except Exception:
        rows = []

row = {
    "instance_id": instance_id,
    "candidate_index": candidate_index,
    "decision": decision,
    "first_attempt_resolved": first_attempt_resolved,
    "candidate_dir": candidate_dir,
    "attempt_summary_path": str(pathlib.Path(metrics_dir) / "attempt_summaries" / f"{instance_id}.attempt_01.summary.json"),
    "model_config": model_config,
    "prompt_profile": prompt_profile,
    "v21_flags": v21_flags,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}

updated = False
for idx, existing in enumerate(rows):
    if isinstance(existing, dict) and existing.get("instance_id") == instance_id:
        rows[idx] = row
        updated = True
        break
if not updated:
    rows.append(row)

history_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

if decision == "selected_for_repeat":
    selected_payload = {
        "run_tag": run_tag,
        "instance_id": instance_id,
        "candidate_index": candidate_index,
        "candidate_dir": candidate_dir,
        "attempt1_summary_path": row["attempt_summary_path"],
        "model_config": model_config,
        "prompt_profile": prompt_profile,
        "v21_flags": v21_flags,
        "selected_at": row["updated_at"],
    }
    selected_path.write_text(json.dumps(selected_payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

write_candidate_run_context() {
  "${PYTHON_BIN}" - <<'PY' \
    "${CURRENT_METRICS_DIR}" \
    "${RUN_TAG}" \
    "${CURRENT_INSTANCE_ID}" \
    "${CURRENT_CANDIDATE_DIR}" \
    "${MODEL_CONFIG}" \
    "${PROMPT_PROFILE}" \
    "${START_WITH_TOOLS}" \
    "${AGENT_MEM_STORAGE_DIR}" \
    "${AGENT_MEM_EVIDENCE_DIR}" \
    "${AGENT_MEM_V21_SIDECAR_DIR}" \
    "${AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS}" \
    "${AGENT_MEM_ENABLE_LLM_EXTRACTION}" \
    "${AGENT_MEM_LLM_EXTRACT_MODE}" \
    "${AGENT_MEM_V21_ENABLE_SUCCESS_FACT_HOTPATH}" \
    "${AGENT_MEM_V21_ENABLE_SIDECAR}" \
    "${AGENT_MEM_V21_ENABLE_SUBTASK_PROJECTION}" \
    "${AGENT_MEM_V21_ENABLE_CARD_COMPILER}" \
    "${AGENT_MEM_V21_ENABLE_GOVERNANCE}" \
    "${AGENT_MEM_V21_HOTPATH_TIMEOUT_MS}" \
    "${AGENT_MEM_V21_COLDPATH_TIMEOUT_MS}" \
    "${AGENT_MEM_V21_MAX_CARDS_PER_QUERY}"
import json
import pathlib
import sys
from datetime import datetime, timezone

metrics_dir = pathlib.Path(sys.argv[1])
out = metrics_dir / "run_context.json"
payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "run_tag": sys.argv[2],
    "instance_id": sys.argv[3],
    "candidate_dir": sys.argv[4],
    "model_config": sys.argv[5],
    "prompt_profile": sys.argv[6],
    "launcher": sys.argv[7],
    "storage": {
        "graph_dir": sys.argv[8],
        "evidence_dir": sys.argv[9],
        "sidecar_dir": sys.argv[10],
    },
    "memory_flags": {
        "enable_online_embeddings": sys.argv[11],
        "enable_llm_extraction": sys.argv[12],
        "llm_extract_mode": sys.argv[13],
        "v21_enable_success_fact_hotpath": sys.argv[14],
        "v21_enable_sidecar": sys.argv[15],
        "v21_enable_subtask_projection": sys.argv[16],
        "v21_enable_card_compiler": sys.argv[17],
        "v21_enable_governance": sys.argv[18],
        "v21_hotpath_timeout_ms": sys.argv[19],
        "v21_coldpath_timeout_ms": sys.argv[20],
        "v21_max_cards_per_query": sys.argv[21],
    },
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(str(out))
PY
}

write_attempt_context() {
  local attempt="$1"
  local attempt_tag="$2"
  local attempt_dir="$3"
  local run_log="$4"
  local event_log="$5"
  local pred_json="$6"
  local raw_report_json="$7"
  local summary_json="$8"
  local feedback_json="$9"
  local eval_run_id="${10}"
  local out_json
  out_json="$(attempt_context_json "${attempt}")"
  "${PYTHON_BIN}" - <<'PY' \
    "${out_json}" \
    "${RUN_TAG}" \
    "${CURRENT_INSTANCE_ID}" \
    "${attempt}" \
    "${attempt_tag}" \
    "${eval_run_id}" \
    "${MODEL_CONFIG}" \
    "${PROMPT_PROFILE}" \
    "${attempt_dir}" \
    "${run_log}" \
    "${event_log}" \
    "${pred_json}" \
    "${raw_report_json}" \
    "${summary_json}" \
    "${feedback_json}" \
    "${AGENT_MEM_V21_ENABLE_SUCCESS_FACT_HOTPATH}" \
    "${AGENT_MEM_V21_ENABLE_SIDECAR}" \
    "${AGENT_MEM_V21_ENABLE_SUBTASK_PROJECTION}" \
    "${AGENT_MEM_V21_ENABLE_CARD_COMPILER}" \
    "${AGENT_MEM_V21_ENABLE_GOVERNANCE}" \
    "${AGENT_MEM_V21_SIDECAR_DIR}"
import json
import pathlib
import sys
from datetime import datetime, timezone

out = pathlib.Path(sys.argv[1])
payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "run_tag": sys.argv[2],
    "instance_id": sys.argv[3],
    "attempt": int(sys.argv[4]),
    "attempt_tag": sys.argv[5],
    "eval_run_id": sys.argv[6],
    "model_config": sys.argv[7],
    "prompt_profile": sys.argv[8],
    "paths": {
        "attempt_dir": sys.argv[9],
        "run_log": sys.argv[10],
        "event_log": sys.argv[11],
        "predictions_json": sys.argv[12],
        "official_eval_json": sys.argv[13],
        "summary_json": sys.argv[14],
        "feedback_json": sys.argv[15],
        "sidecar_dir": sys.argv[21],
    },
    "memory_flags": {
        "v21_enable_success_fact_hotpath": sys.argv[16],
        "v21_enable_sidecar": sys.argv[17],
        "v21_enable_subtask_projection": sys.argv[18],
        "v21_enable_card_compiler": sys.argv[19],
        "v21_enable_governance": sys.argv[20],
    },
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(str(out))
PY
}

finalize_incomplete_attempt() {
  local attempt="$1"
  local attempt_tag="$2"
  local attempt_dir="$3"
  local run_log="$4"
  local event_log="$5"
  local pred_json="$6"
  local raw_report_json="$7"
  local summary_json="$8"
  local feedback_json="$9"
  local eval_run_id="${10}"
  local local_reason="${11}"

  runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:local_eval_stub"
  "${PYTHON_BIN}" "${LOCAL_EVAL_STUB_WRITER}" \
    --instance-id "${CURRENT_INSTANCE_ID}" \
    --outcome incomplete \
    --reason "${local_reason}" \
    --out "${raw_report_json}" >/dev/null

  runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:summarize_eval"
  "${PYTHON_BIN}" "${COMMON_DIR}/summarize_official_eval.py" \
    --report-json "${raw_report_json}" \
    --run-id "${eval_run_id}" \
    --dataset "SWE-bench/SWE-bench ${SWEBENCH_SPLIT}" \
    --scope "Lite failed-case repeat10 with_mem attempt ${attempt_tag}" \
    --summary-out "${summary_json}" >/dev/null

  runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:apply_feedback"
  "${PYTHON_BIN}" "${EVAL_FEEDBACK_APPLIER}" \
    --workspace-root "${WS_ROOT}" \
    --report-json "${raw_report_json}" \
    --output-dir "${attempt_dir}" \
    --cache-file "${tmp_cache}" \
    --run-id "${eval_run_id}" \
    --output-json "${feedback_json}" >/dev/null

  write_attempt_context "${attempt}" "${attempt_tag}" "${attempt_dir}" "${run_log}" "${event_log}" "${pred_json}" "${raw_report_json}" "${summary_json}" "${feedback_json}" "${eval_run_id}" >/dev/null
}

run_attempt() {
  local attempt="$1"
  local attempt_tag
  attempt_tag="$(printf "%02d" "${attempt}")"
  local attempt_dir run_log event_log pred_json raw_report_json summary_json feedback_json context_json eval_run_id tmp_cache pred_count local_reason latest_report
  local env_retry_index=0

  attempt_dir="$(attempt_outdir "${attempt}")"
  run_log="$(attempt_log "${attempt}")"
  event_log="$(attempt_event_log "${attempt}")"
  pred_json="$(attempt_pred_json "${attempt}")"
  raw_report_json="$(attempt_raw_report "${attempt}")"
  summary_json="$(attempt_summary_json "${attempt}")"
  feedback_json="$(attempt_feedback_json "${attempt}")"
  context_json="$(attempt_context_json "${attempt}")"
  eval_run_id="${RUN_TAG}_${CURRENT_INSTANCE_ID}_attempt_${attempt_tag}"

  mkdir -p "${attempt_dir}"
  if [[ "${REDO_EXISTING}" != "1" && -f "${summary_json}" && -f "${feedback_json}" ]]; then
    echo "[skip] instance=${CURRENT_INSTANCE_ID} attempt=${attempt_tag} already summarized"
    if [[ ! -f "${context_json}" ]]; then
      write_attempt_context "${attempt}" "${attempt_tag}" "${attempt_dir}" "${run_log}" "${event_log}" "${pred_json}" "${raw_report_json}" "${summary_json}" "${feedback_json}" "${eval_run_id}" >/dev/null
    fi
    return 0
  fi

  export SWE_AGENT_EXT_TOOLS_LOG_FILE="${event_log}"
  export SWE_AGENT_EXT_INSTANCE_ID="${CURRENT_INSTANCE_ID}"
  export SWE_AGENT_EXT_RUN_ID="${RUN_TAG}"
  export SWE_AGENT_EXT_ATTEMPT_ID="attempt-${attempt_tag}"

  while true; do
    local instance_image image_match_pattern run_status startup_retry_index
    instance_image=""
    image_match_pattern="${CURRENT_INSTANCE_ID}"
    tmp_cache="$(mktemp)"
    printf '[\n  "%s"\n]\n' "${CURRENT_INSTANCE_ID}" > "${tmp_cache}"

    echo "[run] instance=${CURRENT_INSTANCE_ID} attempt=${attempt_tag} try=$((env_retry_index + 1))/$((ENV_ERROR_RETRIES + 1))" | tee "${run_log}"
    runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:preflight"
    if ! runtime_guard_preflight "${run_log}"; then
      runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:preflight_failed"
      if (( env_retry_index < ENV_ERROR_RETRIES )); then
        echo "[retry] environment failure at preflight for ${CURRENT_INSTANCE_ID}; retry=$((env_retry_index + 1))/${ENV_ERROR_RETRIES}; waiting ${ENV_RETRY_WAIT_SEC}s" | tee -a "${run_log}"
        sleep "${ENV_RETRY_WAIT_SEC}"
        archive_attempt_retry_artifacts "${attempt_dir}" "${run_log}" "${event_log}" "${pred_json}" "${raw_report_json}" "${summary_json}" "${feedback_json}" "${context_json}" "$((env_retry_index + 1))"
        rm -f "${tmp_cache}"
        env_retry_index=$((env_retry_index + 1))
        mkdir -p "${attempt_dir}"
        continue
      fi
      rm -f "${tmp_cache}"
      echo "[fatal] docker preflight failed for ${CURRENT_INSTANCE_ID} attempt=${attempt_tag}" | tee -a "${run_log}"
      return "${ENV_FATAL_EXIT_CODE}"
    fi

    runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:resolve_image"
    if instance_image="$(runtime_guard_resolve_swebench_image "${run_log}" "${CURRENT_INSTANCE_ID}" "${SWEBENCH_SUBSET}" "${SWEBENCH_SPLIT}")"; then
      image_match_pattern="${instance_image}"
    else
      echo "[warn] failed to resolve deployment image for ${CURRENT_INSTANCE_ID}; skip runtime prewarm" | tee -a "${run_log}"
      instance_image=""
    fi

    if [[ "${RUNTIME_PREWARM}" == "1" && -n "${instance_image}" ]]; then
      runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:runtime_warmup"
      if ! runtime_guard_prepare_runtime_image "${run_log}" "${instance_image}" "${PYTHON_STANDALONE_DIR}" "linux/amd64"; then
        runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:runtime_warmup_failed"
        echo "[warn] runtime warmup failed for ${CURRENT_INSTANCE_ID}; continue with direct launch" | tee -a "${run_log}"
        runtime_guard_capture_diag "${run_log}" "${image_match_pattern}"
      fi
    fi

    runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:capacity_wait"
    runtime_guard_wait_for_capacity "${run_log}" "${MAX_ACTIVE_RUN_BATCHES}" 10
    runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:cleanup_before_run"
    runtime_guard_cleanup_logged_containers "${run_log}"
    run_status=1
    startup_retry_index=0

    while (( startup_retry_index <= STARTUP_RETRIES )); do
      [[ "${startup_retry_index}" -gt 0 ]] && echo "[retry] startup retry ${startup_retry_index}/${STARTUP_RETRIES} for ${CURRENT_INSTANCE_ID}" | tee -a "${run_log}"
      runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:launch_sweagent_retry_${startup_retry_index}"
      set +e
      local -a run_cmd=(
        "${START_WITH_TOOLS}" --mode with-mem
        "${SWEAGENT_BIN}" run-batch
        --config "${MODEL_CONFIG}"
        --instances.type swe_bench
        --instances.subset "${SWEBENCH_SUBSET}"
        --instances.split "${SWEBENCH_SPLIT}"
        --instances.filter "^${CURRENT_INSTANCE_ID}$"
        --instances.deployment.type docker
        --instances.deployment.startup_timeout "${DEPLOYMENT_STARTUP_TIMEOUT_SEC}"
        --output_dir "${attempt_dir}"
        --num_workers 1
        --env_var_path "${ENV_VAR_PATH}"
        --agent.model.per_instance_call_limit "${PER_INSTANCE_CALL_LIMIT}"
        --redo_existing "$(to_bool_token "${REDO_EXISTING}")"
        --agent.tools.total_execution_timeout "${EFFECTIVE_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC}"
      )
      if [[ "${PYTHON_STANDALONE_DIR}" != "__NONE__" ]]; then
        run_cmd+=(--instances.deployment.python_standalone_dir "${PYTHON_STANDALONE_DIR}")
      fi
      if (( INSTANCE_TOTAL_EXEC_TIMEOUT_SEC > 0 )); then
        # Per-attempt bridge-overhead file: cleared by watchdog at startup.
        _bridge_overhead_file="${run_log%.log}.bridge_overhead.tmp"
        export SWEAGENT_BRIDGE_OVERHEAD_FILE="${_bridge_overhead_file}"
        "${PYTHON_BIN}" "${WATCHDOG_PY}" \
          --budget "${INSTANCE_TOTAL_EXEC_TIMEOUT_SEC}" \
          --overhead-file "${_bridge_overhead_file}" \
          -- "${run_cmd[@]}" 2>&1 | tee -a "${run_log}"
        unset SWEAGENT_BRIDGE_OVERHEAD_FILE
      else
        "${run_cmd[@]}" 2>&1 | tee -a "${run_log}"
      fi
      run_status=$?
      set -e
      if attempt_log_indicates_account_balance_error "${run_log}"; then
        runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:provider_fatal_insufficient_balance"
        runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:cleanup_after_provider_fatal"
        runtime_guard_cleanup_logged_containers "${run_log}"
        rm -f "${tmp_cache}"
        echo "[fatal] provider balance exhausted for ${CURRENT_INSTANCE_ID} attempt=${attempt_tag}" | tee -a "${run_log}"
        return "${ENV_FATAL_EXIT_CODE}"
      fi
      if [[ "${run_status}" -eq 0 ]]; then
        runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:sweagent_exit_0"
        break
      fi
      runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:sweagent_exit_${run_status}"
      if grep -q "Runtime did not start within timeout" "${run_log}"; then
        runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:capture_diag_runtime_timeout"
        runtime_guard_capture_diag "${run_log}" "${image_match_pattern}"
        runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:cleanup_after_runtime_timeout"
        runtime_guard_cleanup_logged_containers "${run_log}"
        startup_retry_index=$((startup_retry_index + 1))
        continue
      fi
      runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:capture_diag_non_runtime_failure"
      runtime_guard_capture_diag "${run_log}" "${image_match_pattern}"
      break
    done

    runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:rebuild_predictions"
    "${PYTHON_BIN}" "${COMMON_DIR}/rebuild_eval_predictions.py" \
      --output-dir "${attempt_dir}" \
      --cache-file "${tmp_cache}" \
      --default-model-name "lite_failedcase_repeat10_withmem_closed_loop" \
      --out "${pred_json}" >/dev/null

    pred_count="$("${PYTHON_BIN}" - <<'PY' "${pred_json}"
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit(0)
print(len(data) if isinstance(data, dict) else 0)
PY
)"

    if [[ "${pred_count}" -eq 0 ]]; then
      local_reason="missing_prediction"
      if [[ "${run_status}" -eq 124 || "${run_status}" -eq 137 ]]; then
        local_reason="timeout_no_prediction"
      elif [[ "${run_status}" -ne 0 ]]; then
        local_reason="run_failure_no_prediction"
      fi
      # Abort immediately on fatal environment errors such as pull failures or segfaults.
      if [[ "${run_status}" -ne 0 ]] && attempt_log_indicates_fatal_environment_error "${run_log}"; then
        rm -f "${tmp_cache}"
        echo "[fatal] docker/runtime failure after run for ${CURRENT_INSTANCE_ID} attempt=${attempt_tag}" | tee -a "${run_log}"
        return "${ENV_FATAL_EXIT_CODE}"
      fi
      # Retry non-fatal failures up to ENV_ERROR_RETRIES times.
      # This covers docker daemon flaps, container startup races, and any non-model crash.
      # Timeout (rc=124/137) and account-balance errors are excluded from retry.
      if [[ "${run_status}" -ne 0 && "${run_status}" -ne 124 && "${run_status}" -ne 137 ]] \
         && ! attempt_log_indicates_account_balance_error "${run_log}" \
         && (( env_retry_index < ENV_ERROR_RETRIES )); then
        echo "[retry] non-timeout run failure (rc=${run_status}) for ${CURRENT_INSTANCE_ID}; retry=$((env_retry_index + 1))/${ENV_ERROR_RETRIES}; waiting ${ENV_RETRY_WAIT_SEC}s" | tee -a "${run_log}"
        runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:cleanup_after_run_failure"
        runtime_guard_cleanup_logged_containers "${run_log}"
        sleep "${ENV_RETRY_WAIT_SEC}"
        archive_attempt_retry_artifacts "${attempt_dir}" "${run_log}" "${event_log}" "${pred_json}" "${raw_report_json}" "${summary_json}" "${feedback_json}" "${context_json}" "$((env_retry_index + 1))"
        rm -f "${tmp_cache}"
        env_retry_index=$((env_retry_index + 1))
        mkdir -p "${attempt_dir}"
        continue
      fi
      finalize_incomplete_attempt "${attempt}" "${attempt_tag}" "${attempt_dir}" "${run_log}" "${event_log}" "${pred_json}" "${raw_report_json}" "${summary_json}" "${feedback_json}" "${eval_run_id}" "${local_reason}"
    else
      runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:official_eval"
      "${PYTHON_BIN}" -m swebench.harness.run_evaluation \
        --dataset_name SWE-bench/SWE-bench \
        --split "${SWEBENCH_SPLIT}" \
        --predictions_path "${pred_json}" \
        --max_workers "${MAX_WORKERS_EVAL}" \
        --timeout "${EVAL_TIMEOUT_SEC}" \
        --cache_level env \
        --clean false \
        --run_id "${eval_run_id}" \
        --instance_ids "${CURRENT_INSTANCE_ID}"
      latest_report="$(resolve_official_eval_report "${eval_run_id}" "${attempt_tag}" "${attempt_dir}" || true)"
      if [[ -z "${latest_report}" ]]; then
        echo "missing raw eval report for attempt ${attempt_tag}" >&2
        finalize_incomplete_attempt "${attempt}" "${attempt_tag}" "${attempt_dir}" "${run_log}" "${event_log}" "${pred_json}" "${raw_report_json}" "${summary_json}" "${feedback_json}" "${eval_run_id}" "missing_raw_eval_report"
        runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:cleanup_after_attempt"
        runtime_guard_cleanup_logged_containers "${run_log}"
        rm -f "${tmp_cache}"
        return 0
      fi
      cp -f "${latest_report}" "${raw_report_json}"
      runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:summarize_eval"
      "${PYTHON_BIN}" "${COMMON_DIR}/summarize_official_eval.py" \
        --report-json "${raw_report_json}" \
        --run-id "${eval_run_id}" \
        --dataset "SWE-bench/SWE-bench ${SWEBENCH_SPLIT}" \
        --scope "Lite failed-case repeat10 with_mem attempt ${attempt_tag}" \
        --summary-out "${summary_json}" >/dev/null

      runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:apply_feedback"
      "${PYTHON_BIN}" "${EVAL_FEEDBACK_APPLIER}" \
        --workspace-root "${WS_ROOT}" \
        --report-json "${raw_report_json}" \
        --output-dir "${attempt_dir}" \
        --cache-file "${tmp_cache}" \
        --run-id "${eval_run_id}" \
        --output-json "${feedback_json}" >/dev/null

      write_attempt_context "${attempt}" "${attempt_tag}" "${attempt_dir}" "${run_log}" "${event_log}" "${pred_json}" "${raw_report_json}" "${summary_json}" "${feedback_json}" "${eval_run_id}" >/dev/null
    fi

    runtime_guard_mark_stage "${run_log}" "attempt_${attempt_tag}:cleanup_after_attempt"
    runtime_guard_cleanup_logged_containers "${run_log}"
    rm -f "${tmp_cache}"
    return 0
  done
}

write_attempt_summary_rollup() {
  "${PYTHON_BIN}" - <<'PY' "${CURRENT_METRICS_DIR}" "${CURRENT_INSTANCE_ID}" "${REPEATS}" "${SELECTED_CASE_JSON}"
import json
import pathlib
import sys

metrics_dir = pathlib.Path(sys.argv[1])
instance_id = sys.argv[2]
repeats = int(sys.argv[3])
selected_case_path = pathlib.Path(sys.argv[4])
selected_case = {}
if selected_case_path.exists():
    try:
        selected_case = json.loads(selected_case_path.read_text(encoding="utf-8"))
    except Exception:
        selected_case = {}
run_context_path = metrics_dir / "run_context.json"
run_context = {}
if run_context_path.exists():
    try:
        run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
    except Exception:
        run_context = {}

rows = []
for attempt in range(1, repeats + 1):
    tag = f"{attempt:02d}"
    summary_path = metrics_dir / "attempt_summaries" / f"{instance_id}.attempt_{tag}.summary.json"
    feedback_path = metrics_dir / "attempt_feedback" / f"{instance_id}.attempt_{tag}.feedback.json"
    context_path = metrics_dir / "attempt_context" / f"{instance_id}.attempt_{tag}.context.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    feedback = json.loads(feedback_path.read_text(encoding="utf-8")) if feedback_path.exists() else {}
    context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.exists() else {}
    rows.append({
        "attempt": attempt,
        "summary_path": str(summary_path),
        "feedback_path": str(feedback_path),
        "context_path": str(context_path),
        "submitted_instances": summary.get("submitted_instances", 0),
        "resolved_instances": summary.get("resolved_instances", 0),
        "incomplete_instances": summary.get("incomplete_instances", 0),
        "solved_rate_on_planned": summary.get("solved_rate_on_planned", 0.0),
        "feedback_outcome_counts": feedback.get("outcome_counts", {}),
        "prompt_profile": context.get("prompt_profile", ""),
        "memory_flags": context.get("memory_flags", {}),
    })

out = metrics_dir / "attempt_summaries.json"
out.write_text(json.dumps({
    "instance_id": instance_id,
    "repeats": repeats,
    "selected_case": selected_case,
    "run_context": run_context,
    "attempts": rows,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(str(out))
PY
}


# mapfile -t CANDIDATE_IDS < <(build_candidate_list)
# if [[ "${#CANDIDATE_IDS[@]}" -eq 0 ]]; then
# fi
#
# echo "candidate_count=${#CANDIDATE_IDS[@]} subset=${SWEBENCH_SUBSET} split=${SWEBENCH_SPLIT} repeats=${REPEATS}"
#
# if [[ "${DRY_RUN}" == "1" ]]; then
# fi
#
# selected=0
# selected_index=0
#
# for idx in "${!CANDIDATE_IDS[@]}"; do
#
#
#
#
# done
#
# if [[ "${selected}" != "1" ]]; then
# fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry_run=1"
  echo "fixed_instance_id=${INSTANCE_ID}"
  exit 0
fi

selected=1
selected_index=1
set_candidate_context "${INSTANCE_ID}"
record_selection_decision "${selected_index}" "selected_for_repeat" "0"
write_candidate_run_context >/dev/null

echo "fixed_instance_id=${INSTANCE_ID} subset=${SWEBENCH_SUBSET} split=${SWEBENCH_SPLIT} repeats=${REPEATS}"
for attempt in $(seq 1 "${REPEATS}"); do
  run_attempt "${attempt}"
done
write_attempt_summary_rollup

echo "done"
echo "run_root=${RUN_ROOT}"
echo "selected_case=${SELECTED_CASE_JSON}"
echo "selected_candidate_index=${selected_index}"
echo "attempt_summaries=${CURRENT_METRICS_DIR}/attempt_summaries.json"
