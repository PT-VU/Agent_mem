#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Run independent no-memory SWE-agent attempts.
#
# Contract:
# 1) Accept a fixed instance ID.
# 2) Run N attempts through SWE-agent run-batch.
# 3) Preserve runtime logs, JSONL records, and trajectories in the private run root.
# 4) Disable Agent-mem for this lane.
# 5) Emit summaries compatible with the with-memory lane.
#
# Flow: arguments -> instance selection -> attempt execution -> evaluation ->
# summary JSON.
# -----------------------------------------------------------------------------
set -euo pipefail

WS_ROOT="${WS_ROOT:-/home/pt/SWE-bench}"
ARTIFACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWE_AGENT_ROOT="${SWE_AGENT_ROOT:-${WS_ROOT}/SWE-agent}"
SWEAGENT_BIN="${SWEAGENT_BIN:-${SWE_AGENT_ROOT}/.venv/bin/sweagent}"
START_WITH_TOOLS="${START_WITH_TOOLS:-${ARTIFACT_ROOT}/framework/sweagent_external_tools_v2/bin/start_with_external_tools.sh}"
COMMON_DIR="${ARTIFACT_ROOT}/experiments/shared/common"
RUNTIME_GUARD_SH="${RUNTIME_GUARD_SH:-${COMMON_DIR}/docker_runtime_guard.sh}"

EXP_DIR="${EXP_DIR:-${WS_ROOT}/PDDL_work_mem/06_artificial_intelligence/experiments/multi-try_experiment/no_mem}"
OUTDIR="${OUTDIR:-${EXP_DIR}/outputs}"
LOG_DIR="${LOG_DIR:-${EXP_DIR}/logs/no_mem}"
CACHE_DIR="${CACHE_DIR:-${EXP_DIR}/cache}"
METRICS_DIR="${METRICS_DIR:-${EXP_DIR}/metrics}"
INSTANCE_LIST_FILE="${INSTANCE_LIST_FILE:-${CACHE_DIR}/instance_ids.json}"
SUMMARY_OUT="${SUMMARY_OUT:-${METRICS_DIR}/summary_multi_try_no_mem_20min.json}"

EXPERIMENT_ID="${EXPERIMENT_ID:-multi_try_no_mem_20min}"
SWEBENCH_SUBSET="${SWEBENCH_SUBSET:-full}"
SWEBENCH_SPLIT="${SWEBENCH_SPLIT:-test}"
START_INDEX="${START_INDEX:-0}"
INSTANCE_COUNT="${INSTANCE_COUNT:-5}"
INSTANCE_IDS_CSV="${INSTANCE_IDS_CSV:-}"
REPEATS="${REPEATS:-5}"
ATTEMPT_START="${ATTEMPT_START:-1}"

MODEL_CONFIG="${MODEL_CONFIG:-${WS_ROOT}/SWE-agent/config/kimi25_siliconflow.yaml}"
NUM_WORKERS="${NUM_WORKERS:-1}"
PER_INSTANCE_CALL_LIMIT="${PER_INSTANCE_CALL_LIMIT:-0}"
INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${INSTANCE_TOTAL_EXEC_TIMEOUT_SEC:-1200}"
ENV_VAR_PATH="${ENV_VAR_PATH:-${SWE_AGENT_ROOT}/.env}"
REDO_EXISTING="${REDO_EXISTING:-1}"
FAIL_FAST="${FAIL_FAST:-0}"
RESUME_MODE="${RESUME_MODE:-1}"
ENV_ERROR_RETRIES="${ENV_ERROR_RETRIES:-1}"
ENV_FATAL_EXIT_CODE="${ENV_FATAL_EXIT_CODE:-86}"
PYTHON_STANDALONE_DIR="${PYTHON_STANDALONE_DIR-__NONE__}"

is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }
to_bool_token() {
  local raw
  raw="$(echo "${1}" | tr '[:upper:]' '[:lower:]')"
  case "${raw}" in
    1|true|yes|on) echo "True" ;;
    *) echo "False" ;;
  esac
}

if ! is_int "${START_INDEX}" || ! is_int "${INSTANCE_COUNT}" || ! is_int "${REPEATS}" || ! is_int "${ATTEMPT_START}" || ! is_int "${ENV_ERROR_RETRIES}" || ! is_int "${ENV_FATAL_EXIT_CODE}"; then
  echo "START_INDEX/INSTANCE_COUNT/REPEATS/ATTEMPT_START/ENV_ERROR_RETRIES/ENV_FATAL_EXIT_CODE must be integers" >&2
  exit 2
fi
if (( ATTEMPT_START < 1 )); then
  echo "ATTEMPT_START must be >= 1" >&2
  exit 2
fi
if [[ ! -f "${RUNTIME_GUARD_SH}" ]]; then
  echo "missing runtime guard: ${RUNTIME_GUARD_SH}" >&2
  exit 2
fi

mkdir -p "${OUTDIR}" "${LOG_DIR}" "${CACHE_DIR}" "${METRICS_DIR}"
source "${RUNTIME_GUARD_SH}"

build_instance_list() {
  "${SWE_AGENT_ROOT}/.venv/bin/python" - <<'PY' \
    "${INSTANCE_LIST_FILE}" "${INSTANCE_IDS_CSV}" "${SWEBENCH_SUBSET}" "${SWEBENCH_SPLIT}" "${START_INDEX}" "${INSTANCE_COUNT}"
import json
import pathlib
import sys
from sweagent.run.batch_instances import SWEBenchInstances

out_path = pathlib.Path(sys.argv[1])
ids_csv = sys.argv[2].strip()
subset = sys.argv[3]
split = sys.argv[4]
start = int(sys.argv[5])
count = int(sys.argv[6])

if ids_csv:
    ids = [x.strip() for x in ids_csv.split(",") if x.strip()]
else:
    stop = start + count
    instances = SWEBenchInstances(subset=subset, split=split, slice=f"{start}:{stop}").get_instance_configs()
    ids = [inst.problem_statement.id for inst in instances]

payload = {
    "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "subset": subset,
    "split": split,
    "start_index": start,
    "instance_count_requested": count,
    "instance_ids": ids,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"instance_ids={len(ids)} -> {out_path}")
PY
}

read_instance_ids() {
  "${SWE_AGENT_ROOT}/.venv/bin/python" - <<'PY' "${INSTANCE_LIST_FILE}"
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
data = json.loads(p.read_text(encoding="utf-8"))
for iid in data.get("instance_ids", []):
    print(iid)
PY
}

attempt_is_complete() {
  local iid="$1"
  local attempt_outdir="$2"
  local instance_dir="${attempt_outdir}/${iid}"
  if [[ -f "${attempt_outdir}/run_batch_exit_statuses.yaml" ]]; then
    return 0
  fi
  if [[ -f "${attempt_outdir}/preds.json" ]]; then
    return 0
  fi
  if [[ -f "${instance_dir}/${iid}.traj" && -f "${instance_dir}/${iid}.pred" ]]; then
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
  local attempt_outdir="$1"
  local attempt_event_log="$2"
  local attempt_run_log="$3"
  local retry_index="$4"

  if [[ -d "${attempt_outdir}" ]]; then
    mv "${attempt_outdir}" "${attempt_outdir}.env_retry_${retry_index}"
  fi
  if [[ -f "${attempt_event_log}" ]]; then
    mv "${attempt_event_log}" "${attempt_event_log%.jsonl}.env_retry_${retry_index}.jsonl"
  fi
  if [[ -f "${attempt_run_log}" ]]; then
    mv "${attempt_run_log}" "${attempt_run_log%.log}.env_retry_${retry_index}.log"
  fi
}

run_single_attempt() {
  local iid="$1"
  local attempt="$2"
  local attempt_outdir="$3"
  local attempt_event_log="$4"
  local attempt_run_log="$5"
  local attempt_tag
  local try_index=0
  local rc=0

  attempt_tag="$(printf "%02d" "${attempt}")"

  while true; do
    mkdir -p "${attempt_outdir}"
    export SWE_AGENT_EXT_TOOLS_LOG_FILE="${attempt_event_log}"

    echo "[${iid}] attempt=${attempt_tag} try=$((try_index + 1))/$((ENV_ERROR_RETRIES + 1)) out=${attempt_outdir}"
    runtime_guard_mark_stage "${attempt_run_log}" "attempt_${attempt_tag}:cleanup_before_run"
    runtime_guard_cleanup_logged_containers "${attempt_run_log}"
    runtime_guard_mark_stage "${attempt_run_log}" "attempt_${attempt_tag}:preflight"
    if ! runtime_guard_preflight "${attempt_run_log}"; then
      runtime_guard_mark_stage "${attempt_run_log}" "attempt_${attempt_tag}:preflight_failed"
      if (( try_index < ENV_ERROR_RETRIES )); then
        echo "[${iid}] attempt=${attempt_tag} docker preflight failed; retry=$((try_index + 1))/${ENV_ERROR_RETRIES}" | tee -a "${attempt_run_log}"
        archive_attempt_retry_artifacts "${attempt_outdir}" "${attempt_event_log}" "${attempt_run_log}" "$((try_index + 1))"
        try_index=$((try_index + 1))
        continue
      fi
      echo "[${iid}] attempt=${attempt_tag} fatal docker runtime failure" | tee -a "${attempt_run_log}"
      return "${ENV_FATAL_EXIT_CODE}"
    fi

    set +e
    local -a run_cmd=(
      "${START_WITH_TOOLS}" --mode no-mem
      "${SWEAGENT_BIN}" run-batch
      --config "${MODEL_CONFIG}"
      --instances.type swe_bench
      --instances.subset "${SWEBENCH_SUBSET}"
      --instances.split "${SWEBENCH_SPLIT}"
      --instances.filter "^${iid}$"
      --output_dir "${attempt_outdir}"
      --num_workers "${NUM_WORKERS}"
      --env_var_path "${ENV_VAR_PATH}"
      --agent.model.per_instance_call_limit "${PER_INSTANCE_CALL_LIMIT}"
      --agent.tools.total_execution_timeout "${INSTANCE_TOTAL_EXEC_TIMEOUT_SEC}"
      --redo_existing "$(to_bool_token "${REDO_EXISTING}")"
    )
    if [[ "${PYTHON_STANDALONE_DIR}" != "__NONE__" ]]; then
      run_cmd+=(--instances.deployment.python_standalone_dir "${PYTHON_STANDALONE_DIR}")
    fi
    if (( INSTANCE_TOTAL_EXEC_TIMEOUT_SEC > 0 )); then
      timeout --foreground -k 45s "${INSTANCE_TOTAL_EXEC_TIMEOUT_SEC}s" "${run_cmd[@]}" 2>&1 | tee -a "${attempt_run_log}"
    else
      "${run_cmd[@]}" 2>&1 | tee -a "${attempt_run_log}"
    fi
    rc=$?
    set -e

    if attempt_log_indicates_account_balance_error "${attempt_run_log}"; then
      echo "[${iid}] attempt=${attempt_tag} fatal provider failure: insufficient balance" | tee -a "${attempt_run_log}"
      runtime_guard_mark_stage "${attempt_run_log}" "attempt_${attempt_tag}:cleanup_after_provider_fatal"
      runtime_guard_cleanup_logged_containers "${attempt_run_log}"
      return "${ENV_FATAL_EXIT_CODE}"
    fi

    if [[ "${rc}" -eq 0 ]]; then
      return 0
    fi
    if ! attempt_log_indicates_environment_error "${attempt_run_log}"; then
      return "${rc}"
    fi
    if (( try_index >= ENV_ERROR_RETRIES )); then
      if attempt_log_indicates_fatal_environment_error "${attempt_run_log}"; then
        echo "[${iid}] attempt=${attempt_tag} fatal environment failure after retries exhausted" | tee -a "${attempt_run_log}"
        return "${ENV_FATAL_EXIT_CODE}"
      fi
      return "${rc}"
    fi

    echo "[${iid}] attempt=${attempt_tag} detected environment failure; retry=$((try_index + 1))/${ENV_ERROR_RETRIES}" | tee -a "${attempt_run_log}"
    runtime_guard_mark_stage "${attempt_run_log}" "attempt_${attempt_tag}:cleanup_after_env_error"
    runtime_guard_cleanup_logged_containers "${attempt_run_log}"
    archive_attempt_retry_artifacts "${attempt_outdir}" "${attempt_event_log}" "${attempt_run_log}" "$((try_index + 1))"
    try_index=$((try_index + 1))
  done
}

build_instance_list
mapfile -t INSTANCE_IDS < <(read_instance_ids)
if [[ "${#INSTANCE_IDS[@]}" -eq 0 ]]; then
  echo "No instance ids selected. Abort." >&2
  exit 1
fi

echo "Selected ${#INSTANCE_IDS[@]} instances, repeats=${REPEATS}, attempt_start=${ATTEMPT_START}, timeout=${INSTANCE_TOTAL_EXEC_TIMEOUT_SEC}s"

for iid in "${INSTANCE_IDS[@]}"; do
  safe_iid="${iid//\//_}"
  instance_log_dir="${LOG_DIR}/${safe_iid}"
  instance_out_root="${OUTDIR}/${iid}"
  mkdir -p "${instance_log_dir}" "${instance_out_root}"

  echo "==== Instance: ${iid} ===="
  for attempt in $(seq "${ATTEMPT_START}" "$((ATTEMPT_START + REPEATS - 1))"); do
    attempt_tag="$(printf "%02d" "${attempt}")"
    attempt_outdir="${instance_out_root}/attempt_${attempt_tag}"
    attempt_event_log="${instance_log_dir}/attempt_${attempt_tag}.jsonl"
    attempt_run_log="${instance_log_dir}/attempt_${attempt_tag}.log"
    mkdir -p "${attempt_outdir}"

    if [[ "$(to_bool_token "${RESUME_MODE}")" == "True" ]]; then
      if attempt_is_complete "${iid}" "${attempt_outdir}"; then
        echo "[${iid}] attempt=${attempt}/${REPEATS} skip (already complete: ${attempt_outdir})"
        continue
      fi
    fi

    set +e
    run_single_attempt "${iid}" "${attempt}" "${attempt_outdir}" "${attempt_event_log}" "${attempt_run_log}"
    rc=$?
    set -e

    if [[ "${rc}" -eq "${ENV_FATAL_EXIT_CODE}" ]]; then
      echo "[${iid}] attempt=${attempt} fatal environment failure rc=${rc}" | tee -a "${attempt_run_log}"
      exit "${rc}"
    fi
    if [[ "${rc}" -ne 0 ]]; then
      echo "[${iid}] attempt=${attempt} failed rc=${rc}" | tee -a "${attempt_run_log}"
      if [[ "$(to_bool_token "${FAIL_FAST}")" == "True" ]]; then
        exit "${rc}"
      fi
    fi
  done
done

"${SWE_AGENT_ROOT}/.venv/bin/python" "${COMMON_DIR}/collect_multi_try_summary.py" \
  --experiment-id "${EXPERIMENT_ID}" \
  --mode "no-mem" \
  --workspace-root "${WS_ROOT}" \
  --output-root "${OUTDIR}" \
  --instance-list-file "${INSTANCE_LIST_FILE}" \
  --repeats "${REPEATS}" \
  --event-log-root "${LOG_DIR}" \
  --summary-out "${SUMMARY_OUT}"

echo "Done: ${SUMMARY_OUT}"
