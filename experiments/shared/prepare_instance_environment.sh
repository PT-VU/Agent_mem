#!/usr/bin/env bash
set -euo pipefail

WS_ROOT="${WS_ROOT:-/home/pt/SWE-bench}"
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/common" && pwd)"
RUNTIME_GUARD_SH="${RUNTIME_GUARD_SH:-${COMMON_DIR}/docker_runtime_guard.sh}"

INSTANCE_ID="${INSTANCE_ID:?INSTANCE_ID is required}"
LOG_FILE="${LOG_FILE:?LOG_FILE is required}"
SWEBENCH_SUBSET="${SWEBENCH_SUBSET:-lite}"
SWEBENCH_SPLIT="${SWEBENCH_SPLIT:-test}"
PLATFORM="${PLATFORM:-linux/amd64}"
PYTHON_STANDALONE_DIR="${PYTHON_STANDALONE_DIR:-__NONE__}"
RUNTIME_PREWARM="${RUNTIME_PREWARM:-0}"
IMAGE_PULL_TIMEOUT_SEC="${IMAGE_PULL_TIMEOUT_SEC:-1800}"
RUNTIME_WARMUP_TIMEOUT_SEC="${RUNTIME_WARMUP_TIMEOUT_SEC:-1800}"
ENV_FATAL_EXIT_CODE="${ENV_FATAL_EXIT_CODE:-86}"

if [[ ! -f "${RUNTIME_GUARD_SH}" ]]; then
  echo "missing runtime guard: ${RUNTIME_GUARD_SH}" >&2
  exit 2
fi

mkdir -p "$(dirname "${LOG_FILE}")"
source "${RUNTIME_GUARD_SH}"

runtime_guard_mark_stage "${LOG_FILE}" "environment_prepare:preflight"
if ! runtime_guard_preflight "${LOG_FILE}"; then
  runtime_guard_mark_stage "${LOG_FILE}" "environment_prepare:preflight_failed"
  exit "${ENV_FATAL_EXIT_CODE}"
fi

runtime_guard_mark_stage "${LOG_FILE}" "environment_prepare:resolve_image"
if ! INSTANCE_IMAGE="$(runtime_guard_resolve_swebench_image "${LOG_FILE}" "${INSTANCE_ID}" "${SWEBENCH_SUBSET}" "${SWEBENCH_SPLIT}")"; then
  runtime_guard_mark_stage "${LOG_FILE}" "environment_prepare:resolve_image_failed"
  exit "${ENV_FATAL_EXIT_CODE}"
fi

runtime_guard_mark_stage "${LOG_FILE}" "environment_prepare:pull_image"
if ! runtime_guard_pull_image "${LOG_FILE}" "${INSTANCE_IMAGE}" "${IMAGE_PULL_TIMEOUT_SEC}"; then
  runtime_guard_mark_stage "${LOG_FILE}" "environment_prepare:pull_image_failed"
  exit "${ENV_FATAL_EXIT_CODE}"
fi

if [[ "${RUNTIME_PREWARM}" == "1" ]]; then
  runtime_guard_mark_stage "${LOG_FILE}" "environment_prepare:runtime_warmup"
  if ! runtime_guard_prepare_runtime_image "${LOG_FILE}" "${INSTANCE_IMAGE}" "${PYTHON_STANDALONE_DIR}" "${PLATFORM}" "${RUNTIME_WARMUP_TIMEOUT_SEC}"; then
    runtime_guard_mark_stage "${LOG_FILE}" "environment_prepare:runtime_warmup_failed"
    exit "${ENV_FATAL_EXIT_CODE}"
  fi
fi

runtime_guard_mark_stage "${LOG_FILE}" "environment_prepare:ready"
echo "instance_id=${INSTANCE_ID}" >>"${LOG_FILE}"
echo "instance_image=${INSTANCE_IMAGE}" >>"${LOG_FILE}"
