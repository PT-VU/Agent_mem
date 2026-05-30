#!/usr/bin/env bash
set -euo pipefail
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

runtime_guard_match_tokens() {
  local image_pattern="$1"
  local no_registry no_tag sanitized instance_token

  printf '%s\n' "${image_pattern}"

  no_registry="${image_pattern#docker.io/}"
  no_tag="${no_registry%:*}"
  sanitized="$(printf '%s' "${image_pattern}" | tr -cd '[:alnum:]_.-')"
  instance_token="$(printf '%s' "${no_tag}" | grep -oE '[A-Za-z0-9]+-[0-9]+$' || true)"

  [[ "${no_registry}" != "${image_pattern}" ]] && printf '%s\n' "${no_registry}"
  [[ "${no_tag}" != "${no_registry}" ]] && printf '%s\n' "${no_tag}"
  [[ -n "${sanitized}" ]] && printf '%s\n' "${sanitized}"
  [[ -n "${instance_token}" ]] && printf '%s\n' "${instance_token}"
}

runtime_guard_list_matching_containers() {
  local image_pattern="$1"
  local token line
  local -a tokens=()

  while IFS= read -r token; do
    [[ -z "${token}" ]] && continue
    tokens+=("${token}")
  done < <(runtime_guard_match_tokens "${image_pattern}" | awk '!seen[$0]++')

  docker ps -a --format '{{.ID}} {{.Image}} {{.Names}}' | while IFS= read -r line; do
    for token in "${tokens[@]}"; do
      if [[ "${line}" == *"${token}"* ]]; then
        printf '%s\n' "${line}"
        break
      fi
    done
  done
}

runtime_guard_extract_logged_container_names() {
  local log_file="$1"
  [[ -f "${log_file}" ]] || return 0
  python3 - <<'PY' "${log_file}"
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
patterns = [
    r"Starting container\s+([A-Za-z0-9._:\-\s]+?)\s+with image",
    r"--name\s+([A-Za-z0-9._:\-\s]+?)\s+(?:sha256:|docker\.io/|/bin/sh|\")",
]
seen = set()
for pattern in patterns:
    for name in re.findall(pattern, text, flags=re.MULTILINE):
        normalized = re.sub(r"\s+", "", name)
        if normalized and normalized not in seen:
            seen.add(normalized)
            print(normalized)
PY
}

runtime_guard_mark_stage() {
  local log_file="$1"
  local stage="$2"
  {
    echo "[stage] $(date -Is) ${stage}"
  } >>"${log_file}" 2>&1
}

runtime_guard_preflight() {
  local log_file="$1"
  local rc=0
  local restore_errexit=0
  case $- in
    *e*)
      restore_errexit=1
      set +e
      ;;
  esac
  {
    echo "[preflight] $(date -Is) docker info"
  } >>"${log_file}" 2>&1
  docker info >/dev/null >>"${log_file}" 2>&1
  rc=$?
  if (( restore_errexit )); then
    set -e
  fi
  if [[ "${rc}" -ne 0 ]]; then
    {
      echo "[preflight] docker info failed rc=${rc}"
    } >>"${log_file}" 2>&1
    return "${rc}"
  fi

  {
    echo "[preflight] $(date -Is) docker ps -a"
  } >>"${log_file}" 2>&1
  if (( restore_errexit )); then
    set +e
  fi
  docker ps -a --format '{{.ID}} {{.Image}} {{.Status}} {{.Names}}' >/dev/null >>"${log_file}" 2>&1
  rc=$?
  if (( restore_errexit )); then
    set -e
  fi
  if [[ "${rc}" -ne 0 ]]; then
    {
      echo "[preflight] docker ps -a failed rc=${rc}"
    } >>"${log_file}" 2>&1
    return "${rc}"
  fi

  {
    echo "[preflight] docker ok"
  } >>"${log_file}" 2>&1
}

runtime_guard_prepare_runtime_image() {
  local log_file="$1"
  local base_image="$2"
  local standalone_dir="$3"
  local platform="${4:-linux/amd64}"
  local timeout_sec="${5:-0}"
  local ws_root="${WS_ROOT:-/home/pt/SWE-bench}"
  local helper="${COMMON_DIR}/prepare_swerex_runtime.py"
  local rc=0

  if [[ -z "${standalone_dir}" || "${standalone_dir}" == "__NONE__" ]]; then
    {
      echo "[prepare] skip runtime image warmup because python_standalone_dir is disabled"
    } >>"${log_file}" 2>&1
    return 0
  fi

  local image_key lock_file
  image_key="$(printf '%s|%s|%s' "${base_image}" "${standalone_dir}" "${platform}" | sha256sum | awk '{print $1}')"
  lock_file="/tmp/agentmem_runtime_prepare_${image_key}.lock"

  {
    echo "[prepare] $(date -Is) begin runtime image warmup base_image=${base_image} standalone_dir=${standalone_dir} platform=${platform}"
  } >>"${log_file}" 2>&1

  (
    flock -w 1800 9 || {
      echo "[prepare] failed to acquire warmup lock: ${lock_file}" >>"${log_file}" 2>&1
      exit 1
    }
    if [[ "${timeout_sec}" =~ ^[0-9]+$ ]] && (( timeout_sec > 0 )); then
      timeout --foreground -k 45s "${timeout_sec}s" \
        "${ws_root}/SWE-agent/.venv/bin/python" "${helper}" \
          --image "${base_image}" \
          --python-standalone-dir "${standalone_dir}" \
          --platform "${platform}" \
          --output-format text
    else
      "${ws_root}/SWE-agent/.venv/bin/python" "${helper}" \
        --image "${base_image}" \
        --python-standalone-dir "${standalone_dir}" \
        --platform "${platform}" \
        --output-format text
    fi
  ) 9>"${lock_file}" >>"${log_file}" 2>&1 || rc=$?

  if [[ "${rc}" -eq 0 ]]; then
    {
      echo "[prepare] $(date -Is) runtime image warmup ok"
    } >>"${log_file}" 2>&1
  else
    {
      echo "[prepare] $(date -Is) runtime image warmup failed rc=${rc}"
    } >>"${log_file}" 2>&1
  fi
  return "${rc}"
}

runtime_guard_pull_image() {
  local log_file="$1"
  local image="$2"
  local timeout_sec="${3:-0}"
  local rc=0

  if docker image inspect "${image}" >/dev/null 2>&1; then
    {
      echo "[pull] image already present: ${image}"
    } >>"${log_file}" 2>&1
    return 0
  fi

  {
    echo "[pull] $(date -Is) docker pull ${image}"
  } >>"${log_file}" 2>&1

  if [[ "${timeout_sec}" =~ ^[0-9]+$ ]] && (( timeout_sec > 0 )); then
    timeout --foreground -k 45s "${timeout_sec}s" docker pull "${image}" >>"${log_file}" 2>&1 || rc=$?
  else
    docker pull "${image}" >>"${log_file}" 2>&1 || rc=$?
  fi

  if [[ "${rc}" -eq 0 ]]; then
    {
      echo "[pull] $(date -Is) docker pull ok: ${image}"
    } >>"${log_file}" 2>&1
  else
    {
      echo "[pull] $(date -Is) docker pull failed rc=${rc}: ${image}"
    } >>"${log_file}" 2>&1
  fi
  return "${rc}"
}

runtime_guard_resolve_swebench_image() {
  local log_file="$1"
  local instance_id="$2"
  local subset="${3:-full}"
  local split="${4:-test}"
  local ws_root="${WS_ROOT:-/home/pt/SWE-bench}"
  local helper="${COMMON_DIR}/resolve_swebench_image.py"
  local image=""

  if ! image="$("${ws_root}/SWE-agent/.venv/bin/python" "${helper}" \
      --instance-id "${instance_id}" \
      --subset "${subset}" \
      --split "${split}" 2>>"${log_file}")"; then
    {
      echo "[resolve-image] failed for instance=${instance_id} subset=${subset} split=${split}"
    } >>"${log_file}" 2>&1
    return 1
  fi

  image="$(printf '%s' "${image}" | tail -n 1 | tr -d '\r')"
  if [[ -z "${image}" ]]; then
    {
      echo "[resolve-image] empty image for instance=${instance_id}"
    } >>"${log_file}" 2>&1
    return 1
  fi

  {
    echo "[resolve-image] ${instance_id} -> ${image}"
  } >>"${log_file}" 2>&1
  printf '%s\n' "${image}"
}

runtime_guard_wait_for_capacity() {
  local log_file="$1"
  local max_active="$2"
  local poll_sec="${3:-10}"
  if [[ -z "${max_active}" || "${max_active}" == "0" ]]; then
    return 0
  fi
  while true; do
    local active_count
    active_count="$(
      ps -eo args= | awk '
        $0 ~ /^[^ ]*python[^ ]* [^ ]*sweagent run-batch( |$)/ { count++; next }
        $0 ~ /^[^ ]*sweagent run-batch( |$)/ { count++; next }
        END { print count + 0 }
      '
    )"
    if [[ "${active_count}" =~ ^[0-9]+$ ]] && (( active_count < max_active )); then
      {
        echo "[capacity] $(date -Is) active_run_batch=${active_count} max=${max_active} -> proceed"
      } >>"${log_file}" 2>&1
      return 0
    fi
    {
      echo "[capacity] $(date -Is) active_run_batch=${active_count} max=${max_active} -> wait ${poll_sec}s"
    } >>"${log_file}" 2>&1
    sleep "${poll_sec}"
  done
}

runtime_guard_capture_diag() {
  local log_file="$1"
  local image_pattern="$2"
  {
    echo "[diag] $(date -Is) docker ps -a"
    docker ps -a --format '{{.ID}} {{.Image}} {{.Status}} {{.Names}}'
    echo "[diag] matching containers for ${image_pattern}"
    runtime_guard_list_matching_containers "${image_pattern}" || true
    while read -r cid _; do
      [ -z "${cid}" ] && continue
      echo "[diag] docker inspect ${cid}"
      docker inspect "${cid}" || true
      echo "[diag] docker logs ${cid}"
      docker logs --tail 200 "${cid}" || true
    done < <(runtime_guard_list_matching_containers "${image_pattern}" || true)
  } >>"${log_file}" 2>&1
}

runtime_guard_cleanup_pattern() {
  local log_file="$1"
  local image_pattern="$2"
  {
    echo "[cleanup] $(date -Is) image pattern=${image_pattern}"
    while read -r cid _ name; do
      [ -z "${cid}" ] && continue
      echo "[cleanup] removing container ${cid} ${name}"
      docker rm -f "${cid}" || true
    done < <(runtime_guard_list_matching_containers "${image_pattern}" || true)
  } >>"${log_file}" 2>&1
}

runtime_guard_cleanup_logged_containers() {
  local log_file="$1"
  {
    echo "[cleanup] $(date -Is) logged_containers_from=${log_file}"
    while IFS= read -r container_name; do
      [[ -z "${container_name}" ]] && continue
      echo "[cleanup] removing logged container ${container_name}"
      docker rm -f "${container_name}" || true
    done < <(runtime_guard_extract_logged_container_names "${log_file}" || true)
  } >>"${log_file}" 2>&1
}
