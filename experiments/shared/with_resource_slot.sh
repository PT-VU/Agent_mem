#!/usr/bin/env bash
set -euo pipefail

SLOT_DIR=""
SLOTS=1
POLL_SEC=5
LABEL="resource_task"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slot-dir)
      SLOT_DIR="$2"
      shift 2
      ;;
    --slots)
      SLOTS="$2"
      shift 2
      ;;
    --poll-sec)
      POLL_SEC="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${SLOT_DIR}" ]]; then
  echo "--slot-dir is required" >&2
  exit 2
fi
if ! [[ "${SLOTS}" =~ ^[0-9]+$ ]] || (( SLOTS < 1 )); then
  echo "--slots must be integer >= 1" >&2
  exit 2
fi
if ! [[ "${POLL_SEC}" =~ ^[0-9]+$ ]] || (( POLL_SEC < 1 )); then
  echo "--poll-sec must be integer >= 1" >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then
  echo "command after -- is required" >&2
  exit 2
fi

mkdir -p "${SLOT_DIR}"

cleanup_stale_slot() {
  local slot_path="$1"
  local pid_file="${slot_path}/pid"
  if [[ ! -d "${slot_path}" ]]; then
    return 0
  fi
  if [[ ! -f "${pid_file}" ]]; then
    rm -rf "${slot_path}"
    return 0
  fi
  local holder_pid
  holder_pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ -z "${holder_pid}" ]] || ! kill -0 "${holder_pid}" 2>/dev/null; then
    rm -rf "${slot_path}"
  fi
}

while true; do
  for idx in $(seq 1 "${SLOTS}"); do
    slot_path="${SLOT_DIR}/slot_${idx}"
    cleanup_stale_slot "${slot_path}"
    if mkdir "${slot_path}" 2>/dev/null; then
      printf '%s\n' "$$" > "${slot_path}/pid"
      trap 'rm -rf "${slot_path}"' EXIT INT TERM
      echo "[slot] acquired idx=${idx} label=${LABEL}" >&2
      "$@"
      rc=$?
      rm -rf "${slot_path}"
      trap - EXIT INT TERM
      exit "${rc}"
    fi
  done
  echo "[slot] waiting label=${LABEL} slots=${SLOTS}" >&2
  sleep "${POLL_SEC}"
done
