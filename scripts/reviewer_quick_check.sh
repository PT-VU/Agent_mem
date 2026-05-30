#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWE_AGENT_ROOT="${SWE_AGENT_ROOT:-}"
STATIC_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reviewer_quick_check.sh --swe-agent-root /path/to/SWE-agent
  bash scripts/reviewer_quick_check.sh --static-only

The default mode runs publication checks and no-API hook smoke tests.
The smoke tests use SWE-agent's dummy runtime. Docker and API keys are not
required. Set KEEP_REVIEWER_CHECK_ARTIFACTS=1 to preserve temporary output.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --swe-agent-root)
      SWE_AGENT_ROOT="${2:?missing value for --swe-agent-root}"
      shift 2
      ;;
    --static-only)
      STATIC_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "[check] public artifact"
bash "${ARTIFACT_ROOT}/scripts/verify_artifact.sh"

if [[ "${STATIC_ONLY}" == "1" ]]; then
  echo "[ok] reviewer static check passed"
  exit 0
fi

if [[ -z "${SWE_AGENT_ROOT}" ]]; then
  if [[ -d "${ARTIFACT_ROOT}/../SWE-agent" ]]; then
    SWE_AGENT_ROOT="${ARTIFACT_ROOT}/../SWE-agent"
  else
    echo "SWE-agent checkout not found. Pass --swe-agent-root /path/to/SWE-agent." >&2
    exit 2
  fi
fi

SWE_AGENT_ROOT="$(cd "${SWE_AGENT_ROOT}" && pwd)"
WS_ROOT="$(cd "${SWE_AGENT_ROOT}/.." && pwd)"
PYTHON_BIN="${SWE_AGENT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing SWE-agent virtual environment: ${PYTHON_BIN}" >&2
  echo "Create SWE-agent/.venv and install SWE-agent first. See README.md." >&2
  exit 2
fi

echo "[check] required Python imports"
"${PYTHON_BIN}" - <<'PY'
import importlib.util

required = ("sweagent", "swerex", "networkx", "numpy", "yaml")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing Python modules: "
        + ", ".join(missing)
        + ". Install framework/sweagent_external_tools_v2/requirements.txt."
    )
print("[ok] Python imports: " + ", ".join(required))
PY

TMP_ROOT="$(mktemp -d -t agent_mem_reviewer_check.XXXXXX)"
cleanup() {
  if [[ "${KEEP_REVIEWER_CHECK_ARTIFACTS:-0}" == "1" ]]; then
    echo "[info] preserved temporary output: ${TMP_ROOT}"
  else
    rm -rf "${TMP_ROOT}"
  fi
}
trap cleanup EXIT

export WS_ROOT SWE_AGENT_ROOT
export SWE_AGENT_PYTHON="${PYTHON_BIN}"
export PYTHONPYCACHEPREFIX="${TMP_ROOT}/pycache"
export AGENT_MEM_ENABLE_LLM_EXTRACTION=0
export AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS=0

run_smoke() {
  local label="$1"
  local package="$2"
  local mode="$3"
  local output="${TMP_ROOT}/${label}.stdout"
  echo "[check] ${label}"
  PERSIST_ROOT="${TMP_ROOT}/${label}_state" \
  SWE_AGENT_EXT_TOOLS_LOG_FILE="${TMP_ROOT}/${label}.jsonl" \
    bash "${ARTIFACT_ROOT}/framework/${package}/bin/start_with_external_tools.sh" \
      --mode "${mode}" --smoke-test >"${output}" 2>&1
  grep -q "SMOKE TEST PASS mode=${mode}" "${output}"
  tail -n 4 "${output}"
}

run_smoke "core_with_mem" "sweagent_external_tools_v2" "with-mem"
run_smoke "core_no_mem" "sweagent_external_tools_v2" "no-mem"
run_smoke "multiagent_with_mem" "sweagent_external_tools_multiagent" "with-mem"

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "[info] Docker daemon is ready for optional real-case verification."
else
  echo "[info] Docker is not ready. Offline checks passed; real-case verification requires Docker."
fi

echo "[ok] reviewer quick-check passed"
