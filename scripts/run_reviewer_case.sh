#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWE_AGENT_ROOT="${SWE_AGENT_ROOT:-}"
MODE="with-mem"
INSTANCE_ID="django__django-16139"
REPEATS="1"
MODEL_CONFIG="${ARTIFACT_ROOT}/config/kimi_k2_5_moonshot.yaml"
RUN_ROOT=""
EXECUTE=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_reviewer_case.sh \
    --swe-agent-root /path/to/SWE-agent \
    [--mode with-mem|no-mem] \
    [--instance django__django-16139] \
    [--repeats 1] \
    [--model-config /path/to/model.yaml] \
    [--run-root /path/to/output] \
    [--execute]

Without --execute, the script prints the resolved configuration and exits.
Execution requires Docker and a MOONSHOT_API_KEY in the environment or in
SWE-agent/.env. A real run consumes model API quota and may pull Docker images.
Use --repeats 2 or more to observe cross-attempt memory reuse.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --swe-agent-root)
      SWE_AGENT_ROOT="${2:?missing value for --swe-agent-root}"
      shift 2
      ;;
    --mode)
      MODE="${2:?missing value for --mode}"
      shift 2
      ;;
    --instance)
      INSTANCE_ID="${2:?missing value for --instance}"
      shift 2
      ;;
    --repeats)
      REPEATS="${2:?missing value for --repeats}"
      shift 2
      ;;
    --model-config)
      MODEL_CONFIG="${2:?missing value for --model-config}"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="${2:?missing value for --run-root}"
      shift 2
      ;;
    --execute)
      EXECUTE=1
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

case "${MODE}" in
  with-mem|no-mem) ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac

if ! [[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--repeats must be an integer greater than zero" >&2
  exit 2
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
ENV_VAR_PATH="${ENV_VAR_PATH:-${SWE_AGENT_ROOT}/.env}"
RUN_ROOT="${RUN_ROOT:-${PWD}/reviewer_runs/${MODE}_${INSTANCE_ID}_$(date +%Y%m%d_%H%M%S)}"

cat <<EOF
=== Agent-mem reviewer case ===
mode          = ${MODE}
instance      = ${INSTANCE_ID}
repeats       = ${REPEATS}
swe_agent     = ${SWE_AGENT_ROOT}
model_config  = ${MODEL_CONFIG}
run_root      = ${RUN_ROOT}
execute       = ${EXECUTE}
EOF

if [[ "${EXECUTE}" != "1" ]]; then
  echo
  echo "Preview only. Add --execute to start Docker and model API work."
  exit 0
fi

for required in \
  "${SWE_AGENT_ROOT}/.venv/bin/python" \
  "${SWE_AGENT_ROOT}/.venv/bin/sweagent" \
  "${MODEL_CONFIG}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required file not found: ${required}" >&2
    exit 2
  fi
done

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required for real-case verification." >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not available to the current user." >&2
  exit 2
fi

if [[ -f "${ENV_VAR_PATH}" ]]; then
  set -a
  source "${ENV_VAR_PATH}"
  set +a
fi
if [[ -z "${MOONSHOT_API_KEY:-}" ]]; then
  echo "MOONSHOT_API_KEY is required for the bundled reviewer model config." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"
export WS_ROOT SWE_AGENT_ROOT ENV_VAR_PATH RUN_ROOT INSTANCE_ID REPEATS MODEL_CONFIG
export PROMPT_PROFILE="reviewer_quick_case"
export SWEBENCH_SUBSET="full"
export SWEBENCH_SPLIT="test"
export INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${INSTANCE_TOTAL_EXEC_TIMEOUT_SEC:-1200}"
export MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-1}"
export EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-1200}"
export WITHMEM_RUNTIME_PREWARM="${WITHMEM_RUNTIME_PREWARM:-0}"
export NOMEM_RUNTIME_PREWARM="${NOMEM_RUNTIME_PREWARM:-0}"
export AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS="${AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS:-0}"
export AGENT_MEM_BUG_INVARIANT_VERBATIM="${AGENT_MEM_BUG_INVARIANT_VERBATIM:-1}"
export AGENT_MEM_BUG_ANTI_PATTERN="${AGENT_MEM_BUG_ANTI_PATTERN:-1}"
export AGENT_MEM_PATCH_CONSISTENCY_GATE="${AGENT_MEM_PATCH_CONSISTENCY_GATE:-enforce}"
export AGENT_MEM_REUSE_EXPLORE="${AGENT_MEM_REUSE_EXPLORE:-auto}"
export AGENT_MEM_LOCAL_EFFECTIVE_FEEDBACK="${AGENT_MEM_LOCAL_EFFECTIVE_FEEDBACK:-1}"
export AGENT_MEM_L3_FORCE_SUBMIT="${AGENT_MEM_L3_FORCE_SUBMIT:-dry_run}"
export AGENT_MEM_FORCE_STRATEGY="${AGENT_MEM_FORCE_STRATEGY:-auto}"

if [[ "${MODE}" == "with-mem" ]]; then
  bash "${ARTIFACT_ROOT}/experiments/shared/run_same_problem_withmem.sh"
else
  bash "${ARTIFACT_ROOT}/experiments/shared/run_same_problem_nomem.sh"
fi

echo "[ok] reviewer case finished: ${RUN_ROOT}"
