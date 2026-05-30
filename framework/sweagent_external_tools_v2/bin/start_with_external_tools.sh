#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Public artifact defaults remain overrideable through environment variables.
WS_ROOT="${WS_ROOT:-/home/pt/SWE-bench}"
SWE_AGENT_ROOT="${SWE_AGENT_ROOT:-${WS_ROOT}/SWE-agent}"
PERSIST_ROOT="${PERSIST_ROOT:-${WS_ROOT}/PDDL_work_mem/05_operations/state/persistence}"
# Keep the package parent importable without requiring a workspace symlink.
V2_FW_DIR="${V2_FW_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
V2_FW_PARENT="$(cd "${V2_FW_DIR}/.." && pwd)"

MODE="${SWE_AGENT_AGENT_MEM_MODE:-with-mem}"
if [[ "${1:-}" == "--mode" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "Missing value for --mode (expected with-mem|no-mem)" >&2
    exit 2
  fi
  MODE="$2"
  shift 2
fi

# Clear stale hook values from parent shell before selecting mode-specific defaults.
unset SWE_AGENT_EXTERNAL_HOOK_CLASS SWE_AGENT_EXTERNAL_HOOK_PYTHONPATH

case "${MODE}" in
  with-mem)
    export SWE_AGENT_EXTERNAL_HOOK_CLASS="sweagent_external_tools_v2.bridge_hook:ExternalToolBridgeHook"
    export SWE_AGENT_EXTERNAL_HOOK_PYTHONPATH="${V2_FW_PARENT}"
    export SWE_AGENT_EXT_TOOL_A_CMD="${SWE_AGENT_EXT_TOOL_A_CMD:-${V2_FW_DIR}/bin/run_tool_a.sh}"
    export SWE_AGENT_EXT_TOOL_B_CMD="${SWE_AGENT_EXT_TOOL_B_CMD:-${V2_FW_DIR}/bin/run_tool_b.sh}"
    export SWE_AGENT_EXT_TOOL_TIMEOUT_SEC="${SWE_AGENT_EXT_TOOL_TIMEOUT_SEC:-8.0}"
    export SWE_AGENT_EXT_TOOL_A_TIMEOUT_SEC="${SWE_AGENT_EXT_TOOL_A_TIMEOUT_SEC:-45.0}"
    export SWE_AGENT_EXT_TOOL_B_TIMEOUT_SEC="${SWE_AGENT_EXT_TOOL_B_TIMEOUT_SEC:-8.0}"
    export SWE_AGENT_EXT_TOOL_RETRY_TIMEOUT_SEC="${SWE_AGENT_EXT_TOOL_RETRY_TIMEOUT_SEC:-90.0}"
    export SWE_AGENT_EXT_TOOL_MAX_RETRIES="${SWE_AGENT_EXT_TOOL_MAX_RETRIES:-1}"
    export SWE_AGENT_EXT_TOOL_ADAPTIVE_TIMEOUT="${SWE_AGENT_EXT_TOOL_ADAPTIVE_TIMEOUT:-1}"
    export SWE_AGENT_EXT_TOOL_ENABLE_STALE_FALLBACK="${SWE_AGENT_EXT_TOOL_ENABLE_STALE_FALLBACK:-1}"
    export SWE_AGENT_EXT_TOOL_CIRCUIT_THRESHOLD="${SWE_AGENT_EXT_TOOL_CIRCUIT_THRESHOLD:-8}"
    export SWE_AGENT_EXT_TOOL_CIRCUIT_COOLDOWN_SEC="${SWE_AGENT_EXT_TOOL_CIRCUIT_COOLDOWN_SEC:-30}"
    export SWE_AGENT_EXT_TOOLS_LOG_FILE="${SWE_AGENT_EXT_TOOLS_LOG_FILE:-/tmp/sweagent_ext_tools.log}"
    export SWE_AGENT_MEM_MAX_HINTS="${SWE_AGENT_MEM_MAX_HINTS:-0}"
    export SWE_AGENT_MEM_HINT_CHAR_BUDGET="${SWE_AGENT_MEM_HINT_CHAR_BUDGET:-2400}"
    export SWE_AGENT_MEM_MIN_ITEM_CONFIDENCE="${SWE_AGENT_MEM_MIN_ITEM_CONFIDENCE:-0.0}"
    export SWE_AGENT_MEM_GATE_MIN_EXTERNAL_RATIO="${SWE_AGENT_MEM_GATE_MIN_EXTERNAL_RATIO:-0.35}"
    export SWE_AGENT_MEM_GATE_MIN_BUFFER_RATIO="${SWE_AGENT_MEM_GATE_MIN_BUFFER_RATIO:-0.15}"
    export SWE_AGENT_MEM_GATE_MIN_ACTION_ERROR_COVERAGE="${SWE_AGENT_MEM_GATE_MIN_ACTION_ERROR_COVERAGE:-0.60}"
    export SWE_AGENT_MEM_GATE_HARD_FAIL="${SWE_AGENT_MEM_GATE_HARD_FAIL:-0}"
    export AGENT_MEM_STORAGE_DIR="${AGENT_MEM_STORAGE_DIR:-${PERSIST_ROOT}/graph_store}"
    export AGENT_MEM_EVIDENCE_DIR="${AGENT_MEM_EVIDENCE_DIR:-${PERSIST_ROOT}/evidence_store}"
    export AGENT_MEM_EMBEDDING_MODEL="${AGENT_MEM_EMBEDDING_MODEL:-sentence-transformers}"
    export AGENT_MEM_EMBEDDING_MODEL_NAME="${AGENT_MEM_EMBEDDING_MODEL_NAME:-all-MiniLM-L6-v2}"
    export AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS="${AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS:-1}"
    export AGENT_MEM_ENABLE_LLM_EXTRACTION="${AGENT_MEM_ENABLE_LLM_EXTRACTION:-1}"
    export AGENT_MEM_LLM_EXTRACT_MODE="${AGENT_MEM_LLM_EXTRACT_MODE:-hybrid}"
    export AGENT_MEM_LLM_MODEL="${AGENT_MEM_LLM_MODEL:-kimi-k2.5}"
    # Keep URL/key overrideable; default key follows Moonshot Kimi via SWE-agent env.
    export AGENT_MEM_LLM_API_URL="${AGENT_MEM_LLM_API_URL:-https://api.moonshot.cn/v1/chat/completions}"
    export AGENT_MEM_LLM_API_KEY="${AGENT_MEM_LLM_API_KEY:-${MOONSHOT_API_KEY:-${SILICONFLOW_API_KEY:-}}}"
    # 0 means no fixed cap: extractor can emit multi-signal outputs.
    export AGENT_MEM_EXTRACT_MAX_SIGNALS="${AGENT_MEM_EXTRACT_MAX_SIGNALS:-0}"
    export AGENT_MEM_EXTRACT_MAX_ABSTRACTS="${AGENT_MEM_EXTRACT_MAX_ABSTRACTS:-0}"
    export AGENT_MEM_EXTRACT_MAX_REPAIR_PATTERNS="${AGENT_MEM_EXTRACT_MAX_REPAIR_PATTERNS:-0}"
    export AGENT_MEM_V21_ENABLE_SUCCESS_FACT_HOTPATH="${AGENT_MEM_V21_ENABLE_SUCCESS_FACT_HOTPATH:-0}"
    export AGENT_MEM_V21_ENABLE_SIDECAR="${AGENT_MEM_V21_ENABLE_SIDECAR:-0}"
    export AGENT_MEM_V21_ENABLE_SUBTASK_PROJECTION="${AGENT_MEM_V21_ENABLE_SUBTASK_PROJECTION:-0}"
    export AGENT_MEM_V21_ENABLE_CARD_COMPILER="${AGENT_MEM_V21_ENABLE_CARD_COMPILER:-0}"
    export AGENT_MEM_V21_ENABLE_GOVERNANCE="${AGENT_MEM_V21_ENABLE_GOVERNANCE:-0}"
    export AGENT_MEM_V21_SIDECAR_DIR="${AGENT_MEM_V21_SIDECAR_DIR:-${PERSIST_ROOT}/sidecar_store}"
    export AGENT_MEM_V21_HOTPATH_TIMEOUT_MS="${AGENT_MEM_V21_HOTPATH_TIMEOUT_MS:-50}"
    export AGENT_MEM_V21_COLDPATH_TIMEOUT_MS="${AGENT_MEM_V21_COLDPATH_TIMEOUT_MS:-5000}"
    export AGENT_MEM_V21_MAX_CARDS_PER_QUERY="${AGENT_MEM_V21_MAX_CARDS_PER_QUERY:-4}"
    mkdir -p "${AGENT_MEM_STORAGE_DIR}" "${AGENT_MEM_EVIDENCE_DIR}"
    if [[ "${AGENT_MEM_V21_ENABLE_SIDECAR}" == "1" || "${AGENT_MEM_V21_ENABLE_SIDECAR,,}" == "true" ]]; then
      mkdir -p "${AGENT_MEM_V21_SIDECAR_DIR}"
    fi
    ;;
  no-mem)
    # No-memory mode retains baseline event logging.
    export SWE_AGENT_EXTERNAL_HOOK_CLASS="sweagent_external_tools_v2.baseline_hook:BaselineLoggingHook"
    export SWE_AGENT_EXTERNAL_HOOK_PYTHONPATH="${V2_FW_PARENT}"
    export SWE_AGENT_EXT_TOOLS_LOG_FILE="${SWE_AGENT_EXT_TOOLS_LOG_FILE:-/tmp/sweagent_ext_tools.log}"
    ;;
  *)
    echo "Unsupported mode: ${MODE} (expected with-mem|no-mem)" >&2
    exit 2
    ;;
esac

export SWE_AGENT_EXTERNAL_HOOK_CLASS SWE_AGENT_EXT_TOOL_A_CMD SWE_AGENT_EXT_TOOL_B_CMD

export SWE_AGENT_EXT_MODE="${MODE}"

if [[ "${1:-}" == "--smoke-test" ]]; then
  exec "${SWE_AGENT_ROOT}/.venv/bin/python" "${V2_FW_DIR}/tests/smoke_test.py"
fi

if [[ $# -eq 0 ]]; then
  cat <<'USAGE'
Usage:
  start_with_external_tools.sh [--mode with-mem|no-mem] --smoke-test
  start_with_external_tools.sh [--mode with-mem|no-mem] <command> [args...]

Examples:
  start_with_external_tools.sh --mode with-mem --smoke-test
  start_with_external_tools.sh --mode no-mem --smoke-test
  start_with_external_tools.sh --mode with-mem sweagent run --config config/default.yaml ...
USAGE
  exit 1
fi

exec "$@"
