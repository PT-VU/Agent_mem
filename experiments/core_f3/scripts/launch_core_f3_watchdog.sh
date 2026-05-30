#!/usr/bin/env bash
# F3: 9 instances x 10 attempts x nomem + with_mem with fair watchdog timing.
# Django: 12284 / 16139 / 12497
# SymPy: 24066 / 13031 / 13551
# Astropy: 12907 / 14995 / 13033
set -euo pipefail

WS_ROOT="${WS_ROOT:-/home/pt/SWE-bench}"
ARTIFACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
V2_FW_REAL="${ARTIFACT_ROOT}/framework/sweagent_external_tools_v2"
V2_FW_LINK="${V2_FW_REAL}"

RUN_TAG="${RUN_TAG:-phase9_f3_9inst_both_watchdog_$(date +%Y%m%d_%H%M%S)}"
RUNS_ROOT="${RUNS_ROOT:-${WS_ROOT}/PDDL_work_mem/06_artificial_intelligence/experiments/final_validation/phase9_v2}"
RUN_ROOT="${RUNS_ROOT}/${RUN_TAG}"
MODEL_CONFIG="${MODEL_CONFIG:-${WS_ROOT}/SWE-agent/config/kimi25_moonshot.yaml}"
PROMPT_PROFILE="${PROMPT_PROFILE:-prompt_base}"
REPEATS="${REPEATS:-10}"
export SWEBENCH_SUBSET="full"
export SWEBENCH_SPLIT="test"

mkdir -p "${RUN_ROOT}/orchestrator_logs" "${RUN_ROOT}/orchestrator_state" \
          "${RUN_ROOT}/notes" "${RUN_ROOT}/agent_mem_logs"

INSTANCE_LIST_JSON="${RUN_ROOT}/same_instances.json"
cat > "${INSTANCE_LIST_JSON}" <<JSON
{
  "generated_at": "$(date -Iseconds)",
  "subset": "full",
  "split": "test",
  "selection_mode": "manual_f3_9instances_nomem_withmem_watchdog",
  "count_selected": 9,
  "repeats": ${REPEATS},
  "instance_ids": [
    "django__django-12284",
    "django__django-16139",
    "django__django-12497",
    "sympy__sympy-24066",
    "sympy__sympy-13031",
    "sympy__sympy-13551",
    "astropy__astropy-12907",
    "astropy__astropy-14995",
    "astropy__astropy-13033"
  ]
}
JSON

cat > "${RUN_ROOT}/agent_mem_logs/v2_config.json" <<'JSON'
{
  "AGENT_MEM_BUG_INVARIANT_VERBATIM": "1",
  "AGENT_MEM_BUG_ANTI_PATTERN": "1",
  "AGENT_MEM_PATCH_CONSISTENCY_GATE": "enforce",
  "AGENT_MEM_REUSE_EXPLORE": "auto",
  "AGENT_MEM_LOCAL_EFFECTIVE_FEEDBACK": "1",
  "AGENT_MEM_L3_FORCE_SUBMIT": "dry_run",
  "AGENT_MEM_FORCE_STRATEGY": "auto"
}
JSON

cat > "${RUN_ROOT}/orchestrator_state/run_manifest.json" <<JSON
{
  "generated_at": "$(date -Iseconds)",
  "experiment_type": "f3_9instances_nomem_withmem_watchdog",
  "run_root": "${RUN_ROOT}",
  "framework_version": "phase9_v2c_6fixes_watchdog",
  "groups": ["nomem", "with_mem"],
  "repeats": ${REPEATS},
  "instances": [
    "django__django-12284", "django__django-16139", "django__django-12497",
    "sympy__sympy-24066", "sympy__sympy-13031", "sympy__sympy-13551",
    "astropy__astropy-12907", "astropy__astropy-14995", "astropy__astropy-13033"
  ],
  "nomem_prior_estimates": {
    "django__django-12284":  "30% (phase8)",
    "django__django-16139":  "26% (phase8)",
    "django__django-12497":  "25% (phase8)",
    "sympy__sympy-24066":    "33% (phase8)",
    "sympy__sympy-13031":    "unknown (Verified)",
    "sympy__sympy-13551":    "unknown (Verified)",
    "astropy__astropy-12907": "23% (phase8)",
    "astropy__astropy-14995": "45% (phase8)",
    "astropy__astropy-13033": "unknown (Verified)"
  }
}
JSON

SWE_AGENT_ENV_FILE="${WS_ROOT}/SWE-agent/.env"
if [[ -f "${SWE_AGENT_ENV_FILE}" ]]; then set -a; source "${SWE_AGENT_ENV_FILE}"; set +a; fi
if [[ -z "${MOONSHOT_API_KEY:-}" ]]; then echo "MOONSHOT_API_KEY required" >&2; exit 2; fi

export START_WITH_TOOLS="${V2_FW_LINK}/bin/start_with_external_tools.sh"
export V2_FW_DIR="${V2_FW_LINK}"
export SWE_AGENT_EXTERNAL_HOOK_PYTHONPATH="${ARTIFACT_ROOT}/framework"
export SWE_AGENT_EXTERNAL_HOOK_CLASS="sweagent_external_tools_v2.bridge_hook:ExternalToolBridgeHook"
export SWE_AGENT_EXT_TOOL_A_CMD="${V2_FW_LINK}/bin/run_tool_a.sh"
export SWE_AGENT_EXT_TOOL_B_CMD="${V2_FW_LINK}/bin/run_tool_b.sh"
export AGENT_MEM_V2_CONFIG_FILE="${RUN_ROOT}/agent_mem_logs/v2_config.json"
export RUN_ROOT RUN_TAG MODEL_CONFIG PROMPT_PROFILE REPEATS

export AGENT_MEM_BUG_INVARIANT_VERBATIM="1"
export AGENT_MEM_BUG_ANTI_PATTERN="1"
export AGENT_MEM_PATCH_CONSISTENCY_GATE="enforce"
export AGENT_MEM_REUSE_EXPLORE="auto"
export AGENT_MEM_LOCAL_EFFECTIVE_FEEDBACK="1"
export AGENT_MEM_L3_FORCE_SUBMIT="dry_run"
export AGENT_MEM_FORCE_STRATEGY="auto"

# Run one nomem lane and one with_mem lane concurrently.
export GLOBAL_HEAVY_SLOTS="2"
export NOMEM_HEAVY_SLOTS="1"
export WITHMEM_HEAVY_SLOTS="1"
export MAX_ACTIVE_RUN_BATCHES="2"
export PER_INSTANCE_CALL_LIMIT="0"
export NOMEM_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="1200"
export WITHMEM_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="1200"
export MAX_WORKERS_EVAL="1"
export EVAL_TIMEOUT_SEC="1200"
export ENV_ERROR_RETRIES="5"
export ENV_RETRY_WAIT_SEC="30"
export ENV_FATAL_EXIT_CODE="86"
export IMAGE_PULL_TIMEOUT_SEC="1800"
export PHASE7_PREPARE_SLOTS="2"
export PHASE7_PREPARE_SLOT_POLL_SEC="10"
export NOMEM_RUNTIME_PREWARM="0"
export WITHMEM_RUNTIME_PREWARM="1"
export NOMEM_RUNTIME_WARMUP_TIMEOUT_SEC="1800"
export WITHMEM_RUNTIME_WARMUP_TIMEOUT_SEC="1800"
export NOMEM_PYTHON_STANDALONE_DIR="/root"
export WITHMEM_PYTHON_STANDALONE_DIR="/root"
export SKIP_INSTANCE_PREPARE="0"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export RESUME_MODE="1"
export REDO_EXISTING="0"
export DRY_RUN="0"
export SWE_AGENT_EXT_TOOLS_LOG_FILE="${RUN_ROOT}/agent_mem_logs/hook_events.jsonl"
# Watchdog: exclude bridge overhead from the with_mem budget.
export WATCHDOG_PY="${V2_FW_LINK}/bin/watchdog.py"

mkdir -p "${RUN_ROOT}/resource_slots/same_global" \
         "${RUN_ROOT}/resource_slots/same_nomem" \
         "${RUN_ROOT}/resource_slots/same_with_mem"

RESUME_SCRIPT="${ARTIFACT_ROOT}/experiments/shared/resume_same_only.sh"
INVENTORY_SCRIPT="${ARTIFACT_ROOT}/experiments/shared/collect_trial_inventory.py"
PYTHON_BIN="${WS_ROOT}/SWE-agent/.venv/bin/python"

MAIN_ENV_FILE="${RUN_ROOT}/orchestrator_state/main_tmux.env"
{
  for v in RUN_ROOT RUN_TAG MODEL_CONFIG PROMPT_PROFILE REPEATS SWEBENCH_SUBSET SWEBENCH_SPLIT \
           START_WITH_TOOLS V2_FW_DIR SWE_AGENT_EXTERNAL_HOOK_PYTHONPATH SWE_AGENT_EXTERNAL_HOOK_CLASS \
           SWE_AGENT_EXT_TOOL_A_CMD SWE_AGENT_EXT_TOOL_B_CMD SWE_AGENT_EXT_TOOLS_LOG_FILE \
           AGENT_MEM_V2_CONFIG_FILE AGENT_MEM_BUG_INVARIANT_VERBATIM AGENT_MEM_BUG_ANTI_PATTERN \
           AGENT_MEM_PATCH_CONSISTENCY_GATE AGENT_MEM_REUSE_EXPLORE AGENT_MEM_LOCAL_EFFECTIVE_FEEDBACK \
           AGENT_MEM_L3_FORCE_SUBMIT AGENT_MEM_FORCE_STRATEGY \
           GLOBAL_HEAVY_SLOTS NOMEM_HEAVY_SLOTS WITHMEM_HEAVY_SLOTS \
           NOMEM_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC WITHMEM_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC \
           EVAL_TIMEOUT_SEC IMAGE_PULL_TIMEOUT_SEC ENV_ERROR_RETRIES ENV_RETRY_WAIT_SEC \
           PHASE7_PREPARE_SLOTS WITHMEM_RUNTIME_PREWARM WITHMEM_PYTHON_STANDALONE_DIR \
           HF_HUB_OFFLINE HF_DATASETS_OFFLINE WATCHDOG_PY; do
    echo "${v}='${!v:-}'"
  done
} > "${MAIN_ENV_FILE}"

export SAME_JSON="${INSTANCE_LIST_JSON}"

SANITIZED_TAG="${RUN_TAG//[^A-Za-z0-9_]/_}"
MAIN_SESSION="${SANITIZED_TAG}_main"
INDEX_SESSION="${SANITIZED_TAG}_idx"
MAIN_LOG="${RUN_ROOT}/orchestrator_logs/main.tmux.log"
INDEX_LOG="${RUN_ROOT}/orchestrator_logs/indexer.tmux.log"

if tmux has-session -t "${MAIN_SESSION}" 2>/dev/null; then
  echo "tmux session already running: ${MAIN_SESSION}" >&2; exit 3
fi

tmux new-session -d -s "${MAIN_SESSION}" \
  "set -a; source '${SWE_AGENT_ENV_FILE}'; source '${MAIN_ENV_FILE}'; set +a; export SAME_JSON='${INSTANCE_LIST_JSON}'; '${RESUME_SCRIPT}' >> '${MAIN_LOG}' 2>&1"

tmux new-session -d -s "${INDEX_SESSION}" \
  "while tmux has-session -t '${MAIN_SESSION}' 2>/dev/null; do '${PYTHON_BIN}' '${INVENTORY_SCRIPT}' --run-root '${RUN_ROOT}' >> '${INDEX_LOG}' 2>&1; sleep 60; done; '${PYTHON_BIN}' '${INVENTORY_SCRIPT}' --run-root '${RUN_ROOT}' >> '${INDEX_LOG}' 2>&1"

cat <<EOF
=== F3: 9-instance nomem+with_mem Watchdog ===
run_root      = ${RUN_ROOT}
run_tag       = ${RUN_TAG}
instances     = 9 (3 django / 3 sympy / 3 astropy)
repeats       = ${REPEATS} x 2 modes = 180 attempts
budget        = nomem: 1200s timeout | with_mem: 1200s Watchdog excluding bridge overhead
framework     = ${V2_FW_REAL}
tmux_main     = ${MAIN_SESSION}
tmux_indexer  = ${INDEX_SESSION}
main_log      = ${MAIN_LOG}

Resume:
  RUN_TAG=${RUN_TAG} bash $0
EOF
