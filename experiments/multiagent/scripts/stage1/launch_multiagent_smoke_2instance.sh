#!/usr/bin/env bash
# Launch a two-instance Level-2 multi-agent smoke test.
# Instances: astropy__astropy-12057 and sympy__sympy-13551.
# Enable T1-A, T1-B, and T1-C together with Kimi.
#
# Smoke criteria:
set -euo pipefail

WS_ROOT="${WS_ROOT:-/home/pt/SWE-bench}"
ARTIFACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# Public artifact multi-agent framework.
MA_FW_REAL="${ARTIFACT_ROOT}/framework/sweagent_external_tools_multiagent"
MA_FW_LINK="${MA_FW_REAL}"

if [[ ! -d "${MA_FW_REAL}" ]]; then
  echo "multiagent framework not found: ${MA_FW_REAL}" >&2
  exit 2
fi
# MA_FW_LINK remains an alias for compatibility with the original launcher.

# Runtime output root.
RUN_TAG="${RUN_TAG:-multiagent_smoke_kimi_$(date +%Y%m%d_%H%M%S)}"
RUNS_ROOT="${RUNS_ROOT:-${WS_ROOT}/PDDL_work_mem/06_artificial_intelligence/experiments/final_validation/multiagent_smoke}"
RUN_ROOT="${RUNS_ROOT}/${RUN_TAG}"
MODEL_CONFIG="${MODEL_CONFIG:-${WS_ROOT}/SWE-agent/config/kimi25_moonshot.yaml}"
PROMPT_PROFILE="${PROMPT_PROFILE:-prompt_base}"
REPEATS="${REPEATS:-3}"
SWEBENCH_SUBSET="${SWEBENCH_SUBSET:-full}"
SWEBENCH_SPLIT="${SWEBENCH_SPLIT:-test}"

mkdir -p "${RUN_ROOT}/orchestrator_logs" "${RUN_ROOT}/orchestrator_state" "${RUN_ROOT}/notes" \
         "${RUN_ROOT}/agent_mem_logs" "${RUN_ROOT}/t1b_interim_cache"

# Write the fixed instance list once.
INSTANCE_LIST_JSON="${RUN_ROOT}/same_instances.json"
if [[ ! -f "${INSTANCE_LIST_JSON}" ]]; then
  cat > "${INSTANCE_LIST_JSON}" <<JSON
{
  "generated_at": "$(date -Iseconds)",
  "subset": "full",
  "split": "test",
  "selection_mode": "manual_multiagent_smoke",
  "count_requested": 2,
  "count_selected": 2,
  "total_available": 2294,
  "repeats": ${REPEATS},
  "instance_ids": [
    "astropy__astropy-12057",
    "sympy__sympy-13551"
  ]
}
JSON
fi

# Write run_manifest.json once.
MANIFEST="${RUN_ROOT}/orchestrator_state/run_manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  cat > "${MANIFEST}" <<JSON
{
  "generated_at": "$(date -Iseconds)",
  "experiment_type": "multiagent_smoke_kimi",
  "level": "Level-2",
  "run_root": "${RUN_ROOT}",
  "same_instances_json": "${INSTANCE_LIST_JSON}",
  "model_config": "${MODEL_CONFIG}",
  "prompt_profile": "${PROMPT_PROFILE}",
  "repeats": ${REPEATS},
  "groups": ["nomem", "with_mem"],
  "framework_version": "sweagent_external_tools_multiagent",
  "framework_path": "${MA_FW_REAL}",
  "framework_symlink": "${MA_FW_LINK}",
  "t1_flags": {"T1A": true, "T1B": true, "T1C": true},
  "smoke_criteria": [
    "zero crashes / uncaught exceptions",
    "hook_events.jsonl contains t1a_reformulation_done events",
    "hook_events.jsonl contains t1c_critic_verdict events",
    "t1b_interim_cache/ contains per-instance .json files"
  ]
}
JSON
fi

# Load runtime credentials from the external SWE-agent environment.
SWE_AGENT_ENV_FILE="${WS_ROOT}/SWE-agent/.env"
if [[ -f "${SWE_AGENT_ENV_FILE}" ]]; then
  set -a; source "${SWE_AGENT_ENV_FILE}"; set +a
fi
if [[ -z "${MOONSHOT_API_KEY:-}" ]]; then
  echo "MOONSHOT_API_KEY required (check ${SWE_AGENT_ENV_FILE})" >&2; exit 2
fi

# Agent hook configuration.
export START_WITH_TOOLS="${MA_FW_LINK}/bin/start_with_external_tools.sh"
export V2_FW_DIR="${MA_FW_LINK}"
export SWE_AGENT_EXTERNAL_HOOK_PYTHONPATH="${ARTIFACT_ROOT}/framework"
export SWE_AGENT_EXTERNAL_HOOK_CLASS="sweagent_external_tools_multiagent.bridge_hook:ExternalToolBridgeHook"
export SWE_AGENT_EXT_TOOL_A_CMD="${MA_FW_LINK}/bin/run_tool_a.sh"
export SWE_AGENT_EXT_TOOL_B_CMD="${MA_FW_LINK}/bin/run_tool_b.sh"

# Hook event log.
export SWE_AGENT_EXT_TOOLS_LOG_FILE="${RUN_ROOT}/agent_mem_logs/hook_events.jsonl"

# Enable the three Tier-1 modules.
export SWE_AGENT_T1A_ENABLED="true"
export SWE_AGENT_T1B_ENABLED="true"
export SWE_AGENT_T1C_ENABLED="true"

# Tier-1 auxiliary model. The client appends the chat-completions endpoint.
export AGENT_MEM_LLM_MODEL="${AGENT_MEM_LLM_MODEL:-kimi-k2.5}"
export AGENT_MEM_LLM_BASE_URL="${AGENT_MEM_LLM_BASE_URL:-https://api.moonshot.cn/v1}"
export AGENT_MEM_LLM_API_KEY="${AGENT_MEM_LLM_API_KEY:-${MOONSHOT_API_KEY}}"

# T1-A reformulation timeout.
export SWE_AGENT_T1A_TIMEOUT_SEC="${SWE_AGENT_T1A_TIMEOUT_SEC:-10.0}"

# T1-B interim-memory cache.
export SWE_AGENT_T1B_CACHE_DIR="${RUN_ROOT}/t1b_interim_cache"
# Write an interim card after this localization threshold.
export SWE_AGENT_T1B_LOCALIZE_THRESHOLD="${SWE_AGENT_T1B_LOCALIZE_THRESHOLD:-3}"

# T1-C critic configuration.
export SWE_AGENT_T1C_TIMEOUT_SEC="${SWE_AGENT_T1C_TIMEOUT_SEC:-12.0}"
export SWE_AGENT_T1C_REVISE_THRESHOLD="${SWE_AGENT_T1C_REVISE_THRESHOLD:-0.4}"
export SWE_AGENT_T1C_REJECT_THRESHOLD="${SWE_AGENT_T1C_REJECT_THRESHOLD:-0.8}"

# Base Agent-mem v2 configuration.
export AGENT_MEM_BUG_INVARIANT_VERBATIM="${AGENT_MEM_BUG_INVARIANT_VERBATIM:-1}"
export AGENT_MEM_BUG_ANTI_PATTERN="${AGENT_MEM_BUG_ANTI_PATTERN:-1}"
export AGENT_MEM_PATCH_CONSISTENCY_GATE="${AGENT_MEM_PATCH_CONSISTENCY_GATE:-enforce}"
export AGENT_MEM_REUSE_EXPLORE="${AGENT_MEM_REUSE_EXPLORE:-auto}"
export AGENT_MEM_LOCAL_EFFECTIVE_FEEDBACK="${AGENT_MEM_LOCAL_EFFECTIVE_FEEDBACK:-1}"
export AGENT_MEM_L3_FORCE_SUBMIT="${AGENT_MEM_L3_FORCE_SUBMIT:-dry_run}"
export AGENT_MEM_FORCE_STRATEGY="${AGENT_MEM_FORCE_STRATEGY:-auto}"

# Resource and timeout limits.
export GLOBAL_HEAVY_SLOTS="${GLOBAL_HEAVY_SLOTS:-2}"
export NOMEM_HEAVY_SLOTS="${NOMEM_HEAVY_SLOTS:-1}"
export WITHMEM_HEAVY_SLOTS="${WITHMEM_HEAVY_SLOTS:-1}"
export MAX_ACTIVE_RUN_BATCHES="${MAX_ACTIVE_RUN_BATCHES:-2}"
export PER_INSTANCE_CALL_LIMIT="${PER_INSTANCE_CALL_LIMIT:-0}"
export NOMEM_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${NOMEM_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC:-1200}"
export WITHMEM_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC="${WITHMEM_INSTANCE_TOTAL_EXEC_TIMEOUT_SEC:-1200}"
export MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-1}"
export EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-1200}"
export ENV_ERROR_RETRIES="${ENV_ERROR_RETRIES:-1}"
export IMAGE_PULL_TIMEOUT_SEC="${IMAGE_PULL_TIMEOUT_SEC:-1800}"
export PHASE7_PREPARE_SLOTS="${PHASE7_PREPARE_SLOTS:-2}"
export PHASE7_PREPARE_SLOT_POLL_SEC="${PHASE7_PREPARE_SLOT_POLL_SEC:-10}"
export NOMEM_RUNTIME_PREWARM="${NOMEM_RUNTIME_PREWARM:-0}"
export WITHMEM_RUNTIME_PREWARM="${WITHMEM_RUNTIME_PREWARM:-1}"
export NOMEM_RUNTIME_WARMUP_TIMEOUT_SEC="${NOMEM_RUNTIME_WARMUP_TIMEOUT_SEC:-1800}"
export WITHMEM_RUNTIME_WARMUP_TIMEOUT_SEC="${WITHMEM_RUNTIME_WARMUP_TIMEOUT_SEC:-1800}"
export NOMEM_PYTHON_STANDALONE_DIR="${NOMEM_PYTHON_STANDALONE_DIR:-/root}"
export WITHMEM_PYTHON_STANDALONE_DIR="${WITHMEM_PYTHON_STANDALONE_DIR:-/root}"
export SKIP_INSTANCE_PREPARE="${SKIP_INSTANCE_PREPARE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# Resume inputs.
export RUN_ROOT RUN_TAG MODEL_CONFIG PROMPT_PROFILE REPEATS SWEBENCH_SUBSET SWEBENCH_SPLIT
export SAME_JSON="${INSTANCE_LIST_JSON}"

# Persist the tmux environment without credentials.
MAIN_ENV_FILE="${RUN_ROOT}/orchestrator_state/main_tmux.env"
{
  echo "RUN_ROOT='${RUN_ROOT}'"
  echo "RUN_TAG='${RUN_TAG}'"
  echo "MODEL_CONFIG='${MODEL_CONFIG}'"
  echo "REPEATS='${REPEATS}'"
  echo "SWEBENCH_SUBSET='${SWEBENCH_SUBSET}'"
  echo "SWEBENCH_SPLIT='${SWEBENCH_SPLIT}'"
  for v in SWE_AGENT_EXTERNAL_HOOK_CLASS SWE_AGENT_EXTERNAL_HOOK_PYTHONPATH \
           SWE_AGENT_EXT_TOOL_A_CMD SWE_AGENT_EXT_TOOL_B_CMD START_WITH_TOOLS V2_FW_DIR \
           SWE_AGENT_EXT_TOOLS_LOG_FILE \
           SWE_AGENT_T1A_ENABLED SWE_AGENT_T1B_ENABLED SWE_AGENT_T1C_ENABLED \
           AGENT_MEM_LLM_MODEL AGENT_MEM_LLM_BASE_URL \
           SWE_AGENT_T1A_TIMEOUT_SEC SWE_AGENT_T1B_CACHE_DIR SWE_AGENT_T1B_LOCALIZE_THRESHOLD \
           SWE_AGENT_T1C_TIMEOUT_SEC SWE_AGENT_T1C_REVISE_THRESHOLD SWE_AGENT_T1C_REJECT_THRESHOLD \
           AGENT_MEM_BUG_INVARIANT_VERBATIM AGENT_MEM_BUG_ANTI_PATTERN \
           AGENT_MEM_PATCH_CONSISTENCY_GATE AGENT_MEM_REUSE_EXPLORE \
           AGENT_MEM_LOCAL_EFFECTIVE_FEEDBACK AGENT_MEM_L3_FORCE_SUBMIT AGENT_MEM_FORCE_STRATEGY \
           GLOBAL_HEAVY_SLOTS NOMEM_HEAVY_SLOTS WITHMEM_HEAVY_SLOTS \
           HF_HUB_OFFLINE HF_DATASETS_OFFLINE; do
    echo "${v}='${!v:-}'"
  done
} > "${MAIN_ENV_FILE}"

# Launch the orchestrator and inventory indexer in tmux.
RESUME_SCRIPT="${ARTIFACT_ROOT}/experiments/shared/resume_same_only.sh"
INVENTORY_SCRIPT="${ARTIFACT_ROOT}/experiments/shared/collect_trial_inventory.py"
PYTHON_BIN="${WS_ROOT}/SWE-agent/.venv/bin/python"

SANITIZED_TAG="${RUN_TAG//[^A-Za-z0-9_]/_}"
MAIN_SESSION="${MAIN_SESSION:-${SANITIZED_TAG}}"
INDEX_SESSION="${INDEX_SESSION:-${SANITIZED_TAG}_idx}"
MAIN_LOG="${RUN_ROOT}/orchestrator_logs/same_only_resume.tmux.log"
INDEX_LOG="${RUN_ROOT}/orchestrator_logs/trial_indexer.tmux.log"

if tmux has-session -t "${MAIN_SESSION}" 2>/dev/null; then
  echo "tmux session already running: ${MAIN_SESSION}" >&2; exit 3
fi

tmux new-session -d -s "${MAIN_SESSION}" \
  "set -a; source '${SWE_AGENT_ENV_FILE}'; source '${MAIN_ENV_FILE}'; set +a; '${RESUME_SCRIPT}' >> '${MAIN_LOG}' 2>&1"

tmux new-session -d -s "${INDEX_SESSION}" \
  "while tmux has-session -t '${MAIN_SESSION}' 2>/dev/null; do '${PYTHON_BIN}' '${INVENTORY_SCRIPT}' --run-root '${RUN_ROOT}' >> '${INDEX_LOG}' 2>&1; sleep 30; done; '${PYTHON_BIN}' '${INVENTORY_SCRIPT}' --run-root '${RUN_ROOT}' >> '${INDEX_LOG}' 2>&1"

cat <<EOF
=== Multi-agent Level-2 smoke launch ===
run_root        = ${RUN_ROOT}
same_json       = ${INSTANCE_LIST_JSON}
framework       = ${MA_FW_REAL}
hook_class      = ${SWE_AGENT_EXTERNAL_HOOK_CLASS}
hook_pythonpath = ${SWE_AGENT_EXTERNAL_HOOK_PYTHONPATH}
t1_flags        = T1A=${SWE_AGENT_T1A_ENABLED} T1B=${SWE_AGENT_T1B_ENABLED} T1C=${SWE_AGENT_T1C_ENABLED}
t1_model        = ${AGENT_MEM_LLM_MODEL} @ ${AGENT_MEM_LLM_BASE_URL}
t1b_cache_dir   = ${SWE_AGENT_T1B_CACHE_DIR}
repeats         = ${REPEATS}
instances       = astropy__astropy-12057, sympy__sympy-13551
v2_flags        = INVARIANT=${AGENT_MEM_BUG_INVARIANT_VERBATIM} ANTI=${AGENT_MEM_BUG_ANTI_PATTERN} GATE=${AGENT_MEM_PATCH_CONSISTENCY_GATE} REUSE=${AGENT_MEM_REUSE_EXPLORE}
main_session    = ${MAIN_SESSION}
index_session   = ${INDEX_SESSION}
main_log        = ${MAIN_LOG}

Smoke checks:
  1. Confirm that the tmux session exits without a non-zero status.
  2. grep t1a_reformulation_done "${SWE_AGENT_EXT_TOOLS_LOG_FILE}" | wc -l  (expected > 0)
  3. grep t1c_critic_verdict "${SWE_AGENT_EXT_TOOLS_LOG_FILE}" | wc -l       (expected > 0)
  4. ls "${SWE_AGENT_T1B_CACHE_DIR}/"*.json 2>/dev/null | wc -l              (expected > 0)
EOF
