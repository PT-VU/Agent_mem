#!/usr/bin/env bash
# Launch Stage2 multi-agent upgrade experiment.
set -euo pipefail

WS_ROOT="${WS_ROOT:-/home/pt/SWE-bench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP_SCRIPT="${SCRIPT_DIR}/../stage1/launch_multiagent_l3_group.sh"
EXP_DIR="${SCRIPT_DIR}"
GROUP_NAME="${GROUP_NAME:-stage2_upgrade_t1a_t1c_gated_2case}"
RUN_TAG="${RUN_TAG:-stage2_upgrade_2case_kimi_$(date +%Y%m%d_%H%M%S)}"
L3_ROOT="${WS_ROOT}/PDDL_work_mem/06_artificial_intelligence/experiments/final_validation/multiagent_l3"
RUN_ROOT="${L3_ROOT}/${GROUP_NAME}/${RUN_TAG}"

mkdir -p "${RUN_ROOT}"
cp "${EXP_DIR}/same_instances.json" "${RUN_ROOT}/same_instances.json"

GROUP_NAME="${GROUP_NAME}" \
RUN_TAG="${RUN_TAG}" \
T1A_ENABLED=true \
T1B_ENABLED=false \
T1C_ENABLED=true \
REPEATS=10 \
MAX_ACTIVE_RUN_BATCHES="${MAX_ACTIVE_RUN_BATCHES:-1}" \
GLOBAL_HEAVY_SLOTS="${GLOBAL_HEAVY_SLOTS:-1}" \
WITHMEM_HEAVY_SLOTS="${WITHMEM_HEAVY_SLOTS:-1}" \
SKIP_NOMEM="${SKIP_NOMEM:-1}" \
SWE_AGENT_T1A_MAX_REFORMATS_PER_ATTEMPT="${SWE_AGENT_T1A_MAX_REFORMATS_PER_ATTEMPT:-3}" \
SWE_AGENT_T1C_USE_PRECHECK_DIFF="${SWE_AGENT_T1C_USE_PRECHECK_DIFF:-1}" \
bash "${GROUP_SCRIPT}"

echo "RUN_ROOT=${RUN_ROOT}"
