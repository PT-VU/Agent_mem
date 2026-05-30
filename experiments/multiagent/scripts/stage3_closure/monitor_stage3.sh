#!/usr/bin/env bash
# Monitor latest Stage3 run.
set -euo pipefail

WS_ROOT="${WS_ROOT:-/home/pt/SWE-bench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}"
ANALYZER="${SCRIPT_DIR}/../analyze_multiagent_results.py"
SUMMARIZER="${SCRIPT_DIR}/../stage2_critic_guard/summarize_validation.py"
L3_ROOT="${WS_ROOT}/PDDL_work_mem/06_artificial_intelligence/experiments/final_validation/multiagent_l3"

if [[ -f "${EXP_DIR}/latest_run.env" ]]; then
  # shellcheck disable=SC1091
  source "${EXP_DIR}/latest_run.env"
else
  GROUP_NAME="${GROUP_NAME:-stage3_closure_sympy5}"
fi

GROUP_NAME="${GROUP_NAME:-stage3_closure_sympy5}"
GROUP_DIR="${L3_ROOT}/${GROUP_NAME}"
if [[ ! -d "${GROUP_DIR}" ]]; then
  echo "No Stage3 group dir yet: ${GROUP_DIR}" >&2
  exit 2
fi

RUN_ROOT="${RUN_ROOT:-$(find "${GROUP_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)}"
echo "GROUP_NAME=${GROUP_NAME}"
echo "RUN_ROOT=${RUN_ROOT}"
echo

python3 "${ANALYZER}" \
  --l3-root "${L3_ROOT}" \
  --group "${GROUP_NAME}" || true

echo
python3 "${SUMMARIZER}" \
  --run-root "${RUN_ROOT}" || true

echo
python3 "${EXP_DIR}/decide_stage3_next.py" \
  --run-root "${RUN_ROOT}" \
  --session "${SESSION_NAME:-}" \
  --write-json "${EXP_DIR}/latest_decision.json" || true
