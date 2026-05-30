#!/usr/bin/env bash
# Monitor Stage2 multi-agent upgrade experiment.
set -euo pipefail

WS_ROOT="${WS_ROOT:-/home/pt/SWE-bench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZER="${SCRIPT_DIR}/../analyze_multiagent_results.py"
GROUP_NAME="${GROUP_NAME:-stage2_upgrade_t1a_t1c_gated_2case}"
L3_ROOT="${WS_ROOT}/PDDL_work_mem/06_artificial_intelligence/experiments/final_validation/multiagent_l3"
GROUP_DIR="${L3_ROOT}/${GROUP_NAME}"

if [[ ! -d "${GROUP_DIR}" ]]; then
  echo "No run directory yet: ${GROUP_DIR}" >&2
  exit 2
fi

RUN_ROOT="${RUN_ROOT:-$(find "${GROUP_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)}"
echo "RUN_ROOT=${RUN_ROOT}"

if command -v tmux >/dev/null 2>&1; then
  tmux ls 2>/dev/null | grep -E "l3_${GROUP_NAME}|${GROUP_NAME}" || true
fi

python3 "${ANALYZER}" \
  --l3-root "${L3_ROOT}" \
  --group "${GROUP_NAME}" || true

echo
echo "T1 event quick count:"
find "${RUN_ROOT}/same_problem/with_mem" -path "*logs/*.jsonl" -type f -print0 2>/dev/null \
  | xargs -0 grep -h '"event":' 2>/dev/null \
  | grep -E 't1a_reformulation_done|t1a_reformulation_skipped|t1c_precheck_diff_captured|t1c_critic_verdict|t1c_critic_skipped' \
  | sed -E 's/.*"event": "([^"]+)".*/\1/' \
  | sort \
  | uniq -c || true
