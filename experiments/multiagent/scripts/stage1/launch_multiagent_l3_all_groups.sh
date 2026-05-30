#!/usr/bin/env bash
# Launch the six Level-3 comparison groups in separate tmux sessions.
#
# Tier-1 matrix:
#
# Examples:
set -euo pipefail

WS_ROOT="${WS_ROOT:-/home/pt/SWE-bench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP_SCRIPT="${SCRIPT_DIR}/launch_multiagent_l3_group.sh"

# Stagger launches to reduce Docker startup contention.
LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-30}"

# Optional single-group filter.
GROUP_FILTER="${GROUP_FILTER:-}"

# Shared experiment defaults.
export WS_ROOT
export REPEATS="${REPEATS:-10}"
export MODEL_CONFIG="${MODEL_CONFIG:-${WS_ROOT}/SWE-agent/config/kimi25_moonshot.yaml}"
export SKIP_NOMEM="${SKIP_NOMEM:-1}"   # 1 = reuse historical nomem control data

# Each matrix entry is: "GROUP_NAME T1A T1B T1C".
L3_GROUPS=(
  "control  false false false"
  "t1a_only true  false false"
  "t1c_only false false true"
  "t1a_t1c  true  false true"
  "t1b_only false true  false"
  "all_t1   true  true  true"
)

echo "================================================================"
echo "  Multi-agent Level-3 matrix launch: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Instances: 5  repeats: ${REPEATS}  model: kimi-k2.5"
echo "  SKIP_NOMEM: ${SKIP_NOMEM}  launch stagger: ${LAUNCH_STAGGER_SEC}s"
echo "================================================================"

LAUNCHED=0
for entry in "${L3_GROUPS[@]}"; do
  read -r gname t1a t1b t1c <<< "${entry}"

  # Apply the optional single-group filter.
  if [[ -n "${GROUP_FILTER}" && "${gname}" != "${GROUP_FILTER}" ]]; then
    echo "  [skip] ${gname} (GROUP_FILTER=${GROUP_FILTER})"
    continue
  fi

  echo ""
  echo "  >> launch [${gname}]  T1A=${t1a} T1B=${t1b} T1C=${t1c}"

  GROUP_NAME="${gname}" \
  T1A_ENABLED="${t1a}" \
  T1B_ENABLED="${t1b}" \
  T1C_ENABLED="${t1c}" \
    bash "${GROUP_SCRIPT}"

  LAUNCHED=$((LAUNCHED + 1))

  # Avoid launching every Docker-heavy group at once.
  if [[ "${LAUNCHED}" -lt "${#L3_GROUPS[@]}" && -z "${GROUP_FILTER}" ]]; then
    echo "  [waiting ${LAUNCH_STAGGER_SEC}s before next launch...]"
    sleep "${LAUNCH_STAGGER_SEC}"
  fi
done

echo ""
echo "================================================================"
echo "  launched groups: ${LAUNCHED}"
echo ""
echo "  tmux monitor:"
echo "    watch -n 60 'tmux ls | grep l3_'"
echo ""
echo "  trial inventory monitor:"
cat <<'MONITOR'

L3_ROOT="/home/pt/SWE-bench/PDDL_work_mem/06_artificial_intelligence/experiments/final_validation/multiagent_l3"
for group in control t1a_only t1c_only t1a_t1c t1b_only all_t1; do
  inv=$(ls -t "${L3_ROOT}/${group}"/*/orchestrator_state/trial_inventory.json 2>/dev/null | head -1)
  if [[ -z "$inv" ]]; then echo "[${group}] no inventory"; continue; fi
  python3 -c "
import json
d = json.load(open('${inv}'))
total = d['trial_count']
done = sum(1 for r in d['records'] if r['status'] not in ('in_progress_or_unfinished','pending'))
resolved = sum(1 for r in d['records'] if r['report'].get('official_eval_status')=='resolved')
print(f'[${group}] {done}/{total} done  resolved={resolved}')
" 2>/dev/null || echo "[${group}] parse error"
done

MONITOR
echo "================================================================"
