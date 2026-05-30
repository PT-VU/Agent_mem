#!/usr/bin/env bash
set -euo pipefail
WS_ROOT="${WS_ROOT:-/home/pt/SWE-bench}"
SWE_AGENT_PYTHON="${SWE_AGENT_PYTHON:-${WS_ROOT}/SWE-agent/.venv/bin/python}"
PACKAGE_PARENT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PACKAGE_PARENT}:${PYTHONPATH:-}"
if [[ ! -x "${SWE_AGENT_PYTHON}" ]]; then
  SWE_AGENT_PYTHON="python3"
fi
"${SWE_AGENT_PYTHON}" -m sweagent_external_tools_multiagent.dispatcher --tool B
