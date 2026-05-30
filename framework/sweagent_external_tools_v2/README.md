# sweagent_ext_tools

External, pluggable Tool A / Tool B implementation for SWE-agent.

## What lives here
- `bridge_hook.py`: Hook class loaded by SWE-agent plugin loader.
- `dispatcher.py`: CLI entry that routes events to Tool A or Tool B.
- `tools/tool_a.py`: Planning event handler (prints + logs).
- `tools/tool_b.py`: Error event handler (prints + logs).
- `bin/run_tool_a.sh`, `bin/run_tool_b.sh`: Shell entrypoints.

## Runtime contract
SWE-agent passes one environment variable to the external command:
- `SWE_AGENT_EXT_EVENT_JSON`: JSON event payload.

Current events:
- `plan_generated` (Tool A)
- `action_error` (Tool B)

## Quick smoke test
```bash
export SWE_AGENT_EXT_EVENT_JSON='{"event":"plan_generated","action":"ls"}'
bash bin/run_tool_a.sh
```

Log file default:
- `/tmp/sweagent_ext_tools.log`

Override:
```bash
export SWE_AGENT_EXT_TOOLS_LOG_FILE=/tmp/my_ext_tools.log
```
