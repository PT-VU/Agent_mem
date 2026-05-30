# Agent-mem: Work-Experience Memory for SWE-agent

Agent-mem is a training-free memory system that provides retrievable, structured, and verifiable work experience for long-horizon software engineering tasks. It integrates with SWE-agent to enhance planning and execution through learned experience.

## Features

- **Structured Experience Storage**: Stores actions as ProblemFiles with evidence pointers
- **Knowledge Graph**: Organizes experience in observation KG with success/failure edges
- **Belief System**: Maintains evolvable beliefs with statistical promotion/demotion
- **Multi-view Embeddings**: Supports hierarchical retrieval from different perspectives
- **Planning-time Injection**: Injects experience during planning phase
- **Execution-time Repair**: Provides repair suggestions during errors
- **Evidence-backed Recommendations**: All suggestions include traceable evidence

## Architecture

Agent-mem follows the architecture described in `Agent-mem .md` with these core components:

1. **ProblemFile**: Action-level experience atom with evidence pointers
2. **ObservationKG**: Fact graph with success_next and fail_retry edges
3. **BeliefGraph**: Evolvable belief system with statistical backing
4. **MemoryAgent**: Coordinates retrieval, distillation, and evidence referencing
5. **SWEAgentAdapter**: Integration layer with SWE-agent external tools

## Installation

```bash
# Clone the repository
cd /home/pt/SWE-bench/PDDL_work_mem/tools/sweagent_ext_tools

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Configuration

Create a configuration file (`config.yaml` or `config.json`):

```yaml
system:
  name: "Agent-mem"
  version: "0.1.0"

storage:
  graph_store_dir: "./agent_mem_data"
  evidence_dir: "./agent_mem_evidence"
  auto_save: true

embeddings:
  dimension: 384
  model: "random"  # or "sentence-transformers"

retrieval:
  max_planning_subgraphs: 3
  max_repair_subgraphs: 5
  similarity_threshold: 0.7
```

Or use environment variables:
```bash
export AGENT_MEM_STORAGE_DIR="./agent_mem_data"
export AGENT_MEM_EVIDENCE_DIR="./agent_mem_evidence"
export AGENT_MEM_LOG_LEVEL="INFO"
```

## Integration with SWE-agent

Agent-mem replaces the existing Tool A/Tool B in `sweagent_ext_tools`:

### 1. Update SWE-agent configuration

In your SWE-agent configuration, set the external tool commands:

```yaml
external_tools:
  tool_a_cmd: "python -m agent_mem_main --tool A"
  tool_b_cmd: "python -m agent_mem_main --tool B"
  timeout_sec: 5.0
```

### 2. Or use environment variables:

```bash
export SWE_AGENT_EXT_TOOL_A_CMD="python -m agent_mem_main --tool A"
export SWE_AGENT_EXT_TOOL_B_CMD="python -m agent_mem_main --tool B"
export SWE_AGENT_EXT_TOOL_TIMEOUT_SEC=5.0
```

### 3. Enable the bridge hook

Make sure the `ExternalToolBridgeHook` is enabled in your SWE-agent run configuration.

## Usage

### Basic Usage

```python
from agent_mem import SWEAgentAdapter, ConfigManager

# Create adapter
config_manager = ConfigManager("config.yaml")
adapter = SWEAgentAdapter(
    storage_dir=config_manager.get("storage.graph_store_dir"),
    evidence_dir=config_manager.get("storage.evidence_dir")
)

# Handle planning event
planning_response = adapter.handle_plan_generated({
    "agent": "swe-agent",
    "thought": "I need to fix the bug in calculate()",
    "action": "First, I'll examine the calculate function"
})

# Handle error event
error_response = adapter.handle_action_error({
    "agent": "swe-agent",
    "error_type": "SyntaxError",
    "error_message": "invalid syntax on line 42",
    "thought": "Trying to fix syntax",
    "action": "edit file.py line 42"
})

# End task
task_report = adapter.end_task(success=True, summary="Fixed calculate() bug")
```

### Command Line Interface

```bash
# Process planning event (Tool A)
export SWE_AGENT_EXT_EVENT_JSON='{"event":"plan_generated","agent":"swe-agent","thought":"...","action":"..."}'
python -m agent_mem_main --tool A --config config.yaml

# Process error event (Tool B)
export SWE_AGENT_EXT_EVENT_JSON='{"event":"action_error","agent":"swe-agent","error_type":"...","error_message":"..."}'
python -m agent_mem_main --tool B --config config.yaml

# Show statistics
python -m agent_mem_main --stats

# Export data
python -m agent_mem_main --export ./export_data
```

### API Reference

#### SWEAgentAdapter

- `handle_plan_generated(event_data)`: Process planning events
- `handle_action_error(event_data)`: Process error events
- `handle_action_success(...)`: Log successful actions
- `end_task(success, summary)`: Complete a task
- `get_task_statistics(task_id)`: Get task statistics
- `export_data(output_dir)`: Export all data

#### MemoryAgent

- `retrieve_for_planning(task_context, current_action, agent_name)`: Planning-time retrieval
- `retrieve_for_repair(error_type, error_message, current_action, problem_file)`: Execution-time repair
- `update_beliefs(task_id, outcome, evidence_refs)`: Update beliefs
- `get_memory_statistics()`: Get memory statistics

## Testing

Run the test suite:

```bash
cd /home/pt/SWE-bench/PDDL_work_mem/tools/sweagent_ext_tools
python -m pytest agent_mem/tests/test_core.py -v
```

## MVP Features Implemented

1.  **Problem File + evidence pointers**: Structured action storage with evidence
2.  **Multi-view embedding system**: Placeholder embeddings for hierarchical retrieval
3.  **Fact graph with two edge types**: success_next and fail_retry edges
4.  **Hierarchical retrieval**: Task-level and action-level retrieval
5.  **Execution-time fail_retry **: Repair suggestions with evidence
6.  **Belief statistics**: Support/uplift tracking with promotion rules
7.  **SWE-agent integration**: Full replacement of Tool A/Tool B

## Next Steps (Phase 2)

1. **Real embedding models**: Replace random embeddings with sentence-transformers
2. **Asynchronous RCA**: Implement root cause analysis agent
3. **PDDL3 constraints**: Add preference/constraint compilation
4. **GUI support**: Multi-modal embeddings for UI states
5. **Production storage**: Neo4j or other graph database backend
6. **Performance optimization**: Caching, batching, and parallel processing

## Project Structure

```
agent_mem/
 core/                    # Core data structures
    problem_file.py     # ProblemFile with evidence pointers
    observation_kg.py   # Observation knowledge graph
    belief_graph.py     # Belief graph system
 storage/                # Storage implementations
    graph_store.py      # Graph storage with NetworkX
 processing/             # Processing modules
    action_logger.py    # Action logging with evidence
    kg_writer.py        # KG writing with embeddings
 retrieval/              # Retrieval modules
    memory_agent.py     # Memory agent service
 integration/            # Integration adapters
    sweagent_adapter.py # SWE-agent integration
 config/                 # Configuration management
    config_manager.py   # Configuration manager
 tests/                  # Test suite
    test_core.py        # Core component tests
 __init__.py            # Package exports
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use Agent-mem in your research, please cite:

```bibtex
@software{agent_mem2025,
  title = {Agent-mem: Work-Experience Memory for SWE-agent},
  author = {Agent-mem Team},
  year = {2025},
  url = {https://github.com/your-org/agent-mem}
}
```