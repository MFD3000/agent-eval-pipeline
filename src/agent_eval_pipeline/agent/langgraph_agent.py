"""
LangGraph-based LabInsightsAgent

BACKWARD COMPATIBILITY NOTICE:
------------------------------
This module has been ELEVATED following code-elevation principles.
The code has been split into focused modules:
- agent/state.py - AgentState TypedDict
- agent/nodes/ - Individual node functions
- agent/graph.py - Graph construction with DI
- agent/langgraph_runner.py - Entry points

This file re-exports for backward compatibility. New code should import
from the specific modules or from agent_eval_pipeline.agent directly.

WHY LANGGRAPH:
--------------
1. STATE MACHINE: Explicit nodes and edges make the workflow visible
2. CHECKPOINTING: Can pause/resume agent execution
3. HUMAN-IN-THE-LOOP: Easy to add approval steps
4. OBSERVABILITY: Each node is traceable
5. TESTABILITY: Can test individual nodes in isolation


"""

# Re-export everything from the elevated modules for backward compatibility

# State
from agent_eval_pipeline.agent.state import AgentState

# Nodes (for advanced use)
from agent_eval_pipeline.agent.nodes import (
    create_retrieve_node,
    create_analyze_node,
    apply_safety,
)

# Graph
from agent_eval_pipeline.agent.graph import (
    build_agent_graph,
    get_agent_graph,
    clear_graph_cache,
)

# Runner (public API)
from agent_eval_pipeline.agent.langgraph_runner import (
    LangGraphAgentResult,
    LangGraphAgentError,
    run_langgraph_agent,
    run_langgraph_agent_raw,
)

__all__ = [
    # State
    "AgentState",
    # Nodes
    "create_retrieve_node",
    "create_analyze_node",
    "apply_safety",
    # Graph
    "build_agent_graph",
    "get_agent_graph",
    "clear_graph_cache",
    # Runner
    "LangGraphAgentResult",
    "LangGraphAgentError",
    "run_langgraph_agent",
    "run_langgraph_agent_raw",
]
