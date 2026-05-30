"""
Integration modules for Agent-mem system.

Adapters for integrating with SWE-agent and other external systems.
"""

from .sweagent_adapter import SWEAgentAdapter
# from .event_handler import EventHandler  # Not implemented in MVP
# from .hook_integration import HookIntegration  # Not implemented in MVP

__all__ = ["SWEAgentAdapter"]  # , "EventHandler", "HookIntegration"