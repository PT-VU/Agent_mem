"""
Configuration management for Agent-mem system.
"""

from .config_manager import ConfigManager

# Default configuration is available via ConfigManager().config
DEFAULT_CONFIG = None  # Placeholder

__all__ = ["ConfigManager", "DEFAULT_CONFIG"]