"""
Configuration manager for Agent-mem system.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ConfigManager:
    """Manages configuration for Agent-mem system."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigManager.

        Args:
            config_path: Path to configuration file. If None, uses defaults.
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if self.config_path and self.config_path.exists():
            return self._load_from_file(self.config_path)
        else:
            return self._get_default_config()

    def _load_from_file(self, filepath: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        suffix = filepath.suffix.lower()

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if suffix == '.json':
                    return json.load(f)
                elif suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {suffix}")
        except Exception as e:
            print(f"Warning: Failed to load config from {filepath}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "system": {
                "name": "Agent-mem",
                "version": "0.1.0",
                "description": "Work-Experience Memory module for SWE-agent",
            },
            "storage": {
                "graph_store_dir": os.getenv("AGENT_MEM_STORAGE_DIR", "./agent_mem_data"),
                "evidence_dir": os.getenv("AGENT_MEM_EVIDENCE_DIR", "./agent_mem_evidence"),
                "persistence_interval": 60,  # seconds
                "auto_save": True,
            },
            "embeddings": {
                "dimension": 384,
                "model": "sentence-transformers",
                "model_name": "all-MiniLM-L6-v2",  # if using sentence-transformers
                "cache_dir": "./embedding_cache",
                "batch_size": 32,
            },
            "retrieval": {
                "max_planning_subgraphs": 3,
                "max_repair_subgraphs": 5,
                "similarity_threshold": 0.7,
                "embedding_view_weights": {
                    "emb_task_sem": 0.4,
                    "emb_file_scope": 0.2,
                    "emb_error_sig": 0.3,
                    "emb_intent": 0.1,
                },
            },
            "processing": {
                "enable_error_handler": True,
                "enable_rca": True,
                "max_suggestions": 3,
            },
            "beliefs": {
                "min_support_for_preference": 10,
                "min_uplift_for_preference": 0.1,
                "min_confidence_for_preference": 0.7,
                "min_support_for_constraint": 50,
                "min_uplift_for_constraint": 0.3,
                "min_confidence_for_constraint": 0.9,
                "forgetting_age_days": 30,
                "forgetting_access_threshold": 5,
            },
            "logging": {
                "level": "INFO",
                "file": os.getenv("AGENT_MEM_LOG_FILE", "./agent_mem.log"),
                "max_size_mb": 10,
                "backup_count": 5,
            },
            "integration": {
                "sweagent": {
                    "tool_a_enabled": True,
                    "tool_b_enabled": True,
                    "timeout_sec": 2.0,
                    "max_response_size": 10000,
                },
                "web": {
                    "enabled": False,
                    "host": "localhost",
                    "port": 8080,
                },
            },
            "performance": {
                "max_concurrent_queries": 10,
                "cache_size": 1000,
                "cache_ttl_seconds": 300,
            },
            "agent_mem": {
                "v21": {
                    "enable_success_fact_hotpath": False,
                    "enable_sidecar": False,
                    "enable_subtask_projection": False,
                    "enable_card_compiler": False,
                    "enable_governance": False,
                    "sidecar_dir": os.getenv("AGENT_MEM_V21_SIDECAR_DIR", "./agent_mem_data/sidecar"),
                    "hotpath_timeout_ms": 50,
                    "coldpath_timeout_ms": 5000,
                    "max_cards_per_query": 4,
                },
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-separated key."""
        keys = key.split('.')
        config = self.config

        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def save(self, filepath: Optional[str] = None) -> bool:
        """Save configuration to file."""
        save_path = Path(filepath) if filepath else self.config_path
        if not save_path:
            return False

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            suffix = save_path.suffix.lower()
            with open(save_path, 'w', encoding='utf-8') as f:
                if suffix == '.json':
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                elif suffix in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False)
                else:
                    # Default to JSON
                    json.dump(self.config, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Error saving config to {save_path}: {e}")
            return False

    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        env_mappings = {
            "AGENT_MEM_STORAGE_DIR": ("storage.graph_store_dir", "str"),
            "AGENT_MEM_EVIDENCE_DIR": ("storage.evidence_dir", "str"),
            "AGENT_MEM_LOG_LEVEL": ("logging.level", "str"),
            "AGENT_MEM_LOG_FILE": ("logging.file", "str"),
            "AGENT_MEM_EMBEDDING_MODEL": ("embeddings.model", "str"),
            "AGENT_MEM_EMBEDDING_MODEL_NAME": ("embeddings.model_name", "str"),
            "AGENT_MEM_EMBEDDING_DIMENSION": ("embeddings.dimension", "int"),
            "AGENT_MEM_MAX_SUBGRAPHS": ("retrieval.max_planning_subgraphs", "int"),
            "AGENT_MEM_TIMEOUT_SEC": ("integration.sweagent.timeout_sec", "float"),
            "AGENT_MEM_V21_ENABLE_SUCCESS_FACT_HOTPATH": ("agent_mem.v21.enable_success_fact_hotpath", "bool"),
            "AGENT_MEM_V21_ENABLE_SIDECAR": ("agent_mem.v21.enable_sidecar", "bool"),
            "AGENT_MEM_V21_ENABLE_SUBTASK_PROJECTION": ("agent_mem.v21.enable_subtask_projection", "bool"),
            "AGENT_MEM_V21_ENABLE_CARD_COMPILER": ("agent_mem.v21.enable_card_compiler", "bool"),
            "AGENT_MEM_V21_ENABLE_GOVERNANCE": ("agent_mem.v21.enable_governance", "bool"),
            "AGENT_MEM_V21_SIDECAR_DIR": ("agent_mem.v21.sidecar_dir", "str"),
            "AGENT_MEM_V21_HOTPATH_TIMEOUT_MS": ("agent_mem.v21.hotpath_timeout_ms", "int"),
            "AGENT_MEM_V21_COLDPATH_TIMEOUT_MS": ("agent_mem.v21.coldpath_timeout_ms", "int"),
            "AGENT_MEM_V21_MAX_CARDS_PER_QUERY": ("agent_mem.v21.max_cards_per_query", "int"),
        }

        for env_var, (config_key, value_type) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if value_type == "int":
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                elif value_type == "float":
                    try:
                        value = float(value)
                    except ValueError:
                        continue
                elif value_type == "bool":
                    value = value.strip().lower() in {"1", "true", "yes", "on"}
                self.set(config_key, value)

    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return validation results."""
        issues = []

        # Check storage directories
        storage_dir = self.get("storage.graph_store_dir")
        if storage_dir:
            try:
                Path(storage_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create storage directory {storage_dir}: {e}")

        if self.get("agent_mem.v21.enable_sidecar", False):
            sidecar_dir = self.get("agent_mem.v21.sidecar_dir")
            if sidecar_dir:
                try:
                    Path(sidecar_dir).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create v2.1 sidecar directory {sidecar_dir}: {e}")

        hotpath_timeout_ms = int(self.get("agent_mem.v21.hotpath_timeout_ms", 0) or 0)
        coldpath_timeout_ms = int(self.get("agent_mem.v21.coldpath_timeout_ms", 0) or 0)
        max_cards_per_query = int(self.get("agent_mem.v21.max_cards_per_query", 0) or 0)
        if hotpath_timeout_ms < 0:
            issues.append(f"agent_mem.v21.hotpath_timeout_ms must be >= 0, got {hotpath_timeout_ms}")
        if coldpath_timeout_ms < 0:
            issues.append(f"agent_mem.v21.coldpath_timeout_ms must be >= 0, got {coldpath_timeout_ms}")
        if max_cards_per_query < 0:
            issues.append(
                f"agent_mem.v21.max_cards_per_query must be >= 0, got {max_cards_per_query}"
            )

        # Check embedding configuration
        embedding_model = self.get("embeddings.model")
        if embedding_model != "sentence-transformers":
            issues.append(
                f"Unsupported embedding model: {embedding_model}. "
                "Only 'sentence-transformers' is allowed."
            )

        # Check retrieval thresholds
        similarity_threshold = self.get("retrieval.similarity_threshold")
        if not (0.0 <= similarity_threshold <= 1.0):
            issues.append(f"Invalid similarity threshold: {similarity_threshold}")

        # Check belief thresholds
        min_support = self.get("beliefs.min_support_for_preference")
        if min_support < 1:
            issues.append(f"min_support_for_preference must be >= 1, got {min_support}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config_summary": {
                "storage_dir": self.get("storage.graph_store_dir"),
                "embedding_model": self.get("embeddings.model"),
                "max_subgraphs": self.get("retrieval.max_planning_subgraphs"),
                "auto_save": self.get("storage.auto_save"),
                "v21_sidecar_enabled": self.get("agent_mem.v21.enable_sidecar", False),
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()

    def get_config(self) -> Dict[str, Any]:
        """Compatibility helper: return full configuration."""
        return self.to_dict()

    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.config, indent=2, ensure_ascii=False)
