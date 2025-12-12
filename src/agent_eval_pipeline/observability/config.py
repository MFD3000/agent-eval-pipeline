"""
Phoenix/OpenTelemetry Configuration

Loads observability settings from environment variables.
Supports graceful degradation when Phoenix is not installed.
"""

import os
from dataclasses import dataclass, field


@dataclass
class PhoenixConfig:
    """Configuration for Phoenix observability.

    Environment Variables:
        PHOENIX_ENABLED: Enable Phoenix tracing (default: false)
        PHOENIX_PROJECT_NAME: Project name in Phoenix UI (default: agent-eval-pipeline)
        PHOENIX_COLLECTOR_ENDPOINT: Remote endpoint (optional, local if empty)
        PHOENIX_CAPTURE_LLM_CONTENT: Log prompts/responses (default: false)

    PRIVACY WARNING:
        Setting PHOENIX_CAPTURE_LLM_CONTENT=true will export raw prompts and
        responses to Phoenix/OTLP endpoints. For healthcare applications, this
        may include sensitive lab values and patient queries. Only enable in
        controlled environments with appropriate data handling agreements.
    """

    enabled: bool = False
    project_name: str = "agent-eval-pipeline"
    collector_endpoint: str | None = None
    capture_llm_content: bool = False  # Default False for privacy - health data sensitivity

    @classmethod
    def from_env(cls) -> "PhoenixConfig":
        """Load config from environment variables."""
        return cls(
            enabled=os.environ.get("PHOENIX_ENABLED", "false").lower() in ("true", "1", "yes"),
            project_name=os.environ.get("PHOENIX_PROJECT_NAME", "agent-eval-pipeline"),
            collector_endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or None,
            # Default to False for privacy - must explicitly opt-in to capture content
            capture_llm_content=os.environ.get("PHOENIX_CAPTURE_LLM_CONTENT", "false").lower() in ("true", "1", "yes"),
        )


# Global config singleton
_config: PhoenixConfig | None = None


def get_config() -> PhoenixConfig:
    """Get the global Phoenix config (lazy-loaded from env)."""
    global _config
    if _config is None:
        _config = PhoenixConfig.from_env()
    return _config


def reset_config() -> None:
    """Reset config (useful for testing)."""
    global _config
    _config = None
