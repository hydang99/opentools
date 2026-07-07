"""Optional integrations for external agent frameworks."""

from .dspy import (
    as_callable,
    as_dspy_tool,
    as_dspy_tools,
    build_dspy_agent,
    optimize_dspy_agent,
)

__all__ = [
    "as_callable",
    "as_dspy_tool",
    "as_dspy_tools",
    "build_dspy_agent",
    "optimize_dspy_agent",
]
