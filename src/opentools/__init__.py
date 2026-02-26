"""
OpenTools - An effective and easy-to-use agentic framework with extendable tools for complex reasoning.

OpenTools provides a comprehensive framework for building AI agents with powerful tool integration,
enabling complex reasoning and task execution through a unified interface.
"""

from . import core, utils
from .core.base import BaseTool
from .core.registry import ToolRegistry, registry
from .core.config import OpenToolsConfig

# Lazy imports - only import when actually accessed
# This prevents auto-loading all tools/agents when just importing the package
_tools_module = None
_agents_module = None
_solver_module = None

def __getattr__(name):
    """Lazy loader for modules that trigger heavy initialization."""
    global _tools_module, _agents_module, _solver_module
    
    if name == "tools":
        if _tools_module is None:
            from . import tools
            _tools_module = tools
        return _tools_module
    elif name == "agents":
        if _agents_module is None:
            from . import agents
            _agents_module = agents
        return _agents_module
    elif name in ("BaseAgent", "create_agent", "list_agents", "register_agent"):
        # Lazy load agents module and return the requested attribute
        if _agents_module is None:
            from . import agents
            _agents_module = agents
        return getattr(_agents_module, name)
    elif name == "UnifiedSolver":
        # Lazy load solver module and return the requested attribute
        if _solver_module is None:
            from . import solver
            _solver_module = solver
        return getattr(_solver_module, name)
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Convenience functions for tool discovery and management
def list_available_tools():
    """List all available tools in the registry."""
    return registry.list_tools()

def get_tool_info(tool_name: str):
    """Get detailed information about a specific tool."""
    return registry.get_tool_info(tool_name)

def search_tools(query: str):
    """Search for tools by name or description."""
    return registry.search_tools(query)

def get_tools_by_category():
    """Get tools grouped by category."""
    return registry.get_tools_by_category()

def create_tool(tool_name: str, **kwargs):
    """Create an instance of a tool by name."""
    return registry.create_tool_instance(tool_name, **kwargs)

def load_all_tools(verbose: bool = True, **kwargs):
    """Manually trigger loading of all available tools."""
    return registry.load_all_tools(verbose=verbose, **kwargs)

__version__ = "0.0.1"
__author__ = "DM2"
__email__ = "hdang@nd.edu"

__all__ = [
    "core",
    "tools", 
    "utils",
    "BaseTool",
    "ToolRegistry", 
    "OpenToolsConfig",
    "registry",
    # Convenience functions
    "list_available_tools",
    "get_tool_info", 
    "search_tools",
    "get_tools_by_category",
    "create_tool",
    "load_all_tools",
    # Agents
    "BaseAgent",
    "create_agent",
    "list_agents", 
    "register_agent",
    "UnifiedSolver",
]
