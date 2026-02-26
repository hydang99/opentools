"""
Tools module for OpenTools framework.

This module contains various tools that can be used with the OpenTools framework.
Tools are automatically discovered and registered with the global registry.
"""
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from opentools.core.registry import registry

# Auto-discover and register all available tools
def _auto_load_tools():
    """Automatically discover and load all tools."""
    # Only load if not already loaded to avoid double loading
    if not registry.is_module_discovered("opentools.tools"):
        return registry.load_all_tools(verbose=False)
    return registry.list_tools()

# Load all tools when the module is imported
_loaded_tools = _auto_load_tools()

# Import specific tools for direct access
__all__ = _loaded_tools
# Also expose the registry for advanced usage
__all__.append("registry")
