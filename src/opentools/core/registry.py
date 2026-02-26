"""
Tool registry for managing and discovering tools in the OpenTools framework.
"""
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from .base import BaseTool
import sys

# Ensure UTF-8 encoding for stdout when possible
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass  # reconfigure not available or failed

class ToolRegistry:
    """Registry for managing tools in the OpenTools framework."""
    
    def __init__(self):
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._tool_instances: Dict[str, BaseTool] = {}
        self._discovered_modules: List[str] = []
    
    def register(self, tool_class: Type[BaseTool]) -> Type[BaseTool]:
        """Register a tool class."""
        if not issubclass(tool_class, BaseTool):
            raise ValueError(f"Tool class must inherit from BaseTool: {tool_class}")
        
        tool_name = tool_class.__name__
        self._tools[tool_name] = tool_class
        return tool_class
    
    def register_instance(self, tool_instance: BaseTool) -> None:
        """Register a tool instance."""
        if not isinstance(tool_instance, BaseTool):
            raise ValueError(f"Tool instance must be an instance of BaseTool: {tool_instance}")
        
        tool_name = tool_instance.tool_name
        self._tool_instances[tool_name] = tool_instance
    
    def get_tool(self, tool_name: str) -> Optional[Type[BaseTool]]:
        """Get a tool class by name."""
        return self._tools.get(tool_name)
    
    def get_tool_instance(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool instance by name."""
        return self._tool_instances.get(tool_name)
    
    def create_tool_instance(self, tool_name: str, **kwargs) -> Optional[BaseTool]:
        """Create a new instance of a tool."""
        tool_class = self.get_tool(tool_name)
        if tool_class is None:
            return None
        
        return tool_class(**kwargs)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def list_tool_instances(self) -> List[str]:
        """List all registered tool instance names."""
        return list(self._tool_instances.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a tool."""
        tool_class = self.get_tool(tool_name)
        if tool_class is None:
            return None
        
        # Create a temporary instance to get tool metadata
        try:
            temp_instance = tool_class()
            meta = getattr(temp_instance, "get_metadata", lambda: {})()
            return {
                "name": tool_name,
                "description": getattr(temp_instance, "description", meta.get("description", "N/A")),
                "version": getattr(temp_instance, "tool_version", "1.0.0"),
                "input_types": meta.get("parameters", {}),
                "output_type": "varies",
                "demo_commands": meta.get("demo_commands", []),
                "require_llm_engine": getattr(temp_instance, "require_llm_engine", False),
            }
        except Exception as e:
            print(e)
            return {
                "name": tool_name,
                "description": "Tool information unavailable",
            }
    
    def discover_tools(self, module_path: str) -> List[str]:
        """Discover and register tools from a module."""
        try:
            module = __import__(module_path, fromlist=['*'])
            discovered_tools = []
            
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseTool) and 
                    obj != BaseTool):
                    self.register(obj)
                    discovered_tools.append(name)
            
            return discovered_tools
        except ImportError:
            return []
    
    def auto_discover_tools(self, tools_dir: str = "opentools.tools", verbose: bool = False) -> List[str]:
        """Automatically discover and register all tools in the tools directory."""
        # Check if already discovered
        if tools_dir in self._discovered_modules:
            return self.list_tools()
        
        discovered_tools = []
        
        try:
            # Get the tools module
            tools_module = __import__(tools_dir, fromlist=['*'])
            tools_path = Path(tools_module.__file__).parent
            
            # Discover tools in subdirectories
            for item in tools_path.iterdir():
                if item.is_dir() and not item.name.startswith('_'):
                    # Try to import the tool module
                    tool_module_path = f"{tools_dir}.{item.name}"
                    try:
                        tool_module = __import__(tool_module_path, fromlist=['*'])
                        
                        # Look for tool classes in the module
                        for name, obj in inspect.getmembers(tool_module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, BaseTool) and 
                                obj != BaseTool):
                                # Only register if not already registered
                                if name not in self._tools:
                                    self.register(obj)
                                    discovered_tools.append(name)
                                    if verbose:
                                        print(f"✅ Discovered and registered: {name}")
                        
                    except ImportError as e:
                        if verbose:
                            print(f"⚠️  Could not import {tool_module_path}: {e}")
                    except Exception as e:
                        if verbose:
                            print(f"❌ Error processing {tool_module_path}: {e}")
            
            self._discovered_modules.append(tools_dir)
            return discovered_tools
            
        except ImportError as e:
            if verbose:
                print(f"❌ Could not import tools module {tools_dir}: {e}")
            return []
    
    def load_all_tools(self, tools_dir: str = "opentools.tools", verbose: bool = False) -> List[str]:
        """Load all available tools automatically."""
        # Check if tools from this directory have already been loaded
        if tools_dir in self._discovered_modules:
            if verbose:
                print(f"📦 Tools from {tools_dir} already loaded ({len(self.list_tools())} tools)")
            return self.list_tools()
        
        if verbose:
            print("🔍 Auto-discovering tools...")
        discovered = self.auto_discover_tools(tools_dir, verbose=verbose)
        if verbose:
            print(f"📦 Loaded {len(discovered)} tools: {discovered}")
        return discovered
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available tools."""
        tools_info = {}
        for tool_name in self.list_tools():
            info = self.get_tool_info(tool_name)
            if info:
                tools_info[tool_name] = info
        return tools_info
    
    def search_tools(self, query: str) -> List[str]:
        """Search for tools by name (case-insensitive).
        
        Args:
            query: Search query string. If any word in the query appears in a tool name, 
                   that tool will be included in results.
        
        Returns:
            List of tool names that match the query.
        """
        if not query or not query.strip():
            return []
        
        query_lower = query.lower().strip()
        query_words = query_lower.split()
        
        matching_tools = []
        all_tools = self.list_tools()
        
        for tool_name in all_tools:
            tool_name_lower = tool_name.lower()
            # Check if any word in the query appears in the tool name
            if any(word in tool_name_lower for word in query_words):
                matching_tools.append(tool_name)
        
        return matching_tools
    
    def search_agents(self, query: str) -> List[Dict[str, Any]]:
        """Search for agents by name or description (case-insensitive).
        
        Args:
            query: Search query string. If any word in the query appears in an agent name 
                   or description, that agent will be included in results.
        
        Returns:
            List of agent info dictionaries that match the query.
        """
        if not query or not query.strip():
            return []
        
        try:
            from ..agents import list_agents
        except ImportError:
            return []
        
        query_lower = query.lower().strip()
        query_words = query_lower.split()
        
        matching_agents = []
        all_agents = list_agents()
        
        for agent in all_agents:
            agent_name = agent.get('name', '').lower()
            agent_display = agent.get('display_name', '').lower()
            agent_desc = agent.get('description', '').lower()
            
            # Check if any word in the query appears in agent name, display name, or description
            searchable_text = f"{agent_name} {agent_display} {agent_desc}"
            if any(word in searchable_text for word in query_words):
                matching_agents.append(agent)
        
        return matching_agents
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._tool_instances.clear()
        self._discovered_modules.clear()
    
    def is_module_discovered(self, tools_dir: str) -> bool:
        """Check if tools from a specific directory have already been discovered."""
        return tools_dir in self._discovered_modules


# Global tool registry instance
registry = ToolRegistry() 