"""
OpenTools Agents Module

This module provides a framework for managing different agent types.
Each agent implements the BaseAgent interface and can have its own
approach to solving tasks.
"""

from .base_agent import BaseAgent
from .tool_based_agent import ToolBasedAgent
from .agent_manager import (
    AgentManager,
    get_agent_manager,
    create_agent,
    list_agents,
    register_agent
)
from .octotools.agent import OctoToolsAgent
from .zero_shot.agent import ZeroShotAgent
from .chain_of_thought.agent import ChainOfThoughtAgent
from .react.agent import ReActAgent
from .opentools.agent import OpenToolsAgent

__all__ = [
    "BaseAgent",
    "ToolBasedAgent",
    "AgentManager", 
    "get_agent_manager",
    "create_agent",
    "list_agents", 
    "register_agent",
    "OctoToolsAgent",
    "DirectLLMAgent",
    "ChainOfThoughtAgent",
    "ReActAgent",
    "OpenToolsAgent"
] 