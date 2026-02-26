"""
Agent Manager for creating and managing different agent types.
Allows users to select which agent to use for solving tasks.
"""

from typing import Dict, Any, List, Type, Optional
from .base_agent import BaseAgent
from .octotools.agent import OctoToolsAgent
from .zero_shot.agent import ZeroShotAgent
from .chain_of_thought.agent import ChainOfThoughtAgent
from .react.agent import ReActAgent
from .opentools.agent import OpenToolsAgent

class AgentManager:
    """
    Manager for creating and managing different agent types.
    Provides a registry of available agents and factory methods.
    """
    
    def __init__(self):
        self._agent_registry: Dict[str, Type[BaseAgent]] = {}
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Register the default agents"""
        self.register_agent("octotools", OctoToolsAgent)
        self.register_agent("zero_shot", ZeroShotAgent)
        self.register_agent("chain_of_thought", ChainOfThoughtAgent)
        self.register_agent("react", ReActAgent)
        self.register_agent("opentools", OpenToolsAgent)
        
    def register_agent(self, agent_name: str, agent_class: Type[BaseAgent]):
        """
        Register a new agent type.
        
        Args:
            agent_name: Name to identify the agent
            agent_class: Class that implements BaseAgent
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"Agent class {agent_class} must inherit from BaseAgent")
        
        self._agent_registry[agent_name.lower()] = agent_class
    
    def list_available_agents(self) -> List[Dict[str, str]]:
        """
        Get a list of all available agents with their information.
        
        Returns:
            List of dictionaries containing agent information
        """
        agents_info = []
        
        for agent_name, agent_class in self._agent_registry.items():
            class_name = agent_class.__name__
            
            # Get metadata from class-level attributes
            display_name = getattr(agent_class, 'AGENT_NAME', None)
            description = getattr(agent_class, 'AGENT_DESCRIPTION', None)
            
            # Fallback if class-level metadata is not available
            if not display_name:
                display_name = agent_name.title()
            if not description:
                description = f"Agent class: {class_name}"
            
            agents_info.append({
                "name": agent_name,
                "display_name": display_name,
                "description": description,
                "class": class_name
            })
        
        return agents_info
    
    def create_agent(self, 
                    agent_name: str,
                    llm_engine_name: str,
                    **kwargs) -> BaseAgent:
        """
        Create an agent instance.
        
        Args:
            agent_name: Name of the agent to create
            llm_engine_name: LLM engine to use
            **kwargs: Additional arguments to pass to the agent
            
        Returns:
            Instance of the requested agent
            
        Raises:
            ValueError: If agent_name is not registered
        """
        agent_name = agent_name.lower()
        
        if agent_name not in self._agent_registry:
            available = list(self._agent_registry.keys())
            raise ValueError(f"Agent '{agent_name}' not found. Available agents: {available}")
        
        agent_class = self._agent_registry[agent_name]
        return agent_class(llm_engine_name=llm_engine_name, **kwargs)
    
    def get_agent_class(self, agent_name: str) -> Type[BaseAgent]:
        """
        Get the agent class for a given name.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent class
            
        Raises:
            ValueError: If agent_name is not registered
        """
        agent_name = agent_name.lower()
        
        if agent_name not in self._agent_registry:
            available = list(self._agent_registry.keys())
            raise ValueError(f"Agent '{agent_name}' not found. Available agents: {available}")
        
        return self._agent_registry[agent_name]
    
    def agent_exists(self, agent_name: str) -> bool:
        """
        Check if an agent is registered.
        
        Args:
            agent_name: Name of the agent to check
            
        Returns:
            True if agent exists, False otherwise
        """
        return agent_name.lower() in self._agent_registry


# Global agent manager instance
_agent_manager = AgentManager()


def get_agent_manager() -> AgentManager:
    """Get the global agent manager instance"""
    return _agent_manager


def create_agent(agent_name: str, llm_engine_name: str, **kwargs) -> BaseAgent:
    """
    Convenience function to create an agent.
    
    Args:
        agent_name: Name of the agent to create
        llm_engine_name: LLM engine to use
        **kwargs: Additional arguments to pass to the agent
        
    Returns:
        Instance of the requested agent
    """
    return _agent_manager.create_agent(agent_name, llm_engine_name, **kwargs)


def list_agents() -> List[Dict[str, str]]:
    """
    Convenience function to list available agents.
    
    Returns:
        List of dictionaries containing agent information
    """
    return _agent_manager.list_available_agents()


def register_agent(agent_name: str, agent_class: Type[BaseAgent]):
    """
    Convenience function to register a new agent.
    
    Args:
        agent_name: Name to identify the agent
        agent_class: Class that implements BaseAgent
    """
    return _agent_manager.register_agent(agent_name, agent_class) 