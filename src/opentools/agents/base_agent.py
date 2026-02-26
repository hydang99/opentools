"""
Minimal base agent class that all agents should inherit from.
Provides basic interface and simple shared utilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from ..core.display import AgentDisplayMixin

class BaseAgent(AgentDisplayMixin, ABC):
    """
    Minimal base class for all agents in the OpenTools framework.
    
    Each agent should implement its own components (memory, executor, planner, etc.)
    and solve method according to its specific requirements.
    """
    
    # Class-level metadata (to be overridden by subclasses)
    AGENT_NAME: str = None
    AGENT_DESCRIPTION: str = None
    
    def __init__(self, 
                 llm_engine_name: str,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize the base agent with minimal shared configuration.
        
        Args:
            llm_engine_name: The LLM engine to use
            verbose: Whether to print verbose output
            **kwargs: Additional arguments specific to the agent
        """
        self.llm_engine_name = llm_engine_name
        self.verbose = verbose
    
    @abstractmethod
    def get_agent_name(self) -> str:
        """Return the name of this agent"""
        pass
    
    @abstractmethod
    def get_agent_description(self) -> str:
        """Return a description of this agent"""
        pass
    
    @abstractmethod
    def solve(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Solve a question/task. Each agent implements this differently.
        
        Args:
            question: The question/task to solve
            **kwargs: Additional arguments specific to the agent
            
        Returns:
            Dictionary containing the solution and metadata
        """
        pass
    
    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging utility with backward compatibility"""
        # Use the enhanced display manager from AgentDisplayMixin
        super().log(message, level)
    
    def get_agent_info(self) -> Dict[str, str]:
        """Get basic information about this agent"""
        return {
            "name": self.get_agent_name(),
            "description": self.get_agent_description(),
            "llm_engine": self.llm_engine_name
        } 