"""
Base class for tool-based agents.
Provides common functionality for agents that use tools (OctoTools, ReAct, etc.)
"""

from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union, Optional
from .base_agent import BaseAgent
from .mixins.tool_capability_mixin import ToolCapabilityMixin

class ToolBasedAgent(BaseAgent, ToolCapabilityMixin):
    """
    Base class for agents that use tools.
    Provides common tool initialization, execution, and memory functionality.
    """
    
    def __init__(self, 
                 llm_engine_name: str,
                 model_family: str = "openai",
                 enabled_tools: List[str] = None,
                 num_threads: int = 1,
                 max_time: int = 120,
                 max_output_length: int = 100000,
                 vllm_config_path: str = None,
                 verbose: bool = True,
                 enable_faiss_retrieval: bool = False,
                 **kwargs):
        """
        Initialize the tool-based agent.
        
        Args:
            llm_engine_name: The LLM engine to use
            model_family: The family of the LLM engine (default: "openai")
            enabled_tools: List of tools to enable (default: ["all"])
            num_threads: Number of threads for execution
            max_time: Maximum execution time per tool
            max_output_length: Maximum output length
            vllm_config_path: Path to VLLM config if using VLLM
            verbose: Whether to print verbose output
            enable_faiss_retrieval: Whether to use FAISS-based tool retrieval (default: False)
        """
        self.model_family = model_family
        self.enabled_tools = enabled_tools or ["all"]
        self.log(f"Enabled tools 🔧: {self.enabled_tools}", "INFO")
        self.num_threads = num_threads
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.vllm_config_path = vllm_config_path
        self.enable_faiss_retrieval = enable_faiss_retrieval
        super().__init__(llm_engine_name=llm_engine_name, verbose=verbose, **kwargs)
        self.setup_tool_components()
    
    def setup_tool_components(self):
        """Setup shared tool components (initializer, executor)"""
        self.log("Initializing tool-based agent components...")
        self.initialize_tool_capabilities(
            enabled_tools=self.enabled_tools,
            num_threads=self.num_threads,
            max_time=self.max_time,
            max_output_length=self.max_output_length,
            vllm_config_path=self.vllm_config_path,
            enable_faiss_retrieval=self.enable_faiss_retrieval,
            model_family=self.model_family,
        )
        self.log("Tool-based agent components initialized successfully")
    
    @abstractmethod
    def setup_reasoning_components(self):
        """Setup agent-specific reasoning components (planner, etc.)"""
        pass
    
    def get_available_tools(self) -> List[str]:
        return super().get_available_tools()
    
    def get_toolbox_metadata(self) -> Dict[str, Any]:
        return super().get_toolbox_metadata()
    
    def get_relevant_tools(self, question: str) -> List[str]:
        return super().get_relevant_tools(question)

    #Parsing arguments with different format    
    def _parse_args(self, args: Union[str, Dict[str, Any], None]) -> Dict[str, Any]:
        return ToolCapabilityMixin._parse_args(self, args)

    #Normalize Tool Call with different format
    def _normalize_tool_call(self, tc: Any) -> Tuple[str, Dict[str, Any]]:
        return ToolCapabilityMixin._normalize_tool_call(self, tc)

    def execute_tool(self, tool_calls: Any, question: str) -> Any:
        return ToolCapabilityMixin.execute_tool(self, tool_calls, question)
    
    def generate_tool_command(self, question: str, image_path: Optional[str], 
                            context: str, sub_goal: str, tool_name: str) -> Any:
        return ToolCapabilityMixin.generate_tool_command(self, question, image_path, context, sub_goal, tool_name)
    
    def generate_and_execute_tool(self, question: str, image_path: Optional[str], 
                                context: str, sub_goal: str, tool_name: str) -> Any:
        return ToolCapabilityMixin.generate_and_execute_tool(self, question, image_path, context, sub_goal, tool_name)