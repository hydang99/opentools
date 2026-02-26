import os
import importlib
import re
from typing import Dict, Any, List
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..')))
from typing import Dict, Any, List, Optional

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

class Executor:
    def __init__(self, llm_engine_name: str, root_cache_dir: str = "solver_cache",  num_threads: int = 1, 
                        max_time: int = 120, 
                        max_output_length: int = 100000, verbose: bool = False, llm_engine: Optional[Any] = None):
        self.llm_engine_name = llm_engine_name
        self.root_cache_dir = root_cache_dir
        self.num_threads = num_threads
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.verbose = verbose
        self.llm_engine = llm_engine

    def set_query_cache_dir(self, query_cache_dir):
        if query_cache_dir:
            self.query_cache_dir = query_cache_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_cache_dir = os.path.join(self.root_cache_dir, timestamp)
        os.makedirs(self.query_cache_dir, exist_ok=True)
    def execute_tool_command(self, tool_name: str, args: dict) -> Any:
        """
        Execute a tool command with timeout protection. If execution exceeds max_time seconds,
        the function will be interrupted and return a timeout message.
        
        Args:
            tool_name (str): Name of the tool to execute
            args (dict): Command string containing tool.run() calls

        Returns:
            Any: List of execution results or error message
        """
        module_name = f"opentools.tools.{tool_name.lower().replace('_tool', '')}.tool"
        print(f"Module name: {module_name}")
        try:
            # Dynamically import the module
            module = importlib.import_module(module_name)
            
            # Get the tool class
            tool_class = getattr(module, tool_name)

            # Check if the tool requires an LLM engine
            # NOTE may need to refine base.py and tool.py to handle this better
            if getattr(tool_class, 'require_llm_engine', False):
                # Instantiate the tool with the model_string
                tool = tool_class(model_string=self.llm_engine_name, llm_engine=self.llm_engine)
            else:
                # Instantiate the tool without model_string for tools that don't require it
                tool = tool_class()
            
            result = tool.run(**args)
            # print(f"Tool execution result: {result}")
            return result
        except Exception as e:
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return f"Error in execute_tool_command: {str(e)} CLMMMMM" 
    
if __name__ == "__main__":
    executor = Executor(llm_engine_name="gpt-4o-mini")
    executor.execute_tool_command(tool_name="Calculator_Tool", args={"operation": "add", "values": [2, 2]})