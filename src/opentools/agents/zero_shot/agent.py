"""
Direct LLM Agent implementation for zero-shot responses.
This agent provides simple LLM responses without any tool usage or planning.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))

from opentools.agents.base_agent import BaseAgent
from opentools.agents.mixins.tool_capability_mixin import ToolCapabilityMixin


class ZeroShotAgent(BaseAgent, ToolCapabilityMixin):
    """
    ZeroShot LLM agent that provides zero-shot responses without tools.
    Perfect for simple questions that don't require complex reasoning or tool usage.
    """
    
    # Class-level metadata
    AGENT_NAME = "ZeroShotLLM"
    AGENT_DESCRIPTION = "Zero-shot LLM responses without tools - fast and simple"
    
    def __init__(self, 
                 llm_engine_name: str,
                 verbose: bool = True,
                 enable_tool_calls: bool = False,
                 enabled_tools: Optional[List[str]] = None,
                 num_threads: int = 1,
                 max_time: int = 120,
                 max_output_length: int = 100000,
                 vllm_config_path: Optional[str] = None,
                 enable_faiss_retrieval: bool = False,
                 **kwargs):
        """
        Initialize the ZeroShot LLM agent.
        
        Args:
            llm_engine_name: The LLM engine to use
            verbose: Whether to print verbose output
            enable_tool_calls: Whether to enable tool calling capability
            enabled_tools: List of tools to enable (default: ["all"])
            num_threads: Number of threads for execution
            max_time: Maximum execution time per tool
            max_output_length: Maximum output length
            vllm_config_path: Path to VLLM config if using VLLM
            enable_faiss_retrieval: Whether to use FAISS-based tool retrieval (default: False)
        """
        # Initialize the base agent
        super().__init__(llm_engine_name=llm_engine_name, verbose=verbose, **kwargs)
        self.enable_tool_calls = enable_tool_calls
        if self.enable_tool_calls:
            self.initialize_tool_capabilities(
                enabled_tools=enabled_tools,
                num_threads=num_threads,
                max_time=max_time,
                max_output_length=max_output_length,
                vllm_config_path=vllm_config_path,
            )
        
        # Setup LLM engines
        self.setup_llm_engines()
        
    
    def setup_llm_engines(self):
        """Setup LLM engines for text and multimodal processing"""
        from opentools.core.factory import create_llm_engine
        
        self.log("Initializing ZeroShot LLM agent...")
        
        self.llm_engine = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=False)
        self.llm_engine_mm = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=True)
        
        self.log("ZeroShot LLM agent initialized successfully")
    
    def get_agent_name(self) -> str:
        """Return the name of this agent"""
        return "ZeroShotLLM"
    
    def get_agent_description(self) -> str:
        """Return a description of this agent"""
        return "Zero-shot LLM responses without tools - fast and simple"

    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Get information about an image file"""
        from PIL import Image
        
        image_info = {}
        if image_path and os.path.isfile(image_path):
            image_info["image_path"] = image_path
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                image_info.update({
                    "width": width,
                    "height": height
                })
            except Exception as e:
                self.log(f"Error processing image file: {str(e)}", level="ERROR")
        return image_info
    
    def generate_response(self, question: str, image_path: Optional[str] = None, max_tokens: int = 4000) -> str:
        """Generate a direct response from the LLM"""
        image_info = self.get_image_info(image_path)

        # Use multimodal engine if we have an image, otherwise use text-only
        if image_info and "image_path" in image_info:
            response = self.llm_engine_mm.generate(question, image=image_path, max_tokens=max_tokens)
        else:
            response = self.llm_engine.generate(question, max_tokens=max_tokens)

        if isinstance(response, dict):
            response = response.get('text')
        else:
            response = str(response)
        return response
    
    def solve(self,
              question: str,
              image_path: Optional[str] = None,
              max_tokens: int = 4000,
              save_result: bool = True,
              **kwargs) -> Dict[str, Any]:
        """
        Solve a question using direct LLM response (zero-shot).
        
        Args:
            question: The question to solve
            image_path: Optional path to an image
            max_tokens: Maximum tokens for LLM response
            save_result: Whether to save result to file
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            Dictionary containing the solution and metadata
        """
        start_time = time.time()
        
        if self.verbose:
            self.log(f"Received question: {question}")
            if image_path:
                self.log(f"Processing image: {image_path}")

        try:

            self.log("Generating direct LLM response...")
            response = self.generate_response(question, image_path, max_tokens)
            
            # Get token usage from LLM engine
            token_usage = self.llm_engine.get_token_usage() if hasattr(self.llm_engine, 'get_token_usage') else {}
            
            # Prepare result data
            result_data = {
                "query": question,
                "image": image_path,
                "direct_output": response,
                "agent": self.get_agent_name(),
                "llm_engine": self.llm_engine_name,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "max_tokens_used": max_tokens,
                "token_usage": token_usage,
            }
            
            if self.verbose:
                self.log(f"LLM Response:\n{response}")
                
                # Log token usage summary
                if token_usage:
                    self.log(f"Token Usage Summary:")
                    self.log(f"  Total tokens: {token_usage.get('total_tokens', 0)}")
                    self.log(f"  Prompt tokens: {token_usage.get('total_prompt_tokens', 0)}")
                    self.log(f"  Completion tokens: {token_usage.get('total_completion_tokens', 0)}")
                    self.log(f"  API calls: {token_usage.get('call_count', 0)}")
                
                self.log(f"Completed in {result_data['execution_time']:.2f} seconds")
            
            # Save result if requested
            if save_result:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cache_dir = f"direct_llm_cache/{timestamp}"
                os.makedirs(cache_dir, exist_ok=True)
                
                result_file = os.path.join(cache_dir, "result.json")
                with open(result_file, 'w') as f:
                    json.dump(result_data, f, indent=2, default=str)
                
                result_data["cache_dir"] = cache_dir
            
            return result_data
            
        except Exception as e:
            error_data = {
                "error": str(e),
                "query": question,
                "image": image_path,
                "agent": self.get_agent_name(),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.log(f"Error occurred: {str(e)}", level="ERROR")
            return error_data 
if __name__ == "__main__":
    agent = ZeroShotAgent(llm_engine_name="gpt-5-mini")
    result = agent.solve(question="What is the capital of France?")
    print(result)