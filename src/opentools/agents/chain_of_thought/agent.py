"""
Chain of Thought Agent implementation for step-by-step reasoning without tools.
This agent prompts the LLM to think through problems step by step.
"""
import os
import re
import json
import time
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.agents.base_agent import BaseAgent
from opentools.agents.mixins.tool_capability_mixin import ToolCapabilityMixin

class ChainOfThoughtAgent(BaseAgent, ToolCapabilityMixin):
    """
    Chain of Thought agent that prompts the LLM for step-by-step reasoning.
    Good for complex questions that need reasoning but don't require tools.
    """
    
    # Class-level metadata
    AGENT_NAME = "ChainOfThought"
    AGENT_DESCRIPTION = "Step-by-step reasoning without tools - good for complex logic problems"
    
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
        Initialize the Chain of Thought agent.
        
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
        
        self.log("Initializing Chain of Thought agent...")
        
        self.llm_engine = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=False)
        self.llm_engine_mm = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=True)
        
        self.log("Chain of Thought agent initialized successfully")
    
    def get_agent_name(self) -> str:
        """Return the name of this agent"""
        return "ChainOfThought"
    
    def get_agent_description(self) -> str:
        """Return a description of this agent"""
        return "Step-by-step reasoning without tools - good for complex logic problems"
    
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
    
    def create_cot_prompt(self, question: str, image_path: Optional[str] = None) -> str:
        """Create a chain of thought prompt"""
        
        base_prompt = f"""Please solve this problem step by step. Think through your reasoning process carefully.

Question: {question}
"""
        
        if image_path:
            base_prompt = f"""Please analyze the provided image and answer the question step by step. Think through your reasoning process carefully.

Question: {question}
"""
        
        return base_prompt
    
    def generate_cot_response(self, question: str, image_path: Optional[str] = None, max_tokens: int = 4000) -> str:
        """Generate a chain of thought response"""
        
        cot_prompt = self.create_cot_prompt(question, image_path)
        image_info = self.get_image_info(image_path)

        # Use multimodal engine if we have an image, otherwise use text-only
        if image_info and "image_path" in image_info:
            response = self.llm_engine_mm.generate(cot_prompt, image=image_path, max_tokens=max_tokens)
        else:
            response = self.llm_engine.generate(cot_prompt, max_tokens=max_tokens)

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
        Solve a question using chain of thought reasoning.
        
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
            
            # Generate chain of thought response
            self.log("Generating step-by-step reasoning...")
            response = self.generate_cot_response(question, image_path, max_tokens)
            
            # Try to extract the final answer
            final_answer = self.extract_final_answer(response)
            
            # Prepare result data
            result_data = {
                "query": question,
                "image": image_path,
                "reasoning": response,
                "direct_output": final_answer,
                "agent": self.get_agent_name(),
                "llm_engine": self.llm_engine_name,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "max_tokens_used": max_tokens,
            }
            
            if self.verbose:
                self.log(f"Chain of Thought Reasoning:\n{response}")
                if final_answer:
                    self.log(f"Extracted Final Answer: {final_answer}")
                self.log(f"Completed in {result_data['execution_time']:.2f} seconds")
            
            # Save result if requested
            if save_result:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cache_dir = f"cot_cache/{timestamp}"
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
    
    def extract_final_answer(self, response: str) -> Optional[str]:
        """Try to extract the final answer from the chain of thought response"""
        # import re
        
        # Look for patterns like "Final Answer:", "Answer:", etc.
        patterns = [
            r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>",
            r"Answer:\s*(.+?)(?:\n|$)",
            r"The answer is:\s*(.+?)(?:\n|$)",
            r"Therefore,?\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Clean up the answer (remove extra punctuation, etc.)
                answer = re.sub(r'^[^\w]*|[^\w]*$', '', answer)
                if answer:
                    return answer
        
        # If no explicit final answer found, try to get the last meaningful line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            return lines[-1]
        
        return None 

if __name__ == "__main__":
    agent = ChainOfThoughtAgent(llm_engine_name="gpt-5-mini", verbose=True)
    result = agent.solve(question="Who is the author of the paper 'Attention is all you need'?")
    print(result)
