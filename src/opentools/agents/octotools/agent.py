"""
OctoTools agent implementation that uses the existing planning, memory, and execution modules.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..tool_based_agent import ToolBasedAgent
from .modules.planner import Planner
from .modules.utils import make_json_serializable_truncated
from .modules.executor import Executor as OctoExecutor
from .modules.memory import Memory

class OctoToolsAgent(ToolBasedAgent):
    """
    OctoTools agent that implements the original solving approach with
    planning, memory, and tool execution.
    """
    
    # Class-level metadata
    AGENT_NAME = "OctoTools"
    AGENT_DESCRIPTION = "Advanced tool-based agent with planning, memory, and step-by-step execution for complex tasks"
    
    def __init__(self, 
                 llm_engine_name: str,
                 enabled_tools: List[str] = None,
                 num_threads: int = 1,
                 max_time: int = 120,
                 max_output_length: int = 100000,
                 vllm_config_path: str = None,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize the OctoTools agent.
        
        Args:
            llm_engine_name: The LLM engine to use
            enabled_tools: List of tools to enable (default: ["all"])
            num_threads: Number of threads for execution
            max_time: Maximum execution time per tool
            max_output_length: Maximum output length
            vllm_config_path: Path to VLLM config if using VLLM
            verbose: Whether to print verbose output
        """
        # Setup LLM engine before calling super().__init__() so it's available for tool initialization
        from opentools.core.factory import create_llm_engine
        self.llm_engine_name = llm_engine_name
        self.llm_engine = create_llm_engine(model_string=llm_engine_name, is_multimodal=True)
        self.memory = Memory()
        # Initialize the tool-based agent (handles tool setup)
        super().__init__(
            llm_engine_name=llm_engine_name,
            enabled_tools=enabled_tools,
            num_threads=num_threads,
            max_time=max_time,
            max_output_length=max_output_length,
            vllm_config_path=vllm_config_path,
            verbose=verbose,
            **kwargs
        )
        
        # Setup OctoTools-specific reasoning components
        self.setup_reasoning_components()
    
    def setup_reasoning_components(self):
        """Setup the OctoTools-specific reasoning components"""
        self.log("Initializing OctoTools reasoning components...")
        
        # Initialize planner
        self.planner = Planner(
            llm_engine=self.llm_engine,
            llm_engine_name=self.llm_engine_name,
            toolbox_metadata=self.initializer.toolbox_metadata,
            available_tools=self.initializer.available_tools,
            verbose=self.verbose
        )
        # Override the generic executor with OctoTools-specific executor
        self.executor = OctoExecutor(
            llm_engine=self.llm_engine,
            llm_engine_name=self.llm_engine_name,
            root_cache_dir="solver_cache",
            num_threads=self.num_threads,
            max_time=self.max_time,
            max_output_length=self.max_output_length,
            verbose=self.verbose,
        )
        
        self.log("OctoTools reasoning components initialized successfully")
    
    def get_agent_name(self) -> str:
        """Return the name of this agent"""
        return "OctoTools"
    
    def get_agent_description(self) -> str:
        """Return a description of this agent"""
        return "Advanced tool-based agent with planning, memory, and step-by-step execution for complex tasks"
    
    def get_tool(self, question: str, image_path: Optional[str] = None):
        """Get tool for a question (matches original solver interface)"""
        self.log(f'Received Question: {question}')
        self.executor.set_query_cache_dir("solver_cache")
        if image_path:
            image_path = os.path.abspath(image_path)
        # Initialize json_data with basic problem information
        json_data = {
            "query": question,
            "image": image_path
        }

        # Generate base response if requested
        output_types = ["base", "final", "direct"]  # Default output types
        if 'base' in output_types:
            base_response = self.planner.generate_base_response(question, image_path, 4000)
            json_data["base_response"] = base_response

        # If only base response is needed, save and return
        if set(output_types) == {'base'}:
            return json_data
    
        # Continue with query analysis and tool execution if final or direct responses are needed
        if {'final', 'direct'} & set(output_types):
            query_start_time = time.time()
            query_analysis = self.planner.analyze_query(question, image_path)
            json_data["query_analysis"] = query_analysis

            # Main execution loop
            step_count = 0

            local_start_time = time.time()
            next_step = self.planner.generate_next_step(
                question, 
                image_path, 
                query_analysis, 
                self.memory, 
                step_count, 
                10  # max_steps
            )
            context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)

            if tool_name is None or tool_name not in self.planner.available_tools:
                self.log(f"Error: Tool '{tool_name}' is not available or not found.", level="ERROR")
                command = "Not command is generated due to the tool not found."
                result = "Not result is generated due to the tool not found."
            else:
                tool_command = self.executor.generate_tool_command(
                    question, 
                    image_path, 
                    context, 
                    sub_goal, 
                    tool_name, 
                    self.planner.toolbox_metadata[tool_name]
                )
                return tool_name, tool_command
        return None, None

    def solve(self,
              question: str,
              image_path: Optional[str] = None,
              output_types: str = "base,final,direct",
              max_steps: int = 10,
              max_time: int = 300,
              max_tokens: int = 4000,
              root_cache_dir: str = "solver_cache",
              query_cache_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Solve a question using the original OctoTools approach with planning and tool execution.
        This matches the original solver implementation exactly.
        """
        # Start a grouped query session for UI streaming
        try:
            self.start_query_session(question, {"agent": self.get_agent_name()})
        except Exception:
            pass
        # Set up cache directory  
        if query_cache_dir:
            cache_dir = query_cache_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_dir = os.path.join(root_cache_dir, timestamp)
        
        os.makedirs(cache_dir, exist_ok=True)
        self.executor.set_query_cache_dir(cache_dir)
        if image_path:
            image_path = os.path.abspath(image_path)
        # Initialize json_data with basic problem information
        json_data = {
            "query": question,
            "image": image_path
        }
        
        if self.verbose:
            self.log(f"Received Query: {question}")
            if image_path:
                self.log(f"Received Image: {image_path}")

        # Parse output types
        if isinstance(output_types, str):
            output_types_list = output_types.lower().split(',')
        else:
            output_types_list = output_types
        assert all(output_type in ["base", "final", "direct"] for output_type in output_types_list), \
            "Invalid output type. Supported types are 'base', 'final', 'direct'."

        # Generate base response if only base is requested
        start_time = time.time()
        if set(output_types_list) == {'base'}:
            self.log("Base response requested - generating direct LLM response without tools")
            try:
                base_response = self.planner.generate_base_response(question, image_path, max_tokens)
                token_usage = self.llm_engine.get_token_usage()
                
                result = {
                    "query": question,
                    "image": image_path,
                    "base_response": base_response,
                    "base_output": base_response,  # Also include base_output for compatibility with solve.py
                    "agent": self.get_agent_name(),
                    "execution_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                    "token_usage": token_usage,
                    "cache_dir": cache_dir
                }
                
                if self.verbose:
                    self.log(f"Base response: {base_response}")
                
                return result
            except Exception as e:
                self.log(f"Error generating base response: {e}", level="ERROR")
                return {
                    "query": question,
                    "image": image_path,
                    "base_response": "",
                    "base_output": "",  # Include empty base_output to prevent KeyError
                    "error": str(e),
                    "agent": self.get_agent_name(),
                    "execution_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat(),
                    "status": "error"
                }
    
        # Continue with query analysis and tool execution if final or direct responses are needed
        if {'final', 'direct'} & set(output_types_list):
            if self.verbose:
                self.log("Reasoning Steps from OctoTools (Deep Thinking...)")

            # [1] Analyze query
            query_start_time = time.time()
            try:
                self.log_step_start("Query Analysis")
            except Exception:
                pass
            query_analysis = self.planner.analyze_query(question, image_path)
            json_data["query_analysis"] = query_analysis
            if self.verbose:
                self.log("Step 0: Query Analysis\n")
                self.log(f"{query_analysis}")
                self.log(f"[Time]: {round(time.time() - query_start_time, 2)}s")
            try:
                self.log_event("thought", json.dumps(query_analysis, ensure_ascii=False))
                self.log_step_end(success=True)
            except Exception:
                pass

            # Main execution loop
            step_count = 0
            action_times = []
            while step_count < max_steps and (time.time() - query_start_time) < max_time:
                step_count += 1
                step_start_time = time.time()

                # [2] Generate next step
                local_start_time = time.time()
                next_step = self.planner.generate_next_step(
                    question, 
                    image_path, 
                    query_analysis, 
                    self.memory, 
                    step_count, 
                    max_steps
                )
                context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)
                if self.verbose:
                    self.log(f"Step {step_count}: Action Prediction ({tool_name})\n")
                    self.log(f"[Context]: {context}\n[Sub Goal]: {sub_goal}\n[Tool]: {tool_name}")
                    self.log(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                try:
                    self.log_step_start(f"Action Prediction ({tool_name})", {"step": step_count})
                    self.log_event("thought", f"Context: {context}")
                    self.log_event("thought", f"Sub Goal: {sub_goal}")
                    # Selection is part of planning, not an action execution
                    self.log_event("thought", f"Selected tool: {tool_name}", tool_name=tool_name)
                except Exception:
                    pass

                if tool_name is None or tool_name not in self.planner.available_tools:
                    self.log(f"Error: Tool '{tool_name}' is not available or not found.", level="ERROR")
                    command = "Not command is generated due to the tool not found."
                    result = "Not result is generated due to the tool not found."
                    try:
                        self.log_event("observation", f"Tool '{tool_name}' not found.")
                        self.log_step_end(success=False)
                    except Exception:
                        pass

                else:
                    # [3] Generate the tool command
                    local_start_time = time.time()
                    tool_command = self.executor.generate_tool_command(
                        question, 
                        image_path, 
                        context, 
                        sub_goal, 
                        tool_name, 
                        self.planner.toolbox_metadata[tool_name]
                    )
                    analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
                    if self.verbose:
                        self.log(f"Step {step_count}: Command Generation ({tool_name})\n")
                        self.log(f"[Analysis]: {analysis}\n[Explanation]: {explanation}\n[Command]: {command}")
                        self.log(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                    try:
                        self.log_event("thought", analysis or "")
                        self.log_event("thought", f"Plan: {explanation}" if explanation else "", tool_name=tool_name)
                        # Only mark the actual command as an action
                        self.log_event("action", f"Command:\n{command}", tool_name=tool_name)
                    except Exception:
                        pass
                    
                    # [4] Execute the tool command
                    local_start_time = time.time()
                    result = self.executor.execute_tool_command(tool_name, command)
                    result = make_json_serializable_truncated(result) # Convert to JSON serializable format
                    if self.verbose:
                        self.log(f"Step {step_count}: Command Execution ({tool_name})\n")
                        self.log(f"[Result]:\n{json.dumps(result, indent=4)}")
                        self.log(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                    try:
                        self.log_event("observation", json.dumps(result, ensure_ascii=False)[:1500], tool_name=tool_name)
                        self.log_step_end(success=True)
                    except Exception:
                        pass
                
                # Track execution time for the current step
                execution_time_step = round(time.time() - step_start_time, 2)
                action_times.append(execution_time_step)

                # Update memory
                self.memory.add_action(step_count, tool_name, sub_goal, command, result)
                memory_actions = self.memory.get_actions()

                # [5] Verify memory (context verification)
                local_start_time = time.time()
                stop_verification = self.planner.verificate_context(
                    question, 
                    image_path, 
                    query_analysis, 
                    self.memory
                )
                context_verification, conclusion = self.planner.extract_conclusion(stop_verification)
                if self.verbose:
                    conclusion_emoji = "✅" if conclusion == 'STOP' else "🛑"
                    self.log(f"Step {step_count}: Context Verification\n")
                    self.log(f"[Analysis]: {context_verification}\n[Conclusion]: {conclusion} {conclusion_emoji}")
                    self.log(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                try:
                    self.log_event("thought", json.dumps(context_verification, ensure_ascii=False))
                    self.log_event("observation", f"Conclusion: {conclusion}")
                except Exception:
                    pass
                
                # Break the loop if the context is verified
                if conclusion == 'STOP':
                    break

            # Add memory and statistics to json_data
            json_data.update({
                "memory": memory_actions,
                "step_count": step_count,
                "execution_time": round(time.time() - query_start_time, 2),
            })

            # Generate final output if requested
            if 'final' in output_types_list:
                final_output = self.planner.generate_final_output(question, image_path, self.memory)
                json_data["final_output"] = final_output
                # Map to UI-consumed key
                json_data["final_answer"] = final_output
                if self.verbose:
                    self.log(f"Detailed Solution:\n\n{final_output}")
                try:
                    self.log_final_answer(final_output)
                except Exception:
                    pass

            # Generate direct output if requested  
            if 'direct' in output_types_list:
                direct_output = self.planner.generate_direct_output(question, image_path, self.memory)
                json_data["direct_output"] = direct_output
                # Map to UI-consumed key
                json_data["direct_answer"] = direct_output
                if self.verbose:
                    self.log(f"Final Answer:\n\n{direct_output}")
                try:
                    self.log_final_answer(direct_output)
                except Exception:
                    pass

            if self.verbose:
                self.log(f"[Total Time]: {round(time.time() - query_start_time, 2)}s")
                self.log("Query Solved!")

        try:
            # End the grouped query session
            self.end_query_session(json_data.get("direct_output") or json_data.get("final_output"))
        except Exception:
            pass
        json_data["token_usage"] = self.llm_engine.get_token_usage()
        return json_data