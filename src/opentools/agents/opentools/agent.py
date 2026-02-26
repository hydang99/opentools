import os, json, time, sys, traceback
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.agents.tool_based_agent import ToolBasedAgent
from opentools.core.factory import create_llm_engine

from opentools.agents.opentools.memory import Global_Memory, Local_Memory
from opentools.agents.opentools.reasoning import Reasoning
from opentools.agents.opentools.generator import Generator
# Executor is not currently used - tool execution is handled by ToolBasedAgent.execute_tool
# from opentools.agents.opentools.executor import Executor
from opentools.agents.opentools.verifier import Verifier

class DirectOutputAndSummary(BaseModel):
    direct_output: str = Field(description="The direct answer to the given query.")
    summary: str = Field(description="A short summary that justifies the answer using steps taken from the memory.")
    stop: bool = Field(description="True if the memory is sufficient to answer the query, False otherwise.")

class OpenToolsAgent(ToolBasedAgent):

    # Class-level metadata
    AGENT_NAME = "OpenTools"
    AGENT_DESCRIPTION = "OpenTools agent - uses tools to solve problems with multi-agent architecture"

    
    def __init__(self, 
                 llm_engine_name: str,
                 enabled_tools: List[str] = None,
                 num_threads: int = 1,
                 max_time: int = 120,
                 max_output_length: int = 100000,
                 verbose: bool = True,
                 enable_faiss_retrieval: bool = False,
                 **kwargs):
        """
        Initialize the OpenTools agent.
        
        Args:
            llm_engine_name: The LLM engine to use
            enabled_tools: List of tools to enable (default: ["all"])
            num_threads: Number of threads for execution
            max_time: Maximum execution time per tool
            max_output_length: Maximum output length
            verbose: Whether to print verbose output
            enable_faiss_retrieval: Whether to use FAISS-based tool retrieval (default: False)
            **kwargs: Additional keyword arguments
        """
        self.llm_engine_name = llm_engine_name
        self.setup_reasoning_components()
        # Initialize the tool-based agent (handles tool setup)
        super().__init__(
            llm_engine_name=llm_engine_name,
            enabled_tools=enabled_tools,
            num_threads=num_threads,
            max_time=max_time,
            max_output_length=max_output_length,
            verbose=verbose,
            enable_faiss_retrieval=enable_faiss_retrieval,
            **kwargs
        )
        self.reasoning_agent = Reasoning(self.llm_engine)
        self.generator_agent = Generator(self.llm_engine)
        self.verifier_agent = Verifier(self.llm_engine)
    
    def setup_reasoning_components(self):
        self.llm_engine = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=True)  

    def get_agent_name(self) -> str:
        """Return the name of this agent"""
        return "OpenTools"
    
    def get_agent_description(self) -> str:
        """Return a description of this agent"""
        return "OpenTools agent - uses tools to solve problems"
    def count_tokens(self, text):
        """
        Lightweight token estimator. Avoids heavy tiktoken encode calls that can
        stall on very large payloads; approximate by string length instead.
        """
        try:
            return len(text) // 4
        except Exception:
            return 0
    def generate_direct_output(self, question: str, image_path: str, memory: dict) -> str:
        try:
            memory_json = json.dumps(memory)
        except Exception as e:
            memory_json = memory
        prompt_generate_final_output = f"""
Context:
Query: {question}
Image: {image_path}
Memory:
{memory_json}
Based on the query and the memory above, provide ONLY the direct answer to the question. 
Extract the key information from the memory and provide a simple, direct answer without any explanations, analysis, or additional context.
Answer:
"""
        if image_path:
            final_output = self.llm_engine.generate(prompt_generate_final_output, image=image_path)
        else:   
            final_output = self.llm_engine.generate(prompt_generate_final_output)
        self.log(f'Direct output: {final_output}')
        return final_output

    def generate_direct_output_and_summary(self, question: str, image_path: str, file_path: str, memory, candidate_answer: str):
        system_prompt = f"""
You are the Final Answer Composer in a multi-step agent system.

Instructions:
- You generate the direct_output STRICTLY from given memory and candidate answer.
- You must output TWO fields only: direct_output and summary.

You are given:
- Original user query
- Candidate answer
- Memory of the system

Your job:

** direct_output field: **
- The content must directly answer the original user query. Do not add any unnecessary information.

** summary field: **
- Provide a short justification for why the answer is correct using information from the memory.
- Keep it short. Mainly summary of the sub-problems and results of each step.
- Do NOT introduce new facts not present in memory.

** stop field: **
- Return True if the memory is sufficient to answer the query, False otherwise.

** Insufficient information rule: **
- If memory is insufficient to answer the query, output:
  - stop = False
  - direct_output = NOT_READY
  - summary = a short statement of what is missing (still using only what you can infer from memory).
  """
        input_prompt = f"""
Original user query: {question}
Candidate answer: {candidate_answer}
Memory: {memory}
        """
        return self.llm_engine.generate(query = input_prompt, system_prompt= system_prompt, response_format=DirectOutputAndSummary)

        
    def prepare_result_data(self, question: str, image_path: str, file_path: str, reasoning_trace, 
    steps_taken: int, agent: str, execution_time, timestamp, cache_dir, status, token_usage, 
    error_executions, local_memory_list):
        return {
            "query": question,
            "image": image_path,
            "file_path": file_path,
            "llm_engine": self.llm_engine_name,
            "steps_taken": steps_taken,
            "agent": agent,
            "token_usage": token_usage,
            "reasoning_trace": reasoning_trace,      
            "error_executions": error_executions,
            "local_memory_list": local_memory_list,   
            "execution_time": execution_time,
            "timestamp": timestamp,
            "cache_dir": cache_dir,
        }
    def solve(self,
              question: str,
              image_path: Optional[str] = None,
              file_path: Optional[str] = None,
              max_steps: int = 10,
              max_time: int = 300,
              max_tokens: int = 4000,
              root_cache_dir: str = "opentools_cache",
              output_types: str = "direct",
              **kwargs) -> Dict[str, Any]:
        """
        Solve a question using the OpenTools approach.
        
        Args:
            question: The question to solve
            image_path: Optional path to an image (not fully supported yet)
            file_path: Optional path to a file (not fully supported yet)
            max_steps: Maximum number of reasoning/acting steps
            max_time: Maximum time in seconds
            max_tokens: Maximum tokens per LLM call
            root_cache_dir: Root directory for caching
            
        Returns:
            Dictionary containing the solution and metadata
        """
        start_time = time.time()
        # Log the received question, image, and file
        if self.verbose:
            self.log(f"Received question: {question}")
            if image_path:
                self.log(f"Received image: {image_path}")
            if file_path:
                self.log(f"Received file: {file_path}")
        
        # Set up cache directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_dir = os.path.join(root_cache_dir, timestamp)
        os.makedirs(cache_dir, exist_ok=True)
        # Note: Executor is currently not used in OpenTools agent (tool execution is handled by ToolBasedAgent.execute_tool)
        # self.executor.set_query_cache_dir(cache_dir)  # Removed - executor not initialized

        try:
            global_memory = Global_Memory(question)
            local_memory = Local_Memory()

            error_executions = []
            local_memory_list = []
            step_count = 0            
            state = True 

            self.log("Starting OpenTools reasoning and acting loop 💭...")
            
            while step_count < max_steps and (time.time() - start_time) < max_time:
                step_count += 1

                self.log(f"#########################################################","INFO")
                self.log_step_start(f"OpenTools Reasoning Cycle {step_count}")

                agent_type = self.get_agent_type()
                """
                Start the reasoning agent to get the next sub-problem and sub-agent
                """
                self.log(f"Start reasoning agent: ", "INFO")
                reasoning_response = self.reasoning_agent.reasoning_next_step(query = question, sub_agent = agent_type, 
                        global_memory = global_memory.get_actions(), state = state, failure_report = local_memory.get_action(), image_path = image_path, file_path = file_path)
                self.log(f"Reasoning agent response: {reasoning_response}", "INFO")
                
                # Check if response is an error dict before trying to access attributes
                if isinstance(reasoning_response, dict) and "error" in reasoning_response:
                    error_type = reasoning_response.get("error", "unknown_error")
                    error_message = reasoning_response.get("message", "Unknown error occurred")
                    error_msg = f"Reasoning agent returned error: {error_type} - {error_message}"
                    self.log(f"{error_msg}", "ERROR")
                    local_memory.update_error(error_msg)
                    local_memory_list.append(local_memory.get_action())
                    state = False
                    continue
                
                try:
                    sub_problem = reasoning_response.sub_problem
                    sub_agent = reasoning_response.sub_agent
                    supporting_documents = reasoning_response.supporting_documents
                    local_memory.update_memory_reasoning(sub_problem, sub_agent, supporting_documents)
                    if state: 
                        stop = reasoning_response.stop
                        if stop:
                            candidate_answer = reasoning_response.answer
                            global_memory.update_answer(candidate_answer)
                            answer = self.generate_direct_output_and_summary(question, image_path, file_path, global_memory.get_actions(), candidate_answer)
                            self.log(f"Direct output and summary: {answer}", "INFO")
                            if answer.stop:
                                self.log(f"Reach final answer: ", "INFO")
                                self.log_final_answer(answer.direct_output)
                                self.log(f"Summary of step taken: {answer.summary}", "INFO")
                                token_usage = self.llm_engine.get_token_usage()
                                result_data = self.prepare_result_data(question, image_path, file_path, global_memory.get_actions(), step_count,
                                        self.get_agent_name(), time.time() - start_time, datetime.now().isoformat(), cache_dir, "success",
                                        token_usage, error_executions, local_memory_list)
                                result_data["direct_output"] = answer.direct_output
                                result_data["summary"] = answer.summary
                                # Save result
                                result_file = os.path.join(cache_dir, "result.json")
                                with open(result_file, 'w') as f:
                                    json.dump(result_data, f, indent=2, default=str)
                                
                                # End the final step and display summary
                                self.log_step_end("Final answer reached", success=True)
                                if token_usage:
                                    self.log(f"Token Usage Summary:")
                                    self.log(f"  Total tokens: {token_usage.get('total_tokens', 0)}")
                                    self.log(f"  Prompt tokens: {token_usage.get('total_prompt_tokens', 0)}")
                                    self.log(f"  Completion tokens: {token_usage.get('total_completion_tokens', 0)}")
                                    self.log(f"  API calls: {token_usage.get('call_count', 0)}")            
                                self.log(f"OpenTools completed in {result_data['execution_time']:.2f} seconds")
                                if hasattr(self.llm_engine, "reset_token_counters"):
                                    try:
                                        self.llm_engine.reset_token_counters()
                                    except Exception:
                                        pass
                                self.display_summary()
                                return result_data
                            else:
                                self.log(f"Reach final answer but not sufficient information: ", "INFO")
                                local_memory = Local_Memory()
                                local_memory.update_insufficient_info_reason(answer.summary)
                                local_memory_list.append(local_memory.get_action())
                                state = True
                                continue
                    else:
                        failure_source = reasoning_response.failure_source
                        local_memory.update_failure_source(failure_source)
                except (AttributeError, TypeError) as e:
                    # Handle case where reasoning_response is a dict (error case) instead of NextStep/RetryStep object
                    error_msg = f"Reasoning agent error: {str(e)}"
                    if isinstance(reasoning_response, dict):
                        error_type = reasoning_response.get("error", "unknown_error")
                        error_message = reasoning_response.get("message", "Unknown error occurred")
                        error_msg = f"Reasoning agent returned error: {error_type} - {error_message}"
                    self.log(f"{error_msg}", "ERROR")
                    local_memory.update_error(error_msg)
                    local_memory_list.append(local_memory.get_action())
                    state = False
                    continue
                
                """
                Calling sub-agent to generate tool calls 
                """
                self.log(f"Start Generator agent to generate tool calls", "INFO")
                available_tools = self.get_tools_by_agent(sub_agent)
                generator_response = self.generator_agent.generate_tool_calls(sub_problem, sub_agent, available_tools, local_memory.get_action(), supporting_documents, image_path, file_path)
                if generator_response is None:
                    self.log(f"Generator agent returned None - likely failed to generate tool calls", "ERROR")
                    local_memory.update_insufficient_info_reason("Generator agent failed to produce tool calls")
                    continue
                if not isinstance(generator_response, dict) or "tool_calls" not in generator_response:
                    self.log(f"Generator agent response missing 'tool_calls' key. Response: {generator_response}", "ERROR")
                    local_memory.update_insufficient_info_reason(f"Generator agent response invalid: {generator_response}")
                    continue
                tool_calls = generator_response["tool_calls"]
                self.log(f"Generated Tool calls: {tool_calls}", "INFO")
                local_memory.update_tool_calls(tool_calls)

                """
                Calling executor agent to execute tool calls
                """
                self.log(f"Start Executor agent to execute tool calls", "INFO")
                result, fail_executions = self.execute_tool(tool_calls, sub_problem)
                self.log(f"Executed Tool calls: {result}", "INFO")
                error_executions.extend(fail_executions)
                if self.count_tokens(str(result)) >= 110000:
                    self.log("Response exceeded model token limit, replaced with error message", "WARNING")
                    local_memory.update_tool_error("Response exceeded model token limit, replaced with error message")
                    local_memory_list.append(local_memory.get_action())
                    state = False
                    continue
                if "error" in result:
                    self.log(f"Tool execution failed: {result['error']}", "ERROR")
                    local_memory.update_tool_error(result['error'])
                    local_memory_list.append(local_memory.get_action())
                    state = False
                    continue
                local_memory.update_tool_result(result)
                
                """
                Calling verifier agent to verify the tool result
                """
                self.log(f"Start Verifier agent to verify the tool result", "INFO")
                verification_response = self.verifier_agent.verify_tool_result(sub_problem, result, sub_agent, tool_calls)
                
                # Check if verification_response is None (error case)
                if verification_response is None:
                    error_msg = "Verifier agent returned None - likely failed to generate verification response"
                    self.log(f"{error_msg}", "ERROR")
                    local_memory.update_tool_error(error_msg)
                    local_memory_list.append(local_memory.get_action())
                    state = False
                    continue
                
                # Check if verification_response is a dict with error (error case)
                if isinstance(verification_response, dict) and "error" in verification_response:
                    self.log(f"Verifier agent returned error: {verification_response['error']}", "ERROR")
                    local_memory.update_tool_error(verification_response['error'])
                    local_memory_list.append(local_memory.get_action())
                    state = False
                    continue
                
                verification = verification_response.verification
                verify_result = verification_response.summary_result
                self.log(f"Verified Tool result: {verification_response}", "INFO")
                if verification:
                    global_memory.update_step(sub_problem, verify_result)
                    self.log(f"Global memory updated: {global_memory.get_actions()}", "INFO")
                    local_memory_list.append(local_memory.get_action())
                    local_memory = Local_Memory()
                    state = True
                    continue
                else:
                    reason = verification_response.reason
                    suggestion = verification_response.suggestion
                    local_memory.update_verification_failure(verification, reason, suggestion)
                    local_memory_list.append(local_memory.get_action())
                    state = False
                    continue
            self.log(f"OpenTools reasoning and acting loop interrupted due to time limit", "INFO")
            result_data = self.prepare_result_data(question, image_path, file_path, global_memory.get_actions(), step_count,
                    self.get_agent_name(), time.time() - start_time, datetime.now().isoformat(), cache_dir, "interrupted",
                    self.llm_engine.get_token_usage(), error_executions, local_memory_list)
            result_data["direct_output"] = self.generate_direct_output(question, image_path, global_memory.get_actions())
            result_data["summary"] = "No summary available because the loop was interrupted"
            
            result_file = os.path.join(cache_dir, "result.json")
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
            token_usage = self.llm_engine.get_token_usage()
            if token_usage:
                self.log(f"Token Usage Summary:")
                self.log(f"  Total tokens: {token_usage.get('total_tokens', 0)}")
                self.log(f"  Prompt tokens: {token_usage.get('total_prompt_tokens', 0)}")
                self.log(f"  Completion tokens: {token_usage.get('total_completion_tokens', 0)}")
                self.log(f"  API calls: {token_usage.get('call_count', 0)}")  
            if hasattr(self.llm_engine, "reset_token_counters"):
                try:
                    self.llm_engine.reset_token_counters()
                except Exception:
                    pass
            return result_data
        except Exception as e:
            self.log(f"Error in OpenTools reasoning and acting loop: {e}", "ERROR")
            print(f"Traceback: {traceback.format_exc()}")
            print(f"Error: {e}")
            return None
if __name__ == "__main__":
    agent = OpenToolsAgent(llm_engine_name="gpt-4o-mini", enabled_tools=["Generalist_Solution_Generator_Tool"])
    result = agent.solve(question="What is the capital of Vietnam?")
    print(result)
