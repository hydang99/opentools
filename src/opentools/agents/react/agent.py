import os, json, time, re, sys, traceback, tiktoken, ast
from typing import Dict, Any, List, Optional
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.agents.tool_based_agent import ToolBasedAgent
from opentools.core.factory import create_llm_engine

"""
ReAct (Reasoning + Acting) Agent implementation.
This agent alternates between reasoning and taking actions with tools.
"""
class ReActAgent(ToolBasedAgent):
    """
    ReAct agent that alternates between reasoning (thinking) and acting (using tools).
    
    The agent follows this pattern:
    1. Thought: Reason about what to do next
    2. Action: Take an action using a tool
    3. Observation: Observe the result
    4. Thought: Reason about the observation
    5. Repeat until the task is complete
    """
    # Class-level metadata
    AGENT_NAME = "ReAct"
    AGENT_DESCRIPTION = "Reasoning and Acting agent - alternates between thinking and tool usage"
    
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
                 device: Optional[str] = None,
                 **kwargs):
        """
        Initialize the ReAct agent.
        
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
            **kwargs: Additional keyword arguments
        """

        # Setup ReAct-specific reasoning components
        self.setup_reasoning_components(llm_engine_name)
        self.model_family = model_family
        self.log(f"Enable FAISS retrieval: {enable_faiss_retrieval} at ReAct", "INFO")
        # Initialize the tool-based agent (handles tool setup)
        super().__init__(
            llm_engine_name=llm_engine_name,
            model_family=model_family,
            enabled_tools=enabled_tools,
            num_threads=num_threads,
            max_time=max_time,
            max_output_length=max_output_length,
            vllm_config_path=vllm_config_path,
            verbose=verbose,
            enable_faiss_retrieval=enable_faiss_retrieval,
            **kwargs
        )

    def setup_reasoning_components(self, llm_engine_name: str):
        """Setup ReAct-specific reasoning components"""
        self.log("Initializing ReAct reasoning components 🧠...")
        self.llm_engine_name = llm_engine_name
        self.llm_engine = create_llm_engine(model_string=llm_engine_name, is_multimodal=True)        
        self.log("ReAct reasoning components initialized successfully 🧠")
    
    def get_agent_name(self) -> str:
        """Return the name of this agent"""
        return "ReAct"
    
    def get_agent_description(self) -> str:
        """Return a description of this agent"""
        return "Reasoning and Acting agent - alternates between thinking and tool usage"

    def _supports_structured_tool_calls(self) -> bool:
        """Check if the current model family natively supports tool call payloads."""
        return (self.model_family or "").lower() in {"openai"}

    def create_react_prompt(self, question: str, image_path: str,file_path: str, available_tools=None, trace: dict = None) -> str:
        """Create a ReAct prompt, adapting format when the model lacks tool-call support."""
        trace = trace or {}
        try:
            trace_json = json.dumps(trace)
        except Exception as e:
            trace_json = trace
        self.log(f'Tracing from previous steps: {trace_json}', "DEBUG")
        if self._supports_structured_tool_calls():
            base_prompt = f"""
            Question: {question}
            Image (if any): {image_path}
            File (if any): {file_path}
            You should follow this exact format:
            Thought: [your reasoning about current step and what to do next.]
            Action: [The action to take in this step]

            When you have enough information to answer the question, return the Final Answer following:
            Final Answer: [your final answer]

            Important:
            - This Thought/Action can repeat as needed.
            - Observation in current trace is the result of the action taken in the previous step.
            - End with "Final Answer:" when you're done
            - Never use placeholder values in arguments; always use real values.
            - Once a tool returns the metric you need, DO NOT call any tool again; restate the key answers and keep reasoning without calling that same tool/arguments again.
            - Only call a tool again if you change the arguments to obtain new information; otherwise switch tools or answer directly using existing observations.
            - For multiple-choice questions, compute the required value, match it to the listed options, and make your last line exactly `Answer: LETTER`.
            - CRITICAL: If previous steps resulted in an error mentioning "repeated" or "same tool command", you MUST NOT try the same tool command again—use the information already gathered or try a different approach.
            Current trace:
            {trace_json}
            """
            self.log(f"ReAct prompt: {base_prompt}", "DEBUG")
            return base_prompt

        manual_prompt = self._create_manual_react_prompt(question, image_path,file_path, available_tools, trace_json)
        self.log(f"ReAct prompt (manual tools): {manual_prompt}", "DEBUG")
        return manual_prompt

    def _create_manual_react_prompt(self, question: str, image_path: str,file_path: str, available_tools=None, trace_json: str = None) -> str:
        """Prompt template that inlines tool metadata for models without tool-call fields."""
        base_prompt = f"""
        Question: {question}
        Available Tools: {available_tools}
        Image (if any): {image_path}
        File (if any): {file_path}
        You should follow this exact format:
        Thought: [your reasoning about what to do next. If any tool or set of tools could help to perform this, use it in Action]
        Action: [tool_name or list of tools] 
        Tool Use: [specific TOOLUSE for executing the tool]
        Observation: [result will be provided after using the tool]

        When you have enough information to answer the question (observations), put Action: [STOP] and put Final Answer following:
        Action: [STOP]
        Final Answer: [your final answer]

        Important:
        - This Thought/Action/Tool Use/Observation can repeat as needed.
        - Always start with "Thought:" to reason about the problem, analyze if any tool could help to solve the problem, or provide information to solve it. If a tool can help to solve, include it in the Action section. 
        - Use "Action:" to specify which tool to use (must be from Available Tools)
        - Use "Tool Use:" to specify a list of function_call dictionaries following JSON format to execute the tool:
            Example Format: 
        [
            {{
                "type": "function",
                "name": "[Tool's name from Available Tools/Action]",
                "arguments": "[Arguments in Json Format]"
            }}, 
            {{
                "type": "function",
                "name": "[Tool's name from Available Tools/Action]",
                "arguments": "[Arguments in Json Format]"
            }}, 
        ]
            a. The name in Tool Use MUST match exactly (case-sensitive) one of the Available Tools' "name" values. 
            b. The "arguments" in Tool Use MUST conform to that tool's "parameters" schema: use only keys defined under "properties", include all keys listed in "required", and ensure each value matches the declared type (string, integer, number, boolean, object, array).
            c. Use appropriate input/values and based on the context.
        - Wait for "Observation:" from the tool execution before continuing
        - End with "Final Answer:" when you're done collecting all evidence and Action is [STOP]
        - If you wish to use the default values for the arguments provided in the tool metadata, you can omit the arguments key in the tool command.
        - Never put documentation or notes inside the Tool Use code. Never use placeholder values in Tool Use; always use real values.
        - CRITICAL: If previous steps resulted in an error mentioning "repeated" or "same tool command", you MUST NOT try the same tool command again. The previous attempts already failed with the same query. You MUST either: (1) Use a COMPLETELY DIFFERENT search query with different keywords/phrasing, (2) Use a DIFFERENT tool that can solve the problem, or (3) Work with the information already gathered. Repeating the exact same command will only waste steps and lead to failure.
        - After a tool returns the metric you need, summarize it once and continue reasoning; do not re-run the identical command unless the arguments change to request new data.
        - If the query asks for multiple choice, compute the value, map it to the matching option, and only provide the letter (e.g., "A", "B"). The final line must be exactly 'Answer: LETTER'.
        - If a tool already returned a successful result with the needed metric, do not call any tool again; use that result to pick the choice letter and answer.
        Current trace:
        {trace_json}
        """

        return base_prompt
    def generate_direct_output(self, question: str, image_path: str, trace=  {}) -> str:
        try:
            trace_json = json.dumps(trace)
        except Exception as e:
            trace_json = trace
        prompt_generate_final_output = f"""
Context:
Query: {question}
Image: {image_path}
Trace:
{trace_json}
Based on the query and the tool results above, provide ONLY the direct answer to the question. 
Extract the key information from the tool results and provide a simple, direct answer without any explanations, analysis, or additional context.
Answer:
"""
        if image_path:
            final_output = self.llm_engine.generate(prompt_generate_final_output, image=image_path)
        else:   
            final_output = self.llm_engine.generate(prompt_generate_final_output)
        self.log(f'Direct output: {final_output}')
        if isinstance(final_output, dict):
            final_output = final_output.get("text")
        else:
            final_output = str(final_output)
        return final_output

    def count_tokens(self, text):
        """
        Counts the number of tokens in a string using tiktoken if available, otherwise estimates by length.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            token_count = len(encoding.encode(text))
            return token_count
        except Exception:
            return len(text) // 4

    def parse_react_response(self, response: str) -> Dict[str, Any]:
        """Parse the response to extract the final answer"""
        result = {
            "final_answer": "",
            "is_final": False
        }
        if not response or not isinstance(response, str):
            return result
        final_answer_match = re.search(r"Final Answer:\s*(.+?)$", response, re.DOTALL)
        if final_answer_match:
            result["final_answer"] = final_answer_match.group(1).strip()
            result["is_final"] = True
            return result
        return result

    def _parse_manual_react_response(self, response: str) -> Dict[str, Any]:
        """Parse manual ReAct text output into structured fields."""
        result = {
            "thought": "",
            "action": "",
            "tool_command": "",
            "tool_calls": None,
            "final_answer": "",
            "is_final": False
        }
        if not response or not isinstance(response, str):
            return result

        thought_matches = re.findall(r"Thought:\s*(.+?)(?=\n(?:Action:|Final Answer:)|$)", response, re.DOTALL)
        if thought_matches:
            result["thought"] = thought_matches[-1].strip()

        final_answer_match = re.search(r"Final Answer:\s*(.+?)$", response, re.DOTALL)
        if final_answer_match:
            result["final_answer"] = final_answer_match.group(1).strip()
            result["is_final"] = True

        action_match = re.search(r"Action:\s*(.+?)(?=\n|$)", response)
        if action_match:
            result["action"] = action_match.group(1).strip()

        tool_command = re.search(r"Tool Use:\s*(.+?)(?=\n(?:Observation:|Thought:|Action:|Final Answer:)|$)", response, re.DOTALL)
        if tool_command:
            cleaned_command = tool_command.group(1).strip()

            code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned_command, re.DOTALL)
            if code_block_match:
                cleaned_command = code_block_match.group(1).strip()

            cleaned_command = re.sub(r'^\*\*\s*', '', cleaned_command)
            cleaned_command = re.sub(r'\s*\*\*$', '', cleaned_command)
            cleaned_command = re.sub(r'\*\*Observation:.*$', '', cleaned_command, flags=re.DOTALL)
            cleaned_command = re.sub(r'Observation:.*$', '', cleaned_command, flags=re.DOTALL)
            cleaned_command = re.sub(r'//.*?$', '', cleaned_command, flags=re.MULTILINE)
            cleaned_command = re.sub(r'\n\s*\n', '\n', cleaned_command).strip()

            result["tool_command"] = cleaned_command
            try:
                result["tool_calls"] = self._parse_tool_calls_from_block(cleaned_command)
            except Exception as parse_err:
                self.log(f"Failed to parse manual tool command: {parse_err}", "ERROR")

        return result

    def _parse_tool_calls_from_block(self, tool_block: str) -> Optional[List[Dict[str, Any]]]:
        """Convert a Tool Use JSON/text block into executable call dictionaries."""
        if not tool_block:
            return None

        parsed_block: Any
        try:
            parsed_block = json.loads(tool_block)
        except json.JSONDecodeError:
            parsed_block = ast.literal_eval(tool_block)

        if isinstance(parsed_block, dict):
            parsed_block = [parsed_block]

        if not isinstance(parsed_block, list):
            raise ValueError("Tool Use command must be a JSON object or list.")

        normalized_calls: List[Dict[str, Any]] = []
        for item in parsed_block:
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except Exception:
                    item = ast.literal_eval(item)
            if not isinstance(item, dict):
                raise ValueError("Each Tool Use entry must be a JSON object.")
            normalized_calls.append(item)
        return normalized_calls

    def _get_token_usage(self) -> Dict[str, Any]:
        """Safely fetch token usage if the engine supports it."""
        if hasattr(self.llm_engine, "get_token_usage"):
            try:
                return self.llm_engine.get_token_usage() or {}
            except Exception:
                return {}
        return {}

    def solve(self,
              question: str,
              image_path: Optional[str] = None,
              file_path: Optional[str] = None,
              max_steps: int = 10,
              max_time: int = 300,
              max_tokens: int = 4000,
              root_cache_dir: str = "react_cache",
              output_types: str = "full",
              **kwargs) -> Dict[str, Any]:
        """
        Solve a question using the ReAct approach.
        
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
        error_executions = []

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
        self.executor.set_query_cache_dir(cache_dir)

        trace = {}
        tool_calling_history =[]
        supports_structured_tools = self._supports_structured_tool_calls()

        try:
            # Get enabled tools
            available_tools = self.get_relevant_tools(question)
            step_count = 0            
            self.log("Starting ReAct reasoning and acting loop 💭...")
            while step_count < max_steps and (time.time() - start_time) < max_time:
                step_count += 1
                self.log_step_start(f"ReAct Reasoning Cycle {step_count}")
                # Generate prompt
                prompt = self.create_react_prompt(question, image_path,file_path, available_tools, trace)
                # Get response from LLM (fall back to prompt-inlined tools when needed)
                try:
                    if supports_structured_tools:
                        response = self.llm_engine.generate(
                            prompt,
                            system_prompt="You are a helpful assistant that can reason and act to solve problems step by step",
                            tools=available_tools,
                            tool_choice="auto",
                            max_tokens=max_tokens
                        )
                    else:
                        response = self.llm_engine.generate(prompt, max_tokens=max_tokens)
                except Exception as e:
                    self.log(f"Error in generating response from LLM: {e}", "ERROR")
                    print(f"Trace: {traceback.format_exc()}")
                    trace[f"Agent Step {step_count}"] = {
                        "error": str(e),
                        "error_trace": traceback.format_exc(),
                        "error_prompt": prompt
                    }
                    continue
                self.log(f"Got response from LLM: {response}", "INFO")

                response_error = None
                response_text = ""
                response_tool_calls = None

                if isinstance(response, dict):
                    response_error = response.get("error")
                    response_text = response.get("text") or ""
                    response_tool_calls = response.get("tool_calls")
                else:
                    response_text = str(response) if response is not None else ""

                if response_error:
                    self.log(f"LLM returned error: {response_error}", "ERROR")
                    trace[f"Agent Step {step_count}"] = {
                        "error": response_error
                    }
                    continue
                if response_text and self.count_tokens(response_text) >= 128000:
                    self.log("Response exceeded model token limit, replaced with error message", "WARNING")
                    trace[f"Agent Step {step_count}"] = {
                        "error": "The response is too large. Please try another tool or simplify this part of reasoning to stay within model limits."
                    }
                    continue
                # Check if the response reached the final answer
                step_key = f"Agent Step {step_count}"
                if supports_structured_tools:
                    parsed = self.parse_react_response(response_text)
                else:
                    parsed = self._parse_manual_react_response(response_text)
                    if parsed.get("tool_calls"):
                        response_tool_calls = parsed["tool_calls"]

                is_final_ready = parsed.get("final_answer") and (supports_structured_tools or parsed.get("is_final"))
                if is_final_ready:
                    try:
                        self.log(f"Reach final answer\n", "INFO")
                        self.log_final_answer(parsed["final_answer"])
                        token_usage = self._get_token_usage()
                        direct_output = parsed.get("final_answer")
                        trace[f"Agent Step {step_count}"] = {
                            "final_response_from_llm": response_text,
                            "final_answer": direct_output
                        }
                        if 'direct' in output_types or direct_output is None:
                            direct_output = self.generate_direct_output(question, image_path, trace)      
                        result_data = {
                            "query": question,
                            "image": image_path,
                            "steps_taken": step_count,
                            "agent": self.get_agent_name(),
                            "llm_engine": self.llm_engine_name,
                            "reasoning_trace": trace,
                            "token_usage": token_usage,
                            "error_executions": error_executions,
                            "execution_time": time.time() - start_time,
                            "timestamp": datetime.now().isoformat(),
                            "cache_dir": cache_dir,
                        }
                    except Exception as e:
                        self.log(f"Error in getting final answer from ReAct reasoning and acting loop: {e}", "ERROR")
                        result_data = {
                            "error": str(e),
                            "error_trace": traceback.format_exc(),
                            "query": question,
                            "image": image_path,
                            "reasoning_trace": trace,
                            "steps_taken": step_count,
                            "agent": self.get_agent_name(),
                            "llm_engine": self.llm_engine_name,
                            "execution_time": time.time() - start_time,
                            "timestamp": datetime.now().isoformat(),
                            "cache_dir": cache_dir,
                            "status": "incomplete"
                        }
                    if 'final' in output_types:
                        result_data["final_output"] = parsed["final_answer"] if "final_answer" in parsed else response_text
                    elif 'direct' in output_types:
                        result_data["direct_output"] = direct_output
                    else:
                        result_data["full_answer"] = parsed["final_answer"] if "final_answer" in parsed else response_text

                    # Ensure a default direct_output is always available for callers
                    if "direct_output" not in result_data:
                        if "final_output" in result_data:
                            result_data["direct_output"] = result_data["final_output"]
                        elif "full_answer" in result_data:
                            result_data["direct_output"] = result_data["full_answer"]
                    # Save result
                    result_file = os.path.join(cache_dir, "result.json")
                    with open(result_file, 'w') as f:
                        json.dump(result_data, f, indent=2, default=str)
                    
                    # End the final step and display summary
                    self.log_step_end("Final answer reached", success=True)
                    # Log token usage summary
                    if token_usage:
                        self.log(f"Token Usage Summary:")
                        self.log(f"  Total tokens: {token_usage.get('total_tokens', 0)}")
                        self.log(f"  Prompt tokens: {token_usage.get('total_prompt_tokens', 0)}")
                        self.log(f"  Completion tokens: {token_usage.get('total_completion_tokens', 0)}")
                        self.log(f"  API calls: {token_usage.get('call_count', 0)}")                    
                    self.log(f"ReAct completed in {result_data['execution_time']:.2f} seconds")
                    if hasattr(self.llm_engine, "reset_token_counters"):
                        try:
                            self.llm_engine.reset_token_counters()
                        except Exception:
                            pass
                    self.display_summary()
                    
                    return result_data
                    
                trace_entry = {}
                if response_tool_calls:
                    trace_entry["tool_calls"] = response_tool_calls
                if response_text:
                    trace_entry["response_text"] = response_text
                if not supports_structured_tools:
                    if parsed.get("thought"):
                        trace_entry["thought"] = parsed["thought"]
                    if parsed.get("action"):
                        trace_entry["action"] = parsed["action"]
                trace[step_key] = trace_entry
                # Execute action if specified with tool command
                observation = None
                tool_calls = response_tool_calls
                if tool_calls:
                    try:
                        if isinstance(tool_calls, str):
                            loggable_command = tool_calls
                        else:
                            try:
                                loggable_command = json.dumps(tool_calls, ensure_ascii=False)
                            except TypeError:
                                loggable_command = str(tool_calls)
                        self.log_tool_command(loggable_command)
                        if not isinstance(tool_calls, list):
                            tool_calls = [tool_calls]            
                        # Check for repetition: if same command appears in last 5 steps
                        if step_count >= 5 and len(tool_calling_history) >= 5:
                            # Check if current tool_calls match any of the last 5
                            recent_history = tool_calling_history[-5:]
                            repetition_count = sum(1 for prev_calls in recent_history if tool_calls == prev_calls)
                            if repetition_count >= 4:  # Current call would be the 5th repetition
                                observation = "Error: The same tool command is repeated 5 times. Force stop the loop."
                                error_executions.append({
                                    "index": step_count,
                                    "error": observation,
                                    "raw": tool_calls,
                                    "ok": False
                                })
                                result_data = {
                                    "query": question,
                                    "image": image_path,
                                    "reasoning_trace": trace,
                                    "steps_taken": step_count,
                                    "agent": self.get_agent_name(),
                                    "llm_engine": self.llm_engine_name,
                                    "execution_time": time.time() - start_time,
                                    "timestamp": datetime.now().isoformat(),
                                    "cache_dir": cache_dir,
                                    "status": "incomplete",
                                    "token_usage": self.llm_engine.get_token_usage() if hasattr(self.llm_engine, 'get_token_usage') else {},
                                    "error_executions": error_executions,
                                    "Error": observation
                                }
                                output = self.generate_direct_output(question, image_path, trace)
                                if 'direct' in self.output_types:
                                    result_data["direct_output"] = output
                                elif 'final' in self.output_types:
                                    result_data["final_output"] = output
                                else:
                                    result_data[f"full_answer"] = output
                                self.log(f"ReAct stopped due to repetition: {observation}", "ERROR")
                                return result_data
                                    
                        # Check for immediate repetition (same as last 2 steps)
                        elif step_count > 2 and len(tool_calling_history) >= 2 and tool_calls == tool_calling_history[-2] and tool_calls == tool_calling_history[-1]:
                            observation = "Error: The same tool command is repeated 2 times without any progress."
                            error_executions.append({
                                "index": step_count,
                                "error": observation,
                                "raw": tool_calls,
                                "ok": False
                            })
                        else:
                            observation, fail_executions = self.execute_tool(tool_calls, response_text)
                            if self.count_tokens(observation) >= 128000:
                                self.log("Response exceeded model token limit, replaced with error message", "WARNING")
                                trace[f"Agent Step {step_count}"] = {
                                    "observation": "The response is too large and exceeded the model token limit. Please try another tool or simplify this part of reasoning to stay within model limits."
                                }
                                continue
                            error_executions.extend(fail_executions)
                            # Sanitize observation to prevent JSON serialization errors
                            tool_calling_history.append(tool_calls)
                    except (json.JSONDecodeError, ValueError, TypeError) as parse_err:
                        self.log(f"Tool command parsing failed: {parse_err}", "ERROR")
                        self.log(f"Raw tool command: {tool_calls}", "DEBUG")
                        observation = f"Error parsing tool command: {str(parse_err)}"
                        error_executions.append({
                            "index": step_count,
                            "error": observation,
                            "raw": tool_calls,
                            "ok": False
                        })
                        self.log(f"Error parsing tool command: {str(parse_err)}", "ERROR")
                    except Exception as e:
                        observation = f"Error executing tool: {str(e)}"
                        self.log(f"Tool execution error: {observation}", level="ERROR")
                else:
                    observation = "No tool calls provided in model response."
                trace[step_key]["observation"] = observation

                self.log_observation(str(observation))

                       
        except Exception as e:
            self.log(f"Error in ReAct reasoning and acting loop: {e}", "ERROR")
            return {
                "error": str(e),
                "error_trace": traceback.format_exc(),
                "query": question,
                "image": image_path,
                "reasoning_trace": trace,
                "steps_taken": step_count,
                "agent": self.get_agent_name(),
                "llm_engine": self.llm_engine_name,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "cache_dir": cache_dir,
                "status": "incomplete"
            }


        result_data = {
            "query": question,
            "image": image_path,
            "reasoning_trace": trace,
            "steps_taken": step_count,
            "agent": self.get_agent_name(),
            "llm_engine": self.llm_engine_name,
            "execution_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "cache_dir": cache_dir,
            "status": "incomplete",
            "token_usage": self._get_token_usage(),
            "error_executions": error_executions,
        }
        if 'final' in output_types:
            result_data["final_output"] = "No final answer found"
        elif 'direct' in output_types:
            result_data["direct_output"] = self.generate_direct_output(question, image_path, trace)
        else:
            result_data["full_answer"] = trace

        # Keep a default direct_output for downstream consumers
        if "direct_output" not in result_data:
            if "final_output" in result_data:
                result_data["direct_output"] = result_data["final_output"]
            elif "full_answer" in result_data:
                result_data["direct_output"] = result_data["full_answer"]
        self.log(f"ReAct completed without final answer in {result_data['execution_time']} seconds")
        return result_data

if __name__ == "__main__":
    try:
        agent = ReActAgent(llm_engine_name="gpt-4o-mini", verbose=False, enable_faiss_retrieval=False, enabled_tools=["URL_Text_Extractor_Tool","Maze_Solving_Tool"])
        query = """ You'll be given an image, a question, and some choices. You have to select the correct one. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.\nThis is maze having 13 * 11 cells. The empty cells are coloured white and the obstacle cells are coloured black. From an empty cell, you can only move up, down, left, or right to another adjacent empty cell. You cannot move diagonally between two empty cells and cannot step into a cell with an obstacle. The entry cell of the maze is shown with the green arrow. The exit cell of the maze is shown with the blue arrow. Suppose you have found the most optimal path in the maze between the entrance and exit, where you need to go through the least number of empty cells and you need to make the least number of left and right turns. What is the total number of left turns do you need to make in this optimal path?\nA. 1\nB. 2\nC. 3\nD. 5"""
        image_path = r"/home/daoqm/opentools/src/opentools/Benchmark/algopuzzlevqa/images/maze_0000.jpg"
        result = agent.solve(question=query, image_path=image_path)
        print(result)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
