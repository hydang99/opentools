import os
import importlib
import re
from typing import Dict, Any, List
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', '..')))
from .formatters import ToolCommand
from typing import Dict, Any, List, Optional

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

class Executor:
    def __init__(self, llm_engine: Any, llm_engine_name: str, root_cache_dir: str = "solver_cache",  num_threads: int = 1, max_time: int = 120, max_output_length: int = 100000, verbose: bool = False):
        self.llm_engine = llm_engine
        self.llm_engine_name = llm_engine_name
        self.root_cache_dir = root_cache_dir
        self.num_threads = num_threads
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.verbose = verbose

    def set_query_cache_dir(self, query_cache_dir):
        if query_cache_dir:
            self.query_cache_dir = query_cache_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_cache_dir = os.path.join(self.root_cache_dir, timestamp)
        os.makedirs(self.query_cache_dir, exist_ok=True)
    
    def generate_tool_command(self, question: str, image: str, context: str, sub_goal: str, tool_name: str, tool_metadata: Dict[str, Any]) -> Any:
        prompt_generate_tool_command = f"""
Task: Generate a precise command to execute the selected tool based on the given information.

Query: {question}
Image: {image}
Context: {context}
Sub-Goal: {sub_goal}
Selected Tool: {tool_name}
Tool Metadata: {tool_metadata}

Instructions:
1. Carefully review all provided information: the query, image path, context, sub-goal, selected tool, and tool metadata.
2. Analyze the tool's parameters from the metadata to understand required and optional parameters.
3. Construct a command or series of commands that aligns with the tool's usage pattern and addresses the sub-goal.
4. Ensure all required parameters are included and properly formatted.
5. Use appropriate values for parameters based on the given context, particularly the `Context` field which may contain relevant information from previous steps.
6. If multiple steps are needed to prepare data for the tool, include them in the command construction.

Output Format:
Provide your response in the following structure:

Analysis: <analysis>
Command Explanation: <explanation>
Generated Command:
```python
<command>
```

Where:
- <analysis> is a step-by-step analysis of the context, sub-goal, and selected tool to guide the command construction.
- <explanation> is a detailed explanation of the constructed command(s) and their parameters.
- <command> is the Python code to run the tool, which can be one of the following types:
    a. A single line command with execution = tool.run() (PREFERRED).
    b. Variable preparation followed by a single-line execution = tool.run() call (PREFERRED for complex cases).
    c. Multiple lines of execution = tool.run() calls for processing multiple items.
    IMPORTANT: The `execution = tool.run(...)` call itself must always be on a SINGLE LINE. Never put newlines inside the parentheses of tool.run().

Rules:
0. Remember to include all required field by the function. If the funtion say input/query is required then remember to always include it with non empty parameters.
1. The command MUST be valid Python code and include at least one call to `tool.run()`.
2. Each `tool.run()` call MUST be assigned to the 'execution' variable in the format `execution = tool.run(...)`.
3. For multiple executions, use separate `execution = tool.run()` calls for each execution.
4. The final output MUST be assigned to the 'execution' variable, either directly from `tool.run()` or as a processed form of multiple executions.
5. Use the exact parameter names as specified in the tool's parameters.
6. Enclose string values in quotes, use appropriate data types for other values (e.g., lists, numbers).
7. Do not include any code or text that is not part of the actual command.
8. Ensure the command directly addresses the sub-goal and query.
9. Include ALL required parameters, data, and paths to run the tool in the command itself.
10. If preparation steps are needed, include them as separate Python statements before the `tool.run()` calls.
11. If there is image path or url path, please use '/' forward-slashes  instead of backward. 
12. Remember to use the absolute path of the image or url which was given in the query.
13. **CRITICAL: The `execution = tool.run(...)` call MUST be on a SINGLE LINE. Do NOT split it across multiple lines with newlines inside the parentheses. If you need to prepare variables first, define them separately, then call `tool.run()` on a single line. For example, use `image_path = "path"; prompt = "text"; execution = tool.run(image_path=image_path, prompt=prompt)` instead of multi-line format.**
Examples (Not to use directly unless relevant):

Example 1 (Single line command):
Analysis: The tool requires an image path and a list of labels for object detection.
Command Explanation: We pass the image path and a list containing "baseball" as the label to detect.
Generated Command:
```python
execution = tool.run(query= "Given query", image = "path/to/image")
execution = tool.run(image="path/to/image", labels=["baseball"])
```

Example 2 (Data preparation with single-line tool.run() call):
Analysis: The tool requires an image path, multiple labels, and a threshold for object detection.
Command Explanation: We prepare the data by defining variables for the image path, labels, and threshold, then pass these to the tool.run() function on a single line.
Generated Command:
```python
image = "path/to/image"
labels = ["baseball", "football", "basketball"]
threshold = 0.5
execution = tool.run(image=image, labels=labels, threshold=threshold)
```
Note: The `execution = tool.run(...)` call is on a SINGLE LINE even though variables were prepared separately.

Example 3 (Multiple executions):
Analysis: We need to process multiple images for baseball detection.
Command Explanation: We call the tool for each image path, using the same label and threshold for all.
Generated Command:
```python
execution = tool.run(image="path/to/image1", labels=["baseball"], threshold=0.5)
execution = tool.run(image="path/to/image2", labels=["baseball"], threshold=0.5)
execution = tool.run(image="path/to/image3", labels=["baseball"], threshold=0.5)
```


Generated Command:
```python
urls = [
    "https://example.com/article1",
    "https://example.com/article2"
]

execution = tool.run(url=urls[0])
execution = tool.run(url=urls[1])
```
Reason: The command should process multiple items in a single execution, not separate executions for each item.

Remember: Your response MUST end with the Generated Command, which should be valid Python code including any necessary data preparation steps and one or more `execution = tool.run(` calls, without any additional explanatory text. The format `execution = tool.run` must be strictly followed, and the last line must begin with `execution = tool.run` to capture the final output."""

        tool_command = self.llm_engine.generate(prompt_generate_tool_command)
        if isinstance(tool_command, dict):
            tool_command = tool_command.get('text')
        else:
            tool_command = str(tool_command)
        return tool_command

    def extract_explanation_and_command(self, response: Any) -> tuple:
        def normalize_code(code: str) -> str:
            # Remove leading and trailing whitespace and triple backticks
            return re.sub(r'^```python\s*', '', code).rstrip('```').strip()
        
        if isinstance(response, ToolCommand):
            analysis = response.analysis.strip()
            explanation = response.explanation.strip()
            command = response.command.strip()
        else:
            # Extract analysis
            analysis_pattern = r"Analysis:(.*?)Command Explanation"
            analysis_match = re.search(analysis_pattern, response, re.DOTALL)
            analysis = analysis_match.group(1).strip() if analysis_match else "No analysis found."
            # Extract explanation
            explanation_pattern = r"Command Explanation:(.*?)Generated Command"
            explanation_match = re.search(explanation_pattern, response, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation found."
            # Extract command
            command_pattern = r"Generated Command:.*?```python\n(.*?)```"
            command_match = re.search(command_pattern, response, re.DOTALL)
            command = command_match.group(1).strip() if command_match else "No command found."
            print(command)

        command = normalize_code(command)

        return analysis, explanation, command

    def execute_tool_command(self, tool_name: str, command: str) -> Any:
        """
        Execute a tool command with timeout protection. If execution exceeds max_time seconds,
        the function will be interrupted and return a timeout message.
        
        Args:
            tool_name (str): Name of the tool to execute
            command (str): Command string containing tool.run() calls

        Returns:
            Any: List of execution results or error message
        """
        
        def split_commands(command: str) -> List[str]:
            # First try to find single-line tool.run() commands (preferred format)
            single_line_pattern = r'.*?execution\s*=\s*tool\.run\([^\n]*\)\s*(?:\n|$)'
            blocks = re.findall(single_line_pattern, command, re.DOTALL)
            
            # If no single-line matches found, try to handle multi-line cases
            if not blocks:
                # Find tool.run( and match until balanced closing parenthesis
                multi_line_pattern = r'execution\s*=\s*tool\.run\('
                matches = list(re.finditer(multi_line_pattern, command, re.MULTILINE))
                
                for match in matches:
                    start = match.start()
                    # Find matching closing parenthesis by counting parentheses
                    paren_count = 0
                    in_string = False
                    string_char = None
                    escape_next = False
                    i = match.end() - 1  # Position of opening '('
                    
                    while i < len(command):
                        char = command[i]
                        
                        if escape_next:
                            escape_next = False
                            i += 1
                            continue
                        
                        if char == '\\':
                            escape_next = True
                            i += 1
                            continue
                        
                        if not in_string:
                            if char in ('"', "'"):
                                in_string = True
                                string_char = char
                            elif char == '(':
                                paren_count += 1
                            elif char == ')':
                                paren_count -= 1
                                if paren_count == 0:
                                    # Found matching closing parenthesis
                                    end = i + 1
                                    # Include any trailing whitespace/newline
                                    while end < len(command) and command[end] in (' ', '\t', '\n'):
                                        end += 1
                                    # Extract the full block from start to end
                                    block = command[start:end].strip()
                                    if block:
                                        blocks.append(block)
                                    break
                        else:
                            if char == string_char:
                                in_string = False
                                string_char = None
                        
                        i += 1
            
            return [block.strip() for block in blocks if block.strip()]
        
        def execute_with_timeout(block: str, local_context: dict) -> Optional[str]:
            # Set up the timeout handler
            # signal.signal(signal.SIGALRM, timeout_handler)
            # signal.alarm(self.max_time)
            
            try:
                # Execute the block in the local context
                exec(block, globals(), local_context)
                result = local_context.get('execution')
                # signal.alarm(0)  # Disable the alarm
                return result
            except TimeoutError:
                return f"Execution timed out after {self.max_time} seconds"
            finally:
                # signal.alarm(0)  # Ensure alarm is disabled even if other exceptions occur
                pass
        # Import the tool module and instantiate it
        module_name = f"opentools.tools.{tool_name.lower().replace('_tool', '')}.tool"

        try:
            # Dynamically import the module
            module = importlib.import_module(module_name)

            # Get the tool class
            tool_class = getattr(module, tool_name)

            # Check if the tool requires an LLM engine
            # NOTE may need to refine base.py and tool.py to handle this better
            if getattr(tool_class, 'require_llm_engine', False):
                # Instantiate the tool with the model_string
                tool = tool_class(model_string=self.llm_engine_name)
            else:
                # Instantiate the tool without model_string for tools that don't require it
                tool = tool_class()
            
            # Set the custom output directory
            # NOTE: May have a better way to handle this
            tool.set_custom_output_dir(self.query_cache_dir)

            # Split the command into blocks, execute each one and store execution results
            command_blocks = split_commands(command)
            executions = []

            for block in command_blocks:
                # Create a local context to safely execute the block
                local_context = {'tool': tool}

                # Execute the block with timeout protection
                result = execute_with_timeout(block, local_context)
                
                if result is not None:
                    executions.append(result)
                else:
                    executions.append(f"No execution captured from block: {block}")

            # Return all the execution results
            return executions
        except Exception as e:
            print(e)
            return f"Error in execute_tool_command: {str(e)}"
    
    def generate_and_execute_tool(self, question: str, image: str, context: str, sub_goal: str, tool_name: str, tool_metadata: Dict[str, Any]) -> Any:
        """Generate JSON parameters and execute a tool in one step (new cleaner approach)"""
        try:
            # Import the JSON tool executor
            from opentools.core.json_tool_executor import JSONToolExecutor
            
            # Create a JSON executor
            json_executor = JSONToolExecutor(verbose=self.verbose)
            
            # Get the tool instance
            tool_instance = self.get_tool_instance(tool_name)
            if not tool_instance:
                return f"Tool {tool_name} not found"
            
            # Generate parameters using LLM
            parameters = self.generate_tool_command(question, image, context, sub_goal, tool_name, tool_metadata)
            
            # Execute the tool with the generated parameters
            result = json_executor.execute_tool_with_json(tool_instance, parameters)
            
            if result.get('success', False):
                return result.get('result', 'No result')
            else:
                return f"Tool execution failed: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"Error in generate_and_execute_tool: {str(e)}"