# source code: https://github.com/inclusionAI/AWorld/blob/main/examples/gaia/mcp_collections/intelligence/code.py
"""
Compared to the source code, we improved the functionality to not only generate code, but also execute the generated code and return its results.
This enhancement allows for a seamless workflow: users can both synthesize new code and immediately run it for validation or output, rather than just generating code snippets for manual execution.
"""

import os, time, traceback, re, sys, signal, threading, platform, contextlib
from io import StringIO
from typing import Literal
from contextlib import contextmanager
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool

DEFAULT_CODE_EXECUTION_TIMEOUT_SECONDS = int(
    os.getenv("CODE_EXECUTION_TIMEOUT_SECONDS", "180")
)

# Custom exception for code execution timeout
class TimeoutException(Exception):
    pass

# Custom context manager for code execution timeout
@contextmanager
def timeout(seconds):
    def is_windows_os():
        system = platform.system()
        return system == 'Windows'
    
    if is_windows_os():
        # Windows timeout using threading.Timer
        def raise_timeout():
            raise TimeoutException("Code execution timed out")
        timer = threading.Timer(seconds, raise_timeout)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
    else:
        def timeout_handler(signum, frame):
            raise TimeoutException("Code execution timed out")

        # Set the timeout handler
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Restore the original handler and disable the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)

class Code_Generate_Execute_Tool(BaseTool):
    # Default args for `opentools test Code_Generate_Execute_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "python_code_generator",
        "file_location": "code_generate_execute",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }

    """Code_Generate_Execute_Tool
    ---------------------
    Purpose:
        A comprehensive code generation and execution tool that generates Python code based on task descriptions and automatically executes it in a safe environment. Supports various code styles, captures execution output and variables, and provides detailed metadata about the generation and execution process.

    Core Capabilities:
        - Python code generation using LLM models
        - Automatic code execution in isolated environment
        - Multiple code styles (minimal, documented, verbose)
        - Output and variable capture
        - Timeout protection (10 seconds)
        - Dangerous function filtering
        - Comprehensive metadata collection
        - Error handling and reporting
        - Safe execution environment with restricted access

    Intended Use:
        Use this tool when you need to generate and execute Python code based on task descriptions.

    Limitations:
        - May not handle complex code generation or execution tasks
    """


    def __init__(self, model_string="gpt-4o-mini", llm_engine=None):
        self.require_llm_engine = True
        self.code_execution_timeout_seconds = DEFAULT_CODE_EXECUTION_TIMEOUT_SECONDS
        super().__init__(
            type='function',
            name="Code_Generate_Execute_Tool",
            description="""A comprehensive code generation and execution tool that generates Python code based on task descriptions and automatically executes it in a 
            safe environment. Supports various code styles, captures execution output and variables, and provides detailed metadata about the generation and execution
            process. CAPABILITIES: Python code generation using LLM models, automatic code execution in isolated environment, multiple code styles (minimal, documented,
            verbose), output and variable capture, timeout protection (10 seconds), dangerous function filtering, comprehensive metadata collection, error handling and 
            reporting, safe execution environment with restricted access. SYNONYMS: code generator, Python code generator, code execution tool, programming assistant,
            code snippet generator, automated coding, code creation tool, programming tool. EXAMPLES: 'Generate code to calculate factorial of 5', 'Create a function
            to sort a list using bubble sort', 'Write code to find prime numbers up to 50', 'Generate documented code for data analysis'.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Description of the programming task or problem to solve - Required for all operations"
                    },
                    "requirements": {
                        "type": "string",
                        "description": "Specific requirements, constraints, or specifications for the code (optional)"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context or background information (optional)"
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=False,
            category="programming",
            tags=["code_generation", "python", "code_execution", "programming", "llm", "automated_coding", "code_snippets", "development_tools", "code_creation", "programming_assistant"],
            limitation="EXECUTION LIMITATIONS: Code execution is limited to 10 seconds timeout, cannot use external libraries or modules, no access to system resources, file operations, or network requests. SAFETY RESTRICTIONS: Dangerous functions (exit, quit, sys.exit) are automatically filtered, execution environment is isolated and restricted. LANGUAGE SUPPORT: Limited to Python code generation and execution only. DEPENDENCIES: Requires LLM engine configuration for code generation functionality.",
            agent_type="Mathematics-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(query='Generate code to calculate factorial of 5')",
                "description": "Generate code to calculate factorial of 5"
            },
            require_llm_engine=True,
            llm_engine=llm_engine,
        )

    def _prepare_code_prompt(self, query: str, requirements: str = "", context: str = "") -> str:
        """Prepare the code generation prompt with task description and optional requirements.

        Args:
            query: The main task for code generation
            requirements: Optional specific requirements or constraints
            context: Optional additional context or background information

        Returns:
            Formatted prompt string
        """
        prompt_parts = [f"Task: {query}"]

        if requirements:
            prompt_parts.append(f"Requirements: {requirements}")

        if context:
            prompt_parts.append(f"Context: {context}")

        return "\n\n".join(prompt_parts)

    def _call_code_model(self, prompt: str) -> str:
        """Call the code generation model with the prepared prompt.

        Args:
            prompt: The formatted prompt for code generation
            temperature: Model temperature for response variability

        Returns:
            Generated code from the model

        Raises:
            Exception: If model call fails
        """
        system_prompt = """You are an expert Python programmer. Generate clean, efficient, and well-documented Python code that solves the given task. 

IMPORTANT REQUIREMENTS:
0. Make sure to import necessary libraries and modules if used in the code (such as numpy, etc.).
1. Include proper error handling and follow Python best practices
2. The code MUST produce output when executed (use print statements to show results)
3. If you create functions, make sure to call them and print the results
4. Return only executable Python code with minimal explanatory comments
5. Ensure the code will run and produce visible output
6. Avoid using 'if __name__ == "__main__":' blocks unless absolutely necessary
7. Write code that executes directly when run

Here is the task: """
        
        response = self.llm_engine.generate(system_prompt + prompt)
        if isinstance(response, dict):
            response = response.get('text')
        else:
            response = str(response)

        return response

    def _extract_python_code(self, response: str) -> str:
        """Extract Python code from the model response.

        Args:
            response: Raw response from the model

        Returns:
            Extracted Python code
        """
        # Look for the first occurrence of a Python code block
        match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
        if not match:
            # If no code block found, try to extract any code-like content
            lines = response.strip().split("\n")
            start_idx = 0
            end_idx = len(lines)

            for i, line in enumerate(lines):
                if line.strip().startswith("```python") or line.strip().startswith("```"):
                    start_idx = i + 1
                    break

            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break

            # Extract the code
            code_lines = lines[start_idx:end_idx]
            return "\n".join(code_lines).strip()
        
        return match.group(1).strip()

    @contextlib.contextmanager
    def capture_output(self):
        """
        Context manager to capture the standard output.

        Yields:
            StringIO: The captured output.
        """
        new_out = StringIO()
        old_out = sys.stdout
        sys.stdout = new_out
        try:
            yield new_out
        finally:
            sys.stdout = old_out

    def execute_code_snippet(self, code: str) -> dict:
        """
        Executes the given Python code snippet.

        Args:
            code (str): The Python code snippet to be executed.

        Returns:
            dict: A dictionary containing the printed output and local variables.
        """
        # Check for dangerous functions and remove them
        dangerous_functions = ['exit', 'quit', 'sys.exit']
        for func in dangerous_functions:
            if func in code:
                # Use regex to remove function calls with any arguments
                code = re.sub(rf'{func}\s*\([^)]*\)', 'break', code)

        try:
            execution_code = self._extract_python_code(code)

            # Execute with configurable timeout (defaults to a few minutes)
            with timeout(self.code_execution_timeout_seconds):
                try:
                    # Capture the output and local variables
                    local_vars = {}
                    with self.capture_output() as output:
                        # Set __name__ to "__main__" to ensure main blocks execute
                        exec(execution_code, {"__name__": "__main__"}, local_vars)
                    printed_output = output.getvalue().strip()

                    # Filter out built-in variables and modules
                    used_vars = {k: v for k, v in local_vars.items() 
                                if not k.startswith('__') and not isinstance(v, type(sys))}
                    
                    return {
                        "success": True,
                        "printed_output": printed_output, 
                        "variables": used_vars,
                        "error": None
                    }
                except TimeoutException:
                    return {
                        "success": False,
                        "printed_output": "",
                        "variables": {},
                        "error": f"Execution timed out after {self.code_execution_timeout_seconds} seconds"
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "printed_output": "",
                        "variables": {},
                        "error": str(e)
                    }
        
        except Exception as e:
            return {
                "success": False,
                "printed_output": "",
                "variables": {},
                "error": f"Code extraction failed: {str(e)}"
            }

    def run(
        self,
        query: str = Field(description="Description of the programming task or problem to solve"),
        requirements: str = Field(
            default="", description="Specific requirements, constraints, or specifications for the code"
        ),
        context: str = Field(default="", description="Additional context or background information"),
        code_style: Literal["minimal", "documented", "verbose"] = Field(
            default="minimal",
            description="Style of generated code: minimal (concise), documented (with comments), verbose (detailed)",
        )
    ) :
        try:
            # Handle FieldInfo objects
            if isinstance(query, FieldInfo):
                query = query.default
            if isinstance(requirements, FieldInfo):
                requirements = requirements.default
            if isinstance(context, FieldInfo):
                context = context.default
            if isinstance(code_style, FieldInfo):
                code_style = code_style.default

            # Validate input
            if not query or not query.strip():
                raise ValueError("Task description is required for code generation")

            print(f"Generating code for: {query[:100]}...")

            start_time = time.time()

            # Prepare the code generation prompt
            prompt = self._prepare_code_prompt(query, requirements, context)

            # Enhance prompt based on code style
            if code_style == "minimal":
                prompt += "\n\nGenerate concise, minimal code without extensive comments."
            elif code_style == "verbose":
                prompt += "\n\nGenerate detailed code with comprehensive comments and explanations."
            elif code_style == "documented":
                prompt += "\n\nGenerate well-documented code with clear comments and docstrings."

            # Call the code generation model
            raw_response = self._call_code_model(prompt)

            # Extract clean Python code
            generated_code = self._extract_python_code(raw_response)

            # Execute the generated code
            execution_result = self.execute_code_snippet(generated_code)
            
            # If no output was produced, try to execute the code without the main block
            if execution_result["success"] and not execution_result["printed_output"]:
                # Try to extract and execute just the main execution part
                lines = generated_code.split('\n')
                main_execution_lines = []
                function_definitions = []
                in_main_block = False
                
                for line in lines:
                    if line.strip().startswith('if __name__ == "__main__":'):
                        in_main_block = True
                        continue
                    elif line.strip().startswith('def ') and not in_main_block:
                        # Collect function definitions that are outside the main block
                        function_definitions.append(line)
                        continue
                    elif in_main_block and line.strip().startswith('#'):
                        continue
                    elif in_main_block and line.strip() == '':
                        continue
                    elif in_main_block:
                        # Remove indentation from main block lines
                        main_execution_lines.append(line.lstrip())
                
                if main_execution_lines:
                    # Combine function definitions with main execution code
                    combined_code = '\n'.join(function_definitions + main_execution_lines)
                    fallback_result = self.execute_code_snippet(combined_code)
                    if fallback_result["success"] and fallback_result["printed_output"]:
                        execution_result = fallback_result

            response_message = {}

            # Prepare the response message
            if execution_result["success"]:
                response_message["Generated Code"] = f"```python\n{generated_code}\n```"
                response_message["result"] = execution_result['printed_output']
                response_message["success"] = True
                if execution_result["variables"]:
                    response_message["Variables"] = execution_result['variables']
            else:
                response_message["Generated Code"] = f"```python\n{generated_code}\n```"
                response_message["error"] = execution_result['error']
                response_message["success"] = False
            response_message['token_usage'] = self.llm_engine.get_token_usage()
            return response_message

        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def test(self, tool_test: str="python_code_generator", file_location: str="code_generate_execute", result_parameter: str="result", search_type: str="search_pattern"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)

# Example usage and entry point
if __name__ == "__main__":

    # Initialize and run the code generation service
    try:
        tool = Code_Generate_Execute_Tool()
        tool.embed_tool()
        tool.test(tool_test="python_code_generator", file_location="code_generate_execute", result_parameter="result", search_type='search_pattern', count_token=True)

    except Exception as e:
        print(f"An error occurred: {e}: {traceback.format_exc()}")
