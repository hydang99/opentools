import json, sys, os, traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', '..')))
from typing import Dict, Any
from opentools.core.base import BaseTool
from opentools.core.config import config
from opentools.core.factory import create_llm_engine
class Math_Solver_Tool(BaseTool):
    """Math_Solver_Tool
    ---------------------
    Purpose:
        A tool that performs math solve based on a given query.

    Core Capabilities:
        - Solves mathematical problems across algebra, calculus, geometry, trigonometry, and more
        - Provides step-by-step solutions with clean, human-readable mathematical notation
        - Follows strict output formatting rules

    Intended Use:
        Use this tool when you need to solve mathematical problems, including algebra, calculus, geometry, trigonometry, and more.

    Limitations:
        - Requires a valid OpenAI API key and internet connectivity
        - May not handle complex mathematical problems
    """
    # Default args for `opentools test Math_Solver_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "llm_judge": True,
    }

    def __init__(self, model_string = "gpt-4o-mini", llm_engine=None):
        super().__init__(
            type='function',
            name="Math_Solver_Tool",
            description="""A tool powered by the defined model to solve a broad range of math problems. Excels at multi-step logical reasoning in algebra, calculus, geometry, and beyond. Provides step-by-step solutions with clean, human-readable mathematical notation and follows strict output formatting rules. CAPABILITIES: Solves mathematical problems across algebra, calculus, geometry, trigonometry, and more, provides step-by-step reasoning, handles complex equations and expressions, supports symbolic and numerical solutions, automatic formatting with clean mathematical notation. SYNONYMS: math solver, equation solver, mathematical problem solver, algebra solver, calculus solver, math calculator, problem solver, mathematical reasoning tool, equation calculator. EXAMPLES: 'Solve the quadratic equation x^2 - 5x + 6 = 0', 'Evaluate the integral of x dx', 'Find the derivative of sin(x)', 'Calculate the area of a circle with radius 5'.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The full problem description, including any equations, expressions, and constraints to solve"
                    },
                    "image": {
                        "type": "string",
                        "description": "Optional path to an image file for math problem analysis (default: None)"
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=False,
            category="mathematics",
            tags=["math_solver", "equation_solver", "mathematics", "algebra", "calculus", "geometry", "problem_solving", "mathematical_reasoning", "equation_calculator"],
            limitation="While generally reliable, the tool may produce incorrect or incomplete reasoning. Submit one problem at a time for clarity, review step-by-step reasoning, and rephrase if results seem incorrect",
            agent_type="Mathematics-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(query='Solve the equation x + 2 = 0')",
                "description": "Solve the equation x + 2 = 0"
            },
            require_llm_engine=True,
            llm_engine=llm_engine,
        )
        self.prompt =  f"""You are a highly-accurate, fast math-solving machine.
                        Solve the following math problem.
                        You may show your reasoning and intermediate steps if you wish, **but follow all the rules below when presenting the final answer**.
                        Output rules (MUST be followed exactly)
                        1. The **very last line** of your response must contain only the final answer(s)—no words, labels, or punctuation after with the following format "Final answer: x=3".
                        2. If the task is an equation or system, list **all** distinct solutions, including imaginary/complex ones whenever they exist.
                        3. Separate multiple solutions with a single comma and a space, ordered by real part, then by imaginary part.
                        4. Write numbers and expressions in clean, human-readable math notation (e.g. 1/3, √2, π). **Do NOT** use frac(), \\frac{{}}, or any other function-style notation that non human-readable.
                        5. If an exact symbolic form exists, present that exact expression, do not round or approximate."""

    def run(self, query, image=None):
        if not self.llm_engine:
            self.llm_engine = create_llm_engine(self.model_string)
        try:
            result = self.llm_engine.generate(self.prompt + query, image=image)
            if isinstance(result, dict):
                result = result.get('text')
            else:
                result = str(result)
            return {"result": result, "success": True}
        except Exception as e:
            print("Error", e)
            return {"error": "Error could not solved", "success": False, "error_type": "math_solver_failed", "traceback": traceback.format_exc()}

    
    def Llm_Judge(self,answer, solution):
        model_name = self.model_string
        llm_engine  = create_llm_engine(model_name)
        
        max_retries = 3
        for attempt in range(max_retries):
            prompt = f"""
                You are a strict math equivalence grading assistant.
                You will be given two strings:
                Answer and Solution are the answer and solution to the math problem.
                Your job is to determine if the Answer is mathematically equivalent to the Solution and return a JSON object in the following format:
                {{
                    "Boolean": true/false,
                    "Reason": "<Very short explanation if not correct, else say 'Equivalent'>"
                }}
                Answer: {answer}
                Solution: {solution}
                Only return the JSON result, nothing else. Do not wrap it in markdown code blocks.
                """
            
            response = llm_engine.generate(prompt).get("text")
            print('---------------------------------------------------------------------------------------------------------------------------------------------------')     
            print("The response is: ", response)
            
            try:
                # Clean the response - remove markdown code blocks if present
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]   # Remove ```
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]  # Remove trailing ```
                cleaned_response = cleaned_response.strip()
                
                result = json.loads(cleaned_response)
                if (result["Boolean"] == True or result["Reason"] == "Equivalent" or result["Boolean"] == "true"):
                    return True
                return False
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    print(f"Failed to parse JSON after {max_retries} attempts. Returning False.")
                    return False
                print("Retrying...")
                continue
        
        return False

    def test(self, llm_judge=True) -> str:
        """Test the math tool with various operations, JSON structured output."""
        try:
            import json
            # Open testbench
            file_test = os.path.join(os.path.dirname(__file__), '..', 'test_file', 'data.json')
            with open(file_test, encoding='utf-8') as f:
                data = json.load(f)['math_solver']
            # Prepare result dict for JSON output
            file_result = os.path.join(os.path.dirname(__file__), 'test_result.json')
            test_result = {}
            test_result['Test-File length'] = len(data)
            run_accuracy = {'run_1': 0, 'run_2': 0, 'run_3': 0}
            
            for i in range(len(data)):
                test = data[i]
                question_result = {"id": f"math_solver_{i + 1}"}
                if 'query' in test:
                    question_result['query'] = test['query']
                if 'answer' in test:
                    question_result['answer'] = test['answer']
                # Run 3 times for each test
                for j in range(0, 3):
                    run_result = {}
                    result = self.run(test['query'])['result']
                    run_result['result'] = result
                    # comparing answer with ground-truth
                    correctness = False
                    if llm_judge and self.Llm_Judge(result, test['answer']):
                        correctness = True
                        run_accuracy[f'run_{j + 1}'] += 1
                        run_result['accuracy'] = 1
                    else:
                        run_result['accuracy'] = 0
                    run_result['tool_call_pass'] = True
                    run_result['expected_solution'] = test['answer']
                    run_result['correctness'] = correctness
                    question_result[f'run_{j + 1}'] = run_result
                test_result[f'Q{i + 1}'] = question_result
            # Calculate the accuracy of the tool
            test_result['Final_Accuracy'] = {
                'run_1': run_accuracy['run_1'] * 100 / len(data),
                'run_2': run_accuracy['run_2'] * 100 / len(data),
                'run_3': run_accuracy['run_3'] * 100 / len(data),
            }
            test_result['token_usage'] = self.llm_engine.get_token_usage()
            print(test_result['Final_Accuracy'])
            with open(file_result, "w", encoding="utf-8") as output_file:
                json.dump(test_result, output_file, indent=2, default=str)
            return test_result['Final_Accuracy']
        except Exception as e:
            print(f"❌ Math tool test failed: {e}")
            return False
    
if __name__ == "__main__":
    # Test the math solver tool
    math_tool = Math_Solver_Tool()
    math_tool.embed_tool()
    math_tool.test(llm_judge=True)