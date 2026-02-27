import sys, traceback, os, requests, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', '..')))
from opentools.core.base import BaseTool
from dotenv import load_dotenv
from opentools.core.factory import create_llm_engine

class Wolfram_Math_Tool(BaseTool):
    """
    Wolfram_Math_Tool
    ---------------------
    Purpose:
        A specialized tool that sends integral or algebraic equation queries to the Wolfram API
        and returns solutions in text format.

    Core Capabilities:
        - Specialize in solving equations (linear, quadratic, cubic, differential equations)
        - Computing integrals (definite and indefinite)
        - Derivatives
        - Limits
        - Series expansions
        - Matrix operations
        - Linear algebra
        - Calculus
        - Algebra
        - Trigonometry
        - Geometry
        - Statistics
        - Probability
        - Mathematical analysis
        - Symbolic computation
        - Numerical evaluation

    Intended Use:
        Use this tool when you need to solve mathematical problems, including equations, integrals, derivatives, limits, series expansions, matrix operations, linear algebra, calculus, algebra, trigonometry, geometry, statistics, probability, and mathematical analysis.

    Limitations:
        - Requires internet connection for Wolfram API access
        - Depends on Wolfram API's availability and rate limits
        - Some mathematical problems may not be solvable by the Wolfram API
        - Some mathematical problems may require a paid subscription to the Wolfram API
    """
    DEFAULT_TEST_ARGS = {
        "tool_test": "math_solver",
        "llm_judge": True,
    }
    def __init__(self):
        super().__init__(
            type='function',
            name="Wolfram_Math_Tool",
            description="""WolframAlpha computational tool for mathematical queries and calculations. This tool specializes in solving mathematical problems using the Wolfram Alpha API.CAPABILITIES: Specialize in solving equations (linear, quadratic, cubic, differential equations), computing integrals (definite and indefinite), derivatives, limits, series expansions, matrix operations, linear algebra, calculus, algebra, trigonometry, geometry, statistics, probability, and mathematical analysis with symbolic computation and numerical evaluation. SYNONYMS: Wolfram Alpha math tool, mathematical equation solver, integral calculator, derivative solver, math problem solver, computational mathematics tool, symbolic math solver, mathematical analysis tool, equation calculator, math computation engine. EXAMPLES: 'Solve the equation x^2 + 5x + 6 = 0', 'Calculate the integral of x^2 dx', 'Find the derivative of sin(x)', 'Solve the differential equation dy/dx = y'.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The mathematical problem description or query to be solved by Wolfram Alpha"
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=True,
            category="mathematics",
            tags=["wolfram_alpha", "mathematical_solver", "equation_solver", "integral_calculator", "derivative_solver", "math_tools", "computational_mathematics", "symbolic_computation", "mathematical_analysis", "problem_solving"],
            limitation="Requires Wolfram Alpha API key, limited to mathematical queries only, depends on Wolfram Alpha's parsing capabilities, some complex queries may not be parsed correctly.",
            agent_type="Mathematics-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(query='Solve the equation x^2 + 5x + 6 = 0')",
                "description": "Solve the equation x^2 + 5x + 6 = 0"
            }
        )
        self.output = "str - The solution returned by Wolfram"
        self.wolfram_api = "https://api.wolframalpha.com/v2/query"

    def run(self,query:str):
        """Get Wolfram|Alpha results using natural query. Queries to getWolframAlphaResults must ALWAYS have this structure: {\"input\": query}. And please directly read the output json. 
        """
        try:
            load_dotenv()
            APPID = os.getenv("WOLFRAM_API_KEY")
            params = {
                        "appid": APPID,
                        "input": query,
                        "output": "JSON"
                    }        
            response = requests.get(self.wolfram_api, params=params).text
            json_data = json.loads(response)

            if 'pods' not in json_data["queryresult"]:
                return {"result": "WolframAlpha API cannot parse the input query.", "success": False}
            rets = json_data["queryresult"]['pods']
            cleaned_rets = []
            blacklist = ["scanner", "id", "position", "error", "numsubpods", "width", "height", "type", "themes","colorinvertable", "expressiontypes",]
            def filter_dict(d, blacklist):
                if isinstance(d, dict):
                    return {k: filter_dict(v, blacklist) for k, v in d.items() if k not in blacklist}
                elif isinstance(d, list):
                    return [filter_dict(i, blacklist) for i in d]
                else:
                    return d
            for ret in rets:
                ret = filter_dict(ret, blacklist=blacklist)
                if "title" in ret:
                    for res in ret['subpods']:
                        if (res['plaintext']):
                            cleaned_rets.append(res['plaintext'])
            return {"result": cleaned_rets, "success": True}
        except Exception as e:
            return {"error": f"Error getting WolframAlpha results: {str(e)}", "success": False, "error_type": "wolfram_math_failed", "traceback": traceback.format_exc()}
        
    def Llm_Judge(self,answer, solution):
        model_name = "gpt-4o-mini"
        llm_engine  = create_llm_engine(model_name)
        prompt = f"""
            You are a strict math-equivalence checker.  
            You’ll be given two strings:

            Answer   = {{answer}}
            Solution = {{solution}}  

            Your job is simply to decide whether the *solution* expression is present in the *answer* expression **after** the following normalizations:

            1. **Whitespace & case**  
            - Trim leading/trailing whitespace.  
            - Collapse runs of whitespace (spaces, tabs, newlines) into a single space.  
            - Convert everything to lowercase.

            2. **Symbol fixes**  
            - Replace any occurrence of the word “constant” with the single character `C`.  
            - Leave all other characters (digits, letters, `+-*/^()`, Greek letters, function names, etc.) intact.

            3. **Match rule**  
            - When comparing assume log and ln point to the same thing.
            - If the entire normalized *solution* appears as a **substring** of the normalized *answer*, it’s a match. In other words, if normalized answer contains all of the element in normalized solution, then it is a match.

            Return **exactly one bare word** on its own line—**True** if it matches, **False** otherwise.  
            Do **NOT** add any extra text, punctuation, quotes, or explanation.
            <BEGIN-EXAMPLES>
            Example-1
            ANSWER   = "x = 8, so log(x)/log(2)=3"
            SOLUTION = "8"
            Result   = True

            Example-2
            ANSWER   = "∫₀³ x² dx = 9"
            SOLUTION = "8"
            Result   = False

            Example-3
            ANSWER   = "indefinite integral 1/(x + 4) dx = log(x + 4) + constant"
            SOLUTION = "log(x + 4) + c"
            Result   = True
            
            Example-4
            ANSWER   = ['Solve the problem A','x = log(5)/2', 'x = 1/5*i*(3*π*n - i*(log(10) + log(2))), n element Z']
            SOLUTION = log(5)/2 and 1/5*i*(3*π*n - i*(log(10) + log(2)))
            Result   = True
            
               Example-4
            ANSWER   = ['integral_-∞^∞ (A) dx = 3.14']
            SOLUTION = pi ≈ 3.14
            Result   = True
            <END-EXAMPLES>
            
            <BEGIN-TASK>

            Answer: {answer}
            Solution: {solution}
            """
        raw_response = llm_engine.generate(prompt)
        # Handle both dict (with "text") and plain-string responses from the engine
        if isinstance(raw_response, dict):
            response_text = raw_response.get("text") or raw_response.get("response") or ""
        else:
            response_text = str(raw_response)

        normalized = response_text.strip()
        if normalized == "True":
            return True
        return False
    
    def test(self, tool_test: str="math_solver", llm_judge=True):
        """Test the Wolfram Math tool with various test samples, run 3 times, and save results in a JSON file."""
        try:
            # Load testbench data
            file_test = os.path.join(os.path.dirname(__file__), '..', 'test_file', 'data.json')
            with open(file_test, encoding='utf-8') as f:
                data = json.load(f)[tool_test]

            # Prepare result file as JSON
            file_result = os.path.join(os.path.dirname(__file__), 'test_result.json')
            test_result = {}
            test_result['Test-File length'] = len(data)
            run_accuracy = {'run_1': 0, 'run_2': 0, 'run_3': 0}
            
            # Iterate over test cases
            for i, test in enumerate(data):
                question_result = {"id": test.get("id", f"wolfram_{i + 1}")}
                if 'query' in test:
                    question_result['query'] = test['query']
                if 'answer' in test:
                    question_result['expected_answer'] = test['answer']

                # Prepare parameters (exclude answer, id, category)
                parameters = {k: v for k, v in test.items() if k not in ['answer', 'id', 'category']}

                # Run and record result for each of 3 runs
                for j in range(1, 4):
                    run_result = {}
                    result = self.run(**parameters)
                    run_result['result'] = result

                    # If failed or none
                    if not result or (isinstance(result, dict) and result.get("success") == False):
                        run_result['accuracy'] = 0
                        run_result['tool_call_pass'] = False
                        question_result[f'run_{j}'] = run_result
                        continue
                    else:
                        run_result['tool_call_pass'] = True

                    # Calculate accuracy for this run using LLM judgment
                    if result.get("result") and test.get('answer'):
                        response = result['result']
                        expected_answer = test['answer']
                        
                        # Use LLM judge to compare response with expected answer
                        if llm_judge:
                            correctness = self.Llm_Judge(response, expected_answer)
                            accuracy_score = 1 if correctness else 0
                        else:
                            # Fallback to simple string comparison if LLM judge is disabled
                            accuracy_score = 1 if str(response) == str(expected_answer) else 0
                        
                        run_result['accuracy'] = accuracy_score
                        run_accuracy[f'run_{j}'] += accuracy_score
                    else:
                        run_result['accuracy'] = 0
                        run_accuracy[f'run_{j}'] += 0

                    question_result[f'run_{j}'] = run_result

                print(f"Finish query: {i + 1}")
                test_result[f'Q{i + 1}'] = question_result

            # Calculate and record overall accuracy for each run
            test_result['Final_Accuracy'] = {
                'run_1': run_accuracy['run_1'] * 100 / len(data) if data else 0,
                'run_2': run_accuracy['run_2'] * 100 / len(data) if data else 0,
                'run_3': run_accuracy['run_3'] * 100 / len(data) if data else 0
            }
            print(f"Accuracy: {test_result['Final_Accuracy']}")

            with open(file_result, "w", encoding="utf-8") as output_file:
                json.dump(test_result, output_file, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"❌ Wolfram Math tool test failed: {e}")
            return False
        return True
    
        
if __name__ == "__main__":
    # print("STARTING WOLFRAM MATH TOOL MAIN")  # Debug print to confirm main block runs
    tool = Wolfram_Math_Tool()
    tool.embed_tool()
    try:
        tool.test(tool_test="math_solver", llm_judge=True)
    except Exception as e:
        print(f"Error: {e}")
