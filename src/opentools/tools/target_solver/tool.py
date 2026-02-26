from itertools import combinations, permutations
import operator, json, os, sys, traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool

class Target_Solver_Tool(BaseTool):
    """Target_Solver_Tool
    ---------------------
    Purpose:
        A mathematical expression solver that finds all possible ways to combine numbers using arithmetic operations (+, -, *, /) to reach a target value. Uses dynamic programming to exhaustively search all possible expressions and eliminates duplicate solutions using canonical form detection.

    Core Capabilities:
        - Finds all possible mathematical expressions to reach a target value
        - Uses dynamic programming for efficient computation
        - Supports basic arithmetic operations (+, -, *, /)
        - Eliminates duplicate solutions using canonical form detection
        - Provides sorted results alphabetically

    Intended Use:
        Use this tool when you need to find all possible ways to combine numbers using arithmetic operations (+, -, *, /) to reach a target value.

    Limitations:
        - May not handle complex mathematical problems

    """
    # Default args for `opentools test Target_Solver_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "target_solver",
    }
    require_llm_engine = True
    def __init__(self, model_string= "gpt-4o", llm_engine=None):
        super().__init__(
            type='function',
            name="Target_Solver_Tool",
            description="""A mathematical expression solver that finds all possible ways to combine numbers using arithmetic operations (+, -, *, /) to reach a target value. Uses dynamic programming to exhaustively search all possible expressions and eliminates duplicate solutions using canonical form detection. CAPABILITIES: Finds all possible mathematical expressions to reach a target value, uses dynamic programming for efficient computation, supports basic arithmetic operations (+, -, *, /), eliminates duplicate solutions using canonical form detection, provides sorted results alphabetically. SYNONYMS: mathematical expression solver, target value finder, arithmetic combination solver, number combination finder, mathematical puzzle solver, expression generator, target number solver, arithmetic expression finder, mathematical combination tool, number puzzle solver. EXAMPLES: 'Find all ways to make 24 using numbers [4, 6, 8, 9]', 'Find all ways to make 10 using numbers [1, 2, 3, 4]', 'Generate expressions to reach target 15 with numbers [2, 3, 5, 7]'.""",
            parameters={
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of numbers to use in calculations (e.g., [4, 6, 8, 9])"
                    },
                    "target": {
                        "type": "integer",
                        "description": "Target integer value to reach (e.g., 24)"
                    }
                },
                "required": ["numbers", "target"],
                "additionalProperties": False,
            },
            strict=True,
            category="mathematics",
            tags=["mathematical_solver", "expression_solver", "target_finder", "arithmetic_solver", "mathematical_puzzle", "expression_generator", "number_combinations", "dynamic_programming", "mathematical_tool", "puzzle_solver"],
            limitation="Only supports basic arithmetic operations (+, -, *, /), computational complexity grows exponentially with more numbers, limited to integer and floating-point arithmetic, cannot handle negative numbers in input, maximum practical input size is around 6-8 numbers",
            agent_type="Mathematics-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(numbers=[4, 6, 8, 9], target=24)",
                "description": "Find all ways to make 24 using numbers [4, 6, 8, 9]"
            },
            require_llm_engine=True,
            llm_engine=llm_engine,
        )
    
    def test(self, tool_test: str="target_solver"):
        """Test the Target Solver tool with various test samples, run 3 times, and save results in a JSON file."""
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
                question_result = {"id": test.get("id", f"target_{i + 1}")}
                if 'numbers' in test and 'target' in test:
                    question_result['query'] = f"numbers={test['numbers']}, target={test['target']}"
                if 'answer' in test:
                    question_result['expected_answer'] = test['answer']

                # Prepare parameters (exclude answer, id, category)
                parameters = {k: v for k, v in test.items() if k not in ['answer', 'id', 'category']}

                # Run and record result for each of 3 runs
                for j in range(1, 4):
                    run_result = {}
                    result = self.run(**parameters)
                    run_result['result'] = result['result']

                    # If failed or none
                    if not result or (isinstance(result, dict) and result.get("success") == False):
                        run_result['error'] = result.get("error")
                        run_result['accuracy'] = 0
                        run_result['tool_call_pass'] = False
                        question_result[f'run_{j}'] = run_result
                        continue
                    else:
                        run_result['tool_call_pass'] = True

                    # Calculate accuracy for this run using LLM evaluation
                    if result.get("result") and test.get('answer'):
                        response = result['result']
                        expected_answer = test['answer']
                        
                        # Use LLM evaluation to compare response with expected answer
                        accuracy_score = self.llm_eval(response, expected_answer)
                        
                        # Normalize accuracy score (assuming llm_eval returns number of correct solutions)
                        if isinstance(expected_answer, list):
                            max_possible = len(expected_answer)
                        else:
                            max_possible = 1
                        
                        normalized_accuracy = min( 1, accuracy_score / max_possible if max_possible > 0 else 0)
                        run_result['accuracy'] = normalized_accuracy
                        run_accuracy[f'run_{j}'] += normalized_accuracy
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
            print(f"❌ Failed to test the tool with this error: {e}")
            return False
        return True

    def llm_eval(self, response, answer):
        query_prompt_open_tools = f"""
    This is a Game of 24 answer verification task. Compare the model's response against the list of correct answers.
    
    Model response: {response}
    Correct answer(s): {answer}

    Evaluation rules:
    1. Extract the arithmetic expression from the model response (ignore explanations)
    2. Check if the extracted expression is mathematically equivalent () to ANY of the correct answers
        When considering equivalence, ignore differences in:
        - Spacing
        - order of commutative operations
        - equivalent fractions
    The response could have more correct solutions than the answers. Ignore extra solutions.
    The expression must be mathematically equivalent to one of the correct answers
    For example, these are equivalent:
    - "(1+2+3)×4" and "(3+2+1)×4"
    - "8×(1+1+1)" and "(1+1+1)×8"
    - "(a + b) × c" and "c × (b + a)" and "(b + a) × c"
    3. The multiplication symbol × (U+00D7) is interchangeable with *
    4. First extract the expression from the response, then check if it's equivalent to any correct answer
    Response Format:
    Return number of correct solutions in the response no explanation just a single number
    """
        response = self.llm_engine.generate(query_prompt_open_tools)
        if isinstance(response, dict):
            response = response.get('text')
        else:
            response = str(response)
        return float(response)

    def run(self, numbers, target) :
        """
        Find all unique ways to reach the target value using the given numbers.
        
        Args:
            numbers: List of numbers to use in calculations
            target: Target value to reach
        
        Returns:
            List of string representations of valid expressions, sorted alphabetically
        """
        try:
            sorted_numbers = tuple(sorted(numbers))
            unique_expressions = set()
            seen_canonical_keys = set()

            # Try all permutations of the numbers
            for number_permutation in set(permutations(sorted_numbers)):
                expression_map = build_expression_map(number_permutation)
                
                for computed_value, expression_set in expression_map.items():
                    # Check if this value equals the target (with small tolerance)
                    if abs(computed_value - target) > EPSILON:
                        continue
                        
                    for expression in expression_set:
                        canonical_key = get_canonical_key(expression)
                        
                        # Skip if we've already seen this algebraic form
                        if canonical_key in seen_canonical_keys:
                            continue
                            
                        seen_canonical_keys.add(canonical_key)

                        # Convert to string and verify the result
                        expression_string = str(expression)
                        # Replace × with * for evaluation
                        evaluatable_string = expression_string.replace("×", "*")
                        
                        try:
                            computed_result = eval(evaluatable_string)
                            if abs(computed_result - target) < EPSILON:
                                unique_expressions.add(expression_string)
                        except (ZeroDivisionError, ValueError):
                            # Skip invalid expressions
                            continue
            return {"result": list(sorted(unique_expressions)), "success": True}
        except (ZeroDivisionError, ValueError) as e:
            return {"error": f"Invalid input: {str(e)}", "success": False, "error_type": "invalid_input", "traceback": traceback.format_exc()}
        except Exception as e:
            return {"error": f"Error solving target: {str(e)}", "success": False, "error_type": "target_solver_failed", "traceback": traceback.format_exc()}

ARITHMETIC_OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
}
# To help decide the order of operations (when to add parentheses)
OPERATOR_PRECEDENCE = {'+': 1, '-': 1, '*': 2, '/': 2}
# Reduce the number of unique expressions by treating a×b and b×a or a+b and b+a as the same
COMMUTATIVE_OPERATORS = {'+', '*'}
# Symbols to use for multiplication (to make the output more readable)
PRETTY_SYMBOLS = {'*': "\u00d7"}

# Constants for numerical comparisons
EPSILON = 1e-9

class ExpressionNode:
    """
    Represents a mathematical expression as a binary tree.
    Each node can be either a number (leaf) or an operation with two children.
    """

    def __init__(self, operator=None, left_child=None, right_child=None, value=None):
        self.operator = operator
        self.left_child = left_child
        self.right_child = right_child
        self.value = value
        self.precedence = OPERATOR_PRECEDENCE.get(operator, 3)

    def simplify(self):
        """
        Return a simplified copy of the expression.
        Eliminates ×1 and ÷1 operations and applies canonical ordering
        for commutative operators.
        """
        if self.operator is None:
            return self  # Leaf nodes stay as-is

        simplified_left = self.left_child.simplify()
        simplified_right = self.right_child.simplify()

        # For commutative operators, ensure consistent ordering
        if (self.operator in COMMUTATIVE_OPERATORS and 
            str(simplified_left) > str(simplified_right)):
            simplified_left, simplified_right = simplified_right, simplified_left

        new_value = ARITHMETIC_OPERATORS[self.operator](simplified_left.value, simplified_right.value)
        return ExpressionNode(self.operator, simplified_left, simplified_right, new_value)

    def __str__(self):
        """Convert the expression to a readable string with proper parentheses."""
        if self.operator is None:
            # Display integers without decimal places
            return str(int(self.value)) if self.value.is_integer() else str(self.value)

        left_text = str(self.left_child)
        right_text = str(self.right_child)

        # Add parentheses for lower precedence operations
        if self.left_child.operator and self.left_child.precedence < self.precedence:
            left_text = f"({left_text})"
        
        if self.right_child.operator and (
            self.right_child.precedence < self.precedence or
            (self.operator in "-/" and self.right_child.operator == self.operator)
        ):
            right_text = f"({right_text})"

        # Use pretty symbols for multiplication
        operator_symbol = PRETTY_SYMBOLS.get(self.operator, self.operator)
        return f"{left_text}{operator_symbol}{right_text}"

    def create_leaf(number):
        """Create a leaf node representing a single number."""
        return ExpressionNode(value=number)

    
   
def get_canonical_key(expression):
    """
    Return a hashable key that treats equivalent expressions as equal.
    For commutative operators (+, *), treats a×b and b×a as the same.
    """
    if expression.operator is None:
        return ("number", expression.value)

    if expression.operator in COMMUTATIVE_OPERATORS:
        def collect_children(node, children):
            """Recursively collect all children of commutative operations."""
            if node.operator == expression.operator:
                collect_children(node.left_child, children)
                collect_children(node.right_child, children)
            else:
                children.append(get_canonical_key(node))

        all_children = []
        collect_children(expression.left_child, all_children)
        collect_children(expression.right_child, all_children)
        return expression.operator, tuple(sorted(all_children))

    return (
        expression.operator,
        get_canonical_key(expression.left_child),
        get_canonical_key(expression.right_child),
    )



def build_expression_map(numbers):
    """
    Build all possible mathematical expressions from the given numbers.
    
    Returns a mapping from computed values to sets of expressions that produce them.
    Uses dynamic programming to avoid redundant calculations.
    """
    if len(numbers) == 1:
        leaf_expression = ExpressionNode.create_leaf(float(numbers[0]))
        return {leaf_expression.value: {leaf_expression}}

    result = {}

    # Try all ways to split the numbers into two groups
    for left_group_size in range(1, len(numbers)):
        for left_indices in combinations(range(len(numbers)), left_group_size):
            left_numbers = tuple(numbers[i] for i in left_indices)
            right_numbers = tuple(numbers[i] for i in range(len(numbers)) 
                                if i not in left_indices)

            # Recursively build expressions for both groups
            left_expressions = build_expression_map(left_numbers)
            right_expressions = build_expression_map(right_numbers)

            # Combine expressions from both groups using all operators
            for left_value, left_expression_set in left_expressions.items():
                for right_value, right_expression_set in right_expressions.items():
                    for operator_symbol, operator_function in ARITHMETIC_OPERATORS.items():
                        # Skip division by zero
                        if operator_symbol == "/" and abs(right_value) < EPSILON:
                            continue

                        combined_value = operator_function(left_value, right_value)

                        # Create all combinations of left and right expressions
                        for left_expr in left_expression_set:
                            for right_expr in right_expression_set:
                                new_expression = ExpressionNode(
                                    operator_symbol, left_expr, right_expr, combined_value
                                ).simplify()
                                
                                if combined_value not in result:
                                    result[combined_value] = set()
                                result[combined_value].add(new_expression)
    return result



if __name__ == "__main__":
    tool = Target_Solver_Tool()
    tool.embed_tool()
    try:
        tool.test(tool_test="target_solver")
    except Exception as e:
        print(f"❌ Failed to test the tool with this error: {e}")
