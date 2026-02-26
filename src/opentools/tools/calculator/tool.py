import sys, os, math, json
from typing import List
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
import traceback
class Calculator_Tool(BaseTool):
    """Calculator_Tool
    ---------------------
    Purpose:
        A comprehensive mathematical operations tool with precise decimal handling. Adopting from easytools/tools/funchub/math.py

    Core Capabilities:
        - Basic arithmetic operations (add, subtract, multiply, divide)
        - Power and root calculations (power, sqrt, exp)
        - Logarithmic functions (log base-10, ln natural log)
        - Combinatorial mathematics (combinations, permutations, factorial)
        - Number theory (GCD, LCM, remainder)
        - Trigonometric and hyperbolic functions (sin, cos, tan, asin, acos, atan, sinh, cosh, tanh)
        - Constant-aware parsing (pi, e, tau, c, h, hbar, k, na, g0, R)
        - Concise unit conversions (length, mass, time, temperature, pressure, energy)

    Intended Use:
        Use this tool when you need to perform mathematical calculations, including basic arithmetic, logarithms, combinatorics, trigonometry, constants parsing, and common unit conversions.

    Limitations:
        - May not handle complex symbolic math or unsupported unit systems
        - Precision limited to 10 decimal places
        - Some operations require specific number of input values
        - No support for complex numbers
    """
    def __init__(self):
        super().__init__(
            type='function',
            name="Calculator_Tool",
            description="""A comprehensive mathematical calculation operations tool that provides basic arithmetic, logarithms, combinatorics, trigonometry, constants parsing,  and common unit conversions with precise decimal handling. CAPABILITIES: Basic arithmetic operations (add, subtract, multiply, divide), power and root calculations  (power, sqrt, exp), logarithmic functions (log base-10, ln natural log), combinatorial mathematics (combinations, permutations, factorial), number theory  (GCD, LCM, remainder), trigonometric and hyperbolic functions (sin, cos, tan, asin, acos, atan, sinh, cosh, tanh), constant-aware parsing (pi, e, tau, c, h, hbar,  k, na, g0, R) and concise unit conversions (length, mass, time, temperature, pressure, energy). SYNONYMS: math calculator, mathematical operations, arithmetic tool,  scientific calculator, math solver, computation tool, mathematical functions, number cruncher, math processor. EXAMPLES: 'Calculate 15.7 + 23.4 + 8.9', 'Find the square  root of 256', 'Compute sin of 45 degrees', 'Convert 5 kilometers to meters', 'Calculate combinations of 10 choose 3', 'Find GCD of 48 and 72'.""",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The mathematical operation to perform",
                        "enum": [
                            "add", "subtract", "multiply", "divide", "power", "sqrt", "log", "ln",
                            "lcm", "gcd", "remainder", "choose", "permutate", "exp", "factorial",
                            "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
                            "convert_units"
                        ]
                    },
                    "values": {
                        "type": "array",
                        "items": {
                            "type": ["number", "string"]
                        },
                        "description": "List of numerical values (numbers or constants as strings). For trig functions: [angle, 'deg' optional]. For convert_units: [value, from_unit, to_unit]."
                    }
                },
                "required": ["operation", "values"],
                "additionalProperties": False,
            },
            strict=True,
            category="mathematics",
            tags=["calculator", "math", "arithmetic", "combinatorics", "logarithms", "number_theory", "computation", "mathematical_operations"],
            limitation="Limited to predefined mathematical operations and a concise unit set; may not handle complex symbolic math or unsupported unit systems; precision limited to 10 decimal places; some operations require specific number of input values; no support for complex numbers",
            agent_type="Mathematics-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(operation='add', values=[1, 2, 3])",
                "description": "Add 1, 2, and 3"
            },
        )
        
        self.func_map = {
            'add': self.add,
            'subtract': self.subtract,
            'multiply': self.multiply,
            'divide': self.divide,
            'power': self.power,
            'sqrt': self.sqrt,
            'log': self.log,
            'ln': self.ln,
            'choose': self.choose,
            'permutate': self.permutate,
            'gcd': self.gcd,
            'lcm': self.lcm,
            'remainder': self.remainder,
            'exp': self.exp,
            'factorial': self.factorial,
            'sin': self.sin,
            'cos': self.cos,
            'tan': self.tan,
            'asin': self.asin,
            'acos': self.acos,
            'atan': self.atan,
            'sinh': self.sinh,
            'cosh': self.cosh,
            'tanh': self.tanh,
            'convert_units': self.convert_units,
        }

        self.constants = {
            "pi": math.pi,
            "e": math.e,
            "tau": math.tau,
            "c": 299_792_458,  # speed of light (m/s)
            "light_speed": 299_792_458,
            "g0": 9.80665,  # standard gravity (m/s^2)
            "k": 1.380_649e-23,  # Boltzmann constant (J/K)
            "na": 6.022_140_76e23,  # Avogadro constant (1/mol)
            "avogadro": 6.022_140_76e23,
            "r": 8.314_462_618,  # Gas constant (J/(mol*K))
            "gas_constant": 8.314_462_618,
            "h": 6.626_070_15e-34,  # Planck constant (J*s)
            "hbar": 1.054_571_817e-34,  # Reduced Planck constant (J*s)
            "phi": (1 + 5 ** 0.5) / 2,  # Golden ratio
        }
    
    def custom_round(self, x: float, decimal_places: int = 2) -> float:
        """Round the result to specified decimal places with special handling for small numbers.
        
        Args:
            x: The number to round
            decimal_places: Number of decimal places (default: 2)
            
        Returns:
            Rounded number
        """
        str_x = f"{x:.10f}"
        before_decimal = str_x.split('.')[0]
        after_decimal = str_x.split('.')[1]
        leading_zeros = len(after_decimal) - len(after_decimal.lstrip('0'))
        
        if leading_zeros >= 1 and before_decimal == "0":
            return round(x, leading_zeros + 2)
        else:
            return round(x, decimal_places)

    def scito_decimal(self, sci_str: str) -> str:
        """Convert a number in scientific notation to decimal notation.
        
        Args:
            sci_str: Number in scientific notation
            
        Returns:
            Number in decimal notation
        """
        def split_exponent(number_str: str):
            parts = number_str.split("e")
            coefficient = parts[0]
            exponent = int(parts[1]) if len(parts) == 2 else 0
            return coefficient, exponent

        def multiplyby_10(number_str: str, exponent: int):
            if exponent == 0:
                return number_str

            if exponent > 0:
                index = number_str.index(".") if "." in number_str else len(number_str)
                number_str = number_str.replace(".", "")
                new_index = index + exponent
                number_str += "0" * (new_index - len(number_str))
                if new_index < len(number_str):
                    number_str = number_str[:new_index] + "." + number_str[new_index:]
                return number_str

            if exponent < 0:
                index = number_str.index(".") if "." in number_str else len(number_str)
                number_str = number_str.replace(".", "")
                new_index = index + exponent
                number_str = "0" * (-new_index) + number_str
                number_str = "0." + number_str
                return number_str

        coefficient, exponent = split_exponent(sci_str)
        decimal_str = multiplyby_10(coefficient, exponent)

        # remove trailing zeros
        if "." in decimal_str:
            decimal_str = decimal_str.rstrip("0")

        return decimal_str

    def normalize(self, res: float, round_to: int = 2) -> str:
        """Normalize the result to specified decimal places and remove trailing zeros.
        
        Args:
            res: The result to normalize
            round_to: Number of decimal places to round to
            
        Returns:
            Normalized result as string
        """
        # we round the result to specified decimal places
        res = self.custom_round(res, round_to)
        res = str(res)
        if "." in res:
            while res[-1] == "0":
                res = res[:-1]
            res = res.strip(".")
        
        # scientific notation
        if "e" in res:
            res = self.scito_decimal(res)

        return res
    
    def _parse_single_number(self, value) -> float:
        """Parse a single value that might be numeric, a numeric string, or a known constant."""
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            cleaned = value.strip().lower().replace("_", "")
            if cleaned in self.constants:
                return float(self.constants[cleaned])
            try:
                return float(cleaned)
            except ValueError as exc:
                raise ValueError(f"Cannot parse numeric value '{value}'") from exc

        raise TypeError(f"Unsupported value type '{type(value).__name__}' for numeric parsing")

    def _parse_numeric_values(self, values: List) -> List[float]:
        """Parse a list of values into floats, resolving constants and numeric strings."""
        return [self._parse_single_number(v) for v in values]

    def _angle_from_values(self, values: List) -> float:
        """Parse an angle value and optionally convert degrees to radians if flagged."""
        if not values:
            raise ValueError("An angle value is required for trigonometric operations")
        angle = self._parse_single_number(values[0])
        if len(values) > 1 and isinstance(values[1], str) and values[1].strip().lower().startswith("deg"):
            angle = math.radians(angle)
        return angle

    def _should_return_degrees(self, values: List) -> bool:
        """Check whether an inverse trig result should be returned in degrees."""
        if len(values) > 1 and isinstance(values[1], str):
            return values[1].strip().lower().startswith("deg")
        return False

    def add(self, values: List[float]) -> str:
        """Add all the arguments passed to it, normalized to 2 decimal places.
        
        Args:
            values: List of numbers to add
            
        Returns:
            Sum of all values, normalized
        """
        numeric_values = self._parse_numeric_values(values)
        return self.normalize(sum(numeric_values))

    def subtract(self, values: List[float]) -> str:
        """Subtract all subsequent arguments from the first argument, normalized to 2 decimal places.
        
        Args:
            values: List of numbers (first number minus all others)
            
        Returns:
            Result of subtraction, normalized
        """
        numeric_values = self._parse_numeric_values(values)
        res = numeric_values[0]
        for arg in numeric_values[1:]:
            res -= arg
        return self.normalize(res)

    def multiply(self, values: List[float]) -> str:
        """Multiply all the arguments passed to it, normalized to 2 decimal places.
        
        Args:
            values: List of numbers to multiply
            
        Returns:
            Product of all values, normalized
        """
        numeric_values = self._parse_numeric_values(values)
        res = numeric_values[0]
        for arg in numeric_values[1:]:
            res *= arg
        return self.normalize(res)

    def divide(self, values: List[float]) -> str:
        """Divide the first argument by all subsequent arguments, normalized to 2 decimal places.
        
        Args:
            values: List of numbers (first number divided by all others)
            
        Returns:
            Result of division, normalized
        """
        numeric_values = self._parse_numeric_values(values)
        res = numeric_values[0]
        for arg in numeric_values[1:]:
            res /= arg
        return self.normalize(res)

    def power(self, values: List[float]) -> str:
        """Raise the first argument to the power of all subsequent arguments, normalized to 2 decimal places.
        
        Args:
            values: List of numbers (first number raised to power of all others)
            
        Returns:
            Result of power operation, normalized
        """
        numeric_values = self._parse_numeric_values(values)
        res = numeric_values[0]
        for arg in numeric_values[1:]:
            res **= arg
        return self.normalize(res)

    def sqrt(self, values: List[float]) -> str:
        """Calculate the square root of the first argument, normalized to 2 decimal places.
        
        Args:
            values: List containing one number to find square root of
            
        Returns:
            Square root, normalized
        """
        numeric_values = self._parse_numeric_values(values)
        res = numeric_values[0]
        return self.normalize(math.sqrt(res))

    def log(self, values: List[float]) -> str:
        """Calculate logarithm with base 10 (if one argument) or custom base (if two arguments).
        
        Args:
            values: List of numbers (first is the number, second is optional base)
            
        Returns:
            Logarithm result, normalized
        """
        if len(values) == 1:
            numeric_values = self._parse_numeric_values(values)
            res = numeric_values[0]
            return self.normalize(math.log10(res))
        elif len(values) == 2:
            numeric_values = self._parse_numeric_values(values)
            res = numeric_values[0]
            base = numeric_values[1]
            return self.normalize(math.log(res, base))
        else:
            raise ValueError("Invalid number of arguments passed to log function")

    def ln(self, values: List[float]) -> str:
        """Calculate the natural logarithm of the first argument, normalized to 2 decimal places.
        
        Args:
            values: List containing one number to find natural log of
            
        Returns:
            Natural logarithm, normalized
        """
        numeric_values = self._parse_numeric_values(values)
        res = numeric_values[0]
        return self.normalize(math.log(res))

    def choose(self, values: List[float]) -> str:
        """Calculate the number of ways to choose 'r' items from 'n' options without regard to order.
        
        Args:
            values: List containing [n, r] for combination calculation
            
        Returns:
            Number of combinations, normalized
        """
        numeric_values = self._parse_numeric_values(values)
        n = int(numeric_values[0])
        r = int(numeric_values[1])
        return self.normalize(math.comb(n, r))

    def permutate(self, values: List[float]) -> str:
        """Calculate the number of ways to arrange 'r' items out of 'n' options.
        
        Args:
            values: List containing [n, r] for permutation calculation
            
        Returns:
            Number of permutations, normalized
        """
        numeric_values = self._parse_numeric_values(values)
        n = int(numeric_values[0])
        r = int(numeric_values[1])
        return self.normalize(math.perm(n, r))

    def gcd(self, values: List[float]) -> str:
        """Calculate the greatest common divisor of all arguments, normalized to 2 decimal places.
        
        Args:
            values: List of numbers to find GCD of
            
        Returns:
            Greatest common divisor, normalized
        """
        # numeric_values = self._parse_numeric_values(values) #usually round up when convert float into str
        res = int(values[0])
        print("first_num", res)
        for arg in values[1:]:
            res = math.gcd(res, int(arg))
        return self.normalize(res)

    def lcm(self, values: List[float]) -> str:
        """Calculate the least common multiple of all arguments, normalized to 2 decimal places.
        
        Args:
            values: List of numbers to find LCM of
            
        Returns:
            Least common multiple, normalized
        """
        numeric_values = self._parse_numeric_values(values)
        res = int(numeric_values[0])
        for arg in numeric_values[1:]:
            res = res * int(arg) // math.gcd(res, int(arg))
        return self.normalize(res)

    def remainder(self, values: List[float]) -> str:
        """Calculate the remainder of the division of the first argument by the second argument.
        
        Args:
            values: List containing [dividend, divisor]
            
        Returns:
            Remainder, normalized
        """
        numeric_values = self._parse_numeric_values(values)
        dividend = numeric_values[0]
        divisor = numeric_values[1]
        return self.normalize(dividend % divisor)
    
    def exp(self, values: List[float]) -> str:
        """Calculate the exponential (e^x) of the first argument."""
        numeric_values = self._parse_numeric_values(values)
        return self.normalize(math.exp(numeric_values[0]))

    def factorial(self, values: List[float]) -> str:
        """Calculate the factorial of an integer argument."""
        numeric_values = self._parse_numeric_values(values)
        n = numeric_values[0]
        if n < 0 or (isinstance(n, float) and not n.is_integer()):
            raise ValueError("Factorial is only defined for non-negative integers")
        return self.normalize(math.factorial(int(n)))

    def sin(self, values: List[float]) -> str:
        """Calculate sine; accepts optional 'deg' flag for degrees."""
        angle = self._angle_from_values(values)
        return self.normalize(math.sin(angle))

    def cos(self, values: List[float]) -> str:
        """Calculate cosine; accepts optional 'deg' flag for degrees."""
        angle = self._angle_from_values(values)
        return self.normalize(math.cos(angle))

    def tan(self, values: List[float]) -> str:
        """Calculate tangent; accepts optional 'deg' flag for degrees."""
        angle = self._angle_from_values(values)
        return self.normalize(math.tan(angle))

    def asin(self, values: List[float]) -> str:
        """Calculate arcsine; optional 'deg' flag returns result in degrees."""
        numeric_values = self._parse_numeric_values([values[0]])
        result = math.asin(numeric_values[0])
        if self._should_return_degrees(values):
            result = math.degrees(result)
        return self.normalize(result)

    def acos(self, values: List[float]) -> str:
        """Calculate arccosine; optional 'deg' flag returns result in degrees."""
        numeric_values = self._parse_numeric_values([values[0]])
        result = math.acos(numeric_values[0])
        if self._should_return_degrees(values):
            result = math.degrees(result)
        return self.normalize(result)

    def atan(self, values: List[float]) -> str:
        """Calculate arctangent; supports atan(x) or atan2(y, x). Optional 'deg' flag for degrees output."""
        if not values:
            raise ValueError("atan requires at least one argument")

        return_degrees = False
        if isinstance(values[-1], str) and values[-1].strip().lower().startswith("deg"):
            return_degrees = True
            values = values[:-1]

        if not values:
            raise ValueError("atan requires numeric input before the degree flag")

        numeric_values = self._parse_numeric_values(values)
        if len(numeric_values) == 1:
            result = math.atan(numeric_values[0])
        else:
            result = math.atan2(numeric_values[0], numeric_values[1])

        if return_degrees:
            result = math.degrees(result)
        return self.normalize(result)

    def sinh(self, values: List[float]) -> str:
        """Calculate hyperbolic sine; accepts optional 'deg' flag for degrees input."""
        angle = self._angle_from_values(values)
        return self.normalize(math.sinh(angle))

    def cosh(self, values: List[float]) -> str:
        """Calculate hyperbolic cosine; accepts optional 'deg' flag for degrees input."""
        angle = self._angle_from_values(values)
        return self.normalize(math.cosh(angle))

    def tanh(self, values: List[float]) -> str:
        """Calculate hyperbolic tangent; accepts optional 'deg' flag for degrees input."""
        angle = self._angle_from_values(values)
        return self.normalize(math.tanh(angle))

    def convert_units(self, values: List) -> str:
        """Convert between common scientific units.
        
        Args:
            values: [value, from_unit, to_unit]
        """
        if len(values) != 3:
            raise ValueError("convert_units expects [value, from_unit, to_unit]")

        magnitude = self._parse_single_number(values[0])
        from_unit = str(values[1]).strip().lower()
        to_unit = str(values[2]).strip().lower()

        # Temperature conversions are non-linear and handled separately
        def convert_temperature(val: float, source: str, target: str) -> float:
            source = source.lower()
            target = target.lower()
            if source in ("c", "celsius", "degc"):
                celsius = val
            elif source in ("f", "fahrenheit", "degf"):
                celsius = (val - 32) * 5 / 9
            elif source in ("k", "kelvin"):
                celsius = val - 273.15
            else:
                raise ValueError(f"Unsupported temperature unit '{source}'")

            if target in ("c", "celsius", "degc"):
                return celsius
            if target in ("f", "fahrenheit", "degf"):
                return celsius * 9 / 5 + 32
            if target in ("k", "kelvin"):
                return celsius + 273.15
            raise ValueError(f"Unsupported temperature unit '{target}'")

        temperature_units = {"c", "celsius", "degc", "f", "fahrenheit", "degf", "k", "kelvin"}
        if from_unit in temperature_units and to_unit in temperature_units:
            return self.normalize(convert_temperature(magnitude, from_unit, to_unit))

        unit_categories = {
            "length": {
                "m": 1.0,
                "km": 1000.0,
                "cm": 0.01,
                "mm": 0.001,
                "um": 1e-6,
                "nm": 1e-9,
                "mi": 1609.344,
                "ft": 0.3048,
                "in": 0.0254,
            },
            "mass": {
                "kg": 1.0,
                "g": 0.001,
                "mg": 1e-6,
                "lb": 0.45359237,
                "oz": 0.028349523125,
            },
            "time": {
                "s": 1.0,
                "ms": 0.001,
                "us": 1e-6,
                "ns": 1e-9,
                "min": 60.0,
                "hr": 3600.0,
                "day": 86400.0,
            },
            "pressure": {
                "pa": 1.0,
                "kpa": 1e3,
                "mpa": 1e6,
                "bar": 1e5,
                "atm": 101325.0,
            },
            "energy": {
                "j": 1.0,
                "kj": 1e3,
                "mj": 1e6,
                "ev": 1.602176634e-19,
            },
        }

        for category, factors in unit_categories.items():
            if from_unit in factors and to_unit in factors:
                base_value = magnitude * factors[from_unit]
                converted = base_value / factors[to_unit]
                return self.normalize(converted)

        raise ValueError(f"Unsupported unit conversion from '{from_unit}' to '{to_unit}'")
    
    def run(self, operation: str, values: List) -> str:
        """Run the specified mathematical operation with the provided values.
        
        Args:
            operation: The mathematical operation to perform
            values: List of numerical values, constants, or unit strings for the operation

        Returns:
            Result of the mathematical operation
        """
        if operation not in self.func_map:
            available_ops = ', '.join(self.func_map.keys())
            return {"error": f"Unknown operation '{operation}'. Available operations: {available_ops}", "success": False}
        try:
            return {"result": self.func_map[operation](values), "success": True}
        except Exception as e:
            return {"error": f"Error occurred while executing '{operation}': {e}", "success": False, "traceback": traceback.format_exc()}
    
    def test(self) -> str:
        """Test the math tool with various operations."""
        try:
            # Open testbench
            file_test = os.path.join(os.path.dirname(__file__), '..', 'test_file', 'data.json')
            with open(file_test, encoding='utf-8') as f:
                data = json.load(f)['calculator']
            
            # Create test_results directory with timestamped filename
            tool_dir = os.path.dirname(__file__)
            test_results_dir = os.path.join(tool_dir, 'test_results')
            os.makedirs(test_results_dir, exist_ok=True)
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_result_{timestamp}.json"
            file_result = os.path.join(test_results_dir, filename)
            
            with open(file_result,"w",encoding="utf-8") as output_file:
                test_result = {}
                # Add metadata
                test_result['metadata'] = {
                    "tool_name": self.name,
                    "test_timestamp": datetime.now().isoformat(),
                    "test_file": "calculator",
                    "file_location": "calculator",
                    "result_file": filename,
                }
                # write the length of the testbench
                test_result['Test-File length'] = len(data)
                run_accuracy = {'run_1': 0, 'run_2': 0, 'run_3': 0}
                
                for i in range (0,len(data)):
                    test = data[i]
                    # Get operation and parameters
                    function, parameters = test['query'].split('(',1)
                    # format parameters
                    op = ['choose','permutate','gcd','lcm']
                    if (function in op):
                        parameters = list(map(int,parameters.rstrip(')').split(',')))
                    else :                        
                        parameters = list(map(float,parameters.rstrip(')').split(',')))
                    
                    # Prepare the question result
                    question_result = {"id": f"calculator_{i + 1}"}
                    if 'query' in test:
                        question_result['query'] = test['query']
                    if 'answer' in test:
                        question_result['answer'] = test['answer']
                    
                    # Run 3 times
                    for j in range(0,3):
                        run_result = {}
                        # Run the tool and retrieve the result
                        result = self.run(function, parameters)
                        # Check if the result is successful
                        if result['success'] == False:
                            run_result['tool_call_pass'] = False
                            run_result['result'] = result
                            run_result['accuracy'] = 0
                            question_result[f'run_{j + 1}'] = run_result
                            continue
                        else: 
                            run_result['tool_call_pass'] = True
                            run_result['result'] = result
                        
                        # Get the response from the result
                        response = result['result']
                        # Check correctness by comparing with expected answer
                        # Compare as strings to avoid floating point precision issues
                        if str(response) == str(test['answer']) or float(response) == test['answer']:
                            run_accuracy[f'run_{j + 1}'] += 1
                            run_result['accuracy'] = 1
                        else:
                            print(f"The query is {test['query']}, id is {test['id']}")
                            print(f"The response is {response} and the expected answer is {test['answer']}")
                            run_result['accuracy'] = 0
                        
                        print(f"Finish query: {i + 1}")
                        question_result[f'run_{j + 1}'] = run_result
                    
                    test_result[f'Q{i + 1}'] = question_result
                
                # Calculate the accuracy of the tool
                test_result['Final_Accuracy'] = {'run_1': run_accuracy['run_1']*100/len(data), 'run_2': run_accuracy['run_2']*100/len(data), 'run_3': run_accuracy['run_3']*100/len(data)}
                
                # Update metadata with final results
                test_result['metadata']['final_accuracy'] = test_result['Final_Accuracy']
                test_result['metadata']['total_questions'] = len(data)
                
                print(test_result['Final_Accuracy'])
                print(f"📁 Test result saved to: {file_result}")
                json.dump(test_result, output_file, indent=2, default=str)
        except Exception as e:
            print(f"❌ Math tool test failed: {e}")

if __name__ == "__main__":
    calculator_tool = Calculator_Tool()
    calculator_tool.embed_tool()
    try:
        calculator_tool.test()
    except Exception as e:
        print(f"❌ Math tool test failed: {e}")
