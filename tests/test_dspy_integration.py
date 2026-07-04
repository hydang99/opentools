import inspect
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))

from opentools.integrations.dspy import as_callable


class DSPyIntegrationTests(unittest.TestCase):
    def test_real_calculator_callable_has_schema_and_executes(self):
        calculator = as_callable("Calculator_Tool")

        signature = inspect.signature(calculator)
        observed = calculator(operation="add", values=[1, 2, 3])

        self.assertEqual(list(signature.parameters), ["operation", "values"])
        self.assertEqual(observed["result"], "6")
        self.assertTrue(observed["success"])


if __name__ == "__main__":
    unittest.main()
