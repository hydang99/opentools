import inspect
import importlib.util
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))

from opentools.integrations.dspy import as_callable, build_dspy_agent


class DSPyIntegrationTests(unittest.TestCase):
    def test_real_calculator_callable_has_schema_and_executes(self):
        calculator = as_callable("Calculator_Tool")

        signature = inspect.signature(calculator)
        observed = calculator(operation="add", values=[1, 2, 3])

        self.assertEqual(list(signature.parameters), ["operation", "values"])
        self.assertEqual(observed["result"], "6")
        self.assertTrue(observed["success"])

    @unittest.skipUnless(importlib.util.find_spec("dspy"), "optional dspy package is not installed")
    def test_builds_compilable_agent_with_real_opentools_callable(self):
        import dspy

        agent = build_dspy_agent(["Calculator_Tool"], max_steps=3)
        observed = agent.functions["Calculator_Tool"](
            operation="multiply", values=[6, 7]
        )

        self.assertIsInstance(agent, dspy.Module)
        self.assertIn("finish", agent.functions)
        self.assertEqual(observed["result"], "42")


if __name__ == "__main__":
    unittest.main()
