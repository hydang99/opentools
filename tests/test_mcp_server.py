import asyncio
import importlib.util
import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path


SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))

MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None


@contextmanager
def environment(values):
    original = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@unittest.skipUnless(MCP_AVAILABLE, "optional mcp package is not installed")
class MCPServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from opentools import mcp_server

        cls.server = mcp_server.create_server()
        cls.tools = {
            name: registered.fn
            for name, registered in cls.server._tool_manager._tools.items()
        }

    def test_registers_bounded_mcp_surface(self):
        self.assertEqual(
            set(self.tools),
            {
                "list_opentools",
                "inspect_opentool",
                "evaluate_opentool",
                "call_opentool",
            },
        )

    def test_execution_is_disabled_by_default(self):
        with environment(
            {
                "OPENTOOLS_MCP_ALLOW_EXECUTION": None,
                "OPENTOOLS_MCP_ALLOWED_TOOLS": "Calculator_Tool",
            }
        ):
            result = self.tools["evaluate_opentool"]("Calculator_Tool")

        self.assertEqual(result["status"], "execution_disabled")

    def test_inspection_reads_real_metadata_without_absolute_paths(self):
        result = self.tools["inspect_opentool"]("Calculator_Tool")

        self.assertEqual(result["tool_card"]["name"], "Calculator_Tool")
        self.assertEqual(result["inspection"]["risk_level"], "low")
        self.assertNotIn(str(Path.cwd()), str(result))

    def test_real_restricted_tool_cannot_run(self):
        with environment(
            {
                "OPENTOOLS_MCP_ALLOW_EXECUTION": "1",
                "OPENTOOLS_MCP_ALLOWED_TOOLS": "Xlsxe_Extraction_Tool",
                "OPENTOOLS_MCP_MAX_RISK": "caution",
            }
        ):
            result = self.tools["evaluate_opentool"]("Xlsxe_Extraction_Tool")

        self.assertEqual(result["status"], "blocked_by_preflight")

    def test_calls_real_allowlisted_calculator_tool(self):
        with environment(
            {
                "OPENTOOLS_MCP_ALLOW_TOOL_CALLS": "1",
                "OPENTOOLS_MCP_ALLOWED_TOOLS": "Calculator_Tool",
                "OPENTOOLS_MCP_MAX_RISK": "low",
            }
        ):
            result = asyncio.run(
                self.tools["call_opentool"](
                    "Calculator_Tool",
                    {"operation": "add", "values": [1, 2, 3]},
                )
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["result"], "6")


if __name__ == "__main__":
    unittest.main()
