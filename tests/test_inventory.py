import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "opentools" / "inventory.py"
SPEC = importlib.util.spec_from_file_location("opentools_inventory", MODULE_PATH)
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


TOOL_SOURCE = '''
from opentools.core.base import BaseTool

class Example_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            name="Example_Tool",
            description="A deterministic example tool.",
            category="test",
            tags=["test"],
            parameters={"type": "object", "properties": {}},
            agent_type="Test-Agent",
            demo_commands={},
            limitation="Tests only",
            execution_type="local",
        )
'''


class InventoryTests(unittest.TestCase):
    def _fixture(self, root: Path):
        tools = root / "src" / "opentools" / "tools"
        tool_dir = tools / "example"
        tool_dir.mkdir(parents=True)
        (tool_dir / "tool.py").write_text(TOOL_SOURCE, encoding="utf-8")
        return tools, tool_dir

    def test_index_reads_real_result_and_has_no_volatile_generation_time(self):
        with tempfile.TemporaryDirectory() as directory:
            tools, tool_dir = self._fixture(Path(directory))
            (tool_dir / "test_result.json").write_text(
                json.dumps(
                    {
                        "Final_Accuracy": {"run_1": 80, "run_2": 100},
                        "Test-File length": 5,
                        "metadata": {"test_timestamp": "2026-06-20T12:00:00+00:00"},
                    }
                ),
                encoding="utf-8",
            )

            index = inventory.build_index(
                tools,
                stale_after_days=30,
                now=datetime(2026, 6, 30, tzinfo=timezone.utc),
            )

            record = index["tools"]["Example_Tool"]
            self.assertEqual(record["average_accuracy"], 90.0)
            self.assertEqual(record["total_questions"], 5)
            self.assertEqual(record["freshness"], "current")
            self.assertNotIn("generated_at", index)

    def test_failed_refresh_preserves_historical_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            tools, tool_dir = self._fixture(Path(directory))
            (tool_dir / "test_result.json").write_text(
                json.dumps({"Final_Accuracy": {"run_1": 75}, "Test-File length": 4}),
                encoding="utf-8",
            )

            index = inventory.build_index(
                tools,
                run_results={"Example_Tool": {"status": "failed", "error": "observed"}},
            )

            record = index["tools"]["Example_Tool"]
            self.assertEqual(record["evaluation_status"], "failed")
            self.assertEqual(record["average_accuracy"], 75.0)

    def test_inventory_table_is_generated_between_stable_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools, tool_dir = self._fixture(root)
            (tool_dir / "test_result.json").write_text(
                json.dumps({"Final_Accuracy": {"run_1": 100}, "Test-File length": 1}),
                encoding="utf-8",
            )
            readme = root / "readme.md"
            readme.write_text(
                "# Tools\n\n"
                "| Tool name (folder) | Short description | Tool type | Evaluated? | Test suite key (`test_file/data.json`) | Evaluation metrics | Current accuracy |\n"
                "|---|---|---|---|---|---|---|\n"
                "| [`example`](./example/) | Existing description. | local_processing | ✅ | `example` | exact_match | 50 |\n\n"
                "## Tool Architecture\n",
                encoding="utf-8",
            )
            index = inventory.build_index(tools)

            inventory.update_inventory_markdown(index, tools, readme)
            first = readme.read_text(encoding="utf-8")
            inventory.update_inventory_markdown(index, tools, readme)
            second = readme.read_text(encoding="utf-8")

            self.assertEqual(first, second)
            self.assertIn(inventory.BEGIN_MARKER, first)
            self.assertIn("| 100.0 | low |", first)
            self.assertIn("Existing description.", first)

    def test_bulk_evaluation_skips_restricted_source_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            tools, tool_dir = self._fixture(Path(directory))
            (tool_dir / "tool.py").write_text(
                TOOL_SOURCE + "\nimport subprocess\n",
                encoding="utf-8",
            )

            results = inventory.run_bulk_evaluations(
                tools,
                selected_tools=["Example_Tool"],
                max_risk="low",
            )

            self.assertEqual(
                results["Example_Tool"]["status"], "skipped_by_risk_policy"
            )

    def test_bulk_evaluation_records_unknown_requested_tool(self):
        with tempfile.TemporaryDirectory() as directory:
            tools, _ = self._fixture(Path(directory))

            results = inventory.run_bulk_evaluations(
                tools,
                selected_tools=["Missing_Tool"],
            )

            self.assertEqual(results["unknown:Missing_Tool"]["status"], "failed")
            self.assertIn("Tool not found", results["unknown:Missing_Tool"]["error"])


if __name__ == "__main__":
    unittest.main()
