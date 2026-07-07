import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))

from opentools.conversion import convert_submission
from opentools.core.base import BaseTool


SOURCE = '''
def add_numbers(left: int, right: int) -> int:
    """Add two integer values."""
    return left + right
'''


class ConversionTests(unittest.TestCase):
    def _files(self, root: Path):
        source = root / "submitted.py"
        readme = root / "README.md"
        source.write_text(SOURCE, encoding="utf-8")
        readme.write_text("# Submitted adder\n", encoding="utf-8")
        return source, readme

    def test_converts_and_runs_real_submitted_function(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, readme = self._files(root)
            result = convert_submission(
                source,
                readme,
                root / "contributions",
                name="Submitted Adder",
                license_name="Apache-2.0",
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                result["manifest"]["functional_evaluation"]["status"], "not_run"
            )
            tool_file = Path(result["bundle"]) / "tool.py"
            spec = importlib.util.spec_from_file_location("converted_adder", tool_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            classes = [
                value
                for _, value in inspect.getmembers(module, inspect.isclass)
                if value is not BaseTool and issubclass(value, BaseTool)
            ]
            tool = classes[0]()
            observed = tool.run(left=2, right=3)

            self.assertEqual(observed, {"result": 5, "success": True})
            manifest = json.loads(
                (Path(result["bundle"]) / "contribution.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["functional_evaluation"]["status"], "not_run")

if __name__ == "__main__":
    unittest.main()
