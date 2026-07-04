import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "opentools" / "evaluation.py"
SPEC = importlib.util.spec_from_file_location("opentools_evaluation", MODULE_PATH)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def test_static_inspection_does_not_execute_source(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            marker = tmp_path / "imported.txt"
            source = tmp_path / "tool.py"
            source.write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
                encoding="utf-8",
            )

            report = evaluation.inspect_source(source)

            self.assertFalse(marker.exists())
            self.assertEqual(report["risk_level"], "caution")
            self.assertTrue(
                any(item["kind"] == "filesystem_write" for item in report["findings"])
            )

    def test_reports_network_credentials_and_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tool.py"
            source.write_text(
                "import os\nimport requests\nimport subprocess\n"
                "token = os.getenv('SERVICE_API_KEY')\n"
                "subprocess.run(['example'], check=True)\n",
                encoding="utf-8",
            )

            report = evaluation.inspect_source(source)

            self.assertEqual(report["risk_level"], "restricted")
            self.assertEqual(report["observed_credentials"], ["SERVICE_API_KEY"])
            self.assertGreaterEqual(
                {item["kind"] for item in report["findings"]},
                {"credential_access", "network_access", "process_execution"},
            )

    def test_metadata_validation_separates_required_and_recommended_fields(self):
        result = evaluation.validate_metadata(
            {
                "name": "Example",
                "description": "Example tool",
                "category": "test",
                "tags": ["example"],
                "parameters": {"type": "object"},
                "limitation": "For tests only",
            }
        )

        self.assertEqual(result["missing_required"], [])
        self.assertIn("source_url", result["missing_recommended"])

    def test_runner_uses_only_written_result_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = tmp_path / "tool.py"
            source.write_text("# test source\n", encoding="utf-8")

            class Tool:
                DEFAULT_TEST_ARGS = {"cases": [(2, 3, 5), (4, 7, 11)]}

                def test(self, cases):
                    passed = sum(left + right == expected for left, right, expected in cases)
                    accuracy = 100.0 * passed / len(cases)
                    output = tmp_path / "test_results" / "test_result_observed.json"
                    output.parent.mkdir()
                    output.write_text(
                        json.dumps(
                            {
                                "Final_Accuracy": {"run_1": accuracy},
                                "Test-File length": len(cases),
                            }
                        ),
                        encoding="utf-8",
                    )
                    return passed == len(cases)

            result = evaluation.run_existing_tests(Tool(), source)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["final_accuracy"], {"run_1": 100.0})
            self.assertEqual(result["total_questions"], 2)

    def test_runner_does_not_claim_results_without_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tool.py"
            source.write_text("# test source\n", encoding="utf-8")

            class Tool:
                def test(self):
                    return True

            result = evaluation.run_existing_tests(Tool(), source)

            self.assertEqual(result["status"], "completed_without_structured_results")
            self.assertNotIn("final_accuracy", result)

    def test_llm_judge_does_not_run_without_tool_metadata(self):
        result = evaluation.judge_evaluation_report(
            {"inspection": {"risk_level": "low"}, "tool_card": None}
        )

        self.assertEqual(result["status"], "not_run")


if __name__ == "__main__":
    unittest.main()
