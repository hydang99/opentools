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
                DEFAULT_TEST_ARGS = {"case": "real"}

                def test(self, case):
                    self.assert_case(case)
                    output = tmp_path / "test_results" / "test_result_observed.json"
                    output.parent.mkdir()
                    output.write_text(
                        json.dumps(
                            {
                                "Final_Accuracy": {"run_1": 75.0},
                                "Test-File length": 4,
                            }
                        ),
                        encoding="utf-8",
                    )
                    return True

                @staticmethod
                def assert_case(case):
                    if case != "real":
                        raise AssertionError(case)

            result = evaluation.run_existing_tests(Tool(), source)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["final_accuracy"], {"run_1": 75.0})
            self.assertEqual(result["total_questions"], 4)

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

    def test_llm_judge_is_advisory_and_cannot_override_restricted_preflight(self):
        class Engine:
            def generate(self, prompt, **kwargs):
                self.prompt = prompt
                return json.dumps(
                    {
                        "recommendation": "approve",
                        "scores": {
                            "documentation": 4,
                            "test_evidence": 3,
                            "output_contract": 4,
                            "maintainability": 3,
                        },
                        "concerns": ["Subprocess capability requires manual review."],
                        "required_actions": ["Review the process invocation."],
                        "rationale": "The metadata is clear, but static findings remain authoritative.",
                    }
                )

        engine = Engine()
        report = {
            "inspection": {
                "risk_level": "restricted",
                "files_scanned": 1,
                "observed_credentials": [],
                "findings": [
                    {
                        "kind": "process_execution",
                        "file": "/private/project/example/tool.py",
                    }
                ],
                "parse_errors": [],
            },
            "tool_card": {"name": "Example", "evaluation": {"status": "completed"}},
        }

        result = evaluation.judge_evaluation_report(report, engine=engine)

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["advisory_only"])
        self.assertFalse(result["can_override_preflight"])
        self.assertTrue(result["preflight_blocked"])
        self.assertFalse(result["eligible_for_automatic_acceptance"])
        self.assertNotIn("secret", engine.prompt.lower())
        self.assertNotIn("/private/project", engine.prompt)

    def test_llm_judge_rejects_unstructured_output(self):
        class Engine:
            def generate(self, prompt, **kwargs):
                return "Looks good to me"

        report = {
            "inspection": {"risk_level": "low"},
            "tool_card": {"name": "Example"},
        }

        result = evaluation.judge_evaluation_report(report, engine=Engine())

        self.assertEqual(result["status"], "failed")
        self.assertIn("JSONDecodeError", result["error"])


if __name__ == "__main__":
    unittest.main()
