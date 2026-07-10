import json
import shutil
import tempfile
import unittest
from pathlib import Path

from opentools.external_scanners import run_external_scanners


class ExternalScannerIntegrationTests(unittest.TestCase):
    def test_installed_scanners_run_for_real_and_redact_values(self):
        installed = {
            name for name in ("gitleaks", "detect-secrets", "bandit", "semgrep")
            if shutil.which(name)
        }
        if not installed:
            self.skipTest("No optional external scanner is installed")

        with tempfile.TemporaryDirectory() as directory:
            secret = "sk-proj-" + "Q7mV9kR2xP4nT8wL6cD3sF5hJ1bN0aZ8uE2iG6oC"
            source = Path(directory) / "tool.py"
            source.write_text(
                "import pickle\nimport subprocess\n"
                f"OPENAI_API_KEY = {secret!r}\n"
                "def run(command, payload):\n"
                "    subprocess.run(command, shell=True)\n"
                "    return pickle.loads(payload)\n",
                encoding="utf-8",
            )

            report = run_external_scanners(source, timeout=30)
            by_name = {item["tool"]: item for item in report["scanners"]}

            for name in installed:
                self.assertNotEqual(by_name[name]["status"], "unavailable")
                self.assertNotEqual(by_name[name]["status"], "failed")
            self.assertNotIn(secret, json.dumps(report))
            self.assertEqual(report["risk_level"], "restricted")


if __name__ == "__main__":
    unittest.main()
