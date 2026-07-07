"""Run optional third-party security scanners without exposing matched secrets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


RISK_ORDER = {"low": 0, "caution": 1, "restricted": 2}


def _scanner_env() -> Dict[str, str]:
    env = {
        **os.environ,
        "NO_COLOR": "1",
        "SEMGREP_SEND_METRICS": "off",
        "SEMGREP_ENABLE_VERSION_CHECK": "0",
        "SEMGREP_LOG_FILE": str(Path(tempfile.gettempdir()) / "opentools-semgrep.log"),
    }
    if "SSL_CERT_FILE" not in env and Path("/etc/ssl/cert.pem").is_file():
        env["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"
    return env


def _relative_file(value: Any, source: Path) -> str:
    path = Path(str(value or "unknown"))
    try:
        base = source if source.is_dir() else source.parent
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        return path.name or "unknown"


def _result(tool: str, status: str, findings: Optional[List[Dict[str, Any]]] = None,
            *, detail: Optional[str] = None) -> Dict[str, Any]:
    items = findings or []
    result: Dict[str, Any] = {
        "tool": tool,
        "status": status,
        "findings_count": len(items),
        "findings": items,
    }
    if detail:
        result["detail"] = detail
    return result


def _run_json(
    tool: str,
    executable: str,
    command: List[str],
    parser: Callable[[Any], List[Dict[str, Any]]],
    *,
    accepted_codes: Iterable[int] = (0,),
    timeout: int,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    binary = shutil.which(executable)
    if not binary:
        return _result(tool, "unavailable", detail=f"{executable} is not installed")
    command[0] = binary
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_scanner_env(),
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return _result(tool, "failed", detail=f"timed out after {timeout} seconds")
    except OSError as exc:
        return _result(tool, "failed", detail=f"{type(exc).__name__}: scanner could not start")
    if completed.returncode not in set(accepted_codes):
        return _result(tool, "failed", detail=f"scanner exited with code {completed.returncode}")
    try:
        payload = json.loads(completed.stdout or "{}")
        findings = parser(payload)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        return _result(tool, "failed", detail="scanner returned invalid JSON")
    return _result(tool, "findings" if findings else "completed", findings)


def _gitleaks(source: Path, timeout: int) -> Dict[str, Any]:
    binary = shutil.which("gitleaks")
    if not binary:
        return _result("gitleaks", "unavailable", detail="gitleaks is not installed")
    with tempfile.TemporaryDirectory(prefix="opentools-gitleaks-") as directory:
        report = Path(directory) / "report.json"
        command = [
            binary, "dir", str(source), "--redact=100", "--report-format", "json",
            "--report-path", str(report), "--no-banner", "--exit-code", "1",
            "--timeout", str(timeout),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout + 2, check=False,
                env={**os.environ, "NO_COLOR": "1"},
            )
        except subprocess.TimeoutExpired:
            return _result("gitleaks", "failed", detail=f"timed out after {timeout} seconds")
        except OSError:
            return _result("gitleaks", "failed", detail="scanner could not start")
        if completed.returncode not in (0, 1):
            return _result("gitleaks", "failed", detail=f"scanner exited with code {completed.returncode}")
        try:
            payload = json.loads(report.read_text(encoding="utf-8")) if report.exists() else []
            findings = [
                {
                    "rule_id": str(item.get("RuleID") or "secret"),
                    "category": str(item.get("Description") or "possible secret")[:200],
                    "file": _relative_file(item.get("File"), source),
                    "line": item.get("StartLine"),
                    "severity": "restricted",
                }
                for item in payload
                if isinstance(item, dict)
            ]
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return _result("gitleaks", "failed", detail="scanner returned invalid JSON")
        return _result("gitleaks", "findings" if findings else "completed", findings)


def _detect_secrets(source: Path, timeout: int) -> Dict[str, Any]:
    def parse(payload: Any) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for filename, records in (payload.get("results") or {}).items():
            for item in records or []:
                findings.append({
                    "rule_id": "detect-secrets",
                    "category": str(item.get("type") or "possible secret")[:200],
                    "file": _relative_file(filename, source),
                    "line": item.get("line_number"),
                    "severity": "restricted",
                })
        return findings

    return _run_json(
        "detect-secrets", "detect-secrets",
        ["detect-secrets", "scan", "--all-files", "--no-verify", source.name],
        parse, timeout=timeout, cwd=source.parent,
    )


def _bandit(source: Path, timeout: int) -> Dict[str, Any]:
    def parse(payload: Any) -> List[Dict[str, Any]]:
        return [
            {
                "rule_id": str(item.get("test_id") or "bandit"),
                "category": str(item.get("test_name") or "Python security issue")[:200],
                "file": _relative_file(item.get("filename"), source),
                "line": item.get("line_number"),
                "severity": str(item.get("issue_severity") or "MEDIUM").lower(),
                "confidence": str(item.get("issue_confidence") or "UNDEFINED").lower(),
            }
            for item in (payload.get("results") or [])
            if isinstance(item, dict)
        ]

    return _run_json(
        "bandit", "bandit", ["bandit", "-r", str(source), "-f", "json", "-q"],
        parse, accepted_codes=(0, 1), timeout=timeout,
    )


def _semgrep(source: Path, timeout: int, config: Path) -> Dict[str, Any]:
    if not config.is_file():
        return _result("semgrep", "failed", detail="local OpenTools ruleset is missing")

    def parse(payload: Any) -> List[Dict[str, Any]]:
        findings = []
        for item in payload.get("results") or []:
            extra = item.get("extra") or {}
            findings.append({
                "rule_id": str(item.get("check_id") or "semgrep"),
                "category": str(extra.get("message") or "matched local security rule")[:200],
                "file": _relative_file(item.get("path"), source),
                "line": (item.get("start") or {}).get("line"),
                "severity": str(extra.get("severity") or "WARNING").lower(),
            })
        return findings

    return _run_json(
        "semgrep", "semgrep",
        ["semgrep", "scan", "--config", str(config), "--json", "--quiet", "--no-autofix", str(source)],
        parse, accepted_codes=(0, 1), timeout=timeout,
    )


def _scanner_risk(result: Dict[str, Any]) -> str:
    if result["status"] != "findings":
        return "low"
    if result["tool"] in {"gitleaks", "detect-secrets"}:
        return "restricted"
    severities = {str(item.get("severity", "")).lower() for item in result["findings"]}
    return "restricted" if severities & {"high", "error", "critical"} else "caution"


def run_external_scanners(
    source: Path | str,
    *,
    semgrep_config: Optional[Path | str] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Run installed scanners and return only sanitized findings."""
    source_path = Path(source).expanduser().resolve()
    seconds = timeout or int(os.getenv("OPENTOOLS_SCANNER_TIMEOUT_SECONDS", "30"))
    config = Path(semgrep_config) if semgrep_config else Path(__file__).with_name("security") / "semgrep.yml"
    scanners = [
        _gitleaks(source_path, seconds),
        _detect_secrets(source_path, seconds),
        _bandit(source_path, seconds),
        _semgrep(source_path, seconds, config),
    ]
    available = [item for item in scanners if item["status"] != "unavailable"]
    failures = [item for item in available if item["status"] == "failed"]
    risk = max((_scanner_risk(item) for item in scanners), key=RISK_ORDER.get, default="low")
    return {
        "status": "unavailable" if not available else "partial" if failures or len(available) < len(scanners) else "completed",
        "risk_level": risk,
        "findings_count": sum(item["findings_count"] for item in scanners),
        "scanners": scanners,
        "caution": "Scanner findings require maintainer review and do not certify that code is safe.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run optional OpenTools security scanners")
    parser.add_argument("source")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fail-on", choices=("caution", "restricted"))
    args = parser.parse_args()
    report = run_external_scanners(args.source, timeout=args.timeout)
    print(json.dumps(report, indent=2) if args.json else f"{report['status']}: {report['risk_level']} ({report['findings_count']} findings)")
    if args.fail_on and RISK_ORDER[report["risk_level"]] >= RISK_ORDER[args.fail_on]:
        return 2
    return 0 if report["status"] not in {"failed", "unavailable"} else 2


if __name__ == "__main__":
    sys.exit(main())
