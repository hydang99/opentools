"""Local tool inspection and evidence-based evaluation reports.

The preflight inspection in this module is intentionally conservative. It surfaces
capabilities that deserve review; it does not claim that Python source is safe.
Tool code is never imported or executed by :func:`inspect_source`.
"""

from __future__ import annotations

import ast
import argparse
import inspect
import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


NETWORK_MODULES = {
    "aiohttp",
    "arxiv",
    "boto3",
    "eutils",
    "google",
    "habanero",
    "httpx",
    "metapub",
    "openai",
    "pymed",
    "requests",
    "socket",
    "urllib",
    "waybackpy",
    "wikipedia",
    "yfinance",
    "youtube_transcript_api",
    "yt_dlp",
}
PROCESS_MODULES = {"commands", "subprocess"}
SECRET_SUFFIXES = ("_API_KEY", "_KEY", "_PASSWORD", "_SECRET", "_TOKEN")
SECRET_NAME_PATTERN = re.compile(
    r"(?:api[_-]?key|password|secret|token|credential|access[_-]?key|private[_-]?key|client[_-]?secret)",
    re.IGNORECASE,
)
KNOWN_SECRET_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("openai_token", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("credential_in_url", re.compile(r"[a-z][a-z0-9+.-]*://[^\s/:]+:[^\s/@]+@", re.IGNORECASE)),
)
PLACEHOLDER_MARKERS = (
    "changeme",
    "dummy",
    "example",
    "placeholder",
    "replace_me",
    "test_only",
    "your_api",
    "your_key",
    "your_token",
)
WRITE_METHODS = {
    "mkdir",
    "rmdir",
    "touch",
    "unlink",
    "write_bytes",
    "write_text",
}


def _python_files(source: Path) -> List[Path]:
    if source.is_file():
        return [source] if source.suffix == ".py" else []
    if source.is_dir():
        return sorted(path for path in source.rglob("*.py") if ".git" not in path.parts)
    return []


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        value = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def _string_value(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _known_secret_category(value: str) -> Optional[str]:
    if any(marker in value.lower() for marker in PLACEHOLDER_MARKERS):
        return None
    for category, pattern in KNOWN_SECRET_PATTERNS:
        if pattern.search(value):
            return category
    return None


def _target_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _string_value(node.slice)
    return None


class _CapabilityVisitor(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.findings: List[Dict[str, Any]] = []
        self.imports: Set[str] = set()
        self.credentials: Set[str] = set()
        self.scope: List[str] = []
        self.secret_locations: Set[tuple[int | None, str]] = set()

    def _add(self, node: ast.AST, kind: str, severity: str, detail: str) -> None:
        finding = {
            "kind": kind,
            "severity": severity,
            "file": self.filename,
            "line": getattr(node, "lineno", None),
            "scope": ".".join(self.scope) or "module",
            "detail": detail,
        }
        key = (kind, self.filename, finding["line"], detail)
        if key not in {
            (item["kind"], item["file"], item["line"], item["detail"])
            for item in self.findings
        }:
            self.findings.append(finding)

    def _add_secret(self, node: ast.AST, category: str, context: str) -> None:
        key = (getattr(node, "lineno", None), category)
        if key in self.secret_locations:
            return
        self.secret_locations.add(key)
        self._add(
            node,
            "hardcoded_secret",
            "restricted",
            f"possible {category} in {context}; value redacted",
        )

    def _inspect_named_value(self, node: ast.AST, name: Optional[str], value: ast.AST) -> None:
        literal = _string_value(value)
        if literal is None:
            return
        category = _known_secret_category(literal)
        if category:
            self._add_secret(node, category, f"{name or 'string literal'}")
        elif (
            name
            and SECRET_NAME_PATTERN.search(name)
            and len(literal.strip()) >= 8
            and not any(marker in literal.lower() for marker in PLACEHOLDER_MARKERS)
        ):
            self._add_secret(node, "credential_literal", name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._inspect_named_value(node, _target_name(target), node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._inspect_named_value(node, _target_name(node.target), node.value)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            self.imports.add(root)
            if root in NETWORK_MODULES:
                self._add(node, "network_access", "caution", f"imports {alias.name}")
            if root in PROCESS_MODULES:
                self._add(node, "process_execution", "restricted", f"imports {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root:
            self.imports.add(root)
        if root in NETWORK_MODULES:
            self._add(node, "network_access", "caution", f"imports from {node.module}")
        if root in PROCESS_MODULES:
            self._add(node, "process_execution", "restricted", f"imports from {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        if name in {"eval", "exec", "compile", "os.system", "os.popen"}:
            self._add(node, "dynamic_execution", "restricted", f"calls {name}")
        elif name.startswith("subprocess."):
            self._add(node, "process_execution", "restricted", f"calls {name}")

        if name.endswith("create_llm_engine"):
            self._add(node, "network_access", "caution", f"initializes an LLM engine via {name}")
        for keyword in node.keywords:
            self._inspect_named_value(node, keyword.arg, keyword.value)
            if (
                keyword.arg == "require_llm_engine"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                self._add(node, "network_access", "caution", "declares an LLM engine requirement")

        if name == "open":
            mode = None
            if len(node.args) > 1:
                mode = _string_value(node.args[1])
            for keyword in node.keywords:
                if keyword.arg == "mode":
                    mode = _string_value(keyword.value)
            if mode and any(flag in mode for flag in ("w", "a", "x", "+")):
                if self.scope and self.scope[-1].startswith("test"):
                    self._add(
                        node,
                        "evaluation_artifact_write",
                        "info",
                        f"test routine opens a file in mode {mode!r}",
                    )
                else:
                    self._add(node, "filesystem_write", "caution", f"opens a file in mode {mode!r}")
        elif name.split(".")[-1] in WRITE_METHODS:
            if self.scope and self.scope[-1].startswith("test"):
                self._add(
                    node,
                    "evaluation_artifact_write",
                    "info",
                    f"test routine calls {name}",
                )
            else:
                self._add(node, "filesystem_write", "caution", f"calls {name}")

        if name in {"os.getenv", "os.environ.get", "getenv"} and node.args:
            credential = _string_value(node.args[0])
            if credential and credential.upper().endswith(SECRET_SUFFIXES):
                self.credentials.add(credential)
                self._add(
                    node,
                    "credential_access",
                    "caution",
                    f"reads environment credential {credential}",
                )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            category = _known_secret_category(node.value)
            if category:
                self._add_secret(node, category, "string literal")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name) and (
                node.value.value.id == "os" and node.value.attr == "environ"
            ):
                credential = _string_value(node.slice)
                if credential and credential.upper().endswith(SECRET_SUFFIXES):
                    self.credentials.add(credential)
                    self._add(
                        node,
                        "credential_access",
                        "caution",
                        f"reads environment credential {credential}",
                    )
        self.generic_visit(node)


def inspect_source(source: Path | str) -> Dict[str, Any]:
    """Statically inspect Python source without importing or executing it."""
    source_path = Path(source).expanduser().resolve()
    files = _python_files(source_path)
    findings: List[Dict[str, Any]] = []
    imports: Set[str] = set()
    credentials: Set[str] = set()
    parse_errors: List[Dict[str, Any]] = []

    for path in files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            parse_errors.append({"file": str(path), "error": str(exc)})
            continue
        visitor = _CapabilityVisitor(str(path))
        visitor.visit(tree)
        findings.extend(visitor.findings)
        imports.update(visitor.imports)
        credentials.update(visitor.credentials)

    severities = {finding["severity"] for finding in findings}
    if parse_errors or "restricted" in severities:
        risk_level = "restricted"
    elif "caution" in severities:
        risk_level = "caution"
    else:
        risk_level = "low"

    return {
        "source": str(source_path),
        "files_scanned": len(files),
        "risk_level": risk_level,
        "caution": (
            "Static inspection identifies review signals only and is not a security guarantee."
        ),
        "observed_credentials": sorted(credentials),
        "observed_imports": sorted(imports),
        "possible_secret_count": sum(
            finding["kind"] == "hardcoded_secret" for finding in findings
        ),
        "findings": sorted(
            findings,
            key=lambda item: (item["file"], item["line"] or 0, item["kind"]),
        ),
        "parse_errors": parse_errors,
    }


def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate core tool-card metadata while keeping legacy tools compatible."""
    required = ("name", "description", "category", "tags", "parameters", "limitation")
    recommended = (
        "version",
        "source_url",
        "license",
        "execution",
        "safety",
        "usage",
    )
    missing_required = [field for field in required if not metadata.get(field)]
    missing_recommended = [field for field in recommended if not metadata.get(field)]
    return {
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
    }


def build_tool_card(
    tool: Any,
    inspection: Optional[Dict[str, Any]] = None,
    evaluation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a serializable tool card from declared and observed evidence."""
    metadata = dict(getattr(tool, "get_metadata", lambda: {})())
    evaluation_data = {
        "status": "not_run",
        "reported_accuracy": metadata.get("accuracy"),
    }
    if isinstance(metadata.get("evaluation_record"), dict):
        evaluation_data.update(metadata["evaluation_record"])
    if evaluation:
        evaluation_data.update(evaluation)
    card = {
        "name": metadata.get("name", tool.__class__.__name__),
        "version": metadata.get("version", "unspecified"),
        "description": metadata.get("description"),
        "category": metadata.get("category"),
        "tags": metadata.get("tags", []),
        "parameters": metadata.get("parameters", {}),
        "provenance": {
            "source_url": metadata.get("source_url"),
            "license": metadata.get("license"),
        },
        "execution": metadata.get("execution", {}),
        "safety": metadata.get("safety", {}),
        "usage": metadata.get("usage", {}),
        "evaluation": evaluation_data,
    }
    if inspection:
        card["safety"] = {
            **card["safety"],
            "observed_risk_level": inspection["risk_level"],
            "observed_credentials": inspection["observed_credentials"],
            "findings": inspection["findings"],
            "caution": inspection["caution"],
        }
    card["metadata_validation"] = validate_metadata({**metadata, **card})
    return card


def _result_files(source: Path) -> Set[Path]:
    root = source.parent if source.is_file() else source
    return set(root.glob("test_results/test_result_*.json"))


def _test_kwargs(tool: Any) -> Dict[str, Any]:
    signature = inspect.signature(tool.test)
    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.name != "self"
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    defaults = getattr(tool.__class__, "DEFAULT_TEST_ARGS", None)
    if defaults is None and hasattr(tool, "get_default_test_args"):
        defaults = tool.get_default_test_args()
    kwargs = dict(defaults or {})
    missing = [parameter.name for parameter in required if parameter.name not in kwargs]
    if missing:
        raise ValueError(
            "test() requires arguments with no defaults: " + ", ".join(missing)
        )
    return kwargs


def run_existing_tests(tool: Any, source: Path | str) -> Dict[str, Any]:
    """Run a tool's real test routine and record only observed result evidence."""
    source_path = Path(source).resolve()
    before = _result_files(source_path)
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        returned = tool.test(**_test_kwargs(tool))
    except Exception as exc:  # The report must preserve the actual failure.
        return {
            "status": "failed",
            "started_at": started_at,
            "error": f"{type(exc).__name__}: {exc}",
        }

    after = _result_files(source_path)
    new_files = sorted(after - before, key=lambda path: path.stat().st_mtime)
    evidence = None
    evidence_file = None
    if new_files:
        evidence_file = new_files[-1]
        try:
            evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            evidence = None

    if returned is False:
        status = "failed"
    elif evidence is not None:
        status = "completed"
    else:
        status = "completed_without_structured_results"

    result: Dict[str, Any] = {
        "status": status,
        "started_at": started_at,
        "test_return_value": returned,
    }
    if evidence_file:
        result["result_file"] = str(evidence_file)
    if evidence is not None:
        result["final_accuracy"] = evidence.get("Final_Accuracy")
        result["total_questions"] = evidence.get("Test-File length")
    return result


def evaluation_report(
    source: Path | str,
    inspection: Dict[str, Any],
    tool_card: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(Path(source).resolve()),
        "inspection": inspection,
        "tool_card": tool_card,
    }


JUDGE_SYSTEM_PROMPT = """You review submitted tools for an open-source LLM tool registry.
Evaluate only the supplied metadata, static-inspection findings, and observed test evidence.
Do not claim that the tool is secure. Do not infer test performance that is not present.
Treat reported historical accuracy separately from a test run observed in this report.
Return one JSON object with exactly these fields:
recommendation (approve, revise, or reject), scores (documentation, test_evidence,
output_contract, maintainability; integers 1-5), concerns (array of strings),
required_actions (array of strings), and rationale (short string).
"""


def model_review_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    """Select bounded, non-secret evidence suitable for an LLM or MCP client."""
    inspection = report.get("inspection") or {}
    tool_card = report.get("tool_card") or {}
    safety = tool_card.get("safety") or {}
    safe_card = {
        key: tool_card.get(key)
        for key in (
            "name",
            "version",
            "description",
            "category",
            "tags",
            "parameters",
            "provenance",
            "execution",
            "usage",
            "metadata_validation",
        )
    }
    evaluation = tool_card.get("evaluation") or {}
    safe_card["evaluation"] = {
        key: evaluation.get(key)
        for key in (
            "status",
            "reported_accuracy",
            "final_accuracy",
            "total_questions",
            "started_at",
        )
    }
    safe_card["safety"] = {
        key: safety.get(key)
        for key in ("cautions", "assessment", "observed_risk_level", "observed_credentials")
    }
    safe_findings = []
    for finding in inspection.get("findings", [])[:50]:
        safe_finding = {
            key: finding.get(key)
            for key in ("kind", "severity", "line", "scope", "detail")
        }
        if finding.get("file"):
            safe_finding["file"] = Path(finding["file"]).name
        safe_findings.append(safe_finding)
    return {
        "tool_card": safe_card,
        "inspection": {
            "risk_level": inspection.get("risk_level"),
            "files_scanned": inspection.get("files_scanned"),
            "observed_credentials": inspection.get("observed_credentials", []),
            "findings": safe_findings,
            "parse_errors": [
                {
                    "file": Path(item.get("file", "unknown")).name,
                    "error_type": "parse_error",
                }
                for item in inspection.get("parse_errors", [])
            ],
        },
    }


def _json_from_model_response(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        if response.get("error"):
            raise ValueError(f"model error: {response.get('error')}")
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if not isinstance(response, str):
        raise ValueError(f"unsupported judge response type: {type(response).__name__}")
    text = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("judge response must be a JSON object")
    return parsed


def _validate_judgment(judgment: Dict[str, Any]) -> None:
    required = {"recommendation", "scores", "concerns", "required_actions", "rationale"}
    if set(judgment) != required:
        raise ValueError(f"judge fields must be exactly: {sorted(required)}")
    if judgment["recommendation"] not in {"approve", "revise", "reject"}:
        raise ValueError("invalid judge recommendation")
    expected_scores = {
        "documentation",
        "test_evidence",
        "output_contract",
        "maintainability",
    }
    scores = judgment["scores"]
    if not isinstance(scores, dict) or set(scores) != expected_scores:
        raise ValueError(f"judge scores must be exactly: {sorted(expected_scores)}")
    if any(type(score) is not int or not 1 <= score <= 5 for score in scores.values()):
        raise ValueError("judge scores must be integers from 1 to 5")
    if not isinstance(judgment["concerns"], list) or not all(
        isinstance(item, str) for item in judgment["concerns"]
    ):
        raise ValueError("judge concerns must be an array of strings")
    if not isinstance(judgment["required_actions"], list) or not all(
        isinstance(item, str) for item in judgment["required_actions"]
    ):
        raise ValueError("judge required_actions must be an array of strings")
    if not isinstance(judgment["rationale"], str):
        raise ValueError("judge rationale must be a string")


def judge_evaluation_report(
    report: Dict[str, Any],
    model: str = "gpt-4o-mini",
    engine: Any = None,
) -> Dict[str, Any]:
    """Request an advisory LLM review of evidence already collected by OpenTools."""
    if report.get("tool_card") is None:
        return {
            "status": "not_run",
            "model": model,
            "reason": "Tool metadata is unavailable; evaluate an installed tool or use --run-tests for a local tool.",
        }
    try:
        if engine is None:
            from .core.factory import create_llm_engine

            engine = create_llm_engine(model, use_cache=False, is_multimodal=False)
        prompt = "Review this OpenTools evidence:\n" + json.dumps(
            model_review_payload(report), sort_keys=True, default=str
        )
        response = engine.generate(
            prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0,
            max_tokens=1200,
        )
        judgment = _json_from_model_response(response)
        _validate_judgment(judgment)
    except Exception as exc:
        return {
            "status": "failed",
            "model": model,
            "error": f"{type(exc).__name__}: {exc}",
        }

    restricted = (report.get("inspection") or {}).get("risk_level") == "restricted"
    return {
        "status": "completed",
        "model": model,
        "advisory_only": True,
        "can_override_preflight": False,
        "preflight_blocked": restricted,
        "eligible_for_automatic_acceptance": False,
        "judgment": judgment,
    }


def main() -> int:
    """Standalone static preflight used by lightweight CI jobs."""
    parser = argparse.ArgumentParser(description="Statically inspect OpenTools source")
    parser.add_argument("source", help="Python file or directory to inspect")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()
    report = inspect_source(args.source)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Risk level: {report['risk_level']}")
        print(f"Files scanned: {report['files_scanned']}")
        print(f"Findings: {len(report['findings'])}")
        print(report["caution"])
    if report["files_scanned"] == 0 or report["parse_errors"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
