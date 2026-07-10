"""MCP adapter for OpenTools inspection, tool cards, and controlled evaluation.

The server accepts registered tool names only. It does not expose arbitrary local
paths, and execution is disabled unless OPENTOOLS_MCP_ALLOW_EXECUTION=1 is set by
the server owner before startup.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import importlib.util
import inspect
import io
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from .evaluation import (
    build_tool_card,
    evaluation_report,
    inspect_source,
    model_review_payload,
    run_existing_tests,
)
from .schema_validation import validate_json


def _configured_allowlist() -> Optional[Set[str]]:
    value = os.getenv("OPENTOOLS_MCP_ALLOWED_TOOLS", "").strip()
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _execution_enabled() -> bool:
    return os.getenv("OPENTOOLS_MCP_ALLOW_EXECUTION", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _tool_calls_enabled() -> bool:
    return os.getenv("OPENTOOLS_MCP_ALLOW_TOOL_CALLS", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _maximum_risk() -> str:
    value = os.getenv("OPENTOOLS_MCP_MAX_RISK", "low").strip().lower()
    if value not in {"low", "caution"}:
        raise ValueError("OPENTOOLS_MCP_MAX_RISK must be 'low' or 'caution'")
    return value


def _tools_root() -> Path:
    configured = os.getenv("OPENTOOLS_MCP_TOOLS_ROOT")
    root = Path(configured).expanduser() if configured else Path(__file__).parent / "tools"
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"OpenTools MCP tools root is not a directory: {root}")
    return root


def _check_allowed(tool_name: str) -> None:
    allowlist = _configured_allowlist()
    if allowlist is not None and tool_name not in allowlist:
        raise ValueError(f"Tool is not in OPENTOOLS_MCP_ALLOWED_TOOLS: {tool_name}")


def _base_name(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _discover_tool_sources() -> Dict[str, Path]:
    """Discover BaseTool subclasses without importing submitted modules."""
    discovered: Dict[str, Path] = {}
    for source in sorted(_tools_root().rglob("tool.py")):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, UnicodeError, SyntaxError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and any(
                _base_name(base) == "BaseTool" for base in node.bases
            ):
                discovered[node.name] = source
    return discovered


def _source_for_tool(tool_name: str) -> Path:
    _check_allowed(tool_name)
    source = _discover_tool_sources().get(tool_name)
    if source is None:
        raise ValueError(f"Registered tool not found: {tool_name}")
    return source


def _literal(node: ast.AST, default: Any = None) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return default


def _metadata_from_source(tool_name: str, source: Path) -> Dict[str, Any]:
    """Read literal BaseTool constructor metadata without importing tool code."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    values: Dict[str, Any] = {}
    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef) or class_node.name != tool_name:
            continue
        for call in ast.walk(class_node):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr != "__init__":
                continue
            for keyword in call.keywords:
                if keyword.arg:
                    values[keyword.arg] = _literal(keyword.value)
            if values:
                break
        break

    limitation = values.get("limitation")
    return {
        "name": values.get("name") or tool_name,
        "version": values.get("version") or "unspecified",
        "description": values.get("description"),
        "category": values.get("category"),
        "tags": values.get("tags") or [],
        "parameters": values.get("parameters") or {},
        "limitation": limitation,
        "source_url": values.get("source_url"),
        "license": values.get("license"),
        "execution": {
            "type": values.get("execution_type") or "unspecified",
            "network_access": values.get("network_access"),
            "required_api_keys": values.get("required_api_keys") or [],
            "side_effects": values.get("side_effects") or [],
            "estimated_cost": values.get("estimated_cost"),
        },
        "safety": {
            "cautions": values.get("cautions") or [],
            "assessment": "declared_by_tool_author",
        },
        "usage": {
            "suitable_for": values.get("suitable_for") or [],
            "limitations": limitation,
        },
        "accuracy": None,
    }


class _StaticToolCard:
    def __init__(self, metadata: Dict[str, Any]) -> None:
        self._metadata = metadata

    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata


def _load_requested_tool(tool_name: str, source: Path) -> Any:
    """Import exactly one allowlisted tool after deterministic policy gates pass."""
    from .core.base import BaseTool

    module_name = f"opentools_mcp_{source.parent.name}_{source.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load tool module: {source.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = [
        candidate
        for _, candidate in inspect.getmembers(module, inspect.isclass)
        if candidate is not BaseTool
        and issubclass(candidate, BaseTool)
        and candidate.__name__ == tool_name
        and candidate.__module__ == module.__name__
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one BaseTool subclass named {tool_name}")
    return candidates[0]()


def _safe_report(tool: Any, source: Path, test_result: Dict[str, Any]) -> Dict[str, Any]:
    inspection_result = inspect_source(source)
    card = build_tool_card(tool, inspection_result, test_result)
    report = evaluation_report(source, inspection_result, card)
    # This removes source code, absolute paths, credential values, and result paths.
    return model_review_payload(report)


def _execution_block_reason(tool_name: str, source: Path, for_tool_call: bool) -> Optional[str]:
    if _configured_allowlist() is None:
        return "An explicit OPENTOOLS_MCP_ALLOWED_TOOLS list is required for execution."
    enabled = _tool_calls_enabled() if for_tool_call else _execution_enabled()
    if not enabled:
        flag = "OPENTOOLS_MCP_ALLOW_TOOL_CALLS" if for_tool_call else "OPENTOOLS_MCP_ALLOW_EXECUTION"
        return f"Set {flag}=1 before startup to enable this operation."
    risk_level = inspect_source(source)["risk_level"]
    if risk_level == "restricted":
        return "Restricted tools cannot be executed through the MCP server."
    if risk_level == "caution" and _maximum_risk() != "caution":
        return "Caution tools require OPENTOOLS_MCP_MAX_RISK=caution."
    metadata = _metadata_from_source(tool_name, source)
    missing = [
        key for key in metadata["execution"].get("required_api_keys", []) if not os.getenv(key)
    ]
    if missing:
        return "Missing server-side credentials: " + ", ".join(sorted(missing))
    return None


def _json_safe_result(value: Any) -> Any:
    """Return JSON-compatible output while redacting credential-shaped fields."""
    secret_fragments = ("api_key", "password", "secret", "token", "credential")
    if isinstance(value, dict):
        return {
            str(key): _json_safe_result(item)
            for key, item in value.items()
            if not any(fragment in str(key).lower() for fragment in secret_fragments)
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_result(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _validate_arguments(schema: Dict[str, Any], arguments: Dict[str, Any]) -> None:
    validate_json(arguments, schema or {"type": "object"})


def create_server(fastmcp_class: Any = None):
    """Create the FastMCP server; dependency injection keeps registration testable."""
    if fastmcp_class is None:
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:
            raise RuntimeError(
                "The MCP server requires the 'mcp' package. Install project dependencies first."
            ) from exc
        fastmcp_class = FastMCP

    server = fastmcp_class(
        "OpenTools",
        instructions=(
            "Inspect registered OpenTools tools and read sanitized tool cards. "
            "Static findings are review signals, not security guarantees. "
            "Test execution may be disabled by server policy."
        ),
    )

    @server.tool()
    def list_opentools() -> Dict[str, Any]:
        """List registered tools permitted by this MCP server."""
        names = sorted(_discover_tool_sources())
        allowlist = _configured_allowlist()
        if allowlist is not None:
            names = [name for name in names if name in allowlist]
        return {
            "tools": names,
            "execution_enabled": _execution_enabled(),
            "tool_calls_enabled": _tool_calls_enabled(),
            "maximum_risk": _maximum_risk(),
        }

    @server.tool()
    def inspect_opentool(tool_name: str) -> Dict[str, Any]:
        """Return static risk signals and a sanitized tool card without running tests."""
        source = _source_for_tool(tool_name)
        tool = _StaticToolCard(_metadata_from_source(tool_name, source))
        return _safe_report(tool, source, {"status": "not_run"})

    @server.tool()
    def evaluate_opentool(tool_name: str) -> Dict[str, Any]:
        """Run existing tests when server policy allows, then return observed evidence."""
        source = _source_for_tool(tool_name)
        static_tool = _StaticToolCard(_metadata_from_source(tool_name, source))
        inspection_result = inspect_source(source)
        blocked_reason = _execution_block_reason(tool_name, source, for_tool_call=False)
        if blocked_reason:
            return {
                "status": (
                    "blocked_by_preflight"
                    if blocked_reason.startswith("Restricted")
                    else "execution_disabled"
                ),
                "reason": blocked_reason,
                "inspection": model_review_payload(
                    evaluation_report(
                        source,
                        inspection_result,
                        build_tool_card(static_tool, inspection_result, {"status": "not_run"}),
                    )
                )["inspection"],
            }
        tool = _load_requested_tool(tool_name, source)
        with contextlib.redirect_stdout(io.StringIO()):
            test_result = run_existing_tests(tool, source)
        return _safe_report(tool, source, test_result)

    @server.tool()
    async def call_opentool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke one explicitly allowlisted OpenTools tool for an MCP application."""
        source = _source_for_tool(tool_name)
        blocked_reason = _execution_block_reason(tool_name, source, for_tool_call=True)
        if blocked_reason:
            return {
                "status": (
                    "blocked_by_preflight"
                    if blocked_reason.startswith("Restricted")
                    else "execution_disabled"
                ),
                "reason": blocked_reason,
            }
        metadata = _metadata_from_source(tool_name, source)
        try:
            _validate_arguments(metadata.get("parameters") or {}, arguments)
        except Exception as exc:
            return {
                "status": "invalid_arguments",
                "error": f"{type(exc).__name__}: {exc}",
            }

        def invoke() -> Dict[str, Any]:
            tool = _load_requested_tool(tool_name, source)
            with contextlib.redirect_stdout(io.StringIO()):
                result = tool.run(**arguments)
            return {
                "status": "completed",
                "tool_name": tool_name,
                "result": _json_safe_result(result),
            }

        try:
            return await asyncio.to_thread(invoke)
        except Exception as exc:
            return {
                "status": "failed",
                "tool_name": tool_name,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return server


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the OpenTools MCP server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument("--host", help="Host for streamable HTTP (default: SDK setting)")
    parser.add_argument("--port", type=int, help="Port for streamable HTTP")
    args = parser.parse_args()
    server = create_server()
    if args.host:
        server.settings.host = args.host
    if args.port:
        server.settings.port = args.port
    server.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
