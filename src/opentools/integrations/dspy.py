"""Optional DSPy adapters for evaluated OpenTools tools."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..evaluation import inspect_source
from ..inventory import _load_tool, discover_tool_sources
from ..schema_validation import validate_json


RISK_ORDER = {"low": 0, "caution": 1, "restricted": 2}


def _python_type(schema: Dict[str, Any]) -> type:
    schema_type = schema.get("type")
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(schema_type, Any)


def _resolve_tool(tool_name: str, tools_root: Optional[Path | str] = None):
    root = Path(tools_root).resolve() if tools_root else Path(__file__).parents[1] / "tools"
    discovered = discover_tool_sources(root)
    for class_name, item in discovered.items():
        if tool_name in {class_name, item["folder"]}:
            return class_name, item
    raise ValueError(f"OpenTools tool not found: {tool_name}")


def as_callable(
    tool_name: str,
    tools_root: Optional[Path | str] = None,
    max_risk: str = "low",
) -> Callable[..., Any]:
    """Return a schema-validated callable suitable for DSPy or ordinary Python."""
    if max_risk not in {"low", "caution"}:
        raise ValueError("max_risk must be 'low' or 'caution'")
    class_name, item = _resolve_tool(tool_name, tools_root)
    risk = inspect_source(item["source"].parent)["risk_level"]
    if risk == "restricted" or RISK_ORDER[risk] > RISK_ORDER[max_risk]:
        raise ValueError(f"Tool {class_name} is not permitted by max_risk={max_risk}: {risk}")
    tool = _load_tool(class_name, item["source"])
    metadata = tool.get_metadata()
    required_keys = metadata.get("execution", {}).get("required_api_keys", [])
    missing = [key for key in required_keys if not os.getenv(key)]
    if missing:
        raise ValueError("Missing credentials: " + ", ".join(sorted(missing)))
    schema = metadata.get("parameters") or {"type": "object"}

    def invoke(**kwargs):
        validate_json(kwargs, schema)
        result = tool.run(**kwargs)
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(str(result.get("error", result)))
        return result

    invoke.__name__ = metadata.get("name") or class_name
    invoke.__doc__ = metadata.get("description") or class_name
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    signature_parameters = []
    for name, property_schema in properties.items():
        default = inspect.Parameter.empty if name in required else None
        signature_parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=_python_type(property_schema),
            )
        )
    invoke.__signature__ = inspect.Signature(signature_parameters)  # type: ignore[attr-defined]
    return invoke


def as_dspy_tool(
    tool_name: str,
    tools_root: Optional[Path | str] = None,
    max_risk: str = "low",
):
    """Wrap an OpenTools callable with the optional ``dspy.Tool`` primitive."""
    try:
        import dspy
    except ImportError as exc:
        raise RuntimeError("Install the optional integration with: pip install 'opentools[dspy]'") from exc
    return dspy.Tool(as_callable(tool_name, tools_root=tools_root, max_risk=max_risk))


def as_dspy_tools(
    tool_names: Iterable[str],
    tools_root: Optional[Path | str] = None,
    max_risk: str = "low",
) -> List[Any]:
    return [
        as_dspy_tool(name, tools_root=tools_root, max_risk=max_risk)
        for name in tool_names
    ]


__all__ = ["as_callable", "as_dspy_tool", "as_dspy_tools"]
