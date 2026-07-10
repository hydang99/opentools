"""Deterministically convert submitted Python functions into OpenTools bundles."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evaluation import inspect_source


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    if not cleaned:
        raise ValueError("Tool name must contain at least one letter or number")
    if cleaned[0].isdigit():
        cleaned = "tool_" + cleaned
    return cleaned


def _class_name(value: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", value) if part]
    name = "_".join(part[:1].upper() + part[1:] for part in parts)
    if not name:
        raise ValueError("Could not derive a class name")
    if name[0].isdigit():
        name = "Tool_" + name
    return name if name.endswith("_Tool") else name + "_Tool"


def _annotation_schema(node: Optional[ast.AST]) -> Dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, ast.Name):
        return {
            "str": {"type": "string"},
            "int": {"type": "integer"},
            "float": {"type": "number"},
            "bool": {"type": "boolean"},
            "dict": {"type": "object"},
            "list": {"type": "array"},
            "Any": {},
        }.get(node.id, {})
    if isinstance(node, ast.Constant) and node.value is None:
        return {"type": "null"}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        choices = [_annotation_schema(node.left), _annotation_schema(node.right)]
        return {"anyOf": choices}
    if isinstance(node, ast.Subscript):
        base = node.value.id if isinstance(node.value, ast.Name) else None
        if base in {"List", "list", "Sequence", "Iterable"}:
            return {"type": "array", "items": _annotation_schema(node.slice)}
        if base in {"Dict", "dict", "Mapping"}:
            return {"type": "object"}
        if base == "Optional":
            return {"anyOf": [_annotation_schema(node.slice), {"type": "null"}]}
        if base == "Literal":
            values = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            literals = [item.value for item in values if isinstance(item, ast.Constant)]
            schema: Dict[str, Any] = {"enum": literals}
            if literals and all(isinstance(item, str) for item in literals):
                schema["type"] = "string"
            return schema
    return {}


def _functions(tree: ast.Module) -> Dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }


def _already_standard(tree: ast.Module) -> Optional[str]:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
            if name == "BaseTool":
                return node.name
    return None


def analyze_submission(source: Path | str, entrypoint: Optional[str] = None) -> Dict[str, Any]:
    source_path = Path(source).resolve()
    text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(source_path))
    standard_class = _already_standard(tree)
    if standard_class:
        return {
            "status": "already_standard",
            "class_name": standard_class,
            "entrypoint": None,
            "parameters": None,
        }
    candidates = _functions(tree)
    if entrypoint:
        if entrypoint not in candidates:
            raise ValueError(
                f"Entrypoint {entrypoint!r} was not found; available functions: {sorted(candidates)}"
            )
        function = candidates[entrypoint]
    elif len(candidates) == 1:
        function = next(iter(candidates.values()))
    else:
        raise ValueError(
            "Specify --entrypoint when source contains zero or multiple public functions; "
            f"available functions: {sorted(candidates)}"
        )
    if isinstance(function, ast.AsyncFunctionDef):
        raise ValueError("Async entrypoints are not converted automatically")

    positional = list(function.args.posonlyargs) + list(function.args.args)
    defaults_start = len(positional) - len(function.args.defaults)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for index, argument in enumerate(positional):
        schema = _annotation_schema(argument.annotation)
        properties[argument.arg] = {
            **schema,
            "description": f"Argument {argument.arg} for {function.name}.",
        }
        if index < defaults_start:
            required.append(argument.arg)
    if function.args.vararg or function.args.kwarg:
        raise ValueError("Variadic *args/**kwargs entrypoints are not converted automatically")
    return {
        "status": "convertible",
        "class_name": None,
        "entrypoint": function.name,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "description": ast.get_docstring(function),
    }


def _wrapper_source(
    class_name: str,
    display_name: str,
    entrypoint: str,
    parameters: Dict[str, Any],
    description: str,
    category: str,
    source_url: Optional[str],
    license_name: Optional[str],
) -> str:
    metadata = {
        "name": class_name,
        "description": description,
        "category": category,
        "tags": [_slug(display_name), "community_contribution"],
        "parameters": parameters,
        "source_url": source_url,
        "license": license_name,
    }
    return f'''"""Generated OpenTools wrapper. Review before contribution."""
import importlib.util
from pathlib import Path
from opentools.core.base import BaseTool

_SOURCE = Path(__file__).with_name("original_tool.py")
_SPEC = importlib.util.spec_from_file_location("opentools_contributed_{_slug(display_name)}", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load {{_SOURCE}}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_ENTRYPOINT = getattr(_MODULE, {entrypoint!r})


class {class_name}(BaseTool):
    def __init__(self):
        super().__init__(
            name={metadata['name']!r},
            description={metadata['description']!r},
            category={metadata['category']!r},
            tags={metadata['tags']!r},
            parameters={metadata['parameters']!r},
            agent_type="Community-Tool",
            demo_commands={{}},
            limitation="Automatically converted wrapper; behavior and dependencies require maintainer review.",
            version="0.1.0",
            source_url={metadata['source_url']!r},
            license={metadata['license']!r},
            execution_type="unspecified",
            network_access=None,
            side_effects=[],
            cautions=["Generated wrapper must pass review and functional tests before publication."],
            suitable_for=[],
        )

    def run(self, **kwargs):
        try:
            output = _ENTRYPOINT(**kwargs)
            if isinstance(output, dict) and "success" in output:
                return output
            return {{"result": output, "success": True}}
        except Exception as exc:
            return {{"error": f"{{type(exc).__name__}}: {{exc}}", "success": False}}
'''


def convert_submission(
    source: Path | str,
    readme: Path | str,
    output_root: Path | str,
    name: str,
    entrypoint: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "community",
    source_url: Optional[str] = None,
    license_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a reviewable contribution bundle without executing submitted code."""
    source_path = Path(source).resolve()
    readme_path = Path(readme).resolve()
    if source_path.suffix != ".py" or not source_path.is_file():
        raise ValueError("source must be an existing Python file")
    if not readme_path.is_file():
        raise ValueError("readme must be an existing file")
    analysis = analyze_submission(source_path, entrypoint=entrypoint)
    slug = _slug(name)
    bundle = Path(output_root).resolve() / f"{slug}_{uuid.uuid4().hex[:8]}"
    bundle.mkdir(parents=True, exist_ok=False)
    submitted_readme = readme_path.read_text(encoding="utf-8")
    (bundle / "README.md").write_text(submitted_readme, encoding="utf-8")

    if analysis["status"] == "already_standard":
        shutil.copy2(source_path, bundle / "tool.py")
        class_name = analysis["class_name"]
        conversion_status = "already_standard"
        tool_description = (
            description
            or "Existing OpenTools BaseTool submission; inspect its metadata during maintainer review."
        )
        parameter_schema = {}
    else:
        shutil.copy2(source_path, bundle / "original_tool.py")
        class_name = _class_name(name)
        tool_description = (
            description
            or analysis.get("description")
            or f"Community-contributed tool wrapping {analysis['entrypoint']}."
        )
        (bundle / "tool.py").write_text(
            _wrapper_source(
                class_name,
                name,
                analysis["entrypoint"],
                analysis["parameters"],
                tool_description,
                category,
                source_url,
                license_name,
            ),
            encoding="utf-8",
        )
        conversion_status = "converted"
        parameter_schema = analysis["parameters"]

    (bundle / "__init__.py").write_text(
        f"from .tool import {class_name}\n\n__all__ = [{class_name!r}]\n",
        encoding="utf-8",
    )
    risk = inspect_source(bundle)
    safe_findings = []
    for finding in risk["findings"]:
        safe_finding = dict(finding)
        if safe_finding.get("file"):
            safe_finding["file"] = Path(safe_finding["file"]).name
        safe_findings.append(safe_finding)
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "conversion_status": conversion_status,
        "class_name": class_name,
        "entrypoint": analysis.get("entrypoint"),
        "tool_card": {
            "name": class_name,
            "description": tool_description,
            "category": category,
            "parameters": parameter_schema,
            "provenance": {"source_url": source_url, "license": license_name},
            "evaluation": {"status": "not_run"},
        },
        "risk": {
            "risk_level": risk["risk_level"],
            "findings": safe_findings,
            "caution": risk["caution"],
        },
        "functional_evaluation": {
            "status": "not_run",
            "reason": "Submitted code is not executed during conversion.",
        },
    }
    (bundle / "contribution.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    archive = shutil.make_archive(str(bundle), "zip", root_dir=bundle)
    return {
        "status": "completed",
        "bundle": str(bundle),
        "archive": archive,
        "manifest": manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Python code into an OpenTools bundle")
    parser.add_argument("source")
    parser.add_argument("--readme", required=True)
    parser.add_argument("--output", default="opentools_contributions")
    parser.add_argument("--name", required=True)
    parser.add_argument("--entrypoint")
    parser.add_argument("--description")
    parser.add_argument("--category", default="community")
    parser.add_argument("--source-url")
    parser.add_argument("--license", dest="license_name")
    args = parser.parse_args()
    result = convert_submission(
        args.source,
        args.readme,
        args.output,
        args.name,
        entrypoint=args.entrypoint,
        description=args.description,
        category=args.category,
        source_url=args.source_url,
        license_name=args.license_name,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
