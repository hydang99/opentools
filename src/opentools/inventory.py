"""Build the canonical tool-evaluation index and generated Markdown inventory."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import inspect
import io
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from .evaluation import inspect_source, run_existing_tests
except ImportError:  # Allow dependency-light direct execution.
    module_path = Path(__file__).with_name("evaluation.py")
    spec = importlib.util.spec_from_file_location("opentools_evaluation", module_path)
    if spec is None or spec.loader is None:
        raise
    evaluation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation_module)
    inspect_source = evaluation_module.inspect_source
    run_existing_tests = evaluation_module.run_existing_tests


INDEX_SCHEMA_VERSION = "1.0"
BEGIN_MARKER = "<!-- BEGIN GENERATED TOOL INVENTORY -->"
END_MARKER = "<!-- END GENERATED TOOL INVENTORY -->"
RISK_ORDER = {"low": 0, "caution": 1, "restricted": 2}


def _base_name(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _literal(node: ast.AST, default: Any = None) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return default


def discover_tool_sources(tools_root: Path | str) -> Dict[str, Dict[str, Any]]:
    """Discover tool classes and literal metadata without importing tool modules."""
    root = Path(tools_root).resolve()
    discovered: Dict[str, Dict[str, Any]] = {}
    for source in sorted(root.rglob("tool.py")):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, UnicodeError, SyntaxError):
            continue
        for class_node in tree.body:
            if not isinstance(class_node, ast.ClassDef) or not any(
                _base_name(base) == "BaseTool" for base in class_node.bases
            ):
                continue
            values: Dict[str, Any] = {}
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
            folder = source.parent.name
            discovered[class_node.name] = {
                "class_name": class_node.name,
                "folder": folder,
                "source": source,
                "description": values.get("description"),
                "category": values.get("category"),
                "execution_type": values.get("execution_type"),
                "required_api_keys": values.get("required_api_keys") or [],
            }
    return discovered


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_result(path: Path) -> Optional[Tuple[Dict[str, Any], Optional[datetime]]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    timestamp = _parse_timestamp(metadata.get("test_timestamp"))
    return data, timestamp


def latest_result(tool_dir: Path) -> Optional[Dict[str, Any]]:
    """Read the newest structured result, preferring embedded timestamps."""
    candidates = [tool_dir / "test_result.json"]
    candidates.extend(sorted((tool_dir / "test_results").glob("test_result_*.json")))
    observed = []
    for path in candidates:
        if not path.is_file():
            continue
        loaded = _read_result(path)
        if loaded is None:
            continue
        data, timestamp = loaded
        observed.append((timestamp, path.name != "test_result.json", path, data))
    if not observed:
        return None
    timestamped = [item for item in observed if item[0] is not None]
    if timestamped:
        _, _, path, data = max(timestamped, key=lambda item: (item[0], item[1]))
    else:
        # The canonical top-level result is deterministic when legacy files lack dates.
        _, _, path, data = min(observed, key=lambda item: item[1])
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    return {
        "status": "historical",
        "final_accuracy": data.get("Final_Accuracy") or metadata.get("final_accuracy"),
        "total_questions": data.get("Test-File length") or metadata.get("total_questions"),
        "evaluated_at": metadata.get("test_timestamp"),
        "result_source": str(path),
    }


def _average_accuracy(final_accuracy: Any) -> Optional[float]:
    if isinstance(final_accuracy, (int, float)) and not isinstance(final_accuracy, bool):
        return round(float(final_accuracy), 2)
    if not isinstance(final_accuracy, dict):
        return None
    values = [
        float(value)
        for value in final_accuracy.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return round(sum(values) / len(values), 2) if values else None


def _freshness(evaluated_at: Any, stale_after_days: int, now: datetime) -> str:
    parsed = _parse_timestamp(evaluated_at)
    if parsed is None:
        return "unknown"
    age_days = max(0, (now - parsed).days)
    return "stale" if age_days > stale_after_days else "current"


def build_index(
    tools_root: Path | str,
    run_results: Optional[Dict[str, Dict[str, Any]]] = None,
    stale_after_days: int = 30,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a deterministic index from source inspection and observed results."""
    root = Path(tools_root).resolve()
    now = now or datetime.now(timezone.utc)
    run_results = run_results or {}
    tools: Dict[str, Any] = {}
    for class_name, discovered in sorted(
        discover_tool_sources(root).items(), key=lambda item: item[1]["folder"]
    ):
        source = discovered["source"]
        inspection_result = inspect_source(source.parent)
        historical = latest_result(source.parent) or {}
        current_run = run_results.get(class_name)
        evidence = {**historical, **current_run} if current_run else historical
        if not evidence:
            evidence = {"status": "not_evaluated"}
        evaluated_at = evidence.get("started_at") or evidence.get("evaluated_at")
        accuracy = _average_accuracy(evidence.get("final_accuracy"))
        status = evidence.get("status", "unknown")
        freshness = (
            "not_evaluated"
            if status == "not_evaluated"
            else _freshness(evaluated_at, stale_after_days, now)
        )
        tools[class_name] = {
            "folder": discovered["folder"],
            "class_name": class_name,
            "risk_level": inspection_result["risk_level"],
            "evaluation_status": status,
            "freshness": freshness,
            "evaluated": accuracy is not None,
            "average_accuracy": accuracy,
            "final_accuracy": evidence.get("final_accuracy"),
            "total_questions": evidence.get("total_questions"),
            "last_evaluated": evaluated_at,
            "result_source": _relative_result_source(
                evidence.get("result_source") or evidence.get("result_file"), root
            ),
        }
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "stale_after_days": stale_after_days,
        "tools": tools,
    }


def _relative_result_source(value: Any, tools_root: Path) -> Optional[str]:
    if not value:
        return None
    path = Path(str(value))
    try:
        return str(path.resolve().relative_to(tools_root.parent.parent.parent))
    except ValueError:
        return path.name


def write_index(index: Dict[str, Any], output: Path | str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _legacy_rows(markdown: str) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    for line in markdown.splitlines():
        if not line.startswith("| [`"):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) < 7:
            continue
        folder_start = fields[0].find("[`") + 2
        folder_end = fields[0].find("`]", folder_start)
        if folder_start < 2 or folder_end < 0:
            continue
        folder = fields[0][folder_start:folder_end]
        rows[folder] = {
            "description": fields[1],
            "tool_type": fields[2],
            "test_key": fields[4],
            "metrics": fields[5],
        }
    return rows


def _short_description(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "OpenTools tool."
    compact = " ".join(value.split())
    first = compact.split(". ", 1)[0].strip()
    if len(first) > 120:
        first = first[:117].rstrip() + "..."
    return first + ("" if first.endswith(".") else ".")


def _derived_type(discovered: Dict[str, Any], record: Dict[str, Any]) -> str:
    explicit = discovered.get("execution_type")
    if explicit == "api":
        return "api_based"
    if explicit == "prompting":
        return "prompting_based"
    if explicit == "local":
        return "local_processing"
    if discovered.get("required_api_keys") or record.get("risk_level") == "caution":
        return "api_based"
    return "local_processing"


def render_inventory_table(
    index: Dict[str, Any],
    tools_root: Path | str,
    existing_markdown: str,
) -> str:
    discovered = discover_tool_sources(tools_root)
    by_folder = {item["folder"]: item for item in discovered.values()}
    legacy = _legacy_rows(existing_markdown)
    records = {record["folder"]: record for record in index.get("tools", {}).values()}
    lines = [
        BEGIN_MARKER,
        "| Tool name (folder) | Short description | Tool type | Evaluated? | Test suite key | Evaluation metrics | Current accuracy | Risk | Last evaluated | Cases | Status |",
        "|---|---|---|---|---|---|---:|---|---|---:|---|",
    ]
    for folder in sorted(by_folder):
        source = by_folder[folder]
        record = records.get(folder, {})
        prior = legacy.get(folder, {})
        description = prior.get("description") or _short_description(source.get("description"))
        tool_type = prior.get("tool_type") or _derived_type(source, record)
        test_key = prior.get("test_key") or f"`{folder}`"
        metrics = prior.get("metrics") or "–"
        accuracy = record.get("average_accuracy")
        accuracy_text = "–" if accuracy is None else str(accuracy)
        evaluated = "✅" if record.get("evaluated") else "☐"
        last_evaluated = record.get("last_evaluated")
        parsed_date = _parse_timestamp(last_evaluated)
        date_text = parsed_date.date().isoformat() if parsed_date else "–"
        cases = record.get("total_questions")
        cases_text = "–" if cases is None else str(cases)
        status = record.get("evaluation_status", "unknown")
        freshness = record.get("freshness")
        if freshness not in {None, "unknown", "not_evaluated"}:
            status = f"{status} / {freshness}"
        lines.append(
            f"| [`{folder}`](./{folder}/) | {description} | {tool_type} | {evaluated} | "
            f"{test_key} | {metrics} | {accuracy_text} | {record.get('risk_level', 'unknown')} | "
            f"{date_text} | {cases_text} | {status} |"
        )
    lines.append(END_MARKER)
    return "\n".join(lines)


def update_inventory_markdown(
    index: Dict[str, Any],
    tools_root: Path | str,
    readme_path: Path | str,
) -> None:
    path = Path(readme_path)
    original = path.read_text(encoding="utf-8")
    generated = render_inventory_table(index, tools_root, original)
    if BEGIN_MARKER in original and END_MARKER in original:
        start = original.index(BEGIN_MARKER)
        end = original.index(END_MARKER, start) + len(END_MARKER)
        updated = original[:start] + generated + original[end:]
    else:
        header = "| Tool name (folder) | Short description | Tool type | Evaluated?"
        start = original.find(header)
        architecture = original.find("## Tool Architecture", start)
        if start < 0 or architecture < 0:
            raise ValueError("Could not locate the existing tool inventory table")
        updated = original[:start] + generated + "\n\n" + original[architecture:]
    if not updated.endswith("\n"):
        updated += "\n"
    path.write_text(updated, encoding="utf-8")


def _load_tool(class_name: str, source: Path) -> Any:
    from .core.base import BaseTool

    module_name = f"opentools_inventory_{source.parent.name}_{source.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load tool module: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = [
        candidate
        for _, candidate in inspect.getmembers(module, inspect.isclass)
        if candidate is not BaseTool
        and issubclass(candidate, BaseTool)
        and candidate.__name__ == class_name
        and candidate.__module__ == module.__name__
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one BaseTool subclass named {class_name}")
    return candidates[0]()


def run_bulk_evaluations(
    tools_root: Path | str,
    selected_tools: Optional[Iterable[str]] = None,
    max_risk: str = "low",
    discard_raw_results: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run real tests for selected tools while recording failures as evidence."""
    selected = {item for item in (selected_tools or []) if item}
    discovered = discover_tool_sources(tools_root)
    results: Dict[str, Dict[str, Any]] = {}
    known_selectors = set(discovered)
    known_selectors.update(item["folder"] for item in discovered.values())
    for unknown in sorted(selected - known_selectors):
        results[f"unknown:{unknown}"] = {
            "status": "failed",
            "error": f"Tool not found: {unknown}",
        }
    for class_name, item in sorted(discovered.items()):
        if selected and class_name not in selected and item["folder"] not in selected:
            continue
        source = item["source"]
        inspection_result = inspect_source(source.parent)
        if RISK_ORDER[inspection_result["risk_level"]] > RISK_ORDER[max_risk]:
            results[class_name] = {
                "status": "skipped_by_risk_policy",
                "risk_level": inspection_result["risk_level"],
            }
            continue
        try:
            tool = _load_tool(class_name, source)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_existing_tests(tool, source)
        except Exception as exc:
            result = {
                "status": "failed",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        results[class_name] = result
        result_file = result.get("result_file")
        if discard_raw_results and result_file:
            path = Path(result_file).resolve()
            expected_root = (source.parent / "test_results").resolve()
            if path.parent == expected_root and path.is_file():
                path.unlink()
                result["result_source"] = None
                result.pop("result_file", None)
    return results


def refresh(
    tools_root: Path | str,
    index_path: Path | str,
    readme_path: Path | str,
    run_tests: bool = False,
    selected_tools: Optional[Iterable[str]] = None,
    max_risk: str = "low",
    stale_after_days: int = 30,
    discard_raw_results: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    run_results = (
        run_bulk_evaluations(
            tools_root,
            selected_tools=selected_tools,
            max_risk=max_risk,
            discard_raw_results=discard_raw_results,
        )
        if run_tests
        else {}
    )
    index = build_index(
        tools_root,
        run_results=run_results,
        stale_after_days=stale_after_days,
    )
    write_index(index, index_path)
    update_inventory_markdown(index, tools_root, readme_path)
    return index, run_results


def main() -> int:
    package_root = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Refresh OpenTools evaluation inventory")
    parser.add_argument("--tools-root", default=str(package_root / "tools"))
    parser.add_argument(
        "--index",
        default=str(package_root / "tools" / "evaluation_index.json"),
    )
    parser.add_argument(
        "--inventory",
        default=str(package_root / "tools" / "readme.md"),
    )
    parser.add_argument("--run-tests", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tools", help="Comma-separated class names or tool folders")
    selection.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--max-risk", choices=["low", "caution"], default="low")
    parser.add_argument("--stale-after-days", type=int, default=30)
    parser.add_argument("--discard-raw-results", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    selected = [item.strip() for item in (args.tools or "").split(",") if item.strip()]
    if args.run_tests and not selected and not args.all_eligible:
        parser.error("--run-tests requires --tools or --all-eligible")
    index, results = refresh(
        args.tools_root,
        args.index,
        args.inventory,
        run_tests=args.run_tests,
        selected_tools=selected,
        max_risk=args.max_risk,
        stale_after_days=args.stale_after_days,
        discard_raw_results=args.discard_raw_results,
    )
    summary = {
        "indexed_tools": len(index["tools"]),
        "evaluations_requested": len(results),
        "completed": sum(result.get("status") == "completed" for result in results.values()),
        "failed": sum(result.get("status") == "failed" for result in results.values()),
        "skipped": sum(
            result.get("status") == "skipped_by_risk_policy" for result in results.values()
        ),
    }
    print(json.dumps(summary, indent=2))
    if any(name.startswith("unknown:") for name in results):
        return 1
    if args.fail_on_error and summary["failed"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
