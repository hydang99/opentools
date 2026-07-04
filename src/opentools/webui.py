"""Optional review-oriented WebUI for OpenTools contributions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .conversion import convert_submission
from .evaluation import inspect_source, judge_evaluation_report
from .inventory import discover_tool_sources


def _uploaded_path(value: Any) -> Path:
    if isinstance(value, (str, Path)):
        return Path(value).resolve()
    name = getattr(value, "name", None)
    if name:
        return Path(name).resolve()
    raise ValueError("A file must be uploaded")


def _check_size(path: Path) -> None:
    maximum = int(os.getenv("OPENTOOLS_MAX_SUBMISSION_BYTES", str(1_000_000)))
    if path.stat().st_size > maximum:
        raise ValueError(f"Submission exceeds the {maximum}-byte limit: {path.name}")


def submit_contribution(
    tool_file: Any,
    readme_file: Any,
    name: str,
    entrypoint: str = "",
    description: str = "",
    category: str = "community",
    source_url: str = "",
    license_name: str = "",
    request_judge: bool = False,
    judge_model: str = "gpt-4o-mini",
    output_root: Optional[Path | str] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Convert and inspect an upload without executing submitted code."""
    try:
        source = _uploaded_path(tool_file)
        readme = _uploaded_path(readme_file)
        _check_size(source)
        _check_size(readme)
        root = Path(
            output_root
            or os.getenv("OPENTOOLS_CONTRIBUTIONS_DIR", "opentools_contributions")
        )
        result = convert_submission(
            source,
            readme,
            root,
            name=name,
            entrypoint=entrypoint.strip() or None,
            description=description.strip() or None,
            category=category.strip() or "community",
            source_url=source_url.strip() or None,
            license_name=license_name.strip() or None,
        )
        manifest = result["manifest"]
        report: Dict[str, Any] = {
            "status": "completed",
            "conversion_status": manifest["conversion_status"],
            "tool_card": manifest["tool_card"],
            "risk": manifest["risk"],
            "functional_evaluation": manifest["functional_evaluation"],
            "llm_judge": {"status": "not_run"},
            "publication_status": "pending_maintainer_review",
        }
        if request_judge:
            judge_report = {
                "inspection": {
                    "risk_level": manifest["risk"]["risk_level"],
                    "files_scanned": inspect_source(result["bundle"])["files_scanned"],
                    "observed_credentials": [],
                    "findings": manifest["risk"]["findings"],
                    "parse_errors": [],
                },
                "tool_card": manifest["tool_card"],
            }
            report["llm_judge"] = judge_evaluation_report(
                judge_report, model=judge_model.strip() or "gpt-4o-mini"
            )
        return report, result["archive"]
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "functional_evaluation": {"status": "not_run"},
        }, None


def inspect_registered_tool(tool_name: str) -> Dict[str, Any]:
    root = Path(__file__).parent / "tools"
    discovered = discover_tool_sources(root)
    if tool_name not in discovered:
        return {"status": "failed", "error": f"Tool not found: {tool_name}"}
    item = discovered[tool_name]
    index_path = root / "evaluation_index.json"
    index_record = None
    if index_path.is_file():
        index_record = json.loads(index_path.read_text(encoding="utf-8")).get("tools", {}).get(tool_name)
    inspection = inspect_source(item["source"].parent)
    inspection.pop("source", None)
    for finding in inspection.get("findings", []):
        if finding.get("file"):
            finding["file"] = Path(finding["file"]).name
    return {
        "status": "completed",
        "tool_name": tool_name,
        "inspection": inspection,
        "evaluation": index_record,
    }


def build_app():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the WebUI with: pip install 'opentools[webui]'") from exc

    tool_names = sorted(discover_tool_sources(Path(__file__).parent / "tools"))
    with gr.Blocks(title="OpenTools Contribution Review") as app:
        gr.Markdown(
            "# OpenTools Contribution Review\n"
            "Uploads are converted and statically inspected. Submitted code is not executed, "
            "and publication always requires maintainer review."
        )
        with gr.Tab("Submit and standardize"):
            tool_file = gr.File(label="tool.py", file_types=[".py"], type="filepath")
            readme_file = gr.File(label="README", file_types=[".md", ".txt"], type="filepath")
            name = gr.Textbox(label="Tool name")
            entrypoint = gr.Textbox(label="Function entrypoint (required when multiple functions exist)")
            description = gr.Textbox(label="Description", lines=3)
            category = gr.Textbox(label="Category", value="community")
            source_url = gr.Textbox(label="Source URL (optional)")
            license_name = gr.Textbox(label="License/SPDX identifier")
            request_judge = gr.Checkbox(label="Request advisory LLM review", value=False)
            judge_model = gr.Textbox(label="Judge model", value="gpt-4o-mini")
            submit = gr.Button("Create review bundle", variant="primary")
            contribution_report = gr.JSON(label="Observed review report")
            archive = gr.File(label="Contribution bundle")
            submit.click(
                fn=submit_contribution,
                inputs=[
                    tool_file,
                    readme_file,
                    name,
                    entrypoint,
                    description,
                    category,
                    source_url,
                    license_name,
                    request_judge,
                    judge_model,
                ],
                outputs=[contribution_report, archive],
            )
        with gr.Tab("Inspect published tools"):
            selected = gr.Dropdown(choices=tool_names, label="OpenTools tool")
            inspect_button = gr.Button("Inspect")
            published_report = gr.JSON(label="Current evidence")
            inspect_button.click(inspect_registered_tool, inputs=[selected], outputs=[published_report])
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the OpenTools contribution WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_app().launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
