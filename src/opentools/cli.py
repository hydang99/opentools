"""
Command-line interface for OpenTools framework.

This module provides a convenient CLI for discovering and managing tools.
"""

import argparse
import contextlib
import importlib.util
import inspect
import io
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from . import (
    list_available_tools,
    get_tool_info,
    create_tool,
    load_all_tools,
    registry,
    search_tools,
)
from .core.config import OpenToolsConfig
from .evaluation import (
    build_tool_card,
    evaluation_report,
    inspect_source,
    judge_evaluation_report,
    run_existing_tests,
)
from .inventory import refresh as refresh_inventory
from .conversion import convert_submission


def list_command(args):
    """List all available tools."""
    load_all_tools(verbose=False)
    tools = list_available_tools()
    if not tools:
        if args.json:
            print(json.dumps([], indent=2))
        else:
            print("No tools found.")
        return
    
    if args.json:
        print(json.dumps(tools, indent=2))
    else:
        print("Available tools:")
        for tool in tools:
            print(f"  - {tool}")


def info_command(args):
    """Get detailed information about a tool."""
    load_all_tools(verbose=False)
    tool_name = args.tool_name
    info = get_tool_info(tool_name)
    
    if not info:
        print(f"Tool '{tool_name}' not found.")
        return
    
    if args.json:
        print(json.dumps(info, indent=2, default=str))
    else:
        print(f"Tool: {tool_name}")
        print(f"Description: {info.get('description', 'N/A')}")
        print(f"Version: {info.get('version', 'N/A')}")
        print(f"Input Types: {info.get('input_types', {})}")
        print(f"Output Type: {info.get('output_type', 'N/A')}")
        print(f"Demo Commands: {info.get('demo_commands', [])}")
        print(f"Provenance: {info.get('provenance', {})}")
        print(f"Execution: {info.get('execution', {})}")
        print(f"Safety/Cautions: {info.get('safety', {})}")
        print(f"Usage: {info.get('usage', {})}")
        print(f"Evaluation: {info.get('evaluation', {})}")


def search_command(args):
    """Search for tools."""
    query = args.query
    results = search_tools(query)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        if results:
            print(f"Found {len(results)} tool(s) matching '{query}':")
            for tool in results:
                print(f"  - {tool}")
        else:
            print(f"No tools found matching '{query}'")



def reload_command(args):
    """Reload all tools."""
    loaded_tools = registry.load_all_tools()
    print(f"✅ Reloaded {len(loaded_tools)} tools: {loaded_tools}")


def create_env_template(args):
    """Create a .env template file."""
    env_path = args.output or ".env"
    
    if Path(env_path).exists() and not args.force:
        print(f"❌ File {env_path} already exists. Use --force to overwrite.")
        return 1
    
    # Create .env template
    env_content = """# OpenTools API Keys
# Add your API keys here (one per line)
# Format: KEY_NAME=your_api_key_here

GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_token_here
SERPAPI_API_KEY=your_serpapi_key_here

# Optional: Other OpenTools settings
OPENTOOLS_LOG_LEVEL=INFO
WOLFRAM_API_KEY=your_worlframe_api_key_here
GOOGLE_CX_ID=your_google_cx_id_here
GOOGLE_CX_NAME=your_google_cx_name_here

"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"✅ .env template created: {env_path}")
    print("💡 Edit the file to add your actual API keys")
    print("💡 Make sure to add .env to your .gitignore file")
    
    return 0


def load_env_file(args):
    """Load API keys from a .env file and test them."""
    env_file = args.env_file or ".env"
    
    if not Path(env_file).exists():
        print(f"❌ .env file not found: {env_file}")
        print("💡 Create one with: opentools create-env-template")
        return 1
    
    try:
        # Load config from .env file
        config = OpenToolsConfig.from_env(env_file)
        
        print(f"✅ Loaded API keys from {env_file}:")
        print("-" * 50)
        
        if not config.api_keys:
            print("No API keys found in .env file")
        else:
            for key_name, key in config.api_keys.items():
                masked_key = key[:4] + "..." + key[-4:] if len(key) > 12 else "***"
                print(f"{key_name:25} {masked_key}")
        
        return 0
        
    except ImportError as e:
        print(f"❌ {e}")
        print("💡 Install python-dotenv: pip install python-dotenv")
        return 1
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
        return 1


def run_command(args):
    """Run a tool with given arguments."""
    load_all_tools(verbose=False)
    tool_name = args.tool_name
    tool = create_tool(tool_name)
    
    if not tool:
        print(f"❌ Tool '{tool_name}' not found.")
        return 1
    
    args_json = args.args or "{}"
    try:
        kwargs = json.loads(args_json)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON for --args: {e}")
        return 1
    
    try:
        result = tool.run(**kwargs)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            if isinstance(result, dict):
                if result.get("success"):
                    print(result.get("result", result))
                else:
                    print(f"❌ {result.get('error', result)}")
            else:
                print(result)
        return 0
    except TypeError as e:
        print(f"❌ Invalid arguments for tool: {e}")
        meta = getattr(tool, "get_metadata", lambda: {})()
        demo = meta.get("demo_commands", {})
        if isinstance(demo, dict) and "command" in demo:
            print(f"💡 Example: {demo.get('command', '')}")
        elif isinstance(demo, list) and demo:
            print(f"💡 Example: {demo[0]}")
        return 1
    except Exception as e:
        print(f"❌ Error running tool: {e}")
        return 1


def solve_command(args):
    """Solve a question using an agent."""
    from .solver import UnifiedSolver
    
    # Parse enabled_tools from --tools argument
    agent_kwargs = {}
    if args.tools:
        # Parse comma-separated list of tools
        enabled_tools = [tool.strip() for tool in args.tools.split(",") if tool.strip()]
        if enabled_tools:
            agent_kwargs["enabled_tools"] = enabled_tools
    
    solver = UnifiedSolver(
        agent_name=args.agent or "opentools",
        llm_engine_name=args.llm or "gpt-4o-mini",
        verbose=args.verbose,
        **agent_kwargs
    )
    solve_kwargs = {
        "output_types": getattr(args, "output_types", "final") or "final",
        "max_steps": args.max_steps or 10,
        "max_time": args.max_time or 300,
    }
    if args.cache_dir:
        solve_kwargs["root_cache_dir"] = args.cache_dir
    
    result = solver.solve(
        question=args.question,
        image_path=args.image,
        **solve_kwargs,
    )
    
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(json.dumps(result, indent=2, default=str))
    return 0


def test_command(args):
    """Run a tool's test routine."""
    load_all_tools(verbose=False)
    tool_name = args.tool_name
    tool = create_tool(tool_name)
    
    if not tool:
        print(f"❌ Tool '{tool_name}' not found.")
        return 1

    if not hasattr(tool, "test"):
        print(f"❌ Tool '{tool_name}' has no test() method.")
        return 1

    try:
        source = Path(inspect.getfile(tool.__class__)).resolve()
        result = run_existing_tests(tool, source)
    except Exception as exc:
        print(f"❌ Test failed: {exc}")
        return 1

    if result["status"] == "failed":
        print(f"❌ Test failed: {result.get('error', 'test() returned False')}")
        return 1
    print(f"✅ Test routine completed for {tool_name}: {result['status']}")
    if result.get("result_file"):
        print(f"📁 Test result: {result['result_file']}")
    return 0


def _load_local_tool(source: Path):
    """Load one BaseTool subclass after the caller has reviewed preflight results."""
    from .core.base import BaseTool

    module_file = source / "tool.py" if source.is_dir() else source
    if not module_file.is_file() or module_file.suffix != ".py":
        raise ValueError("Local tool must be a Python file or a directory containing tool.py")
    module_name = f"opentools_local_{module_file.parent.name}_{module_file.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module specification for {module_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if obj is not BaseTool
        and issubclass(obj, BaseTool)
        and obj.__module__ == module.__name__
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one local BaseTool subclass, found {len(candidates)}"
        )
    return candidates[0](), module_file


def _resolve_evaluation_target(target: str):
    """Resolve a local source path or an installed registry tool."""
    path = Path(target).expanduser()
    if path.exists():
        return None, path.resolve(), "local"

    # Some legacy tool modules print during discovery. Keep evaluation JSON stable.
    with contextlib.redirect_stdout(io.StringIO()):
        load_all_tools(verbose=False)
        tool = create_tool(target)
    if tool is None:
        raise ValueError(f"Tool or local path not found: {target}")
    return tool, Path(inspect.getfile(tool.__class__)).resolve(), "installed"


def evaluate_command(args):
    """Inspect a tool and optionally run its existing real test routine."""
    try:
        tool, source, target_type = _resolve_evaluation_target(args.target)
    except Exception as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    inspection_result = inspect_source(source)
    test_result = {"status": "not_run"}
    blocked = False

    if args.run_tests:
        if inspection_result["risk_level"] == "restricted" and not args.allow_risky:
            test_result = {
                "status": "blocked_by_preflight",
                "reason": "Restricted source signals require --allow-risky before execution.",
            }
            blocked = True
        else:
            try:
                if tool is None:
                    tool, source = _load_local_tool(source)
                test_result = run_existing_tests(tool, source)
            except Exception as exc:
                test_result = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    card = build_tool_card(tool, inspection_result, test_result) if tool else None
    report = evaluation_report(source, inspection_result, card)
    report["target_type"] = target_type
    report["execution_requested"] = bool(args.run_tests)
    report["llm_judge"] = {"status": "not_run"}

    if args.judge:
        report["llm_judge"] = judge_evaluation_report(report, model=args.judge_model)
        if card is not None:
            card["llm_review"] = report["llm_judge"]

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        report["report_file"] = str(output)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"Target: {source}")
        print(f"Risk level: {inspection_result['risk_level']}")
        print(f"Files scanned: {inspection_result['files_scanned']}")
        print(f"Findings: {len(inspection_result['findings'])}")
        print(f"Test status: {test_result['status']}")
        print(f"LLM judge status: {report['llm_judge']['status']}")
        print(f"Caution: {inspection_result['caution']}")
        if args.output:
            print(f"Report: {report['report_file']}")

    if blocked or test_result["status"] == "failed":
        return 2
    if args.judge and report["llm_judge"]["status"] != "completed":
        return 2
    if inspection_result["parse_errors"]:
        return 2
    return 0


def _inventory_paths(args):
    package_root = Path(__file__).parent
    tools_root = Path(args.tools_root or package_root / "tools").resolve()
    index_path = Path(args.index or tools_root / "evaluation_index.json").resolve()
    inventory_path = Path(args.inventory or tools_root / "readme.md").resolve()
    return tools_root, index_path, inventory_path


def _print_inventory_summary(index, results):
    summary = {
        "indexed_tools": len(index.get("tools", {})),
        "evaluations_requested": len(results),
        "completed": sum(item.get("status") == "completed" for item in results.values()),
        "failed": sum(item.get("status") == "failed" for item in results.values()),
        "skipped": sum(
            item.get("status") == "skipped_by_risk_policy" for item in results.values()
        ),
    }
    print(json.dumps(summary, indent=2))
    return summary


def update_inventory_command(args):
    """Refresh the index and Markdown table from source and existing evidence."""
    tools_root, index_path, inventory_path = _inventory_paths(args)
    index, results = refresh_inventory(
        tools_root,
        index_path,
        inventory_path,
        stale_after_days=args.stale_after_days,
    )
    _print_inventory_summary(index, results)
    return 0


def evaluate_all_command(args):
    """Evaluate an explicit set of tools and refresh the canonical inventory."""
    selected = [item.strip() for item in (args.tools or "").split(",") if item.strip()]
    if not selected and not args.all_eligible:
        print("❌ Pass --tools or explicitly select --all-eligible.", file=sys.stderr)
        return 1
    tools_root, index_path, inventory_path = _inventory_paths(args)
    index, results = refresh_inventory(
        tools_root,
        index_path,
        inventory_path,
        run_tests=True,
        selected_tools=selected,
        max_risk=args.max_risk,
        stale_after_days=args.stale_after_days,
        discard_raw_results=args.discard_raw_results,
    )
    summary = _print_inventory_summary(index, results)
    if any(name.startswith("unknown:") for name in results):
        return 1
    if args.fail_on_error and summary["failed"]:
        return 2
    return 0


def convert_tool_command(args):
    """Create a statically inspected OpenTools contribution bundle."""
    try:
        result = convert_submission(
            args.source,
            args.readme,
            args.output,
            name=args.name,
            entrypoint=args.entrypoint,
            description=args.description,
            category=args.category,
            source_url=args.source_url,
            license_name=args.license_name,
        )
    except Exception as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


def stats_command(args):
    """Show registry statistics."""
    load_all_tools(verbose=False)
    tools = list_available_tools()
    
    if args.json:
        data = {
            "total_tools": len(tools),
            "tools": tools,
        }
        print(json.dumps(data, indent=2))
    else:
        print(f"Total tools: {len(tools)}")
        for t in tools:
            print(f"  - {t}")
    return 0


def list_agents_command(args):
    """List available agents."""
    from .agents import list_agents
    
    agents = list_agents()
    if args.json:
        print(json.dumps(agents, indent=2))
    else:
        print("Available agents:")
        for a in agents:
            print(f"  - {a.get('name', '?')}: {a.get('description', '')}")
    return 0


def search_command(args):
    """Search for tools or agents."""
    load_all_tools(verbose=False)
    
    if args.type == 'tools':
        results = registry.search_tools(args.query)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            if results:
                print(f"Found {len(results)} tool(s) matching '{args.query}':")
                for tool in results:
                    print(f"  - {tool}")
            else:
                print(f"No tools found matching '{args.query}'")
        return 0
    
    elif args.type == 'agents':
        results = registry.search_agents(args.query)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            if results:
                print(f"Found {len(results)} agent(s) matching '{args.query}':")
                for agent in results:
                    name = agent.get('name', '?')
                    desc = agent.get('description', '')
                    print(f"  - {name}: {desc}")
            else:
                print(f"No agents found matching '{args.query}'")
        return 0
    
    else:
        print(f"❌ Invalid search type: {args.type}. Use 'tools' or 'agents'.")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OpenTools - A framework for building and managing AI tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  opentools list                    # List all available tools #tested
  opentools info Calculator_Tool    # Get tool information #tested
  opentools search tools calculator # Search relevant tools #tested
  opentools search agents zero # Search relevant tools
  opentools run Calculator_Tool --args '{"operation":"add","values":[1,2,3]}'
  opentools solve "What is 2+2?" --agent react --llm gpt-4o-mini --tools Calculator_Tool
  opentools test Calculator_Tool    # Run tool's test
  opentools evaluate ./my_tool      # Static preflight; does not execute tool code
  opentools evaluate Calculator_Tool --run-tests --output report.json
  opentools update-inventory        # Refresh index and generated tool table
  opentools evaluate-all --tools Calculator_Tool --discard-raw-results
  opentools convert-tool function.py --readme README.md --name Example
  opentools stats                   # Show registry stats
  opentools list-agents             # List available agents
  opentools reload                  # Reload all tools
  opentools create-env-template     # Create .env template
  opentools load-env .env           # Load and test .env file
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all available tools')
    list_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    list_parser.set_defaults(func=list_command)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get detailed information about a tool')
    info_parser.add_argument('tool_name', help='Name of the tool')
    info_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    info_parser.set_defaults(func=info_command)
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a tool with given arguments')
    run_parser.add_argument('tool_name', help='Name of the tool to run')
    run_parser.add_argument('--args', default='{}', help='JSON object of arguments (e.g. \'{"operation":"add","values":[1,2,3]}\')')
    run_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    run_parser.set_defaults(func=run_command)
    
    # Solve command (run agent)
    solve_parser = subparsers.add_parser('solve', help='Solve a question using an agent')
    solve_parser.add_argument('question', help='Question to solve')
    solve_parser.add_argument('--agent', default='opentools', help='Agent to use (default: opentools)')
    solve_parser.add_argument('--llm', default='gpt-4o-mini', help='LLM engine (default: gpt-4o-mini)')
    solve_parser.add_argument('--tools', help='Comma-separated list of tools to enable (e.g., "Calculator_Tool,Search_Engine_Tool")')
    solve_parser.add_argument('--image', help='Path to image file')
    solve_parser.add_argument('--max-steps', type=int, default=10, help='Maximum steps')
    solve_parser.add_argument('--max-time', type=int, default=300, help='Maximum time (seconds)')
    solve_parser.add_argument('--cache-dir', help='Cache directory')
    solve_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    solve_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    solve_parser.set_defaults(func=solve_command)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run a tool\'s test routine')
    test_parser.add_argument('tool_name', help='Name of the tool to test')
    test_parser.set_defaults(func=test_command)

    # Evaluate command. Static inspection is the safe default; execution is explicit.
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Inspect a local or installed tool and optionally run its existing tests',
    )
    evaluate_parser.add_argument('target', help='Registered tool name, tool.py, or tool directory')
    evaluate_parser.add_argument(
        '--run-tests',
        action='store_true',
        help='Import the tool and execute its existing test() routine after preflight',
    )
    evaluate_parser.add_argument(
        '--allow-risky',
        action='store_true',
        help='Allow test execution when preflight reports restricted capabilities',
    )
    evaluate_parser.add_argument('--output', help='Write the evidence report to this JSON file')
    evaluate_parser.add_argument(
        '--judge',
        action='store_true',
        help='Request an advisory LLM review of metadata and collected evidence',
    )
    evaluate_parser.add_argument(
        '--judge-model',
        default='gpt-4o-mini',
        help='Configured OpenTools model used by --judge (default: gpt-4o-mini)',
    )
    evaluate_parser.add_argument('--json', action='store_true', help='Print the full JSON report')
    evaluate_parser.set_defaults(func=evaluate_command)

    def add_inventory_arguments(command_parser):
        command_parser.add_argument('--tools-root', help='Tools directory (default: packaged tools)')
        command_parser.add_argument('--index', help='Evaluation index JSON output path')
        command_parser.add_argument('--inventory', help='Generated Markdown inventory path')
        command_parser.add_argument(
            '--stale-after-days',
            type=int,
            default=30,
            help='Mark dated evidence stale after this many days (default: 30)',
        )

    update_inventory_parser = subparsers.add_parser(
        'update-inventory',
        help='Regenerate the evaluation index and tool table from existing evidence',
    )
    add_inventory_arguments(update_inventory_parser)
    update_inventory_parser.set_defaults(func=update_inventory_command)

    evaluate_all_parser = subparsers.add_parser(
        'evaluate-all',
        help='Run existing tests for selected eligible tools and refresh the inventory',
    )
    add_inventory_arguments(evaluate_all_parser)
    selection = evaluate_all_parser.add_mutually_exclusive_group()
    selection.add_argument('--tools', help='Comma-separated class names or tool folders')
    selection.add_argument(
        '--all-eligible',
        action='store_true',
        help='Attempt every tool permitted by --max-risk',
    )
    evaluate_all_parser.add_argument(
        '--max-risk',
        choices=['low', 'caution'],
        default='low',
        help='Highest risk classification eligible for execution (default: low)',
    )
    evaluate_all_parser.add_argument(
        '--discard-raw-results',
        action='store_true',
        help='Keep summaries in the index but remove raw result files created by this run',
    )
    evaluate_all_parser.add_argument(
        '--fail-on-error',
        action='store_true',
        help='Return a failing exit code after recording evaluation failures',
    )
    evaluate_all_parser.set_defaults(func=evaluate_all_command)

    convert_parser = subparsers.add_parser(
        'convert-tool',
        help='Convert annotated Python code into a reviewable OpenTools bundle',
    )
    convert_parser.add_argument('source', help='Submitted Python file')
    convert_parser.add_argument('--readme', required=True, help='Submitted README file')
    convert_parser.add_argument('--name', required=True, help='Display name for the tool')
    convert_parser.add_argument('--entrypoint', help='Public function to wrap')
    convert_parser.add_argument('--description')
    convert_parser.add_argument('--category', default='community')
    convert_parser.add_argument('--source-url')
    convert_parser.add_argument('--license', dest='license_name')
    convert_parser.add_argument('--output', default='opentools_contributions')
    convert_parser.set_defaults(func=convert_tool_command)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show registry statistics')
    stats_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    stats_parser.set_defaults(func=stats_command)
    
    # List agents command
    list_agents_parser = subparsers.add_parser('list-agents', help='List available agents')
    list_agents_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    list_agents_parser.set_defaults(func=list_agents_command)
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for tools or agents with keywords')
    search_parser.add_argument('type', choices=['tools', 'agents'], help='Type to search: tools or agents')
    search_parser.add_argument('query', help='Search query string')
    search_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    search_parser.set_defaults(func=search_command)
    
    # Reload command
    reload_parser = subparsers.add_parser('reload', help='Reload all tools')
    reload_parser.set_defaults(func=reload_command)
    
    # Create .env template command
    env_template_parser = subparsers.add_parser('create-env-template', help='Create a .env template file')
    env_template_parser.add_argument('--output', help='Output file path (default: .env)')
    env_template_parser.add_argument('--force', action='store_true', help='Overwrite existing file')
    env_template_parser.set_defaults(func=create_env_template)
    
    # Load .env file command
    load_env_parser = subparsers.add_parser('load-env', help='Load and test API keys from a .env file')
    load_env_parser.add_argument('env_file', nargs='?', help='Path to .env file (default: .env)')
    load_env_parser.set_defaults(func=load_env_file)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
