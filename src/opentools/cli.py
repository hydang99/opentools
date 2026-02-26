"""
Command-line interface for OpenTools framework.

This module provides a convenient CLI for discovering and managing tools.
"""

import argparse
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
    
    import inspect
    sig = inspect.signature(tool.test)
    params = list(sig.parameters.keys())
    if "self" in params:
        params.remove("self")
    
    test_kwargs = {}
    if params:
        # Tool has required params: use DEFAULT_TEST_ARGS or get_default_test_args() if defined
        default_args = getattr(tool.__class__, "DEFAULT_TEST_ARGS", None)
        if default_args is None and hasattr(tool, "get_default_test_args"):
            default_args = tool.get_default_test_args()
        if default_args:
            test_kwargs = dict(default_args)
        else:
            print(f"⚠️  Tool '{tool_name}.test()' requires parameters: {params}")
            print("   Define DEFAULT_TEST_ARGS on the tool class or implement get_default_test_args() for CLI testing.")
            return 1
    
    try:
        tool.test(**test_kwargs)
        print(f"✅ Test completed for {tool_name}")
        return 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1


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