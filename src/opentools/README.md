# OpenTools

OpenTools is an agentic framework with pluggable tools for complex reasoning and task execution. It provides a unified interface to run different agents (ReAct, OpenTools, OctoTools, Chain-of-Thought, Zero-shot) with shared tool discovery, execution, and optional retrieval.

---

## Package layout

```
src/opentools/
  __init__.py       # Package exports and lazy loading
  solver.py         # UnifiedSolver, solve_with_agent()
  cli.py            # Command-line interface (tools + env)

  core/             # BaseTool, config, registry, LLM factory, tool retrieval, display
  agents/           # Agent implementations and mixins (react, opentools, octotools, cot, zeroshot)
  tools/            # Tool packages (calculator, search, visual_ai, ...)
  models/           # Initializer, Executor (used by agents); Planner/Memory (optional/unused)
  Benchmark/        # Benchmark runner and datasets
  utils/            # Shared utilities (if present)
```

- **Core** — Tool base class, config, registry, LLM engines (OpenAI, Gemini), FAISS tool retrieval, display. See `core/README.md`.
- **Agents** — BaseAgent, ToolBasedAgent, agent registry, and agent types (ReAct, OpenTools, OctoTools, CoT, ZeroShot). See `agents/README.md`.
- **Tools** — All tool packages; each exposes a `BaseTool` subclass. See `tools/README.md`.
- **Models** — Initializer (tool discovery) and Executor (tool execution) used by the agent mixin. See `models/README.md`.
- **Benchmark** — Run and score agents on datasets. See `Benchmark/README.md`.

---

## Quick start

### 1. Create an agent and solve a question

```python
from opentools import create_agent, list_agents

# List available agents
for a in list_agents():
    print(a["name"], "—", a["description"])

# Create an agent (e.g. ReAct with all tools)
agent = create_agent(
    "react",
    llm_engine_name="gpt-4o-mini",
    enabled_tools=["all"],
)

# Solve
result = agent.solve(question="What is the capital of France?")
print(result.get("direct_output") or result.get("final_output"))
```

### 2. Use the unified solver

```python
from opentools import UnifiedSolver, solve_with_agent

solver = UnifiedSolver(
    agent_name="opentools",
    llm_engine_name="gpt-4o-mini",
    enabled_tools=["Calculator_Tool", "Search_Engine_Tool"],
)
result = solver.solve(question="What is 2 + 2?")

```

### 3. Run a single tool

```python
from opentools import create_tool, load_all_tools

load_all_tools()  # discover tools under opentools.tools
tool = create_tool("Calculator_Tool")
out = tool.run(operation="add", values=[1, 2, 3])
```

### 4. Environment and API keys

Set API keys (e.g. `OPENAI_API_KEY`, `GOOGLE_API_KEY`) in the environment or in a `.env` file. The core config loads keys whose names end with `_API_KEY`. Use the CLI to create a template or load and test:

```bash
opentools create-env-template --output .env
opentools load-env .env
```

---


## Main exports (from `opentools`)

- **Core:** `BaseTool`, `ToolRegistry`, `registry`, `OpenToolsConfig`
- **Tools:** `list_available_tools`, `get_tool_info`, `create_tool`, `load_all_tools`, `search_tools`, `get_tools_by_category`
- **Agents:** `BaseAgent`, `create_agent`, `list_agents`, `register_agent`
- **Solver:** `UnifiedSolver`, `solve_with_agent`

Subpackages (`opentools.core`, `opentools.agents`, `opentools.tools`, etc.) are imported on demand to avoid loading all tools and agents at once.

---

## Where to read more

- **Agents (difference between ReAct, OpenTools, OctoTools, CoT, ZeroShot):** `agents/README.md`
- **Tools (inventory, how to add a tool, test cases):** `tools/README.md`
- **Core (BaseTool, config, registry, LLM factory, tool retrieval):** `core/README.md`
- **Models (Initializer, Executor):** `models/README.md`
- **Benchmarks (how to run and add benchmarks):** `Benchmark/README.md`
