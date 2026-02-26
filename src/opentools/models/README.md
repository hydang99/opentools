# OpenTools Models

This directory contains modules for **tool discovery**, **tool execution**, and (optionally) **planning and memory**. The parts that are used by the main agent pipeline are **Initializer** and **Executor**; they are wired in via `agents/mixins/tool_capability_mixin.py` for all tool-based agents (ReAct, OpenTools, OctoTools, etc.).

---

## Directory layout

| File | Purpose | Used by |
|------|--------|--------|
| **`initializer.py`** | Discover tools under `opentools/tools`, build `toolbox_metadata`, optionally start vLLM. | `ToolCapabilityMixin` (all tool-based agents). |
| **`executor.py`** | Dynamically import a tool and run `tool.run(**args)` with optional LLM injection. | `ToolCapabilityMixin` (ReAct, OpenTools; OctoTools overrides with its own executor). |
| **`memory.py`** | Store query, files, and action history (query, files, actions). | **Not used** by agents; OctoTools uses `agents/octotools/modules/memory.py`. |
| **`planner.py`** | LLM-based query analysis, next-step planning, memory verification, final/direct output. | **Not used** by agents; OctoTools uses `agents/octotools/modules/planner.py`. |
| **`formatters.py`** | Pydantic schemas for planner/executor (`QueryAnalysis`, `NextStep`, `MemoryVerification`, `ToolCommand`). | Only imported by `planner.py` in this package. |
| **`utils.py`** | `make_json_serializable` and `make_json_serializable_truncated`. | Only referenced in `solver.ipynb`; OctoTools uses `agents/octotools/modules/utils.py`. |

---

## Initializer (`initializer.py`)

**Role:** Discover tool modules, instantiate tools to collect metadata, and optionally start a vLLM server. Used by `ToolCapabilityMixin.initialize_tool_capabilities()` to build `self.initializer`.

### Constructor

- **`enabled_tools`**: List of tool names or `["all"]`. If `["all"]`, every `tool.py` under `opentools/tools` is loaded.
- **`model_string`**: LLM model name (e.g. for tools that need an engine). If it starts with `"vllm-"`, a vLLM server is started.
- **`verbose`**: Extra logging.
- **`vllm_config_path`**: Optional path passed to `vllm serve --config`.

### Behavior

- **`_set_up_tools()`** (called from `__init__`): Normalizes `enabled_tools` → list of folder names (e.g. `Calculator_Tool` → `calculator`), then:
  1. **`load_tools_and_get_metadata()`** — Walks `src/opentools/tools`, imports each `tool.py`, instantiates each `*_Tool` class (with `model_string` if `require_llm_engine`), calls `get_metadata()`, and fills **`toolbox_metadata`** (tool class name → metadata dict).
  2. **`run_demo_commands()`** — Determines which tools are actually available and sets **`available_tools`** (list of tool class names). `toolbox_metadata` is then filtered to these only.

### Attributes

- **`toolbox_metadata`**: Dict mapping tool class name (e.g. `Calculator_Tool`) to metadata from `get_metadata()`.
- **`available_tools`**: List of tool class names that were loaded and kept.
- **`load_all`**: `True` when `enabled_tools == ["all"]`.
- **`model_string`**, **`vllm_config_path`**, **`vllm_server_process`**: Used for vLLM when `model_string.startswith("vllm-")`.

### vLLM

When `model_string` starts with `"vllm-"`:

- Runs `vllm serve <model> --port 8888` (or `vllm serve --config <vllm_config_path> --port 8888`).
- Waits until the server reports startup; stores the process in `vllm_server_process`.

---

## Executor (`executor.py`)

**Role:** Resolve a tool by name, instantiate it (with optional `llm_engine`), and call `tool.run(**args)`. Used by `ToolCapabilityMixin` so agents can run tools via `self.executor.execute_tool_command(name, args)`.

### Constructor

- **`llm_engine_name`**: Model string passed to tools that need an LLM.
- **`llm_engine`**: Optional engine instance injected into tools with `require_llm_engine`.
- **`root_cache_dir`**: Default `"solver_cache"`; used by `set_query_cache_dir`.
- **`num_threads`**, **`max_time`**, **`max_output_length`**, **`verbose`**: Stored for use by subclasses or future behavior.

### Methods

- **`execute_tool_command(tool_name, args)`**  
  - Derives module path: `opentools.tools.<name_lower_without_suffix>.tool` (e.g. `Calculator_Tool` → `opentools.tools.calculator.tool`).  
  - Imports the module, gets the tool class, instantiates it (with `model_string` and `llm_engine` if `require_llm_engine`), and returns `tool.run(**args)`.  
  - On exception, returns an error string.

- **`set_query_cache_dir(query_cache_dir)`**  
  - Sets the cache directory for this run (or a timestamped subdir under `root_cache_dir` if not provided). Creates the directory.

### Tool naming convention

- Tool class name ends with `_Tool` (e.g. `Calculator_Tool`).
- Package name is lowercase without the `_tool` suffix (e.g. `calculator`).
- Import path: `opentools.tools.<package>.tool`, and the class name in that module must match (e.g. `Calculator_Tool`).

---

## Memory (`memory.py`)

**Status:** Not used by the current agent code. OctoTools uses its own `agents/octotools/modules/memory.py`.

Stores query, attached files (with optional descriptions), and an ordered map of actions (tool name, sub_goal, command, result). Methods: `set_query`, `add_file`, `add_action`, `get_query`, `get_files`, `get_actions`. File type descriptions are inferred from extension when `add_file` is called without a description.

---

## Planner (`planner.py`)

**Status:** Not used by any agent. OctoTools uses `agents/octotools/modules/planner.py`, which has a similar role (query analysis, next step, memory verification, final/direct output).

This module uses `formatters.py` for structured outputs (`QueryAnalysis`, `NextStep`, `MemoryVerification`) and would drive a “plan → execute → verify” loop if wired in.

---

## Formatters (`formatters.py`)

**Status:** Only used by `models/planner.py`. Not referenced elsewhere.

Pydantic models for planner/executor prompts:

- **`QueryAnalysis`**: `concise_summary`, `required_skills`, `relevant_tools`, `additional_considerations`.
- **`NextStep`**: `justification`, `context`, `sub_goal`, `tool_name`.
- **`MemoryVerification`**: `analysis`, `stop_signal`.
- **`ToolCommand`**: `analysis`, `explanation`, `command`.

---

## Utils (`utils.py`)

**Status:** Only referenced in `solver.ipynb`. Production OctoTools code uses `agents/octotools/modules/utils.py` (same helpers, different package).

- **`make_json_serializable(obj)`**: Recursively converts objects to JSON-serializable form (dict/list/primitive or `obj.__dict__` or `str(obj)`).
- **`make_json_serializable_truncated(obj, max_length=100000)`**: Same idea but truncates long strings and large representations to `max_length`.

---

## How the used parts fit

1. **`ToolCapabilityMixin.initialize_tool_capabilities()`** creates:
   - **`Initializer(...)`** → `self.initializer` with `toolbox_metadata` and `available_tools`.
   - **`Executor(...)`** → `self.executor` used to run tool commands.

2. Agents use **`self.initializer.toolbox_metadata`** and **`self.initializer.available_tools`** for tool lists and metadata (e.g. prompts, FAISS retrieval).

3. When the agent decides to call a tool, it uses **`self.executor.execute_tool_command(tool_name, args)`** to run it. (OctoTools replaces `self.executor` with its own executor from `octotools/modules` but still uses the same Initializer.)

The **Planner**, **Memory**, and **Formatters** in this directory are currently unused by the main pipeline; they mirror logic that lives under `agents/octotools/modules/` for the OctoTools agent.
