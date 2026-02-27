# OpenTools Core

This directory contains the core runtime used across the framework: the base tool class, tool registry, configuration, LLM engine factory and adapters, FAISS-based tool retrieval, and display/logging for agents.

---

## Directory layout

| File | Purpose |
|------|--------|
| **`base.py`** | `BaseTool`: canonical tool interface and metadata. All tools inherit from it. |
| **`config.py`** | `OpenToolsConfig`: global config and API keys (from env or `.env`). |
| **`display.py`** | `DisplayManager` and `AgentDisplayMixin`: structured logging and step tracking for agents. |
| **`factory.py`** | `create_llm_engine()`: builds the right LLM adapter from a model string. |
| **`openai.py`** | `ChatOpenAI`: OpenAI API (text, multimodal, caching, embeddings, structured output). |
| **`gemini.py`** | `ChatGemini`: Google Gemini API (text, optional multimodal, caching). |
| **`registry.py`** | `ToolRegistry`: register and discover tools; create instances; load from `opentools.tools`. |
| **`tool_retrieval.py`** | `ToolRetriever`: FAISS retrieval over tool embeddings for “relevant tools” given a query. |

---

## BaseTool (`base.py`)

All tools inherit from **`BaseTool`**. It defines the common interface and metadata used by discovery, planning, and execution.

### Required constructor arguments

- **`name`** — Unique tool name.
- **`description`** — Full description (capabilities, examples).
- **`category`** — Category for grouping (e.g. `image`, `web_search`).
- **`tags`** — List of tags for search/routing.
- **`parameters`** — OpenAI-style JSON schema for inputs.
- **`agent_type`** — Agent routing label (e.g. `Visual-Agent`, `Search-Agent`).
- **`demo_commands`** — Example usage (e.g. `{"command": "...", "description": "..."}`).

### Optional constructor arguments

- **`limitation`** — Known limitations.
- **`type`** — Tool type, default `"function"`.
- **`strict`** — Strict parameter validation, default `True`.
- **`accuracy`** — Accuracy metrics.
- **`model_string`** — Model name when the tool uses an LLM.
- **`required_api_keys`** — List of required env key names (e.g. `["OPENAI_API_KEY"]`).
- **`is_multimodal`** — Whether the tool is multimodal.
- **`llm_engine`** — Injected LLM engine instance.
- **`require_llm_engine`** — If `True`, an engine is created when not provided (via `create_llm_engine`).

### Utility methods

- **`get_metadata()`** — Returns the tool metadata dict (name, description, parameters, category, tags, etc.).
- **`get_api_key(name)`** — Reads an API key from the global config.
- **`require_api_key(name)`** — Same but raises if missing.
- **`check_required_api_keys()`** — Validates all `required_api_keys`.
- **`set_custom_output_dir(path)`** — Sets the tool’s output directory.
- **`embed_tool()`** — Builds and stores tool embeddings (needs an LLM engine); used for FAISS tool retrieval.
- **`test(tool_test, file_location, result_parameter, search_type, ...)`** — Runs tests from `tools/test_file/data.json`, writes `tools/<file_location>/temp_result.json`.

---

## Configuration (`config.py`)

**`OpenToolsConfig`** holds global settings and API keys.

- **API keys**: Typically loaded from environment variables whose names end with `_API_KEY`. Use **`from_env(env_file=None)`** to load a `.env` file (requires `python-dotenv`) and collect those variables into `api_keys`.
- **Methods**: `get_api_key(name)`, `set_api_key(name, value)`, `has_api_key(name)`.
- **Other fields**: `default_timeout`, `max_retries`, `log_level`, `log_file`, `enable_cache`, `cache_ttl`.

A global instance **`config`** is created with `OpenToolsConfig.from_env()` so tools and engines can call `config.get_api_key(...)`.

---

## LLM factory (`factory.py`)

**`create_llm_engine(model_string, use_cache=False, is_multimodal=True, **kwargs)`** returns an engine instance:

- **OpenAI**: model strings containing `gpt`, `o1`, `o3`, or `o4` → **`ChatOpenAI`** (from `openai.py`).
- **Gemini**: model strings containing `gemini` → **`ChatGemini`** (from `gemini.py`).
- Anything else raises **`ValueError`** (with a hint to add support in `factory.py`).

Pass **`is_multimodal=True`** for image/file inputs; **`use_cache=True`** for response caching when the engine supports it.

---

## OpenAI engine (`openai.py`)

**`ChatOpenAI`** wraps the OpenAI API and provides:

- **Text and multimodal** generation (images/files via upload, then delete after use).
- **Caching** of responses when `use_cache=True` (e.g. diskcache).
- **Structured output** when `response_format` is a Pydantic model and the model supports it.
- **Embeddings**: `embed_text(text)` and `embed_with_normalization(text)` (e.g. for tool retrieval).

**Requirements**: `OPENAI_API_KEY` in the environment. Embeddings use a fixed model (e.g. `text-embedding-3-large`).

**Behavior**: Detects reasoning models (`o1`, `o3`, `o4`, `gpt-5` variants) and uses the appropriate API shape. Tracks token usage via `get_token_usage()`.

---

## Gemini engine (`gemini.py`)

**`ChatGemini`** wraps the Google Gemini API:

- Text generation; multimodal when the chosen model supports it.
- Optional response caching.

**Requirements**: `GOOGLE_API_KEY` in the environment; `google-genai` (or the project’s Gemini client package) installed.

---

## Tool registry (`registry.py`)

**`ToolRegistry`** manages tool **classes** and optional **instances**.

### Registration and discovery

- **`register(tool_class)`** — Register a `BaseTool` subclass by class name.
- **`register_instance(tool_instance)`** — Register an instance (uses `tool_instance.tool_name` if present; opentools tools use `name`).
- **`discover_tools(module_path)`** — Import a module and register all `BaseTool` subclasses in it.
- **`auto_discover_tools(tools_dir="opentools.tools", verbose=True)`** — Scan `tools_dir` subpackages and register each tool class once.
- **`load_all_tools(tools_dir="opentools.tools", verbose=True)`** — Idempotent load: if the directory was already discovered, returns existing list; otherwise runs `auto_discover_tools`.

### Queries and creation

- **`list_tools()`** — Registered tool class names.
- **`list_tool_instances()`** — Registered instance names.
- **`get_tool(tool_name)`** — Get the tool class.
- **`get_tool_instance(tool_name)`** — Get a registered instance.
- **`create_tool_instance(tool_name, **kwargs)`** — Instantiate a tool by name with given kwargs.
- **`get_tool_info(tool_name)`** — Build a small info dict by creating a temporary instance and reading attributes (e.g. description, version, require_llm_engine). Expects legacy attributes like `tool_description` / `tool_version` if present; otherwise falls back to a minimal description.
- **`get_available_tools()`** — Dict of tool name → `get_tool_info(name)`.
- **`clear()`** — Clear all registered classes and instances.
- **`is_module_discovered(tools_dir)`** — Whether that directory has already been auto-discovered.

A global **`registry`** instance is defined so agents and the initializer can call `registry.load_all_tools()` and `registry.create_tool_instance(...)`.

---

## Tool retrieval (`tool_retrieval.py`)

**`ToolRetriever`** returns the most relevant tools for a query using **FAISS** over precomputed tool embeddings, so agents can pass a subset of tools instead of the full list.

### Setup

- **Embeddings file**: By default, `agents/embeddings/tool_embeddings.json` (each key = tool name, value = embedding vector). Populated by **`BaseTool.embed_tool()`**.
- **`llm_engine`**: Used to embed the user query (and optionally to expand the query). Must support something like `embed_with_normalization(query)`.

### Main methods

- **`set_toolbox_metadata(metadata)`** — Set a dict mapping tool name → metadata (used when formatting retrieved tools for the agent).
- **`retrieve_tools(query)`** — Returns a list of `(tool_name, score)` for the top-k tools. Internally may expand the query and embed it, then search the FAISS index.
- **`get_tool_names(query)`** — Same as `retrieve_tools` but returns only the list of tool names.
- **`get_retrieved_tools_metadata(query)`** — Retrieves tools, then formats their metadata (from `toolbox_metadata`) as a string for prompting.

Agents that use **`enable_faiss_retrieval=True`** typically call `set_toolbox_metadata(get_toolbox_metadata())` then `get_tool_names(question)` or `get_retrieved_tools_metadata(question)` to obtain the tool list or metadata string for the LLM.

---

## Display and logging (`display.py`)

**`DisplayManager`** provides structured, level-based logging for agents:

- **Log levels**: `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`, `STEP`, `RESULT`.
- **Format**: Optional ANSI colors (`use_colors=True`), timestamp, agent name, level, message.
- **Steps**: `start_step(description)` / `end_step(result, success)` to track duration and history.
- **Helpers**: `log_thought`, `log_action`, `log_observation`, `log_tool_command`, `log_error`, `log_warning`, `log_final_answer`, `display_progress_summary`, `get_trace_formatted`.

**`AgentDisplayMixin`** is mixed into **`BaseAgent`** so agents get a `DisplayManager` and methods like `log(...)`, `log_step_start`, `log_step_end`, `log_final_answer`. The mixin expects the agent to implement `get_agent_name()`.

---

## How the pieces fit together

1. **Config** — `config = OpenToolsConfig.from_env()` loads API keys; tools and engines use `config.get_api_key(...)`.
2. **Factory** — Agents and tools call `create_llm_engine(model_string, is_multimodal=...)` to get `ChatOpenAI` or `ChatGemini`.
3. **Registry** — The agents’ initializer (or equivalent) calls `registry.load_all_tools()` to discover tools from `opentools.tools`, then builds instances and metadata for the chosen agent/tool set.
4. **Tool retrieval** — When FAISS retrieval is enabled, the agent passes toolbox metadata to `ToolRetriever`, which uses `tool_embeddings.json` and the LLM embedder to return relevant tool names or metadata for the user question.
5. **Display** — Agents use `AgentDisplayMixin` / `DisplayManager` for consistent logging and step traces.

---

## Core exports

**`opentools.core`** exposes:

- **`BaseTool`** — From `base.py`.
- **`OpenToolsConfig`** — From `config.py`.
- **`ToolRegistry`** — From `registry.py`.

Engines and other modules are used via the factory or direct imports (e.g. `from opentools.core.openai import ChatOpenAI`).
