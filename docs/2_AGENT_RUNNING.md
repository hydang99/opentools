## Running agents via `UnifiedSolver`

This guide explains how to drive OpenTools agents through the shared `UnifiedSolver` interface and how to interpret the results.

You can run agents in two ways:
- **From Python**: import `UnifiedSolver` and call `.solve(...)`.
- **From the CLI**: use `opentools solve` to answer questions from the terminal.

### Basic usage (Python)

```python
from opentools import UnifiedSolver

solver = UnifiedSolver(
    agent_name="opentools",      # or "react", "octotools", "chain_of_thought", "zero_shot"
    llm_engine_name="gpt-4o-mini",
    verbose=True,
    enabled_tools=["all"],       # only used by tool-based agents
    enable_faiss_retrieval=False,
)

result = solver.solve(
    question="What is the capital of the United States?",
    # image_path="path/to/image.png",   # optional, if the agent supports images
    output_types="direct",             # agent-specific; common values: "direct", "final"
    max_steps=10,
    max_time=300,
)
```

### Key constructor arguments

- **`agent_name`**  
  Name registered in the agent registry. Built‑ins include:
  - `react`
  - `opentools`
  - `octotools`
  - `chain_of_thought`
  - `zero_shot`

- **`llm_engine_name`**  
  Passed through to the internal LLM engine factory. Must match a configured engine (e.g. `"gpt-4o-mini"`).

- **`enabled_tools`** (tool‑based agents only)  
  - List of tool class names (`["Wolfram_Math_Tool", "Search_Engine_Tool"]`) or `["all"]`.
  - If a name is misspelled or the tool cannot be loaded, it will silently drop from the available set.

- **`enable_faiss_retrieval`** (via agent kwargs)  
  Enables FAISS‑based tool retrieval when embeddings are present; otherwise the agent will see the full toolbox.

Any extra keyword arguments you pass to `UnifiedSolver(...)` are forwarded to the underlying agent’s constructor.

### `solve(...)` arguments to watch

`UnifiedSolver.solve(question, image_path=None, **kwargs)` forwards `**kwargs` directly to the agent’s `solve` method. Common ones:

- **`output_types`**  
  Controls which primary output field the agent focuses on:
  - `"direct"` → agents tend to fill `direct_output`
  - `"final"` → agents tend to fill `final_output`
  - others may be agent‑specific

- **`max_steps` / `max_time`**  
  Upper bounds for multi‑step agents (ReAct, OpenTools, OctoTools). If either is too small, the agent may stop with an incomplete trace.

- **`root_cache_dir`**  
  Where result files and traces are written. Each run gets a timestamped subfolder (e.g. `react_cache/...`).

Check the agent’s own docstring or `agents/read_me.md` for additional, agent‑specific arguments.

### Running agents from the CLI

The `opentools solve` subcommand wraps `UnifiedSolver` so you can run agents directly from the command line:

```bash
opentools solve "What is the capital of the United States?" \
  --agent opentools \
  --llm gpt-4o-mini \
  --verbose \
  --max-steps 10 \
  --max-time 300
```

Key flags:

- **`user_question`**: user question to solve (required).
- **`--agent`**: agent name (`opentools`, `react`, `octotools`, `chain_of_thought`, `zero_shot`, ...).
- **`--llm`**: LLM engine name (e.g. `gpt-4o-mini`).
- **`--tools`**: optional comma‑separated list of tool class names to enable (tool‑based agents only), e.g. `--tools "Wolfram_Math_Tool,Search_Engine_Tool"`.
- **`--max-steps`, `--max-time`**: bounds on multi‑step reasoning.
- **`--cache-dir`**: where to write traces and results.
- **`--image`**: optional path to an input image (if the agent supports images).
- **`--json`**: return the full result as JSON instead of pretty‑printed text.

### Result structure

Most agents return a dictionary shaped like:

- **Core metadata**
  - `query`
  - `image` / `file_path` (if used)
  - `agent`
  - `llm_engine`
  - `steps_taken`
  - `execution_time`
  - `timestamp`
  - `cache_dir`
  - `solver_type` (added by `UnifiedSolver`)
  - `agent_used`

- **Primary outputs** (agent‑dependent)
  - `direct_output`
  - `final_output`
  - `full_answer`

- **Tracing & diagnostics**
  - `reasoning_trace` (per‑step details; often a nested dict)
  - `token_usage` (prompt / completion / total tokens)
  - `error_executions` (failed tool calls)

When post‑processing results, treat `direct_output` / `final_output` / `full_answer` as interchangeable fallbacks and keep the full `reasoning_trace` for debugging or visualization (as in `docs/demo/agent_running.ipynb`).

