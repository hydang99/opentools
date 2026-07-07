# 🛠️🧰 OpenTools: Open, Reliable, and Collective: A Community-Driven Framework for Tool-Using AI Agents

[![Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github&logoColor=white)](https://github.com/hydang99/opentools)
[![Demo](https://img.shields.io/badge/Hugging%20Face-Demo%20WebUI-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/opentools/opentools)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

**OpenTools** is a community-driven framework for building, evaluating, and deploying tools for tool-integrated language models. It treats end-to-end agent performance as a combination of **tool-use accuracy** (selecting/calling tools correctly) and **intrinsic tool accuracy** (tools staying correct and stable as APIs and environments drift). To support both, OpenTools provides two complementary workflows: a **Tool Accuracy / Maintenance Loop** for continuous validation, regression testing, and reliability reporting, and an **Agentic Workflow** for integrating curated tool collections into LLM agents to solve real user tasks. The project emphasizes standardized tool schemas, continuous community-driven evaluation, clear separation between tools and agent policies, and transparent, debuggable execution via structured tool-call and error logs.

---

## What Is Included

- Tool framework: `BaseTool`, registry, configuration, and LLM engine adapters.
- Agents: OpenTools, OctoTools, ReAct, Chain-of-Thought, Zero-Shot.
- Benchmarks: multi-agent runner and dataset folders.
- CLI: tool discovery, metadata, and environment helpers.

---

## Setup

Create a Python environment, install dependencies, and install OpenTools in editable mode.

### 1. Clone the repository

Fork the repo on GitHub, then clone your fork:

```bash
git clone https://github.com/your-username/opentools.git
cd opentools
```

### 2. Create and activate a virtual environment

Example with conda:

```bash
conda create -n opentools python=3.11 -y
conda activate opentools
```

### 3. (Optional) Windows: install dependencies

On Windows, shell and dependency behavior can differ from macOS/Linux. If you're on Windows, run the provided script first (e.g. in Git Bash or WSL):

```bash
bash set_up_package_window.sh
```

### 4. Install OpenTools in editable mode

From the repo root:

```bash
pip install -e .
```


## API Keys and Environment

OpenTools loads API keys from environment variables ending with `_API_KEY`. If `python-dotenv` is installed, `.env` files can be loaded automatically.

Example `.env`:

```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
...
```

CLI helpers for env setup:

```bash
opentools create-env-template
opentools load-env .env
```

### Evaluate a tool locally

Run a static preflight before importing or executing a local tool:

```bash
opentools evaluate ./src/opentools/tools/calculator
```

The preflight reports observable network, credential, filesystem, subprocess, and
dynamic-execution signals. It is a review aid, not a security guarantee. Tool code
is executed only when explicitly requested:

```bash
opentools evaluate Calculator_Tool --run-tests --output evaluation-report.json
```

An optional LLM judge can review the tool card and observed evidence for
documentation quality, test adequacy, output-contract clarity, and
maintainability:

```bash
opentools evaluate Calculator_Tool --run-tests --judge --judge-model gpt-4o-mini
```

The judge is advisory, cannot override restricted preflight findings, and is not
used as a security certification. CI does not invoke it automatically, avoiding
credential requirements, API costs, and nondeterministic acceptance decisions.
Invoking `--judge` sends selected tool metadata and sanitized findings—but not
source code, credential values, or absolute local paths—to the configured model
provider.

Restricted findings block test execution unless the user also passes
`--allow-risky`. Reports contain only observed test results; when a test routine
does not write structured evidence, OpenTools reports
`completed_without_structured_results` rather than inferring an accuracy.

The same preflight and unit checks run on tool-related pull requests and on a
weekly schedule through GitHub Actions, supporting regression and drift review.

### Refresh evaluations and the tool inventory

`evaluation_index.json` is the canonical summary used by tool cards and the
generated table in `src/opentools/tools/readme.md`. Refresh both from existing
evidence without executing tools:

```bash
opentools update-inventory
```

Run real existing tests for an explicit low-risk set, update the index, and
regenerate the table:

```bash
opentools evaluate-all \
  --tools Calculator_Tool \
  --max-risk low \
  --discard-raw-results
```

Bulk execution requires `--tools` or the explicit `--all-eligible` flag.
Restricted tools are never eligible. The weekly GitHub workflow evaluates its
configured low-risk tools and opens a reviewable automation pull request when the
index or table changes; it does not push directly to `main`. API and LLM tools
should be evaluated manually or in a separately configured workflow with the
necessary credentials and cost controls.

### MCP server

OpenTools exposes the same inspection and evaluation layer through an MCP server:

```bash
opentools-mcp --transport stdio
```

The server provides four MCP tools: `list_opentools`, `inspect_opentool`,
`evaluate_opentool`, and `call_opentool`. It accepts registered tool names only and returns sanitized
evidence without source code, credential values, absolute paths, or result-file
paths. Listing, inspection, and tool-card extraction use AST parsing and do not
import submitted tool modules. Test execution is disabled by default.

Restrict the visible toolbox and explicitly enable tests when starting a trusted,
local server:

```bash
export OPENTOOLS_MCP_ALLOWED_TOOLS="Calculator_Tool,Search_Engine_Tool"
export OPENTOOLS_MCP_ALLOW_EXECUTION=1
export OPENTOOLS_MCP_ALLOW_TOOL_CALLS=1
export OPENTOOLS_MCP_MAX_RISK=low
opentools-mcp --transport stdio
```

Test execution and application tool calls have separate enable flags. Both
require an explicit allowlist; only the requested module is then imported.
Restricted tools remain blocked, and caution tools require the explicit
`OPENTOOLS_MCP_MAX_RISK=caution` policy. A generic MCP client configuration is:

```json
{
  "mcpServers": {
    "opentools": {
      "command": "opentools-mcp",
      "args": ["--transport", "stdio"],
      "env": {
        "OPENTOOLS_MCP_ALLOWED_TOOLS": "Calculator_Tool",
        "OPENTOOLS_MCP_ALLOW_TOOL_CALLS": "1",
        "OPENTOOLS_MCP_MAX_RISK": "low"
      }
    }
  }
}
```

For local development, Streamable HTTP is also available through
`opentools-mcp --transport streamable-http`. Do not expose an unauthenticated
development server publicly.

A minimal read-only container deployment exposing only the calculator is
included:

```bash
docker compose -f docker-compose.mcp.yml up --build
```

The Streamable HTTP endpoint is available at `http://localhost:8000/mcp`.
Extend the image with a tool's dependencies before adding that tool to the
allowlist. Authentication and TLS must be added at the deployment boundary before
public hosting.

### Contribute and standardize a tool

Convert a README plus an annotated Python function into a reviewable OpenTools
bundle:

```bash
opentools convert-tool submitted.py \
  --readme README.md \
  --name "My Tool" \
  --entrypoint run_my_tool \
  --license Apache-2.0
```

The converter infers a JSON parameter schema from supported type annotations,
preserves the submitted source, generates a `BaseTool` wrapper, and performs
static risk inspection. It does **not** execute submitted code or report a
functional score. Unsupported or ambiguous entrypoints fail with an explicit
error rather than guessed behavior.

The optional contribution WebUI exposes the same flow:

```bash
pip install -e '.[webui]'
opentools-webui --host 127.0.0.1 --port 7860
```

Users upload `tool.py` and a README, review risk and metadata findings, and
download a contribution bundle. An optional LLM review evaluates sanitized
metadata and evidence only. Web submissions remain
`pending_maintainer_review`; they are never merged or executed automatically.

### Optional DSPy integration

```bash
pip install -e '.[dspy]'
```

```python
import dspy

from opentools.integrations.dspy import build_dspy_agent, optimize_dspy_agent

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

agent = build_dspy_agent(["Calculator_Tool"])

trainset = [
    dspy.Example(question="What is 2 + 3?", answer="5").with_inputs("question"),
    dspy.Example(question="What is 6 times 7?", answer="42").with_inputs("question"),
]

def exact_answer(example, prediction, trace=None):
    return str(example.answer).strip() == str(prediction.answer).strip()

optimized_agent = optimize_dspy_agent(
    agent,
    trainset=trainset,
    metric=exact_answer,
    batch_size=2,
    max_steps=8,
    max_demos=4,
)
```

Here DSPy optimizes the agent's tool-selection policy—its instructions and
few-shot demonstrations—not the implementation of Calculator. The lower-level
`as_callable` and `as_dspy_tool` helpers remain available when an existing DSPy
program only needs an OpenTools bridge. All paths reuse OpenTools schemas,
credential declarations, and risk policy. Restricted tools cannot be adapted,
and caution tools require `max_risk="caution"`.

An executable notebook covering static inspection, inventory generation,
conversion, MCP invocation, DSPy, the contribution WebUI, and the opt-in real
LLM judge is available at
[`docs/demo/4_evaluation_mcp_contribution.ipynb`](docs/demo/4_evaluation_mcp_contribution.ipynb).
The focused
[`docs/demo/5_dspy_agent_optimization.ipynb`](docs/demo/5_dspy_agent_optimization.ipynb)
shows baseline evaluation and opt-in SIMBA compilation without fabricating an
optimization result.

---

OpenTools can be used in two ways: **from the CLI** (direct command-line interface) or **inside a Python environment** (import and call from your code). Both modes use the same tools and agents.

### Using OpenTools in Python

**Tools** — load the registry, list tools, and run a tool:

```python
from opentools import load_all_tools, list_available_tools, create_tool

load_all_tools()
print(list_available_tools())

tool = create_tool("Calculator_Tool")
result = tool.run(operation="add", values=[1, 2, 3])
print(result)
```

**Agents** — use `UnifiedSolver` to run an agent:

```python
from opentools import UnifiedSolver

solver = UnifiedSolver(agent_name="opentools", llm_engine_name="gpt-4o-mini")
result = solver.solve(question="What is 2 + 2?")
print(result.get("direct_output") or result.get("final_output"))
```

### Using OpenTools from the CLI

Run tools and inspect metadata directly from the terminal:

```bash
opentools list
opentools info Calculator_Tool
opentools reload
```

--- 
## Documentation Structure

- **[`docs/`](./docs/)**: High-level guides and runnable examples for users and contributors.
  - **[`docs/demo/`](./docs/demo/)**: Jupyter notebooks that demonstrate how to run agents and tools  
    (e.g., [`2_agent_running.ipynb`](./docs/demo/2_agent_running.ipynb), [`3_agent_running_with_tool_retrieval.ipynb`](./docs/demo/3_agent_running_with_tool_retrieval.ipynb), [`1_tool_running.ipynb`](./docs/demo/1_tool_running.ipynb)).
  - **Top-level docs files** (e.g., [`2_AGENT_RUNNING.md`](./docs/2_AGENT_RUNNING.md), [`1_TOOL_RUNNING.md`](./docs/1_TOOL_RUNNING.md), [`3_CONTRIBUTION.md`](./docs/3_CONTRIBUTION.md)): walk through how the system works, how to run the demos, and how to contribute or extend OpenTools.
  - If you are unsure where to start, open the markdown files in [`docs/`](./docs/)—they contain step-by-step instructions and background explanations.


- **[`src/`](./src/)**: Source code and in-code documentation.
  - **[`src/opentools/`](./src/opentools/)**: Main Python package that powers everything in this repo.
    - **[`core/`](./src/opentools/core/)**: Config, base classes, and core orchestration logic.
    - **[`tools/`](./src/opentools/tools/)**: Built-in tools, each usually documented by a local `README.md`.
    - **[`models/`](./src/opentools/models/)**: Model and LLM engine integrations.
    - **[`agents/`](./src/opentools/agents/)**: Agent definitions and orchestrators (e.g., the unified solver).
    - **[`Benchmark/`](./src/opentools/Benchmark/)**: Benchmark runners and datasets (e.g., VQAv2) with their own `README.md` files and assets.
  - For deeper technical details, see the `README.md` files inside each subfolder and the inline docstrings/comments in the code.

## Contributing

For how to run things and contribute in practice, start with the docs:

- [`docs/3_CONTRIBUTION.md`](./docs/3_CONTRIBUTION.md): overview of contribution guidelines and how work typically flows in this repo.
- [`docs/demo/`](./docs/demo/) notebooks (e.g., [`2_agent_running.ipynb`](./docs/demo/2_agent_running.ipynb), [`1_TOOL_RUNNING.ipynb`](./docs/demo/1_TOOL_RUNNING.ipynb), [`3_agent_running_with_tool_retrieval.ipynb`](./docs/demo/3_agent_running_with_tool_retrieval.ipynb)): step-by-step, executable demos that show how the pieces fit together.

## License

Apache License 2.0. See [`LICENSE`](./LICENSE).

## Support

- Issues: https://github.com/hydang99/opentools/issues
