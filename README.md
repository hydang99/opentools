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
- Discussions: https://github.com/hydang99/opentools/discussions
