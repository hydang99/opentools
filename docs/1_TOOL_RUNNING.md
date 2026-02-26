## Running tools directly (without agents)

OpenTools tools live under `src/opentools/tools/<tool_name>/tool.py` and are implemented as `BaseTool` subclasses with a `run(...)` method.

You can work with tools in two ways:
- **From Python**: import the tool class and call `.run(...)` (examples below).
- **From the CLI**: use the `opentools` command to list, inspect, run, and test tools.

### Quick pattern (Python)

```python
from opentools.tools.wolfram_math.tool import Wolfram_Math_Tool

tool = Wolfram_Math_Tool()
response = tool.run(query="What is the derivative of x^2 + 3*x?")
print(response)
```

Most tools follow the same pattern:

- Construct the tool class (no arguments in `__init__`).
- Call `.run(...)` with named parameters that match the tool’s JSON schema.
- Get back a **dict** with at least:
  - `result`: tool‑specific payload (string / list / dict)
  - `success`: `True` / `False`
  - optionally `error`, `error_type`, `metadata`, etc.

### Examples (Python)

#### `Wolfram_Math_Tool`

```python
from opentools.tools.wolfram_math.tool import Wolfram_Math_Tool

wolfram = Wolfram_Math_Tool()
res = wolfram.run(query="integrate x^2 from 0 to 3")
print(res["result"])
```

#### `Search_Engine_Tool`

```python
from opentools.tools.search_engine.tool import Search_Engine_Tool

search = Search_Engine_Tool()
res = search.run(
    query="Pan Lu multimodal reasoning thought chains",
    num_results=3,
    output_format="markdown",
)
print(res["result"])
```

#### `Wiki_Search_Tool`

```python
from opentools.tools.wiki_search.tool import Wiki_Search_Tool

wiki = Wiki_Search_Tool()
res = wiki.run(
    operation="search",
    query="OctoTools (agentic framework)",
    limit=3,
    output_format="markdown",
)
print(res["result"])
```

### Running tools from the CLI

When OpenTools is installed, the `opentools` CLI lets you discover and run tools without writing Python:

- **List all tools**

  ```bash
  opentools list
  opentools list --json   # json-readable
  ```

- **Inspect a tool’s metadata**

  ```bash
  opentools info Wolfram_Math_Tool
  opentools info Wolfram_Math_Tool --json
  ```

- **Search for tools by name/description**

  ```bash
  opentools search tools "math"
  opentools search tools "search engine" --json
  ```

- **Run a tool with JSON arguments**

  ```bash
  opentools run Wolfram_Math_Tool --args '{"query": "integrate x^2 from 0 to 3"}'
  opentools run Search_Engine_Tool --args '{"query": "reasoning and ReAct reasoning", "num_results": 3}'
  ```

- **Reload all tools (e.g., after editing/adding tools)**

  ```bash
  opentools reload
  ```

- **Run a tool’s test routine (if implemented)**

  ```bash
  opentools test Calculator_Tool
  ```

### Arguments and pitfalls

- **JSON‑schema parameters**  
  Each tool defines its input schema in `parameters` inside `__init__`. Use **named arguments** that match those property names; unexpected arguments will be rejected when `strict=True`.

- **API keys**  
  Some tools require external APIs:
  - `Wolfram_Math_Tool` → `WOLFRAM_API_KEY`
  - `Search_Engine_Tool` → `GOOGLE_API_KEY`, `GOOGLE_CX_ID`
  - others as documented in their `limitation` field  
  If a key is missing, `run(...)` usually returns `{"success": False, "error": "...", "error_type": ...}`.

- **Result size**  
  Web/search tools can return large payloads. For debugging, you may want to print only a slice, e.g. `res["result"][:2]` or summarize.

### Built‑in testing hooks

Most tools can be tested against the shared test set under `src/opentools/tools/test_file/`:

- **Generic `BaseTool.test(...)`**  
  For tools that don’t override `test`, you can call:

  ```python
  tool.test(
      tool_test="your_tool_key_in_data_json",
      file_location="your_tool_folder_name",
      result_parameter="result",     # which key in the tool result to score
      search_type=None,             # optional, tool‑specific
      count_token=False,
  )
  ```

  This reads cases from `tools/test_file/data.json`, runs the tool over them, and writes a `temp_result.json` plus accuracy.

- **Custom `test()` implementations**  
  Some tools (e.g. `Wolfram_Math_Tool`) override `test()` to support LLM‑based judgment. See the tool’s `test` docstring for details.

### Tool embeddings for FAISS retrieval

Tool‑based agents can select tools semantically using a FAISS index built from tool metadata. To add your tool to that index:

```python
from opentools.tools.wolfram_math.tool import Wolfram_Math_Tool

tool = Wolfram_Math_Tool()
tool.embed_tool()  # writes embeddings to agents/embeddings/tool_embeddings.json
```

Once embedded, agents with `enable_faiss_retrieval=True` can discover and rank your tool automatically.

