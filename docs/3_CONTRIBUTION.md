## Contributing tools, agents, and test cases

This guide explains how to add new tools and agents, and how to contribute test cases so behavior can be evaluated consistently.

### 1. Adding a new tool

1. **Create a tool module**

   Place your code under:

   ```text
   src/opentools/tools/<your_tool_name>/tool.py
   ```

   Export a class that inherits from `BaseTool`:

   ```python
   from opentools.core.base import BaseTool

   class My_Custom_Tool(BaseTool):
       def __init__(self):
           super().__init__(
               type="function",
               name="My_Custom_Tool",
               description="What this tool does.",
               category="my_category",
               tags=["tag1", "tag2"],
               parameters={
                   "type": "object",
                   "properties": {
                       "query": {"type": "string", "description": "User query"}
                   },
                   "required": ["query"],
               },
               agent_type="Search-Agent",   # or other agent type
               demo_commands={
                   "command": "result = tool.run(query='example')",
                   "description": "Short demo of the tool",
               },
               limitation="Any constraints on usage",
               accuracy=self.find_accuracy(
                   os.path.join(os.path.dirname(__file__), "test_result.json")
               ),
           )

       def run(self, query: str, **kwargs):
           # Implement the actual behavior here and return a dict:
           return {"result": f"Echo: {query}", "success": True}
   ```

2. **Make it discoverable**

   Tool discovery walks `src/opentools/tools/**/tool.py` and imports all `BaseTool` subclasses, so you don’t need to edit a registry. Just follow the folder and class pattern above.

3. **Optional: embeddings**

   To include your tool in FAISS‑based retrieval:

   ```python
   tool = My_Custom_Tool()
   tool.embed_tool()  # writes to agents/embeddings/tool_embeddings.json
   ```

### 2. Adding test cases

Shared test cases live in:

```text
src/opentools/tools/test_file/data.json
```

1. **Add a new section** keyed by your tool’s test name:

   ```json
   {
     "my_custom_tool": [
       {
         "id": 0,
         "query": "2 + 2",
         "answer": "4",
         "category": "my_custom_tool"
       },
       {
         "id": 1,
         "query": "5 - 3",
         "answer": "2",
         "category": "my_custom_tool"
       }
     ]
   }
   ```

2. **Implement or reuse a `test()` method**

   - Easiest: call `BaseTool.test(...)` from your tool:

     ```python
     def test(self, tool_test: str = "my_custom_tool"):
         return super().test(
             tool_test=tool_test,
             file_location="<your_tool_name>",
             result_parameter="result",
         )
     ```

   - Or follow patterns from tools like `Wolfram_Math_Tool.test()` if you need a custom judge.

3. **Generate `test_result.json`**

   Run your tool’s test entry point:

   ```bash
   python -m opentools.tools.<your_tool_name>.tool
   # or in Python
   from opentools.tools.<your_tool_name>.tool import My_Custom_Tool
   My_Custom_Tool().test()
   ```

   This should write `<your_tool_folder>/test_result.json` with per‑case results and `Final_Accuracy`, which is used in tool metadata.

3. **Read the full testing guide**

   For a detailed walkthrough of how the shared test file works, how to structure assets under `test_file/`, and how the evaluation metrics are computed, see the testing section in `src/opentools/tools/read_me.md` (the tools documentation in the `src` folder).

### 3. Guidelines

- **Return shape**  
  Tools should return a dict containing at least:

  - `result`: core payload
  - `success`: boolean
  - optional `error`, `error_type`, `metadata`

- **No secrets in the repo**  
  Do not hardcode API keys or secrets. Always read them from environment variables and document the required keys in your tool’s `limitation` field.

- **Keep test assets small**  
  When adding images / PDFs / other files under `tools/test_file/`, prefer small, representative examples to keep the repo lightweight.

### 4. Adding an agent

Agents live under `src/opentools/agents/<agent_name>/` and are registered in the agent manager. See `agents/read_me.md` for a detailed overview; the summary here is the minimal path.

1. **Create the agent module**

   ```text
   src/opentools/agents/my_agent/
     __init__.py
     agent.py
   ```

   In `agent.py`, implement either a simple `BaseAgent` or a tool‑based `ToolBasedAgent`:

   ```python
   from opentools.agents.base_agent import BaseAgent

   class MyAgent(BaseAgent):
       AGENT_NAME = "MyAgent"
       AGENT_DESCRIPTION = "Custom agent example"

       def get_agent_name(self):
           return self.AGENT_NAME

       def get_agent_description(self):
           return self.AGENT_DESCRIPTION

       def solve(self, question: str, **kwargs):
           return {
               "query": question,
               "agent": self.get_agent_name(),
               "llm_engine": self.llm_engine_name,
               "direct_output": f"Echo: {question}",
           }
   ```

   For tool‑using agents, inherit `ToolBasedAgent` and use `ToolCapabilityMixin` helpers (`get_relevant_tools`, `execute_tool`, etc.).

2. **Export from `__init__.py`**

   ```python
   from .agent import MyAgent

   __all__ = ["MyAgent"]
   ```

3. **Register the agent**

   Edit `src/opentools/agents/agent_manager.py` and register your class in `_register_default_agents()`:

   ```python
   from .my_agent import MyAgent

   def _register_default_agents(self):
       ...
       self.register_agent("my_agent", MyAgent)
   ```

   After this, `UnifiedSolver(agent_name="my_agent", ...)` will work.

4. **Document and test**

   - Add a short row for your agent to `agents/README.md` (what it does, how it reasons, whether it uses tools).
   - If the agent depends on tools, reuse the existing tool tests (via `BaseTool.test(...)`) and add higher‑level agent tests if needed (e.g. scripted calls that check expected shapes in `reasoning_trace` / `direct_output`).


