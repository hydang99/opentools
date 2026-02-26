"""
Tool capability mixin that can be composed into any agent to enable tool usage.

Provides initialization for tool discovery/execution and helper methods to
normalize tool calls and run tools. Agents opt-in by calling
initialize_tool_capabilities(...), typically gated by an enable_tool_calls flag.
"""
from typing import Any, Dict, List, Tuple, Union, Optional
import json, os, sys
from datetime import datetime
import traceback
from pydantic import BaseModel, Field
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.tool_retrieval import ToolRetriever

class DecisionRetrieval(BaseModel):
    """Pydantic model for LLM retrieval decision: whether to run retrieval and the query to use."""
    retrieve: bool = Field(description="True if retrieval is needed, False otherwise.")
    query: str = Field(description="A short phrase describing what to retrieve when retrieve is True.")
class ToolCapabilityMixin:
    """Reusable tooling capabilities for agents.

    This mixin provides methods for initializing tool discovery/execution,
    normalizing tool calls, and running tools within agents. 

    Note: This mixin also includes logic to handle retrieval cases when a tool's result is too large (e.g., >30,000 tokens).
    If the tool's output is too long, an automatic retrieval step can extract only the relevant information for the user query.
    """

    def initialize_tool_capabilities(
        self,
        enabled_tools: Optional[List[str]] = None,
        num_threads: int = 1,
        max_time: int = 120,
        max_output_length: int = 100000,
        vllm_config_path: Optional[str] = None,
        enable_faiss_retrieval: bool = False,
        model_family: str = "openai",
    ) -> None:
        """Set up tool discovery, executor, and optional FAISS retrieval for this agent."""
        from ...models.initializer import Initializer
        from ...models.executor import Executor
        from ...models.memory import Memory

        self.enabled_tools = enabled_tools or ["all"]
        self.num_threads = num_threads
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.vllm_config_path = vllm_config_path
        self.model_family = model_family
        if getattr(self, "verbose", False):
            self.log("Initializing tool capabilities...")
        self.initializer = Initializer(
            enabled_tools=self.enabled_tools,
            model_string=self.llm_engine_name,
            verbose=self.verbose,
            vllm_config_path=self.vllm_config_path,
        )
        self.log(f"Available tools that is successfully loaded 🔧: {self.get_available_tools()}", "INFO")

        self.executor = Executor(
            llm_engine_name=self.llm_engine_name,
            llm_engine=self.llm_engine,
            num_threads=self.num_threads,
            max_time=self.max_time,
            max_output_length=self.max_output_length,
            verbose=self.verbose,
        )
        self.agent_description = {
            "Visual-Agent": "A visual agent that can extraction information/text from image and answer simple questions about the image.",
            "Browser_Extraction-Agent": "An agent specialized in extracting information or text directly from web pages or URLs. It works best when used after the Search-Agent to identify relevant URLs, then Page_Extraction-Agent retrieves the desired content from those pages.",
            "General-Agent": "A general agent that can answer questions about any topic.",
            "File_Extraction-Agent": "A file extraction agent that can extract information/text from any type of files. Only local files are supported. Work best with Download_File-Agent to download the file first.",
            "Media-Agent": "A media agent that can extract information from media files like images, videos, audio.",
            "Search-Agent": "A search agent that can search for credibility source of information includes google search, wikipedia search, chemistry search, paper search, news search, etc. It can just search for the result, it can not also extract the information from the search result.",
            "Download_File-Agent": "A download file agent that can download files from the web.",
            "Puzzle-Agent": "A puzzle agent that can solve puzzles like maze, rubik, n_queens, etc. It can not solve math problem just puzzle problems.",
            "Mathematics-Agent": "A mathematics agent that can solve mathematics problems like algebra, geometry, calculus, etc or just simple arithmetic calculations. It can not solve puzzle problems but just pure mathematics problems.",
         }

        if getattr(self, "verbose", False):
            self.log("Tool capabilities initialized successfully")

        self.enable_faiss_retrieval = enable_faiss_retrieval
        try:
            if self.enable_faiss_retrieval:
                self.tool_retriever = ToolRetriever(top_k=3, llm_engine=self.llm_engine)
                if self.verbose:
                    self.log("FAISS tool retrieval enabled")
            else:
                self.tool_retriever = None
                if self.verbose:
                    self.log("FAISS tool retrieval disabled - using all available tools")
        except Exception as e:
            print("Error initializing FAISS tool retrieval:", e)
            self.tool_retriever = None
            self.enable_faiss_retrieval = False
            if self.verbose:
                self.log("FAISS tool retrieval disabled")       

    def get_available_tools(self) -> List[str]:
        """Return the list of tool names currently loaded and available to the agent."""
        return getattr(self, "initializer", None).available_tools if getattr(self, "initializer", None) else []
    
    def get_agent_type(self) -> str:
        """Return a dict mapping each agent type to its description and list of tools."""
        available_tools = self.get_available_tools()
        agent = {}
        metadata = self.get_toolbox_metadata()
        for tool in available_tools:
            agent_type = metadata[tool].get("agent_type")
            if agent_type not in agent:
                agent[agent_type] = {"tools": []}
            agent[agent_type]["tools"].append(tool)
        if "Generalist_Solution_Generator_Tool" in metadata.keys():
            if "Puzzle-Agent" in agent.keys():
                agent["Puzzle-Agent"]["tools"].append("Generalist_Solution_Generator_Tool")
            if "File_Extraction-Agent" in agent.keys():
                agent["File_Extraction-Agent"]["tools"].append("Generalist_Solution_Generator_Tool")

        return agent
    
    def get_tools_by_agent(self, agent_type):
        """Return tool definitions (name, description, parameters, etc.) for the given agent type."""
        available_tools = self.get_available_tools()
        metadata = self.get_toolbox_metadata()
        tools_list = []
        for tool in available_tools:
            if metadata[tool].get("agent_type") == agent_type:
                tools_list.append({
                    "type": metadata[tool].get("type"),
                    "name": metadata[tool].get("name"),
                    "description": metadata[tool].get("description"),
                    "parameters": metadata[tool].get("parameters"),
                    "strict": metadata[tool].get("strict"),
                })

        # Ensure Generic reasoning agents always have access to the generalist tool
        needs_generalist = (
            "Puzzle-Agent" in agent_type
            or "File_Extraction-Agent" in agent_type
            or "Mathematics-Agent" in agent_type
            or "General-Agent" in agent_type
        )
        if needs_generalist and "Generalist_Solution_Generator_Tool" in metadata.keys():
            gen_metadata = metadata["Generalist_Solution_Generator_Tool"]
            tools_list.append({
                "type": gen_metadata.get("type"),
                "name": gen_metadata.get("name"),
                "description": gen_metadata.get("description"),
                "parameters": gen_metadata.get("parameters"),
                "strict": gen_metadata.get("strict"),
            })
        return tools_list

    
    def get_relevant_tools(self, question: str) -> List[str]:
        """Return tools relevant to the question (FAISS-retrieved or all available) as metadata list or JSON string."""
        available_tools = self.get_available_tools()
        try:
            if self.enable_faiss_retrieval and self.tool_retriever is not None:
                # Use FAISS retrieval to get relevant tools
                self.tool_retriever.set_toolbox_metadata(self.get_toolbox_metadata())
                retrieved_tools = self.tool_retriever.get_tool_names(question)
                relevant_tools = [tool for tool in retrieved_tools if tool in available_tools]
                # If no relevant tools found, fall back to a all available tools
                if not relevant_tools:
                    relevant_tools = available_tools if available_tools else []
                self.log(f"Using FAISS retrieval to get relevant tools: {relevant_tools}", "INFO")
            else:
                self.log(f"Using all available tools: {available_tools}", "INFO")
                relevant_tools = available_tools
        except Exception as e:
            print("Error using FAISS tool retrieval:", e)
            relevant_tools = available_tools

        # Build metadata for only relevant tools
        metadata = self.get_toolbox_metadata() or {}
        if self.model_family == "openai":
            tools_list = []
            for tool in relevant_tools:
                if tool in metadata:
                    tools_list.append({
                        "type": metadata[tool].get("type"),
                        "name": metadata[tool].get("name"),
                        "description": metadata[tool].get("description"),
                        "parameters": metadata[tool].get("parameters"),
                        "strict": metadata[tool].get("strict"),
                    })
            return tools_list
        else:
            tools_list = {}
            for tool in relevant_tools:
                if tool in metadata:
                    tools_list[tool] = {
                        "type": metadata[tool].get("type"),
                        "name": metadata[tool].get("name"),
                        "description": metadata[tool].get("description"),
                        "parameters": metadata[tool].get("parameters"),
                    }
        try:
            return json.dumps(tools_list, ensure_ascii=False, indent=2)
        except Exception:
            return str(tools_list)

    def get_toolbox_metadata(self) -> Dict[str, Any]:
        """Return the full toolbox metadata dict (tool name -> metadata) from the initializer."""
        return getattr(self, "initializer", None).toolbox_metadata if getattr(self, "initializer", None) else {}

    def _parse_args(self, args: Union[str, Dict[str, Any], None]) -> Dict[str, Any]:
        """Parse tool arguments from a string (JSON), dict, or None into a single dict."""
        if args is None:
            return {}
        if isinstance(args, dict):
            return args
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
            return {"_": parsed}
        except Exception:
            return {"_raw": str(args)}

    def _normalize_tool_call(self, tc: Any) -> Tuple[str, Dict[str, Any]]:
        """Normalize a tool call (dict or JSON string) into (tool_name, arguments) tuple."""
        if isinstance(tc, str):
            tc = json.loads(tc)

        if not isinstance(tc, dict):
            raise ValueError("Tool call must be a dict or JSON string")

        if tc.get("type") == "function" and isinstance(tc.get("function"), dict):
            name = tc["function"].get("name")
            args = self._parse_args(tc["function"].get("arguments"))
            if not name:
                raise ValueError("Missing function.name")
            return name, args

        if tc.get("type") == "function" and "name" in tc:
            name = tc["name"]
            args = self._parse_args(tc.get("arguments"))
            # Always return the tool name as a string; downstream code expects a hashable key
            return name, args

        if "name" in tc:
            name = tc["name"]
            args = self._parse_args(tc.get("arguments"))
            # Always return the tool name as a string; downstream code expects a hashable key
            return name, args

        raise ValueError("Unrecognized tool call shape")

    def execute_tool(self, tool_calls: Any, question: str) -> Any:
        """Run a list of tool calls; optionally run retrieval on large results. Returns (executions, fail_executions)."""
        executions: List[Dict[str, Any]] = []
        fail_executions: List[Dict[str, Any]] = []
        for i, tool_call in enumerate(tool_calls):
            result = None
            try:
                name, args = self._normalize_tool_call(tool_call)
                result = self.executor.execute_tool_command(name, args)
                if (result.get("success") == False):
                    error = result.get("error") or result.get("message") or result.get("result") or result
                    executions.append({
                        "index": i,
                        "error": error,
                        "traceback": result.get("traceback"),
                        "raw": tool_call,
                        "ok": False,
                    })
                    fail_executions.append({
                        "index": i,
                        "error": error,
                        "traceback": result.get("traceback"),
                        "raw": tool_call,
                        "ok": False,
                    })
                    continue
                try:
                    if result.get("success") and question and self.count_tokens(result) > 30000:
                        # Check if we need retrieval for massive data
                        retrieval_decision = self.decide_retrieval_need(question=question)
                        if retrieval_decision.retrieve == True:
                            # Use retrieval to get specific information
                            retrieval_query = retrieval_decision.get("query")
                            if self.verbose:
                                self.log(f"Retrieval needed")
                            from .file_retrieval import TextHybridRetriever
                            response = result.get("result") if "result" in result else ", ".join(f"{key}: {value}" for key, value in result.items())
                            file_retriever = TextHybridRetriever(response)
                            result["result"] = file_retriever.retrieve(retrieval_query)
                        else:
                            if self.verbose:
                                self.log("No retrieval needed")
                except Exception as e:
                    if self.verbose:
                        self.log(f"Error in retrieval: {e}")
                    result = result
                executions.append({
                    "index": i,
                    "name": name,
                    "args": args,
                    "execution_result": result,
                    "ok": True,
                })
            except Exception as e:
                executions.append({
                    "index": i,
                    "error": f"{type(e).__name__}: {e}",
                    "result": result,
                    "traceback": traceback.format_exc(),
                    "raw": tool_call,
                    "ok": False,
                })
        return executions, fail_executions

    def generate_tool_command(
        self,
        question: str,
        image_path: Optional[str],
        context: str,
        sub_goal: str,
        tool_name: str,
    ) -> Any:
        """Ask the LLM to generate a single tool call (name + arguments) for the given question and tool."""
        if not getattr(self, "initializer", None):
            raise RuntimeError("Initializer not available. Ensure tool capabilities are initialized.")
        tool_metadata = self.initializer.toolbox_metadata.get(tool_name, {})
        return self.executor.generate_tool_command(
            question, image_path, context, sub_goal, tool_name, tool_metadata
        )

    def generate_and_execute_tool(
        self,
        question: str,
        image_path: Optional[str],
        context: str,
        sub_goal: str,
        tool_name: str,
    ) -> Any:
        """Generate a tool call for the given tool and question, then execute it and return the result."""
        if not getattr(self, "initializer", None):
            raise RuntimeError("Initializer not available. Ensure tool capabilities are initialized.")
        tool_metadata = self.initializer.toolbox_metadata.get(tool_name, {})
        return self.executor.generate_and_execute_tool(
            question, image_path or "", context, sub_goal, tool_name, tool_metadata
        )

    def count_tokens(self, metadata):
        """Estimate token count for the given metadata (dict) using tiktoken or a fallback."""
        try:
            import tiktoken
            metadata_str = json.dumps(metadata, ensure_ascii=False)
            encoding = tiktoken.get_encoding("cl100k_base")
            token_count = len(encoding.encode(metadata_str))
            return token_count            
        except Exception as e:
            metadata_str = json.dumps(metadata, ensure_ascii=False)
            return len(metadata_str) // 4
    class RetrievalDecision(BaseModel):
        """Structured response: whether to run retrieval on large tool output and which query to use."""
        retrieve: bool
        query: str

    def decide_retrieval_need(self, question: str):
        """
        Decide whether retrieval is needed for massive data and what to retrieve.
        
        Args:
            result: Tool execution result containing metadata and data
            question: Original user question
            
        Returns:
            Dictionary with 'retrieve' (bool) and 'query' (str) fields
        """
        try:
            # Create prompt for LLM to decide on retrieval
            retrieval_prompt = self._build_retrieval_decision_prompt(
                question
            )
            
            # Use LLM to decide on retrieval
            if hasattr(self, 'llm_engine') and self.llm_engine:
                response = self.llm_engine.generate(retrieval_prompt, response_format=self.RetrievalDecision)
                return response

            else:
                return {"retrieve": False, "query": ""}
                
        except Exception as e:
            if self.verbose:
                self.log(f"Error in retrieval decision: {e}")
            return {"retrieve": False, "query": ""}

    def _build_retrieval_decision_prompt(self, question: str) -> str:
        """
        Build a prompt to help the LLM decide whether retrieval is needed.
        
        Args:
            question: Original user question
            trace: Current trace
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are a helpful assistant that can anaylyze the question and decide whether retrieval is needed to answer the user's question.
The response of the question is too large for the model to handle, so we need you to decide whether retrieval is needed to answer the user's question.
CONTEXT:
- Original Question: {question}

TASK:
Decide:
- Should we use retrieval on the tool result?
- If yes, what short query should we search for?

RETRIEVAL GUIDELINES:
- Use retrieval ONLY when the user needs specific information that could be extracted.
- Don't use retrieval for general questions that can only be answered with the full content or summarization-only questions. For example, "What is the main content of the file?" or "What is the main conclusion?"
- Focus on extracting the most relevant piece of information for the user's question rather than the full content.

FIELD MEANINGS:
- `retrieve`: true if retrieval on the tool result would help answer the question; false otherwise.
- `query`: a short phrase describing what to retrieve when `retrieve` is true.
  - If `retrieve` is false, set `query` to an empty string "".

EXAMPLES:
- For "What is the main conclusion?": {{"retrieve": false, "query": ""}}
- For "Summarize this document": {{"retrieve": false, "query": ""}}
- For "What is the type of the experiment?" {{"retrieve": true, "query": "experiment type"}}
- For "What is the formula for linear regression?" {{"retrieve": true, "query": "linear regression formula"}}
"""
        return prompt
    #  High-level helpers for prompts and parsing (for zero_shot and chain_of_thought agent)
    def build_tools_metadata_block(self) -> str:
        """Return a compact JSON string of available tools' metadata for inclusion in LLM prompts."""
        metadata = self.get_toolbox_metadata() or {}
        # Keep only the most relevant fields for prompting
        concise: Dict[str, Any] = {}
        for tool_cls_name, md in metadata.items():
            concise[tool_cls_name] = {
                "tool_name": md.get("tool_name"),
                "description": md.get("tool_description"),
                "parameters": md.get("parameters"),
                "input_types": md.get("input_types"),
                "output_type": md.get("output_type"),
                "require_llm_engine": md.get("require_llm_engine", False),
                "function_map": md.get("function_map", {}),
                "demo_commands": md.get("demo_commands", []),
            }
        try:
            return json.dumps(concise, ensure_ascii=False, indent=2)
        except Exception:
            return str(concise)

    def build_tool_instruction(self) -> str:
        """Return the standard instruction text telling the LLM how to format tool calls (JSON array)."""
        return (
            "You may call tools if they will improve the answer. "
            "If calling tools, return ONLY a JSON array of objects with the format:\n"
            "[\n  {\n    'type': 'function',\n    'name': '<ToolClassName>',\n    'arguments': { ... }\n  },\n  {\n    'type': 'function',\n    'name': '<ToolClassName>',\n    'arguments': { ... }\n  }\n]\n"
            "Use exact tool class names from the metadata. If not calling tools, respond with final answer text."
        )

    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Parse the model response (string or object) into a list of normalized tool-call dicts."""
        parsed = response
        if isinstance(response, str):
            try:
                parsed = json.loads(response)
            except Exception:
                return []

        tool_calls: List[Dict[str, Any]] = []
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict) and (parsed.get('type') == 'function' or parsed.get('name')):
            candidates = [parsed]
        else:
            candidates = []

        for tc in candidates:
            try:
                name, args = self._normalize_tool_call(tc)
                tool_calls.append({"type": "function", "name": name, "arguments": args})
            except Exception:
                continue
        return tool_calls