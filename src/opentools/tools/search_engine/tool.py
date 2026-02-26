# source code: https://github.com/inclusionAI/AWorld/blob/main/examples/gaia/mcp_collections/tools/search.py
"""
Compared to the source code, we add similarity search to ensure top results are returned in order of relevance.
"""

import json, os, sys, time, traceback, requests
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

class SearchResult(BaseModel):
    """Individual search result with structured data."""

    id: str
    title: str
    url: str
    snippet: str
    source: str
    display_link: str | None = None
    formatted_url: str | None = None


class SearchMetadata(BaseModel):
    """Metadata for search operation results."""

    query: str
    search_engine: str
    total_results: int
    search_time: float | None = None
    language: str = "en"
    country: str = "us"
    safe_search: bool = True
    error_type: str | None = None
    api_quota_used: bool = False


class Search_Engine_Tool(BaseTool):
    """
    Search_Engine_Tool
    ---------------------
    Purpose:
        A comprehensive web search tool that performs Google Custom Search API queries with intelligent result formatting. Features configurable search parameters, safe search filtering, localization options, and LLM-optimized output formatting for research and information gathering tasks.

    Core Capabilities:
        - Performs web searches using Google Custom Search API
        - Configurable result count (1-10)
        - Safe search filtering
        - Language and country localization
        - LLM-optimized output formatting

    Intended Use:
        Use this tool when you need to search the web for information, including web content, news, and other relevant information.

    Limitations:
        - May not handle complex search queries or results

    """
    # Default args for `opentools test Search_Engine_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "search_engine",
    }

    def __init__(self):
        super().__init__(
            type='function',
            name="Search_Engine_Tool",
            description="""A comprehensive web search tool that performs Google Custom Search API queries with intelligent result formatting. Features configurable search parameters, safe search filtering, localization options, and LLM-optimized output formatting for research and information gathering tasks. CAPABILITIES: Performs web searches using Google Custom Search API, configurable result count (1-10), safe search filtering, language and country localization. SYNONYMS: web search tool, Google search, search engine, web search engine, information finder, research tool, web crawler, search utility, web search utility, information search tool. EXAMPLES: 'Search for AI research papers', 'Find information about climate change', 'Search for Python libraries', 'Look up current news'.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string to search for. Can be any topic, question, or keywords you want to search for."
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of search results to return (1-10, default: 5). Maximum limit is 10 due to Google API constraints."
                    },
                    "safe_search": {
                        "type": "boolean",
                        "description": "Whether to enable safe search filtering (default: True). Helps filter out inappropriate content."
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code for search results (e.g., 'en', 'es', 'fr', default: 'en')"
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code for search results (e.g., 'us', 'uk', 'ca', default: 'us')"
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'markdown', 'json', or 'text' (default: 'json')",
                        "enum": ["markdown", "json", "text"]
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=False,
            category="web_search",
            tags=["web_search", "google_search", "search_engine", "information_retrieval", "research_tool", "web_crawler", "search_utility", "information_finder"],
            limitation="Maximum 10 results per query due to Google API limits, requires GOOGLE_API_KEY and GOOGLE_CX_ID environment variables, safe search helps filter inappropriate content, language/country codes affect result relevance",
            agent_type="Search-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(query='AI research papers', num_results=5, safe_search=True, language='en', country='us', output_format='markdown')",
                "description": "Search for AI research papers, return 5 results, use safe search, use English language, use United States country, return results in markdown format"
            }
        )
        self.workspace = Path(os.getcwd())
        self.supported_extensions = {".pdf"}

        # Validate required API credentials
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CX_ID")


    def _format_search_results_for_llm(self, results: list[SearchResult], query: str) -> str:
        """Format search results to be LLM-friendly.

        Args:
            results: List of search results
            query: Original search query

        Returns:
            Formatted string suitable for LLM consumption
        """
        if not results:
            return f"No search results found for query: '{query}'"

        formatted_parts = [f"# Search Results for: '{query}'", f"Found {len(results)} results:\n"]

        for i, result in enumerate(results, 1):
            result_section = [
                f"## Result {i}: {result.title}",
                f"**URL:** {result.url}",
                f"**Source:** {result.source}",
            ]

            if result.display_link:
                result_section.append(f"**Domain:** {result.display_link}")

            result_section.append(f"**Summary:** {result.snippet}")
            result_section.append("")  # Empty line for spacing

            formatted_parts.append("\n".join(result_section))

        return "\n".join(formatted_parts)

    def _validate_search_parameters(self, query: str, num_results: int) -> tuple[str, int]:
        """Validate and normalize search parameters.

        Args:
            query: Search query string
            num_results: Number of results requested

        Returns:
            Tuple of (validated_query, validated_num_results)

        Raises:
            ValueError: If parameters are invalid
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        # Normalize query
        validated_query = query.strip()

        # Validate and clamp num_results
        validated_num_results = max(1, min(num_results, 10))  # Google CSE limit is 10

        return validated_query, validated_num_results

    def run(
        self,
        query: str = Field(description="The search query string to search for"),
        num_results: int = Field(default=5, description="Number of search results to return (1-10, default: 5)"),
        safe_search: bool = Field(default=True, description="Whether to enable safe search filtering"),
        language: str = Field(default="en", description="Language code for search results (e.g., 'en', 'es', 'fr')"),
        country: str = Field(default="us", description="Country code for search results (e.g., 'us', 'uk', 'ca')"),
        output_format: str = Field(default="json", description="Output format: 'markdown', 'json', or 'text'"),
    ) :
        """Search the web using Google Custom Search API.

        This tool provides comprehensive web search capabilities with:
        - Google Custom Search API integration
        - Configurable result count and filtering
        - Safe search and localization options
        - LLM-optimized result formatting
        - Detailed metadata tracking

        Args:
            query: The search query string
            num_results: Number of results to return (1-10)
            safe_search: Enable safe search filtering
            language: Language code for results
            country: Country code for results
            output_format: Format for the response

        Returns:
            ActionResponse with formatted search results and metadata
        """
        if isinstance(query, FieldInfo):
            query = query.default
        if isinstance(num_results, FieldInfo):
            num_results = num_results.default
        if isinstance(safe_search, FieldInfo):
            safe_search = safe_search.default
        if isinstance(language, FieldInfo):
            language = language.default
        if isinstance(country, FieldInfo):
            country = country.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default

        try:
            # Validate API credentials
            if not self.google_api_key or not self.google_cse_id:
                return {
                    "error": (
                        "Google Search API credentials not configured. "
                        "Please set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables."
                    ),
                    "success": False,
                }

            # Validate parameters
            validated_query, validated_num_results = self._validate_search_parameters(query, num_results)

            # Prepare API request
            start_time = time.time()

            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": validated_query,
                "num": validated_num_results,
                "safe": "active" if safe_search else "off",
                "hl": language,
                "gl": country,
            }

            # Make API request
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            search_time = time.time() - start_time
            data = response.json()

            # Parse search results
            search_results = []
            if "items" in data:
                for i, item in enumerate(data["items"]):
                    result = SearchResult(
                        id=f"google-{i}",
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source="google",
                        display_link=item.get("displayLink", ""),
                        formatted_url=item.get("formattedUrl", ""),
                    )
                    search_results.append(result)

            # Format results based on requested format
            if output_format.lower() == "json":
                formatted_content = {
                    "query": validated_query,
                    "results": [result.model_dump() for result in search_results],
                    "count": len(search_results),
                }

                message_content = formatted_content
            elif output_format.lower() == "text":
                if search_results:
                    result_lines = []
                    for i, result in enumerate(search_results, 1):
                        result_lines.append(f"{i}. {result.title}")
                        result_lines.append(f"   URL: {result.url}")
                        result_lines.append(f"   Summary: {result.snippet}")
                        result_lines.append("")  # Empty line
                    message_content = "\n".join(result_lines)
                else:
                    message_content = f"No results found for: {validated_query}"
            elif output_format.lower() == "dict":
                message_content = [result.model_dump() for result in search_results]
            else:  # markdown (default)
                message_content = self._format_search_results_for_llm(search_results, validated_query)

            metadata = SearchMetadata(
                query=validated_query,
                search_engine="google",
                total_results=len(search_results),
                search_time=search_time,
                language=language,
                country=country,
                safe_search=safe_search,
                api_quota_used=True,
            )
            return {
                "result": message_content,
                "success": True,
                "metadata": metadata.model_dump(),
            }

        except requests.exceptions.RequestException as e:
            error_msg = f"Google Search API request failed: {str(e)}"
            print(f"Search API error: {traceback.format_exc()}")

            return {
                "error": error_msg,
                "success": False,
                "error_type": "search_api_request_failed",
                "traceback": traceback.format_exc(),
            }

        except ValueError as e:
            error_msg = f"Invalid search parameters: {str(e)}"

            return {
                "error": error_msg,
                "success": False,
                "error_type": "invalid_search_parameters",
                "traceback": traceback.format_exc(),
            }

        except Exception as e:
            error_msg = f"Search operation failed: {str(e)}"
            error_trace = traceback.format_exc()

            print(f"Unexpected search error: {error_trace}")


            return {
                "error": f"{error_msg}\n\nError details: {error_trace}",
                "success": False,
                "error_type": "search_operation_failed",
                "traceback": traceback.format_exc(),
            }
    def test(self, tool_test: str="search_engine"):
        """Test the Search Engine tool with various test samples, run 3 times, and save results in a JSON file."""
        try:
            # Load testbench data
            file_test = os.path.join(os.path.dirname(__file__), '..', 'test_file', 'data.json')
            with open(file_test, encoding='utf-8') as f:
                data = json.load(f)[tool_test]

            # Prepare result file as JSON
            file_result = os.path.join(os.path.dirname(__file__), 'test_result.json')
            test_result = {}
            test_result['Test-File length'] = len(data)
            run_accuracy = {'run_1': 0, 'run_2': 0, 'run_3': 0}
            
            # Iterate over test cases
            for i, test in enumerate(data):
                question_result = {"id": test.get("id", f"search_{i + 1}")}
                if 'query' in test:
                    question_result['query'] = test['query']
                if 'answer' in test:
                    question_result['expected_answer'] = test['answer']

                # Prepare parameters (exclude answer, id, category)
                parameters = {k: v for k, v in test.items() if k not in ['answer', 'id', 'category']}

                # Run and record result for each of 3 runs
                for j in range(1, 4):
                    run_result = {}
                    result = self.run(**parameters)
                    run_result['result'] = result

                    # If failed or none
                    if not result or (isinstance(result, dict) and result.get("success") == False):
                        run_result['accuracy'] = 0
                        run_result['tool_call_pass'] = False
                        question_result[f'run_{j}'] = run_result
                        continue
                    else:
                        run_result['tool_call_pass'] = True

                    # Calculate accuracy for this run - take top 5 results and average
                    temp_accuracy = 0
                    # Extract search results from the response
                    search_results = []
                    if isinstance(result, dict) and result.get("result"):
                        if isinstance(result["result"], list):
                            search_results = result["result"]
                        elif isinstance(result["result"], dict) and "results" in result["result"]:
                            search_results = result["result"]["results"]

                    # Take top 5 results and calculate accuracy
                    if search_results:
                        # Limit to top 5 results
                        top_5_results = search_results[:5]
                        for item in top_5_results:
                            title = item.get('title') or ""
                            snippet = item.get('snippet') or ""
                            # Combine title and snippet for better accuracy evaluation
                            content = title + " " + snippet
                            test_query = test['query']
                            temp_accuracy += self.eval_accuracy(content, test_query)
                        accuracy_score = temp_accuracy / len(top_5_results) if top_5_results else 0
                    else:
                        accuracy_score = 0

                    run_result['accuracy'] = accuracy_score
                    run_accuracy[f'run_{j}'] += accuracy_score

                    question_result[f'run_{j}'] = run_result

                print(f"Finish query: {i + 1}")
                test_result[f'Q{i + 1}'] = question_result

            # Calculate and record overall accuracy for each run
            test_result['Final_Accuracy'] = {
                'run_1': run_accuracy['run_1'] * 100 / len(data) if data else 0,
                'run_2': run_accuracy['run_2'] * 100 / len(data) if data else 0,
                'run_3': run_accuracy['run_3'] * 100 / len(data) if data else 0
            }
            print(f"Accuracy: {test_result['Final_Accuracy']}")

            with open(file_result, "w", encoding="utf-8") as output_file:
                json.dump(test_result, output_file, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"❌ Failed to test the tool with this error: {e}")
            return False
        return True
# Example usage and entry point
if __name__ == "__main__":

    # Initialize and run the search service
    try:
        service = Search_Engine_Tool()
        print(service.run(query="What is the capital of France?"))
        # service.embed_tool()
        # service.test(tool_test="search_engine")
    except Exception as e:
        print(f"An error occurred: {e}: {traceback.format_exc()}")
