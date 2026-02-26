# source code: https://github.com/octotools/octotools/blob/main/octotools/tools/google_search/tool.py
import os, sys, requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Google_Search_Octotools_Tool(BaseTool):
    """Google_Search_Octotools_Tool
    ---------------------
    Purpose:
        A tool that performs Google searches using the Custom Search API based on a given text query. Provides detailed metadata about search operations, results, and error handling. Supports configurable number of results and Google's indexed web content.

    Core Capabilities:
        - Search Google using Custom Search API
        - Retrieve search results with titles, links, and snippets
        - Configure number of results returned
        - Access Google's indexed web content

    Intended Use:
        Use this tool when you need to search Google for information, including web content, news, and other relevant information.

    Limitations:
        - May not handle complex search queries or results
    """
    def __init__(self):
        super().__init__(
            type='function',
            name="Google_Search_Octotools_Tool",
            description="""A tool that performs Google searches using the Custom Search API based on a given text query. CAPABILITIES: Search Google using Custom Search API, retrieve search results with titles, links, and snippets, configure number of results returned, access Google's indexed web content. SYNONYMS: Google search, web search, internet search, Google Custom Search, search engine, web query tool, Google API search. EXAMPLES: 'Search Google for Python programming tutorials', 'Find information about machine learning with 5 results', 'Search for recent news about artificial intelligence'.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to be used for the Google search."
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "The number of search results to return (default: 10)."
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=False,
            category="web_search",
            tags=["google_search", "web_search", "search_engine", "custom_search", "internet_search", "web_query", "google_api", "search_api"],
            limitation="Requires GOOGLE_API_KEY and GOOGLE_CX environment variables, subject to Google Custom Search API quota limits, requires internet connection, API usage may incur costs, limited to results available in configured custom search engine.",
            agent_type="Search-Agent",
            demo_commands={
                "command": 'execution = tool.run(query="Python programming", num_results=10)',
                "description": "Perform a Google search for 'Python programming' and return 10 results."
            }
        )
        self.api_key = os.getenv("GOOGLE_API_KEY")  # NOTE: Replace with your own API key (Ref: https://developers.google.com/custom-search/v1/introduction)
        self.cx = os.getenv("GOOGLE_CX_ID")  # NOTE: Replace with your own custom search (Ref: https://programmablesearchengine.google.com/controlpanel/all)
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def google_search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """
        Performs a Google search using the provided query.

        Parameters:
            query (str): The search query.
            num_results (int): The number of search results to return.

        Returns:
            Dict[str, Any]: The raw search results from the Google API.
        """
        params = {
            'q': query,
            'key': self.api_key,
            'cx': self.cx,
            'num': num_results
        }
        
        response = requests.get(self.base_url, params=params)
        return response.json()

    def run(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """
        Executes a Google search based on the provided query.

        Parameters:
            query (str): The search query.
            num_results (int): The number of search results to return (default: 10).

        Returns:
            dict: A dictionary containing the query, search results, and success status.
        """
        if not self.api_key or not self.cx:
            return {
                "query": query,
                "result": [],
                "success": False,
                "error": "Google API key or CX is not set. Please set the GOOGLE_API_KEY and GOOGLE_CX environment variables."
            }

        try:
            results = self.google_search(query, num_results)
            
            if 'items' in results:
                search_results = [
                    {
                        "title": item['title'],
                        "link": item['link'],
                        "snippet": item['snippet']
                    }
                    for item in results['items']
                ]
                return {
                    "query": query,
                    "result": search_results,
                    "success": True
                }
            else:
                return {
                    "query": query,
                    "result": [],
                    "success": False,
                    "error": "No results found."
                }
        except Exception as e:
            return {
                "query": query,
                "result": [],
                "success": False,
                "error": f"An error occurred: {str(e)}"
            }

    def execute(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """Legacy method name for backward compatibility. Use run() instead."""
        return self.run(query, num_results)

    def get_metadata(self):
        """
        Returns the metadata for the Google_Search_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        return super().get_metadata()


if __name__ == "__main__":
    import json

    # Example usage of the Google_Search_Octotools
    tool = Google_Search_Octotools_Tool()

    # Execute the tool to perform a Google search
    query = "nobel prize winners in chemistry 2024"
    try:
        execution = tool.run(query=query, num_results=5)
        print(json.dumps(execution, indent=2))
    except Exception as e:
        print(f"Execution failed: {e}")