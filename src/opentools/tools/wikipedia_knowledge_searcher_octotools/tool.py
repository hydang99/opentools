# source code: https://github.com/octotools/octotools/blob/main/octotools/tools/wikipedia_knowledge_searcher/tool.py
import os, sys, wikipedia
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool

class Wikipedia_Knowledge_Searcher_Octotools_Tool(BaseTool):
    """Wikipedia_Knowledge_Searcher_Octotools_Tool
    ---------------------
    Purpose:
        A tool that searches Wikipedia and returns article content based on a given query.

    Core Capabilities:
        - Search Wikipedia database for articles
        - Retrieve article content and text
        - Handle disambiguation pages
        - Extract full or truncated article text
        - Format search results with extracted content

    Intended Use:
        Use this tool when you need to search Wikipedia for articles, retrieve article content, and handle disambiguation pages.

    Limitations:
        - Depends on Wikipedia's availability and rate limits
        - Some articles may not have content
        - Auto-suggest may not always find exact matches
        - Language support depends on Wikipedia's available languages

    """
    def __init__(self):
        super().__init__(
            type='function',
            name="Wikipedia_Knowledge_Searcher_Octotools_Tool",
            description="""A tool that searches Wikipedia and returns article content based on a given query. CAPABILITIES: Search Wikipedia database for articles, retrieve article content and text, handle disambiguation pages, extract full or truncated article text, format search results with extracted content. SYNONYMS: Wikipedia search, knowledge search, encyclopedia search, wiki lookup, Wikipedia article finder, knowledge base search, Wikipedia content extractor. EXAMPLES: 'Search Wikipedia for information about Python programming language', 'Find Wikipedia article about Artificial Intelligence', 'Get Wikipedia content about Theory of Relativity'.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for Wikipedia."
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "The maximum length of the returned text (default: 2000). Use -1 for full text."
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=False,
            category="knowledge_search",
            tags=["wikipedia", "knowledge_search", "encyclopedia", "web_search", "information_retrieval", "text_extraction", "wikipedia_search", "knowledge_base"],
            limitation="Requires internet connection and Wikipedia API access, depends on Wikipedia availability, may fail with ambiguous queries requiring disambiguation, max_length parameter truncates content which may lose context, rate limits may apply with frequent requests.",
            agent_type="Search-Agent",
            demo_commands={
                "command": 'execution = tool.run(query="Python programming language", max_length=2000)',
                "description": "Search Wikipedia for information about Python programming language."
            }
        )

    def search_wikipedia(self, query, max_length=2000):
        """
        Searches Wikipedia based on the given query and returns the text.

        Parameters:
            query (str): The search query for Wikipedia.
            max_length (int): The maximum length of the returned text. Use -1 for full text.

        Returns:
            tuple: (search_results, page_text)
        """
        try:
            search_results = wikipedia.search(query)
            if not search_results:
                return [], "No results found for the given query."

            page = wikipedia.page(search_results[0])
            text = page.content

            if max_length != -1:
                text = text[:max_length]

            return search_results, text
        except wikipedia.exceptions.DisambiguationError as e:
            return e.options, f"DisambiguationError: {str(e)}"
        except wikipedia.exceptions.PageError:
            return [], f"PageError: No Wikipedia page found for '{query}'."
        except Exception as e:
            return [], f"Error searching Wikipedia: {str(e)}"

    def run(self, query, max_length=2000):
        """
        Searches Wikipedia based on the provided query and returns the results.

        Parameters:
            query (str): The search query for Wikipedia.
            max_length (int): The maximum length of the returned text. Use -1 for full text.

        Returns:
            dict: A dictionary containing the search results, extracted text, and formatted output.
        """
        search_results, text = self.search_wikipedia(query, max_length)
        
        formatted_output = f"Search results for '{query}':\n"
        formatted_output += "\n".join(f"{i}. {result}" for i, result in enumerate(search_results, 1))
        formatted_output += f"\n\nExtracted text:\n{text}"

        return {
            "query": query,
            "result": formatted_output,
            "success": True
        }

    def execute(self, query, max_length=2000):
        """Legacy method name for backward compatibility. Use run() instead."""
        return self.run(query, max_length)

    def get_metadata(self):
        """
        Returns the metadata for the Wikipedia_Knowledge_Searcher_Octotools_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        return super().get_metadata()


if __name__ == "__main__":
    tool = Wikipedia_Knowledge_Searcher_Octotools_Tool()
    print(tool.run(query="Python programming language", max_length=2000))