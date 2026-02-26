# Simple ArXiv Paper Search Tool
# Searches arXiv for academic papers based on query
# source code: https://github.com/octotools/octotools/blob/main/octotools/tools/arxiv_paper_searcher/tool.py
import os, sys, re, traceback, requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
from bs4 import BeautifulSoup

class Simple_ArXiv_Paper_Search_Tool(BaseTool):
    """Simple_ArXiv_Paper_Search_Tool
    ---------------------
    Purpose:
        A tool that searches arXiv for academic papers based on a given query. Returns paper titles, authors, abstracts, and links.

    Core Capabilities:
        - Search arXiv database for academic papers
        - Retrieve paper metadata (title, authors, abstract, links)
        - Configurable results per page (25, 50, 100, 200)
        - Limit maximum results returned

    Intended Use:
        Use this tool when you need to search arXiv for academic papers, retrieve paper metadata, and configure results per page.

    Limitations:
        - May not handle complex search queries or results

    """
    def __init__(self):
        super().__init__(
            type='function',
            name="Simple_ArXiv_Paper_Search_Tool",
            description="""A tool that searches arXiv for academic papers based on a given query. Returns paper titles, authors, abstracts, and links. CAPABILITIES: Search arXiv database for academic papers, retrieve paper metadata (title, authors, abstract, links), configurable results per page (25, 50, 100, 200), limit maximum results returned. SYNONYMS: arXiv search, academic paper search, research paper finder, arXiv query tool, paper search engine, academic search tool. EXAMPLES: 'Search for papers about machine learning', 'Find quantum computing papers with max 50 results', 'Search arXiv for tool agents with large language models'.""",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for arXiv papers."
                    },
                    "size": {
                        "type": "integer",
                        "description": "The number of results per page (25, 50, 100, or 200). If None, use 25.",
                        "enum": [25, 50, 100, 200]
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "The maximum number of papers to return (default: 25). Should be less than or equal to 100."
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=False,
            category="academic_search",
            tags=["arxiv", "paper_search", "academic_search", "research_papers", "scientific_literature", "bibliography", "academic_research"],
            limitation="Maximum 100 results per query for traffic reasons, requires internet connection, depends on arXiv website availability and structure, HTML parsing may break if arXiv changes their page structure, size parameter must be one of the valid values (25, 50, 100, 200).",
            agent_type="Search-Agent",
            demo_commands={
                "command": 'execution = tool.run(query="tool agents with large language models", size=50, max_results=25)',
                "description": "Search for papers about tool agents with large language models, with 50 results per page, returning a maximum of 25 papers."
            }
        )
        self.valid_sizes = [25, 50, 100, 200]
        self.base_url = "https://arxiv.org/search/"

    def run(self, query, size=None, max_results=25):
        """
        Main execution method for the tool.
        
        Parameters:
            query (str): The search query for arXiv papers.
            size (int): The number of results per page (25, 50, 100, or 200).
            max_results (int): The maximum number of papers to return.
        
        Returns:
            dict: A dictionary with search results and success status.
        """
        try:
            if size is None:
                size = 25
            elif size not in self.valid_sizes:
                size = min(self.valid_sizes, key=lambda x: abs(x - size))

            results = []
            start = 0

            max_results = min(max_results, 100)  # NOTE: For traffic reasons, limit to 100 results

            while len(results) < max_results:
                params = {
                    "searchtype": "all",
                    "query": query,
                    "abstracts": "show",
                    "order": "",
                    "size": str(size),
                    "start": str(start)
                }

                try:
                    response = requests.get(self.base_url, params=params)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')

                    papers = soup.find_all("li", class_="arxiv-result")
                    if not papers:
                        break

                    for paper in papers:
                        if len(results) >= max_results:
                            break

                        title_elem = paper.find("p", class_="title")
                        authors_elem = paper.find("p", class_="authors")
                        abstract_elem = paper.find("span", class_="abstract-full")
                        link_elem = paper.find("p", class_="list-title")
                        
                        if not all([title_elem, authors_elem, abstract_elem, link_elem]):
                            continue

                        title = title_elem.text.strip()
                        authors = authors_elem.text.strip()
                        authors = re.sub(r'^Authors:\s*', '', authors)
                        authors = re.sub(r'\s+', ' ', authors).strip()
                        
                        abstract = abstract_elem.text.strip()
                        abstract = abstract.replace("△ Less", "").strip()
                        
                        link = link_elem.find("a")["href"]

                        results.append({
                            "title": title,
                            "authors": authors,
                            "abstract": abstract,
                            "link": link
                        })

                    start += size

                except requests.RequestException as e:
                    print(f"Error fetching results from arXiv: {e}")
                    break
                except Exception as e:
                    print(f"Error parsing arXiv results: {e}")
                    break

            return {
                "result": results[:max_results],
                "success": True
            }

        except Exception as e:
            print(f"Error searching arXiv: {e}")
            print(traceback.format_exc())
            return {"error": f"Error searching arXiv: {e}", "success": False, "error_type": "arxiv_search_failed", "traceback": traceback.format_exc()}

    def execute(self, query, size=None, max_results=25):
        """Legacy method name for backward compatibility. Use run() instead."""
        return self.run(query, size, max_results)

    def get_metadata(self):
        """
        Returns the metadata for the Simple_ArXiv_Paper_Search_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        return super().get_metadata()

if __name__ == "__main__":
    import json

    tool = Simple_ArXiv_Paper_Search_Tool()
    query = "enhance mathematical reasoning with large language models"
    result = tool.run(query=query, size=50, max_results=10)
    print(json.dumps(result, indent=2))
