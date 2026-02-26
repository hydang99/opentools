# source code: https://github.com/octotools/octotools/blob/main/octotools/tools/pubmed_search/tool.py
import os, sys, time, traceback, json
from pymed import PubMed
from metapub import PubMedFetcher
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', '..')))

from opentools.core.base import BaseTool
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Suppress stderr by redirecting it to /dev/null
sys.stderr = open(os.devnull, 'w')

import warnings
warnings.filterwarnings("ignore")


class Pubmed_Search_Tool(BaseTool):
    """Pubmed_Search_Tool
    ---------------------
    Purpose:
        A comprehensive PubMed Central search tool that retrieves relevant scientific article abstracts based on text queries. Searches biomedical and life sciences literature from the National Library of Medicine database, providing access to millions of research papers, clinical studies, and scientific publications.

    Core Capabilities:
        - Searches PubMed Central database for scientific articles
        - Retrieves article titles, abstracts, keywords, and URLs
        - Supports multiple query terms with OR logic
        - Filters results to include only articles with abstracts
        - Provides comprehensive biomedical literature access

    Intended Use:
        Use this tool when you need to search for scientific articles, retrieve their abstracts, keywords, and URLs.

    Limitations:
        - May not handle complex search queries or results

    """
    # Default args for `opentools test Pubmed_Search_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "pubmed_extraction",
    }

    def __init__(self):
        super().__init__(
            type='function',
            name="Pubmed_Search_Tool",
            description="""A comprehensive PubMed Central search tool that retrieves relevant scientific article abstracts based on text queries. Searches biomedical and life sciences literature from the National Library of Medicine database, providing access to millions of research papers, clinical studies, and scientific publications.Use this ONLY if you cannot use the other more specific ontology tools. CAPABILITIES: Searches PubMed Central database for scientific articles, retrieves article titles, abstracts, keywords, and URLs, supports multiple query terms with OR logic, filters results to include only articles with abstracts, provides comprehensive biomedical literature access. SYNONYMS: PubMed searcher, biomedical literature search, scientific paper finder, medical research search, academic literature search, biomedical database search, research paper finder, scientific article search, medical literature tool, academic paper search. EXAMPLES: 'Search for articles about COVID vaccine research', 'Find papers on scoliosis treatment', 'Search for occupational health studies'.""",
            parameters={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search query terms for searching PubMed Central. Multiple terms are combined with OR logic."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10, maximum: 50)"
                    }
                },
                "required": ["queries"],
                "additionalProperties": False,
            },
            strict=False,
            category="research_search",
            tags=["pubmed_search", "biomedical_literature", "scientific_papers", "medical_research", "academic_literature", "research_search", "biomedical_database", "scientific_articles", "medical_literature", "academic_papers"],
            limitation="Try to use shorter and more general search queries, API can only retrieve most recent results, maximum 50 results per query, requires stable internet connection, some articles may not have abstracts available, search results depend on PubMed indexing",
            agent_type="Search-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(queries=['COVID-19 vaccine', 'SARS-CoV-2'], max_results=10)",
                "description": "Search for articles about COVID vaccine research, find papers on scoliosis treatment, search for occupational health studies"
            }
        )
        self.pubmed = PubMed(tool="MyTool", email="my@email.address")
        self.fetch = PubMedFetcher()

    def _build_query(self, queries):
        # (term)[Title/Abstract] OR ... AND filters
        ta_terms = [f'({q})[Title/Abstract]' for q in queries if q and q.strip()]
        if not ta_terms:
            raise ValueError("At least one non-empty query is required.")
        return f"({' OR '.join(ta_terms)}) AND hasabstract"

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def _search(self, query_str, max_results=10):
        return self.pubmed.query(query_str, max_results=max_results)

    def run(self, queries, max_results=10):
        return self.search_article(queries=queries, max_results=max_results)

    def search_article(self, queries, max_results=10):
        try:
            q = self._build_query(queries)
            max_results = min(int(max_results or 10), 50)

            results = self._search(q, max_results=max_results)

            items = []
            for art in results:
                # Use pymed data directly (avoid metapub extra calls)
                d = art.toDict()  # already a dict
                pmid_raw = d.get("pubmed_id", "") or ""
                pmid = pmid_raw.split()[0].strip(";")  # sanitize if it contains "; PMC..."
                items.append({
                    "pubmed_id": pmid or None,
                    "title": d.get("title"),
                    "abstract": d.get("abstract"),
                    "keywords": d.get("keywords"),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
                })

            if not items:
                return {"error": "No results found for search query. Try another query or tool.", "success": False, "error_type": "no_results_found"}
            return {"result": items, "success": True}

        except Exception as e:
            return {"error": f"Error searching PubMed: {str(e)}", "success": False, "error_type": "pubmed_search_failed", "traceback": traceback.format_exc()}


    
    def test(self, tool_test: str="pubmed_extraction"):
        """Test the PubMed tool with various test samples, run 3 times, and save results in a JSON file."""
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
                question_result = {"id": test.get("id", f"pubmed_{i + 1}")}
                if 'query' in test:
                    question_result['query'] = test['queries']
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
                    if not result or (isinstance(result, dict) and not isinstance(result, list)):
                        run_result['accuracy'] = 0
                        run_result['tool_call_pass'] = False
                        question_result[f'run_{j}'] = run_result
                        continue
                    else:
                        run_result['tool_call_pass'] = True

                    # Calculate accuracy for this run
                    temp_accuracy = 0
                    for item in result:
                        title = item.get('title') or ""
                        keywords = " ".join(item.get('keywords', []))
                        test_query = " ".join(test['queries']) if isinstance(test['queries'], list) else str(test['queries'] or "")
                        temp_accuracy += self.eval_accuracy(title + " " + keywords, test_query)
                    accuracy_score = temp_accuracy / len(result) if result else 0
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

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/pubmed_search
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage
    tool = Pubmed_Search_Tool()

    # Execute the tool
    try:
        tool.embed_tool()
        tool.test(tool_test="pubmed_extraction")
    except ValueError as e: 
        print(f"Execution failed: {e}")