# source code: https://github.com/inclusionAI/AWorld/blob/main/examples/gaia/mcp_collections/tools/wiki.py
"""
Compared to the source code, we add similarity search to just retrieve the top result(s) in order of relevance. 
This enhancement ensures that the most relevant Wikipedia article(s) are prioritized and returned for the query.
"""


import calendar, json, os, time, traceback, requests, wikipedia, sys
from datetime import datetime, timezone
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
from pydantic import Field, BaseModel
from pydantic.fields import FieldInfo

class WikipediaSearchResult(BaseModel):
    """Model representing a Wikipedia search result."""
    title: str
    snippet: str | None = None
    url: str | None = None


class WikipediaArticle(BaseModel):
    """Model representing a Wikipedia article."""

    title: str
    pageid: int | None = None
    url: str
    content: str
    summary: str
    images: list[str] | None = None
    categories: list[str] | None = None
    links: list[str] | None = None
    references: list[str] | None = None
    sections: list[dict[str, str]] | None = None
    # History-specific fields
    original_query: str | None = None
    requested_date: str | None = None
    actual_date: str | None = None
    is_exact_date: bool | None = None
    editor: str | None = None
    edit_comment: str | None = None


class WikipediaMetadata(BaseModel):
    """Metadata for Wikipedia operation results."""

    query: str
    language: str
    count: int
    operation_type: str
    execution_time: float | None = None
    error_type: str | None = None
    article_id: int | None = None
    requested_date: str | None = None
    actual_date: str | None = None

class Wiki_Search_Tool(BaseTool):
    """Wiki_Search_Tool
    ---------------------
    Purpose:
        A comprehensive Wikipedia search tool that provides article search, categories, links, and historical versions.

    Core Capabilities:
        - Wikipedia article search with intelligent ranking and result processing
        - Category and link extraction from articles
        - Historical version access with date-based lookup
        - Multi-language Wikipedia support across major languages

    Intended Use:
        Use this tool when you need to search Wikipedia for articles, get categories, links, and historical versions.

    Limitations:
        - Depends on Wikipedia's availability and rate limits
        - Some articles may not have historical versions
        - Auto-suggest may not always find exact matches
        - Language support depends on Wikipedia's available languages
    """
    # Default args for `opentools test Wiki_Search_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "wiki_search",
    }

    def __init__(self) -> None:
        super().__init__(
            type='function',
            name="Wiki_Search_Tool",
            description="""A comprehensive Wikipedia search tool that provides article search, categories, links, and historical versions. CAPABILITIES: Wikipedia article search with intelligent ranking and result processing, category and link extraction from articles, historical version access with date-based lookup, multi-language Wikipedia support across major languages, SYNONYMS: Wikipedia search tool, wiki article finder, Wikipedia content extractor, wiki category tool, Wikipedia link extractor, wiki history tool, Wikipedia API tool, wiki search engine, Wikipedia data extractor, wiki content analyzer. EXAMPLES: 'Search Wikipedia for articles about artificial intelligence', 'Get all categories for the deep learning article', 'Extract all links from the neural network article', 'Get historical version of AI article from January 15, 2020'.""",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform: 'search', 'get_categories', 'get_links', 'get_history'",
                        "enum": ["search", "get_categories", "get_links", "get_history"]
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query for search operation (required for search operation)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Article title for content operations (required for get_categories, get_links, get_history operations)"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date for get_history operation in YYYY/MM/DD format (required for get_history operation)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results for search operation (1-20, default: 5)"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code for Wikipedia (e.g., 'en', 'es', 'fr', default: 'en')"
                    },
                    "auto_suggest": {
                        "type": "boolean",
                        "description": "Use auto-suggest for content operations (default: false)"
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'markdown', 'json', or 'text' (default: 'json')",
                        "enum": ["markdown", "json", "text"]
                    }
                },
                "required": ["operation"],
                "additionalProperties": False,
            },
            strict=False,
            category="information_retrieval",
            tags=["wikipedia_search", "wiki_tools", "article_search", "category_extraction", "link_extraction", "historical_versions", "multi_language", "llm_optimized", "information_retrieval", "knowledge_base"],
            limitation="Requires internet connection for Wikipedia API access, depends on Wikipedia's availability and rate limits, some articles may not have historical versions, auto-suggest may not always find exact matches, language support depends on Wikipedia's available languages",
            agent_type="Search-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(operation='search', query='artificial intelligence')",
                "description": "Search Wikipedia for articles about artificial intelligence"
            }
        )
        self.default_language = "en"
        self.max_search_results = 20
        self.max_random_articles = 10
        self.default_summary_sentences = 5

        wikipedia.set_lang(self.default_language)

    def run(
        self,
        operation: str = Field(description="The operation to perform: 'search', 'get_content', 'get_summary', 'get_categories', 'get_links', 'get_history'"),
        query: str | None = Field(default=None, description="Search query for search operation"),
        title: str | None = Field(default=None, description="Article title for content operations"),
        date: str | None = Field(default=None, description="Date for get_history operation (YYYY/MM/DD)"),
        limit: int = Field(default=5, description="Maximum number of results for search (1-20)"),
        language: str = Field(default="en", description="Language code for Wikipedia"),
        auto_suggest: bool = Field(default=False, description="Use auto-suggest for content operations"),
        output_format: str = Field(default="json", description="Output format: 'markdown', 'json', or 'text'"),
    ):
        """Unified function for Wikipedia operations.

        This function handles all Wikipedia operations based on the operation parameter.

        Args:
            operation: The operation to perform
            query: Search query for search operation
            title: Article title for content operations
            date: Date for get_history operation (YYYY/MM/DD)
            limit: Maximum number of results for search
            language: Language code for Wikipedia
            auto_suggest: Use auto-suggest for content operations
            output_format: Format for the response

        Returns:
            Dictionary with operation results and metadata
        """
        # Handle FieldInfo objects
        if isinstance(operation, FieldInfo):
            operation = operation.default
        if isinstance(query, FieldInfo):
            query = query.default
        if isinstance(title, FieldInfo):
            title = title.default
        if isinstance(date, FieldInfo):
            date = date.default
        if isinstance(limit, FieldInfo):
            limit = limit.default
        if isinstance(language, FieldInfo):
            language = language.default
        if isinstance(auto_suggest, FieldInfo):
            auto_suggest = auto_suggest.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default

        start_time = time.time()

        try:
            # Set language
            wikipedia.set_lang(language)

            if operation == "search":
                if not query:
                    return {
                        "error": "Error: 'query' parameter is REQUIRED for 'search' operation. Example: tool.run(operation='search', query='artificial intelligence')",
                        "success": False
                    }
                return self._handle_search(query, limit, language, output_format, start_time)

            elif operation == "get_categories":
                if not title:
                    return {
                        "error": "Error: 'title' parameter is REQUIRED for 'get_categories' operation. Example: tool.run(operation='get_categories', title='Deep learning')",
                        "success": False
                    }
                return self._handle_get_categories(title, language, output_format, start_time)

            elif operation == "get_links":
                if not title:
                    return {
                        "error": "Error: 'title' parameter is REQUIRED for 'get_links' operation. Example: tool.run(operation='get_links', title='Neural network')",
                        "success": False
                    }
                return self._handle_get_links(title, language, output_format, start_time)

            elif operation == "get_history":
                if not title:
                    return {
                        "error": "Error: 'title' parameter is REQUIRED for 'get_history' operation. Example: tool.run(operation='get_history', title='Artificial intelligence', date='2020/01/15')",
                        "success": False
                    }
                if not date:
                    return {
                        "error": "Error: 'date' parameter is REQUIRED for 'get_history' operation. Example: tool.run(operation='get_history', title='Artificial intelligence', date='2020/01/15')",
                        "success": False
                    }
                return self._handle_get_history(title, date, auto_suggest, language, output_format, start_time)
            else:   
                return {
                    "error": f"Error: Unknown operation '{operation}'. Supported operations: 'search', 'get_content', 'get_summary', 'get_categories', 'get_links', 'get_history', 'get_full_content'",
                    "success": False
                }

        except Exception as e:
            error_msg = f"Wikipedia operation failed: {str(e)}"
            print(f"Error in run: {traceback.format_exc()}")

            return {
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "success": False
            }

    
    def _handle_search(self, query: str, limit: int, language: str, output_format: str, start_time: float):
        """Handle search operation."""
        try:
            print(f"🔍 Searching Wikipedia for: {query} (language: {language})")

            # Limit the number of results to prevent excessive API calls
            if limit > self.max_search_results:
                limit = self.max_search_results

            # Search Wikipedia
            search_results = wikipedia.search(query, results=limit)

            # Format results
            formatted_results = []
            for title in search_results:
                try:
                    # Get a summary to use as a snippet
                    summary = wikipedia.summary(title, sentences=1, auto_suggest=False)
                    # Create URL
                    url = f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}"

                    result = WikipediaSearchResult(title=title, snippet=summary, url=url)
                    formatted_results.append(result)
                except Exception as e:
                    print(f"Error getting details for '{title}': {str(e)}")
                    # Still include the result, but without a snippet
                    url = f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    result = WikipediaSearchResult(title=title, url=url)
                    formatted_results.append(result)

            # Format output for LLM
            formatted_output = self._format_search_results(formatted_results, output_format)
            metadata = WikipediaMetadata(
                    query=query,
                    language=language,
                    count=len(formatted_results),
                    operation_type="search",
                    execution_time=time.time() - start_time,
                )

            print(f"✅ Found {len(formatted_results)} results for query: {query}")

            return {
                "result": formatted_output,
                "success": True,
                "metadata": metadata.model_dump(),
            }
        except Exception as e:
            error_msg = f"Wikipedia search failed: {str(e)}"
            print(f"Error in _handle_search: {traceback.format_exc()}")
            return {
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "success": False
            }


    def _handle_get_categories(self, title: str, language: str, output_format: str, start_time: float):
        """Handle get_categories operation."""
        try:
            print(f"🏷️ Retrieving categories for Wikipedia article: {title} (language: {language})")

            # Get the page
            page = wikipedia.page(title, auto_suggest=True, redirect=True)

            # Format output for LLM
            if output_format == "json":
                formatted_output = json.dumps(page.categories, indent=2)
            elif output_format == "text":
                formatted_output = f"Categories for {title}:\n" + "\n".join(f"- {cat}" for cat in page.categories)
            else:  # markdown
                formatted_output = f"# Categories for {title}\n\n" + "\n".join(f"- {cat}" for cat in page.categories)

            print(f"✅ Retrieved {len(page.categories)} categories for: {title}")
            metadata = WikipediaMetadata(
                    query=title,
                    language=language,
                    count=len(page.categories),
                    operation_type="categories_retrieval",
                    execution_time=time.time() - start_time,
                    article_id=page.pageid,
                )
            return {
                "result": formatted_output,
                "success": True,
                "metadata": metadata.model_dump(),
            }
        except Exception as e:
            error_msg = f"Wikipedia get categories failed: {str(e)}"
            print(f"Error in _handle_get_categories: {traceback.format_exc()}")
            return {
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "success": False
            }

    def _handle_get_links(self, title: str, language: str, output_format: str, start_time: float):
        """Handle get_links operation."""
        try:
            print(f"🔗 Retrieving links from Wikipedia article: {title} (language: {language})")

            # Get the page
            page = wikipedia.page(title, auto_suggest=True, redirect=True)

            # Format results
            formatted_results = []
            for link_title in page.links:
                try:
                    url = f"https://{language}.wikipedia.org/wiki/{link_title.replace(' ', '_')}"
                    result = WikipediaSearchResult(title=link_title, url=url)
                    formatted_results.append(result)
                except Exception as e:
                    print(f"Error formatting link '{link_title}': {str(e)}")

            # Format output for LLM
            if output_format == "json":
                formatted_output = json.dumps([result.model_dump() for result in formatted_results], indent=2)
            elif output_format == "text":
                formatted_output = f"Links from {title}:\n" + "\n".join(
                    f"- {result.title}" for result in formatted_results
                )
            else:  # markdown
                formatted_output = f"# Links from {title}\n\n"
                # Limit to first 50 links to avoid overwhelming output
                for i, result in enumerate(formatted_results[:50], 1):
                    formatted_output += f"{i}. [{result.title}]({result.url})\n"
                if len(formatted_results) > 50:
                    formatted_output += f"\n... and {len(formatted_results) - 50} more links"

            print(f"✅ Retrieved {len(formatted_results)} links from: {title}")
            metadata = WikipediaMetadata(
                    query=title,
                    language=language,
                    count=len(formatted_results),
                    operation_type="links_retrieval",
                    execution_time=time.time() - start_time,
                    article_id=page.pageid,
                )
            return {
                "result": formatted_output,
                "success": True,
                "metadata": metadata.model_dump(),
            }
        except Exception as e:
            error_msg = f"Wikipedia get links failed: {str(e)}"
            print(f"Error in _handle_get_links: {traceback.format_exc()}")
            return {
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "success": False
            }

    def _handle_get_history(self, title: str, date: str, auto_suggest: bool, language: str, output_format: str, start_time: float):
        """Handle get_history operation."""
        try:
            print(f"📅 Retrieving historical version of Wikipedia article: {title} for date: {date}")
            def _resolve_title(user_title: str) -> str:
                """
                Resolve a user-provided title to a canonical page title.
                Prefer an exact (case-insensitive) match from search results instead of
                blindly taking the first result (which can be wrong, e.g. "Leg" for "Lego").
                """
                if not auto_suggest:
                    return user_title
                try:
                    results = wikipedia.search(user_title, results=5)
                    if not results:
                        return user_title
                    exact = next((r for r in results if r.strip().lower() == user_title.strip().lower()), None)
                    chosen = exact or results[0]
                    # Use auto_suggest=False here because we've already chosen a candidate.
                    page = wikipedia.page(chosen, auto_suggest=False, redirect=True)
                    print(f"Found matching page: {page.title} for query: {user_title}")
                    return page.title
                except Exception as e:
                    print(f"Auto-suggest failed for {user_title}: {str(e)}")
                    return user_title

            def _parse_history_date(s: str) -> datetime:
                """
                Accept dates as:
                - YYYY/MM/DD (legacy)
                - YYYY-MM-DD (ISO)
                - YYYY/MM or YYYY-MM (interpreted as end of that month)
                - YYYY (interpreted as end of that year)
                Returns a timezone-aware UTC datetime at 23:59:59Z for inclusive boundary.
                """
                raw = (s or "").strip()
                if not raw:
                    raise ValueError("date is required")

                m = re.fullmatch(r"(\d{4})(?:[/-](\d{1,2})(?:[/-](\d{1,2}))?)?", raw)
                if not m:
                    raise ValueError(f"Invalid date format: {s}. Expected YYYY/MM/DD, YYYY-MM-DD, YYYY-MM, or YYYY.")

                year = int(m.group(1))
                month_s = m.group(2)
                day_s = m.group(3)

                if month_s is None:
                    month = 12
                    day = 31
                else:
                    month = int(month_s)
                    if not (1 <= month <= 12):
                        raise ValueError(f"Invalid month in date: {s}")
                    if day_s is None:
                        day = calendar.monthrange(year, month)[1]
                    else:
                        day = int(day_s)
                        max_day = calendar.monthrange(year, month)[1]
                        if not (1 <= day <= max_day):
                            raise ValueError(f"Invalid day in date: {s}")

                return datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)

            actual_title = _resolve_title(title)
            target_date = _parse_history_date(date)

            # Get page revisions
            params = {
                "action": "query",
                "prop": "revisions",
                "titles": actual_title,
                # Prefer lightweight props; content extraction is brittle across MW slot formats.
                "rvprop": "ids|timestamp|user|comment|content",
                "rvlimit": 1,
                "rvdir": "older",
                "rvstart": target_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "format": "json",
            }

            # Make API request
            API_URL = f"https://{language}.wikipedia.org/w/api.php"
            # Wikipedia requests a descriptive User-Agent; many environments get 403 without it.
            # See: https://meta.wikimedia.org/wiki/User-Agent_policy
            headers = {
                "User-Agent": os.getenv(
                    "OPENTOOLS_WIKIMEDIA_USER_AGENT",
                    "OpenToolsWikiSearch/1.0 (contact: set OPENTOOLS_WIKIMEDIA_USER_AGENT)",
                ),
                "Accept": "application/json",
            }

            response = requests.get(API_URL, params=params, headers=headers, timeout=10)
            if response.status_code == 403:
                # One quick retry with a slightly different UA in case the env default is blocked.
                headers["User-Agent"] = headers["User-Agent"] + " retry"
                time.sleep(0.5)
                response = requests.get(API_URL, params=params, headers=headers, timeout=10)

            if response.status_code != 200:
                raise ValueError(f"MediaWiki API HTTP {response.status_code}")
            try:
                data = response.json()
            except Exception as e:
                # Helpful debug when Wikipedia returns HTML (rate limit / block / captive portal)
                sample = (response.text or "")[:200].replace("\n", " ")
                raise ValueError(f"MediaWiki API returned non-JSON response. Sample: {sample}") from e

            # Process response
            page = next(iter(data["query"]["pages"].values()))
            if "revisions" in page:
                revision = page["revisions"][0]
                actual_date = datetime.fromisoformat(revision["timestamp"].replace("Z", "+00:00"))

                # Create URL for this version
                page_id = page["pageid"]
                rev_id = revision["revid"]
                url = f"https://{language}.wikipedia.org/w/index.php?oldid={rev_id}"

                # Create article object
                article = WikipediaArticle(
                    title=actual_title,
                    pageid=page_id,
                    url=url,
                    # MW content can be under "*" (legacy) or under slots.main["*"] (modern).
                    content=revision.get("*")
                    or ((revision.get("slots") or {}).get("main") or {}).get("*")
                    or "",
                    summary=f"Historical version from {actual_date.strftime('%Y/%m/%d')}",
                    images=[],  # Historical versions don't include images
                    categories=[],
                    links=[],
                    references=[],
                    sections=[],
                    original_query=title,
                    requested_date=target_date.strftime("%Y/%m/%d"),
                    actual_date=actual_date.strftime("%Y/%m/%d"),
                    is_exact_date=actual_date.date() == target_date.date(),
                    editor=revision["user"],
                    edit_comment=revision.get("comment", ""),
                )

                # Format output for LLM
                formatted_output = self._format_article(article, output_format, include_full_content=True)
                metadata = WikipediaMetadata(
                        query=title,
                        language=language,
                        count=1,
                        operation_type="history_retrieval",
                        execution_time=time.time() - start_time,
                        article_id=page_id,
                        is_redirect=actual_title != title,
                        requested_date=target_date.strftime("%Y/%m/%d"),
                        actual_date=actual_date.strftime("%Y/%m/%d"),
                    )



                print(f"✅ Retrieved historical version from {actual_date.strftime('%Y/%m/%d')}")

                return {
                    "result": formatted_output,
                    "success": True,
                    "metadata": metadata.model_dump(),
                }

            # No revision found
            error_msg = f"No revision found for {actual_title} (original query: {title}) before {date}"

            return {
                "result": error_msg,
                "success": False,
            }
        
        except Exception as e:
            error_msg = f"Wikipedia get history failed: {str(e)}"
            print(f"Error in _handle_get_history: {traceback.format_exc()}")
            return {
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "success": False
            }

    def _format_search_results(self, results: list[WikipediaSearchResult], output_format: str = "markdown") -> str:
        """Format search results for LLM consumption.

        Args:
            results: List of search results
            output_format: Format type ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if output_format == "json":
            return json.dumps([result.model_dump() for result in results], indent=2)
        elif output_format == "dict":
            return [result.model_dump() for result in results]
        elif output_format == "text":
            if not results:
                return "No results found."

            output_parts = [f"Found {len(results)} results:"]

            for i, result in enumerate(results, 1):
                output_parts.append(f"{i}. Title of the paper: [{result.title}] - ")
                if result.snippet:
                    output_parts.append(f"Snippet: {result.snippet}")
                if result.url:
                    output_parts.append(f"Link to the paper: {result.url}")

            return "\n".join(output_parts)

        else:  # markdown (default)
            if not results:
                return "No Wikipedia search results found."

            output_parts = [f"# Wikipedia Search Results\n\nFound {len(results)} results:\n"]

            for i, result in enumerate(results, 1):
                output_parts.append(f"## {i}. Title of the paper: [{result.title}] - Link to the paper: ({result.url})")
                if result.snippet:
                    output_parts.append(f"- Snippet: {result.snippet}\n")

            return "\n".join(output_parts)

    def _format_article(
        self, article: WikipediaArticle, output_format: str = "markdown", include_full_content: bool = False
    ) -> str:
        """Format article for LLM consumption.

        Args:
            article: Wikipedia article
            output_format: Format type ('markdown', 'json', 'text')
            include_full_content: Whether to include the full article content

        Returns:
            Formatted string suitable for LLM consumption
        """
        if output_format == "json":
            return json.dumps(article.model_dump(), indent=2)

        elif output_format == "text":
            output_parts = [f"Title: {article.title}"]

            if article.url:
                output_parts.append(f"URL: {article.url}")

            if article.summary:
                output_parts.append(f"\nSummary:\n{article.summary}")

            if include_full_content and article.content:
                output_parts.append(f"\nContent:\n{article.content}")

            # Add sections if available
            if article.sections and len(article.sections) > 0:
                output_parts.append(f"\nSections:")
                for section in article.sections:
                    output_parts.append(f"- {section['title']}")
                    if section.get('content'):
                        output_parts.append(f"  {section['content']}")

            if article.categories:
                output_parts.append(f"\nCategories: {', '.join(article.categories)}")

            # Add references if available
            if article.references and len(article.references) > 0:
                output_parts.append(f"\nReferences:")
                for i, ref in enumerate(article.references[:10], 1):
                    output_parts.append(f"{i}. {ref}")
                if len(article.references) > 10:
                    output_parts.append(f"... and {len(article.references) - 10} more references")

            # Add images if available
            if article.images and len(article.images) > 0:
                output_parts.append(f"\nImages: {', '.join(article.images[:5])}")
                if len(article.images) > 5:
                    output_parts.append(f"... and {len(article.images) - 5} more images")

            if article.requested_date:
                output_parts.append("\nHistorical Version:")
                output_parts.append(f"Requested Date: {article.requested_date}")
                output_parts.append(f"Actual Date: {article.actual_date}")

            if article.links and len(article.links) > 0:
                output_parts.append(f"\nRelated Links: {', '.join(article.links[:10])}")
                if len(article.links) > 10:
                    output_parts.append(f"... and {len(article.links) - 10} more links")

            return "\n".join(output_parts)

        else:  # markdown (default)
            output_parts = [f"# {article.title}"]

            if article.url:
                output_parts.append(f"**Wikipedia:** [{article.title}]({article.url})")

            if article.summary:
                output_parts.append(f"\n## Summary\n{article.summary}")

            if include_full_content and article.content:
                output_parts.append(f"\n## Content\n{article.content}")

            # Add sections if available
            if article.sections and len(article.sections) > 0:
                output_parts.append(f"\n## Article Sections\n")
                for section in article.sections:
                    output_parts.append(f"### {section['title']}")
                    if section.get('content'):
                        output_parts.append(f"{section['content']}\n")

            if article.categories and len(article.categories) > 0:
                output_parts.append(f"\n## Categories\n{', '.join(article.categories)}")

            # Add references if available
            if article.references and len(article.references) > 0:
                output_parts.append(f"\n## References\n")
                for i, ref in enumerate(article.references[:15], 1):
                    output_parts.append(f"{i}. {ref}")
                if len(article.references) > 15:
                    output_parts.append(f"\n... and {len(article.references) - 15} more references")

            # Add images if available
            if article.images and len(article.images) > 0:
                output_parts.append(f"\n## Images and Media\n")
                for i, img in enumerate(article.images[:10], 1):
                    output_parts.append(f"{i}. {img}")
                if len(article.images) > 10:
                    output_parts.append(f"\n... and {len(article.images) - 10} more images")

            if article.links and len(article.links) > 0:
                output_parts.append("\n## Related Links\n")
                # Limit to first 30 links to avoid overwhelming output
                for i, link in enumerate(article.links[:30], 1):
                    output_parts.append(f"{i}. {link}")
                if len(article.links) > 30:
                    output_parts.append(f"\n... and {len(article.links) - 30} more links")

            if article.requested_date:
                output_parts.append("\n## Historical Version Information")
                output_parts.append(f"**Requested Date:** {article.requested_date}")
                output_parts.append(f"**Actual Date:** {article.actual_date}")
                if article.editor:
                    output_parts.append(f"**Editor:** {article.editor}")
                if article.edit_comment:
                    output_parts.append(f"**Edit Comment:** {article.edit_comment}")

            return "\n".join(output_parts)

    def test(self, tool_test: str="wiki_search"):
        """Test the Wiki Search tool with various test samples, run 3 times, and save results in a JSON file."""
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
                question_result = {"id": test.get("id", f"wiki_{i + 1}")}
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

                    # Calculate accuracy for this run - take top 2 results and average
                    if result.get("result"):
                        response = result['result']
                        
                        # Handle different response formats
                        if isinstance(response, list):
                            # Calculate similarity scores for all results and get top 2 with highest similarity
                            result_scores = []
                            for result_item in response:
                                if isinstance(result_item, dict) and 'title' in result_item:
                                    title = result_item.get('title', '')
                                    snippet = result_item.get('snippet', '') or ''
                                    # Combine title and snippet for better accuracy evaluation
                                    content = title + " " + snippet
                                    similarity_score = self.eval_accuracy(content, test['query'])
                                    result_scores.append((similarity_score, result_item))
                            
                            # Sort by similarity score (highest first) and take top 2
                            result_scores.sort(key=lambda x: x[0], reverse=True)
                            top_2_results = result_scores[:2]
                            
                            # Calculate average accuracy from top 2 results
                            temp_accuracy = 0
                            for score, result_item in top_2_results:
                                temp_accuracy += score
                            accuracy_score = temp_accuracy / len(top_2_results) if top_2_results else 0
                        elif isinstance(response, str):
                            # If response is a string, use eval_accuracy directly
                            accuracy_score = self.eval_accuracy(response, test['query'])
                        else:
                            accuracy_score = 0
                        
                        run_result['accuracy'] = accuracy_score
                        run_accuracy[f'run_{j}'] += accuracy_score
                    else:
                        run_result['accuracy'] = 0
                        run_accuracy[f'run_{j}'] += 0

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

# Default arguments for testing
if __name__ == "__main__":
    try:
        service = Wiki_Search_Tool()
        service.embed_tool()
        print(service.run(operation="get_history", title="Lego", date="2022/12/31", language="en", auto_suggest=True, output_format="json"))
        # service.test(tool_test="wiki_search")
    except Exception as e:  
        print(f"An error occurred: {e}")
