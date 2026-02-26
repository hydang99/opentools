# source code: https://github.com/inclusionAI/AWorld/blob/main/examples/gaia/mcp_collections/tools/mcparxiv.py
"""
Improvements Over Baseline arXiv Search Tool:

Compared to a basic ArXiv search tool that typically only supports simple keyword-based queries,
this implementation introduces several advanced features to make scholarly queries richer and more precise:

- **Advanced Filtering**: Supports structured search by author, publication category, publication date range 
  (`date_from`/`date_to`), and combinations thereof, not just single 'query' text.
- **Fielded Search**: Allows specification of multiple filters simultaneously (for example: search for papers 
  authored by 'Andrew Ng' in 'cs.LG' category published after 2023-01-01).

"""

import json, traceback, os, sys, arxiv
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
class PaperResult(BaseModel):
    """Individual paper search result with structured data."""
    entry_id: str
    title: str
    authors: list[str]
    summary: str
    published: str
    updated: str | None = None
    categories: list[str]
    primary_category: str
    pdf_url: str | None = None
    doi: str | None = None
    journal_ref: str | None = None
    comment: str | None = None

class ArxivMetadata(BaseModel):
    """Metadata for ArXiv operation results."""
    operation: str
    query: str | None = None
    max_results: int | None = None
    sort_by: str | None = None
    sort_order: str | None = None
    total_results: int | None = None
    execution_time: float | None = None
    error_type: str | None = None
    paper_id: str | None = None
    download_path: str | None = None
    file_size: int | None = None

class Arxiv_Paper_Search_Tool(BaseTool):
    """
    Arxiv_Paper_Search_Tool
    ---------------------
    Purpose:
        Searches and downloads academic papers from ArXiv (arXiv.org) using comprehensive search criteria. Supports paper search by keywords, authors, categories, and date ranges, plus article downloads.

    Core Capabilities:
        - Paper search with multiple filters
        - Paper downloads
        - Paper content extraction with category filtering
        - Author filtering and date range filtering

    Intended Use:
        Use this tool when you need to search and download academic papers from ArXiv, including multiple filters, paper downloads, and paper content extraction.

    Limitations:
        - Requires internet connection for ArXiv API access
        - Depends on ArXiv's availability and rate limits
        - Some papers may not be available for download
        - Some papers may have limited download availability
    """
    # Default args for `opentools test Arxiv_Paper_Search_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "arxiv_paper_search",
        "file_location": "arxiv_paper_search",
        "result_parameter": "title",
        "search_type": "exact_match",
    }

    def __init__(self):
        super().__init__(
            type='function',
            name="Arxiv_Paper_Search_Tool",
            description="""Searches and downloads academic papers from ArXiv (arXiv.org) using comprehensive search criteria. Supports paper search by keywords, authors, categories, and date ranges, plus article downloads. Use when you need to find academic papers, research articles, or scientificpublications. CAPABILITIES: Paper search with multiple filters, paper downloads, paper content extraction with category filtering, author filtering anddate range filtering. SYNONYMS: arxiv search, academic paper search, research paper finder, scientific paper download, arxiv paper retrieval, academic literature search. EXAMPLES: 'Search for machine learning papers from 2024', 'Find papers by author Stephen Hawking', 'Download the PDF of paper 2301.07041','Get papers in computer science category'.""",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform: 'search' to find papers, or 'download' to download a specific paper PDF.",
                        "enum": ["search", "download"]
                    },
                    "query": {
                        "type": "string",
                        "description": "Search terms for finding papers. REQUIRED for search operations. Use a SIMPLE, DIRECT query string with key words. DO NOT use complex query syntax. Extract the most important keywords directly from the search request."
                    },
                    "author": {
                        "type": "string",
                        "description": "Author name(s) to search for. IMPORTANT: Use 'Lastname, Firstname' format for best results (e.g., 'Vaswani, Ashish' not 'Ashish Vaswani'). Multiple authors separated by semicolon (e.g., 'Hawking, S; Einstein, A'). The tool will auto-convert 'Firstname Lastname' format but results may be limited."
                    },
                    "paper_id": {
                        "type": "string",
                        "description": "ArXiv paper ID (e.g., '2301.07041' or 'arxiv:2301.07041'). Required for download operations."
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Sort criteria: 'relevance' (default), 'lastUpdatedDate', or 'submittedDate'.",
                        "enum": ["relevance", "lastUpdatedDate", "submittedDate"]
                    },
                    "sort_order": {
                        "type": "string",
                        "description": "Sort order: 'descending' (default) or 'ascending'.",
                        "enum": ["descending", "ascending"]
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by ArXiv category (e.g., 'cs.AI' for computer science AI, 'math.CO' for mathematics combinatorics)."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of papers to return. Default is 50, minimum should be more than 50. If the result is not enough, increase each time by 50->100."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date filter in 'YYYY-MM-DD' format (UTC timezone). Must be provided with end_date."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date filter in 'YYYY-MM-DD' format (UTC timezone). Must be provided with start_date and at least 1 day after start_date."
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": "Whether to extract text content from downloaded PDFs. Default is true."
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'markdown' for readable formatting, 'json' for structured data, or 'text' for plain text. Default is 'json'.",
                        "enum": ["markdown", "json", "text"]
                    }
                },
                "required": ["operation"],
                "additionalProperties": False,
            },
            strict=False,
            category="research",
            tags=["arxiv", "academic_papers", "research_search", "paper_download", "scientific_literature"],
            limitation = "Do NOT use for general web search, news articles, or non-academic content",
            agent_type="Search-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(operation='search', query='machine learning')",
                "description": "Search for machine learning papers"
            }
        )
        try:
            self.workspace = Path(os.getcwd())
            self.supported_extensions = {".pdf"}
            # ArXiv client configuration
            self.client = arxiv.Client(
                page_size=100,
                delay_seconds=3.0,
                num_retries=3,
            )
            # Create downloads directory
            self._downloads_dir = self.workspace / "arxiv_downloads"
        except Exception as e:
            print(f"An error occurred: {e}: {traceback.format_exc()}")
            
    def run(
        self,
        operation: str = Field(default="search_papers", description="Operation type: 'search_papers', 'get_paper_details', 'download_paper', 'get_categories', 'get_capabilities'"),
        query: str = Field(default=None, description="Search query (keywords, title, etc.)"),
        author: str = Field(default=None, description="Author name(s) to search for. Use format 'Lastname, Firstname' or 'Lastname, F' for best results. Multiple authors separated by semicolon"),
        paper_id: str = Field(default=None, description="ArXiv paper ID (e.g., '2301.07041' or 'arxiv:2301.07041')"),
        sort_by: str = Field(default="relevance", description="Sort by: 'relevance', 'lastUpdatedDate', 'submittedDate'"),
        sort_order: str = Field(default="descending", description="Sort order: 'ascending' or 'descending'"),
        category: str = Field(default=None, description="Filter by ArXiv category (e.g., 'cs.AI', 'math.CO')"),
        start_date: str = Field(default=None, description="REQUIRED for date filtering. Start date filter in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' (UTC timezone). Must be provided with end_date."),
        end_date: str = Field(default=None, description="REQUIRED for date filtering. End date filter in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' (UTC timezone). Must be provided with start_date and must be at least 1 day after start_date."),
        extract_text: bool = Field(default=True, description="Whether to extract text content from PDF"),
        max_results: int = Field(default=50, description="Maximum number of papers to return (default: 50)"),
        output_format: str = Field(default="json", description="Output format: 'markdown', 'json', or 'text'"),
    ):
        """Unified ArXiv paper management interface.

        This tool provides comprehensive ArXiv paper operations through a single interface:
        - Paper search with flexible criteria (default operation)
        - PDF download and text extraction

        Args:
            query: Search query (required for search_papers operation)
            paper_id: ArXiv paper ID (required for get_paper_details and download_paper operations)
            operation: The operation to perform (default: search_papers)
            sort_by: Sort criteria for search results
            sort_order: Sort order for search results
            category: Category filter for search
            extract_text: Whether to extract text from PDF
            output_format: Format for the response output

        Returns:
            ActionResponse with operation results and metadata
        """
        # Handle FieldInfo objects
        if isinstance(query, FieldInfo):
            query = query.default
        if isinstance(author, FieldInfo):
            author = author.default
        if isinstance(paper_id, FieldInfo):
            paper_id = paper_id.default
        if isinstance(operation, FieldInfo):
            operation = operation.default
        if isinstance(sort_by, FieldInfo):
            sort_by = sort_by.default
        if isinstance(sort_order, FieldInfo):
            sort_order = sort_order.default
        if isinstance(category, FieldInfo):
            category = category.default
        if isinstance(max_results, FieldInfo):
            max_results = max_results.default
        if isinstance(start_date, FieldInfo):
            start_date = start_date.default
        if isinstance(end_date, FieldInfo):
            end_date = end_date.default
        if isinstance(extract_text, FieldInfo):
            extract_text = extract_text.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default
        
        try:
            # Validate required parameters for each operation
            if operation == "search" or operation == "search_papers":
                if not query and not author:
                    return {"error": "Error: Either 'query' or 'author' parameter is REQUIRED for 'search' operation. Examples: tool.run(operation='search', query='machine learning') or tool.run(operation='search', author='Hawking, S')",
                            "metadata": ArxivMetadata(
                                operation=operation,
                                query=query,
                                total_results=0,
                                error_type="missing_required_parameter",
                            ).model_dump(),
                            "success": False,
                        }
                return self.mcp_search_papers(query, author, sort_by, sort_order, category, max_results, output_format, start_date, end_date)
            
            elif operation == "download" or operation == "download_paper":
                if not paper_id:
                    return {"error": "Error: 'paper_id' parameter is REQUIRED for 'download' operation. Example: tool.run(operation='download', paper_id='2301.07041')",
                            "metadata": ArxivMetadata(
                                operation=operation,
                                paper_id=paper_id,
                                total_results=0,
                                error_type="missing_required_parameter",
                            ).model_dump(),
                            "success": False,
                        }
                return self.mcp_download_paper(paper_id, extract_text, output_format)

            else:
                return {"error": f"Error: Unknown operation '{operation}'. Supported operations: 'search'/'search_papers' (requires query), 'read'/'get_paper_details' (requires paper_id), 'download'/'download_paper' (requires paper_id), 'categories'/'get_categories' (no params), 'capabilities'/'get_capabilities' (no params)",
                        "metadata": ArxivMetadata(
                            operation=operation,
                            query=query,
                            max_results=max_results,
                            sort_by=sort_by,
                            sort_order=sort_order,
                            error_type="unknown_operation",
                            total_results=0,
                        ).model_dump(),
                        "success": False,
                    }
                
        except Exception as e:
            error_msg = f"Failed to execute {operation} operation: {str(e)}"
            tb_str = traceback.format_exc()
            print(f"ArXiv operation error: {tb_str}")

            return {
                "error": error_msg,
                "traceback": tb_str,
                "metadata": ArxivMetadata(
                    operation=operation,
                    query=query,
                    max_results=max_results,
                    sort_by=sort_by,
                    sort_order=sort_order,
                    error_type=e,
                    total_results=0,
                ).model_dump(),
                "success": False
            }


    def _format_paper_result(self, paper: arxiv.Result) :
        """Convert arxiv.Result to structured PaperResult.

        Args:
            paper: ArXiv paper result object

        Returns:
            Structured PaperResult object
        """
        return PaperResult(
            entry_id=paper.entry_id,
            title=paper.title.strip(),
            authors=[author.name for author in paper.authors],
            summary=paper.summary.strip(),
            published=paper.published.isoformat(),
            updated=paper.updated.isoformat() if paper.updated else None,
            categories=paper.categories,
            primary_category=paper.primary_category,
            pdf_url=paper.pdf_url,
            doi=paper.doi,
            journal_ref=paper.journal_ref,
            comment=paper.comment,
        )

    def _format_search_results(self, results: list[PaperResult], output_format: str = "markdown") -> str:
        """Format paper search results for LLM consumption.

        Args:
            results: List of paper results
            output_format: Format type ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if not results:
            return ""

        elif output_format == "json":
            return json.dumps([result.model_dump() for result in results], indent=2)

        elif output_format == "dict":
            return [result.model_dump() for result in results][0]
        
        elif output_format == "text":
            output_parts = [f"Found {len(results)} papers:\n"]

            for i, paper in enumerate(results, 1):
                authors_str = ", ".join(paper.authors[:3])
                if len(paper.authors) > 3:
                    authors_str += f" et al. ({len(paper.authors)} total)"

                output_parts.extend(
                    [
                        f"Title: {paper.title}",
                        f"Authors: {', '.join(paper.authors)}",
                        f"Published: {paper.published}",
                        f"Updated: {paper.updated or 'N/A'}",
                        f"Primary Category: {paper.primary_category}",
                        f"All Categories: {', '.join(paper.categories)}",
                        f"ArXiv ID: {paper.entry_id.split('/')[-1]}",
                        f"PDF URL: {paper.pdf_url or 'N/A'}",
                        f"DOI: {paper.doi or 'N/A'}",
                        f"Journal Reference: {paper.journal_ref or 'N/A'}",
                        f"Comment: {paper.comment or 'N/A'}",
                        "",
                        "Abstract:",
                        paper.summary,
                    ]
                )
            return "\n".join(output_parts)

        else:  # markdown (default)
            output_parts = [f"# ArXiv Search Results\n\nFound **{len(results)}** papers:\n"]
            for i, paper in enumerate(results, 1):
                authors_str = ", ".join(paper.authors[:3])
                if len(paper.authors) > 3:
                    authors_str += f" *et al.* ({len(paper.authors)} total)"

                arxiv_id = paper.entry_id.split("/")[-1]

                output_parts.extend(
                    [
                        f"# {paper.title}",
                        "",
                        f"**Authors:** {', '.join(paper.authors)}",
                        f"**Published:** {paper.published[:10]}",
                        f"**Updated:** {paper.updated[:10] if paper.updated else 'N/A'}",
                        f"**Primary Category:** {paper.primary_category}",
                        f"**All Categories:** {', '.join(paper.categories)}",
                        f"**ArXiv ID:** `{arxiv_id}`",
                        f"**PDF:** [Download]({paper.pdf_url})" if paper.pdf_url else "**PDF:** N/A",
                        f"**DOI:** {paper.doi}" if paper.doi else "**DOI:** N/A",
                        f"**Journal Reference:** {paper.journal_ref}" if paper.journal_ref else "**Journal Reference:** N/A",
                        f"**Comment:** {paper.comment}" if paper.comment else "**Comment:** N/A",
                        "",
                        "## Abstract",
                        "",
                        paper.summary,
                    ]
                )
            return "\n".join(output_parts)

    def mcp_search_papers(
        self,
        query: str = Field(default=None, description="Search query (keywords, title, etc.)"),
        author: str = Field(default=None, description="Author name(s) to search for. Use format 'Lastname, Firstname' or 'Lastname, F' for best results"),
        sort_by: str = Field(
            default="relevance", description="Sort by: 'relevance', 'lastUpdatedDate', 'submittedDate'"
        ),
        sort_order: str = Field(default="descending", description="Sort order: 'ascending' or 'descending'"),
        category: str | None = Field(default=None, description="Filter by ArXiv category (e.g., 'cs.AI', 'math.CO')"),
        max_results: int = Field(default=50, description="Maximum number of papers to return (default: 50)"),
        output_format: str = Field(default="json", description="Output format: 'markdown', 'json', or 'text'"),
        start_date: str = Field(default=None, description="REQUIRED for date filtering. Start date filter in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' (UTC timezone). Must be provided with end_date."),
        end_date: str = Field(default=None, description="REQUIRED for date filtering. End date filter in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' (UTC timezone). Must be provided with start_date and must be at least 1 day after start_date."),
    ) :
        """Search ArXiv papers with flexible criteria.

        This tool provides comprehensive ArXiv paper search with:
        - Keyword, title, and author search capabilities
        - Category filtering for specific subject areas
        - Flexible sorting options (relevance, date)
        - Configurable result limits
        - LLM-optimized result formatting

        Args:
            query: Search terms (can include keywords, titles, author names)
            sort_by: Sorting criteria for results
            sort_order: Order of sorting (ascending/descending)
            category: Optional category filter (e.g., 'cs.AI' for AI papers)
            output_format: Format for the response output

        Returns:
            ActionResponse with search results and metadata
        """
        # Handle FieldInfo objects
        if isinstance(query, FieldInfo):
            query = query.default
        if isinstance(author, FieldInfo):
            author = author.default
        if isinstance(sort_by, FieldInfo):
            sort_by = sort_by.default
        if isinstance(sort_order, FieldInfo):
            sort_order = sort_order.default
        if isinstance(category, FieldInfo):
            category = category.default
        if isinstance(max_results, FieldInfo):
            max_results = max_results.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default
        if isinstance(start_date, FieldInfo):
            start_date = start_date.default
        if isinstance(end_date, FieldInfo):
            end_date = end_date.default
        if max_results < 50:
            max_results = 50
        start_time = datetime.now()
        try:
            # print(f"🔍 Searching ArXiv for: {query}")

            # Build search query
            search_query = query
            
            # Handle date filtering if provided
            if start_date or end_date:
                from datetime import timezone
                
                def to_utc_str(dt):
                    return dt.strftime("%Y%m%d%H%M")
                
                def parse_date(date_str):
                    """Parse date string in either 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' format"""
                    if len(date_str) == 10:  # YYYY-MM-DD format
                        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    else:  # YYYY-MM-DD HH:MM format
                        return datetime.strptime(date_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                
                try:
                    # Parse start and end dates
                    if start_date and end_date:
                        start_dt = parse_date(start_date)
                        end_dt = parse_date(end_date)
                    elif start_date:
                        start_dt = parse_date(start_date)
                        end_dt = datetime.now()
                    elif end_date:
                        end_dt = parse_date(end_date)
                        from datetime import timedelta
                        years_delta = 20
                        start_dt = end_dt.replace(hour=0, minute=0, second=0) - timedelta(days=365 * years_delta)
                    if start_date == end_date:
                        start_dt = start_dt.replace(hour=0, minute=0, second=0)
                        end_dt = start_dt.replace(hour=23, minute=59, second=59)
                    start_str = to_utc_str(start_dt)
                    end_str = to_utc_str(end_dt)
                    
                    # Create date range filter
                    date_range = f"submittedDate:[{start_str} TO {end_str}]"
                    
                    # Combine with base query
                    query_parts = []
                    
                    # Add date range
                    query_parts.append(date_range)
                    
                    # Add category if provided
                    if category:
                        query_parts.append(f'cat:{category}')
                    
                    # Add query if provided
                    if query:
                        query_parts.append(f'all:"{query}"')
                    
                    # Add author if provided
                    if author:
                        query_parts.append(f'au:{author}')
                    
                    search_query = " AND ".join(query_parts)
                        
                except ValueError as e:
                    invalid_date = start_date if start_date else end_date
                    tb_str = traceback.format_exc()
                    print(f"Date parsing error: {tb_str}")
                    return {"error": f"Error: Invalid date format '{invalid_date}'. Expected format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' (UTC timezone)",
                            "traceback": tb_str,
                            "metadata": ArxivMetadata(
                                operation="search_papers",
                                error_type=type(e).__name__,
                                total_results=0,
                            ).model_dump(),
                            "success": False
                        }
            else:
                # No date filter, use improved logic
                query_parts = []
                
                # Add query if provided
                if query:
                    query_parts.append(f'all:"{query}"')
                
                # Add author if provided - use correct ArXiv syntax
                if author:
                    # Use ArXiv's correct author search syntax (without quotes around the field)
                    query_parts.append(f'au:{author}')
                
                # Combine query parts
                if query_parts:
                    combined_query = " AND ".join(query_parts)
                else:
                    combined_query = "all:*"  # Search all if no specific query
                
                # Add category filter if provided
                if category:
                    search_query = f'cat:"{category}" AND ({combined_query})'
                else:
                    search_query = combined_query

            # Configure sort criteria
            sort_criterion = arxiv.SortCriterion.Relevance
            if sort_by == "lastUpdatedDate":
                sort_criterion = arxiv.SortCriterion.LastUpdatedDate
            elif sort_by == "submittedDate":
                sort_criterion = arxiv.SortCriterion.SubmittedDate

            sort_order_enum = arxiv.SortOrder.Descending
            if sort_order == "ascending":
                sort_order_enum = arxiv.SortOrder.Ascending
            search = arxiv.Search(
                query=search_query, max_results=max_results, sort_by=sort_criterion, sort_order=sort_order_enum
            )
            # Execute search and collect results
            results = []
            try:
                for paper in self.client.results(search):
                    results.append(self._format_paper_result(paper))
                    # Continue until we hit the specified limit or natural end
            except Exception as search_error:
                print(f"Search error (continuing with available results): {traceback.format_exc()}")

            execution_time = (datetime.now() - start_time).total_seconds()
            metadata = ArxivMetadata(
                operation="search_papers",
                query=query,
                sort_by=sort_by,
                sort_order=sort_order,
                total_results=len(results),
                execution_time=execution_time,
            )
            # Format output
            formatted_output = self._format_search_results(results, output_format)
            if formatted_output == "":
                formatted_output = {"error": "No papers found matching the search criteria.", "metadata": metadata.model_dump(), "success": False}
            elif isinstance(formatted_output, dict):
                formatted_output['success'] = True
                formatted_output['metadata'] = metadata.model_dump()
            else:
                formatted_output = {"result": formatted_output, "metadata": metadata.model_dump(), "success": True}
            return formatted_output
            
        except Exception as e:
            error_msg = f"Failed to search ArXiv papers: {str(e)}"
            tb_str = traceback.format_exc()
            print(f"ArXiv search error: {tb_str}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "error": error_msg,
                "traceback": tb_str,
                "metadata": ArxivMetadata(
                    operation="search_papers",
                    error_type=e,
                    query=query,
                    max_results=max_results,
                    sort_by=sort_by,
                    sort_order=sort_order,
                    total_results=0,
                    execution_time=execution_time,
                ).model_dump(),
                "success": False
            }

    def mcp_download_paper(
        self,
        paper_id: str = Field(description="ArXiv paper ID (e.g., '2301.07041' or 'arxiv:2301.07041')"),
        extract_text: bool = Field(default=True, description="Whether to extract text content from PDF"),
        output_format: str = Field(default="json", description="Output format: 'markdown', 'json', or 'text'"),
    ) :
        """Download ArXiv paper PDF and optionally extract text content.

        Args:
            paper_id: ArXiv paper identifier
            extract_text: Whether to extract and return text content
            output_format: Format for the response output

        Returns:
            ActionResponse with download status and optional text content
        """
        # Handle FieldInfo objects
        if isinstance(paper_id, FieldInfo):
            paper_id = paper_id.default
        if isinstance(extract_text, FieldInfo):
            extract_text = extract_text.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default
        self._downloads_dir.mkdir(exist_ok=True)

        try:
            # Clean paper ID
            clean_id = paper_id.replace("arxiv:", "").strip()

            print(f"📥 Downloading paper: {clean_id}")

            start_time = datetime.now()

            # Search for the paper
            search = arxiv.Search(id_list=[clean_id])
            paper = next(self.client.results(search), None)

            if not paper:
                return {
                    "error": f"Paper not found: {clean_id}",
                    "metadata": ArxivMetadata(
                        operation="download_paper",
                        paper_id=clean_id,
                        error_type="paper_not_found",
                    ).model_dump(),
                    "success": False
                }

            # Download PDF
            filename = f"{clean_id.replace('/', '_')}.pdf"
            download_path = self._downloads_dir / filename
            print(f"Downloading paper to: {download_path}")
            paper.download_pdf(dirpath=str(self._downloads_dir), filename=filename)

            execution_time = (datetime.now() - start_time).total_seconds()
            file_size = download_path.stat().st_size if download_path.exists() else 0

            # Prepare response message
            if output_format == "json":
                response_data = {
                    "paper_id": clean_id,
                    "title": paper.title,
                    "download_path": str(download_path),
                    "file_size": file_size,
                    "download_time": execution_time,
                }

                if extract_text:
                    try:
                        # Basic text extraction (would need additional libraries like PyPDF2 or pdfplumber)
                        response_data["text_extraction"] = (
                            "Text extraction requires additional PDF processing libraries"
                        )
                    except Exception:
                        response_data["text_extraction"] = "Text extraction failed"

                formatted_output = json.dumps(response_data, indent=2)

            elif output_format == "text":
                output_parts = [
                    "Paper Downloaded Successfully",
                    f"Paper ID: {clean_id}",
                    f"Title: {paper.title}",
                    f"Download Path: {download_path}",
                    f"File Size: {file_size:,} bytes",
                    f"Download Time: {execution_time:.2f} seconds",
                ]

                if extract_text:
                    output_parts.append("\nNote: Text extraction requires additional PDF processing libraries")

                formatted_output = "\n".join(output_parts)

            else:  # markdown (default)
                output_parts = [
                    "# 📥 Paper Download Complete",
                    "",
                    f"**Paper ID:** `{clean_id}`",
                    f"**Title:** {paper.title}",
                    f"**Download Path:** `{download_path}`",
                    f"**File Size:** {file_size:,} bytes",
                    f"**Download Time:** {execution_time:.2f} seconds",
                ]

                if extract_text:
                    output_parts.extend(
                        [
                            "",
                            "## 📄 Text Extraction",
                            (
                                "*Note: Text extraction requires additional "
                                "PDF processing libraries like PyPDF2 or pdfplumber*"
                            ),
                        ]
                    )

                formatted_output = "\n".join(output_parts)

            # Create metadata
            metadata = ArxivMetadata(
                operation="download_paper",
                paper_id=clean_id,
                download_path=str(download_path),
                file_size=file_size,
                execution_time=execution_time,
            )

            print(f"✅ Downloaded paper in {execution_time:.2f}s ({file_size:,} bytes)")

            return {
                "result": formatted_output,
                "success": True,
                "metadata": metadata.model_dump(),
            }

        except Exception as e:
            error_msg = f"Failed to download paper: {str(e)}"
            tb_str = traceback.format_exc()
            print(f"ArXiv download error: {tb_str}")
            return {
                "error": error_msg,
                "traceback": tb_str,
                "metadata": ArxivMetadata(
                    operation="download_paper",
                    error_type=type(e).__name__,
                    paper_id=clean_id if 'clean_id' in locals() else paper_id,
                ).model_dump(),
                "success": False
            }

    def test(self, tool_test: str="arxiv_paper_search", file_location: str="arxiv_paper_search", result_parameter: str="title", search_type: str="exact_match"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)

if __name__ == "__main__":
    try:
        tool = Arxiv_Paper_Search_Tool()
        tool.embed_tool()
        tool.test(tool_test='arxiv_paper_search',file_location='arxiv_paper_search',result_parameter='title', search_type='exact_match')
    except Exception as e:  
        print(f"An error occurred: {e}")
        print(f"Traceback: {traceback.format_exc()}")