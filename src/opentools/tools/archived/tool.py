# source code: https://github.com/inclusionAI/AWorld/blob/main/examples/gaia/mcp_collections/tools/wayback.py
import json, os, sys, time, traceback, json, requests
from datetime import datetime
from bs4 import BeautifulSoup
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from waybackpy import WaybackMachineCDXServerAPI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
class ArchivedVersion(BaseModel):
    """Individual archived version with structured data."""
    timestamp: str
    url: str
    status_code: str
    digest: str
    length: str
    mime_type: str

class WaybackMetadata(BaseModel):
    """Metadata for Wayback Machine operation results."""
    url: str
    operation: str
    timestamp: str | None = None
    total_versions: int | None = None
    date_range: dict[str, str | None] | None = None
    content_length: int | None = None
    text_extracted: bool = False
    truncated: bool = False
    execution_time: float | None = None
    error_type: str | None = None
    user_agent: str = "AWorld/1.0 (https://github.com/inclusionAI/AWorld; qintong.wqt@antgroup.com)"

class Archived_Tool(BaseTool):
    """
    Archived_Tool: Comprehensive Access to the Internet Archive's Wayback Machine

    Purpose:
        Enables users to retrieve and analyze historical snapshots of any publicly accessible website by leveraging the Internet Archive’s Wayback Machine. Ideal for viewing deleted content, investigating how web pages have evolved, digital forensics, or recovering information lost from the web.

    Main Capabilities:
        - List all archived versions (snapshots) of a given website, with powerful filtering options by date range and snapshot count.
        - Retrieve the full or text-only content of a website exactly as it appeared at a specific historical timestamp.
        - Supports text extraction for clean reading, content truncation for large pages, and customizable output formats (Markdown, JSON, plain text).

    Functionality:
        • Works with any public URL, automatically handling protocol normalization.
        • Returns rich metadata about archives, including exact dates, status, content size, and more.
        • Flexible for research, digital history projects, compliance/audit, and LLM-based downstream workflows.

    Limitations:
        - Requires internet connection for Wayback Machine API access
        - Depends on Wayback Machine's availability and rate limits
        - Some websites may not have archived versions
        - Some websites may have limited archived versions
        - Some websites may have auto-generated archived versions only
    """
    def __init__(self):
        super().__init__(
            type='function',
            name="Archived_Tool",
            description="""Accesses the Internet Archive's Wayback Machine to retrieve historical versions of websites. Lists archived snapshots with date filtering and fetches content from specific timestamps. Use when you need to see how websites looked in the past or access deleted content. CAPABILITIES: Lists archived versions with date filtering, fetches content from specific timestamps, extracts content, truncates long content. SYNONYMS: wayback machine, internet archive, archived website, historical website, website snapshot, deleted content recovery, website history. EXAMPLES: 'Show me archived versions of google.com from 2000', 'Get the content of facebook.com from 2004', 'List snapshots of twitter.com between 2010-2015'.""",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform: 'list_versions' to find archived snapshots, or 'get_content' to retrieve content from a specific timestamp.",
                        "enum": ["list_versions", "get_content"]
                    },
                    "url": {
                        "type": "string",
                        "description": "The website URL to operate on (e.g., 'example.com' or 'https://example.com'). Will automatically add https:// if needed."
                    },
                    "timestamp": {
                        "type": "string",
                        "description": "Required for get_content operation. Timestamp in YYYYMMDDhhmmss format (e.g., '20230101' for January 1, 2023). Must be provided when operation is 'get_content'."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of archived versions to return for list_versions operation. Use 0 for all versions, default is 10."
                    },
                    "from_date": {
                        "type": "string",
                        "description": "Start date filter for list_versions in YYYYMMDDhhmmss format. Only show versions from this date onwards."
                    },
                    "to_date": {
                        "type": "string",
                        "description": "End date filter for list_versions in YYYYMMDDhhmmss format. Only show versions up to this date."
                    },
                    "extract_text_only": {
                        "type": "boolean",
                        "description": "For get_content operation. Extract only text content, removing HTML tags for cleaner reading. Default is true."
                    },
                    "truncate_content": {
                        "type": "boolean",
                        "description": "For get_content operation. Truncate content to manageable length for LLM consumption. Default is false."
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'markdown' for readable formatting, 'json' for structured data, or 'text' for plain text. Default is 'json'."
                    }
                },
                "required": ["operation", "url"],
                "additionalProperties": False,
            },
            strict=False,
            category="web",
            tags=["wayback_machine", "website_archives", "historical_snapshots", "website_history"],
            limitation = "Do NOT use for current website analysis, web scraping, or general internet browsing",
            agent_type="Search-Agent",
            demo_commands= {
                "command": "reponse = tool.run(operation='list_versions', url='example.com')",
                "description": "List archived versions of example.com"
            }
        )
        self.output = "dict - A dictionary containing operation results, metadata, and formatted output based on the requested operation and format."
        self.workspace = Path(os.getcwd())
        self.user_agent = "AWorld/1.0 (https://github.com/inclusionAI/AWorld; qintong.wqt@antgroup.com)"
        self.default_timeout = 30
        self.max_content_length = 8192

    def _format_versions_for_llm(self, versions: list[ArchivedVersion], query_info: dict) -> str:
        """Format archived versions list for LLM consumption.

        Args:
            versions: List of archived versions
            query_info: Query information including URL and filters

        Returns:
            Formatted string suitable for LLM consumption
        """
        if not versions:
            return f"No archived versions found for URL: {query_info.get('url', 'Unknown')}"

        output_parts = [
            f"# Wayback Machine Archives for {query_info.get('url', 'Unknown')}",
            f"\nFound **{len(versions)}** archived versions",
        ]

        if query_info.get("from_date") or query_info.get("to_date"):
            date_filter = []
            if query_info.get("from_date"):
                date_filter.append(f"From: {query_info['from_date']}")
            if query_info.get("to_date"):
                date_filter.append(f"To: {query_info['to_date']}")
            output_parts.append(f"\n**Date Filter:** {' | '.join(date_filter)}")

        output_parts.append("\n## Available Versions:")

        for i, version in enumerate(versions[:10], 1):  # Show first 10
            timestamp_formatted = self._format_timestamp(version.timestamp)
            output_parts.append(
                f"\n{i}. **{timestamp_formatted}**\n"
                f"   - Archive URL: {version.url}\n"
                f"   - Status: {version.status_code} | Size: {version.length} bytes\n"
                f"   - Type: {version.mime_type}"
            )

        if len(versions) > 10:
            output_parts.append(f"\n... and {len(versions) - 10} more versions")

        return "\n".join(output_parts)

    def _format_content_for_llm(self, content_data: dict, output_format: str = "markdown") -> str:
        """Format archived content for LLM consumption.

        Args:
            content_data: Content data dictionary
            output_format: Format type ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if output_format == "json":
            return json.dumps(content_data, indent=2)

        elif output_format == "text":
            return content_data.get("content", "")

        else:  # markdown (default)
            output_parts = [
                f"# Archived Content from {content_data.get('url', 'Unknown')}",
                f"\n**Requested Timestamp:** {content_data.get('timestamp', 'Unknown')}",
                f"**Actual Timestamp:** {self._format_timestamp(content_data.get('fetched_timestamp', ''))}",
                f"**Content Length:** {content_data.get('original_content_length', 0):,} characters",
            ]

            if content_data.get("truncated"):
                output_parts.append(f"**Note:** Content truncated to {self.max_content_length:,} characters")

            if content_data.get("extract_text_only"):
                output_parts.append("**Note:** Text-only extraction applied")

            output_parts.extend(["\n## Content:", "\n---\n", content_data.get("content", ""), "\n---"])

            return "\n".join(output_parts)

    def _format_timestamp(self, timestamp: str) -> str:
        """Format Wayback Machine timestamp to human-readable format.

        Args:
            timestamp: Wayback timestamp (YYYYMMDDhhmmss)

        Returns:
            Human-readable timestamp
        """
        try:
            if len(timestamp) >= 14:
                dt = datetime.strptime(timestamp[:14], "%Y%m%d%H%M%S")
                return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            return timestamp
        except (ValueError, TypeError):
            return timestamp or "Unknown"

    def _validate_wayback_parameters(self, url: str, timestamp: str = None) -> tuple[str, str | None]:
        """Validate and normalize Wayback Machine parameters.

        Args:
            url: URL to validate
            timestamp: Optional timestamp to validate

        Returns:
            Tuple of (validated_url, validated_timestamp)

        Raises:
            ValueError: If parameters are invalid
        """
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        validated_timestamp = None
        if timestamp:
            timestamp = timestamp.strip()
            if len(timestamp) < 8:
                raise ValueError("Timestamp must be at least 8 characters (YYYYMMDD)")
            validated_timestamp = timestamp

        return url, validated_timestamp

    def run(
        self,
        operation: str = Field(description="The operation to perform: 'list_versions' or 'get_content'"),
        url: str = Field(description="The URL to operate on"),
        timestamp: str | None = Field(default=None, description="Timestamp for get_content operation (YYYYMMDDhhmmss)"),
        limit: int = Field(default=10, description="Maximum number of versions to return for list_versions (0 for all)"),
        from_date: str | None = Field(default=None, description="Start date filter for list_versions (YYYYMMDDhhmmss)"),
        to_date: str | None = Field(default=None, description="End date filter for list_versions (YYYYMMDDhhmmss)"),
        extract_text_only: bool = Field(default=True, description="Extract only text content for get_content"),
        truncate_content: bool = Field(default=False, description="Truncate content for get_content"),
        output_format: str = Field(default="json", description="Output format: 'markdown', 'json', or 'text'"),
    ):
        """Unified function for Wayback Machine operations.

        This function handles both listing archived versions and fetching archived content
        based on the operation parameter.

        Args:
            operation: The operation to perform ('list_versions' or 'get_content')
            url: The URL to operate on
            timestamp: Timestamp for get_content operation (YYYYMMDDhhmmss)
            limit: Maximum number of versions to return for list_versions
            from_date: Start date filter for list_versions (YYYYMMDDhhmmss)
            to_date: End date filter for list_versions (YYYYMMDDhhmmss)
            extract_text_only: Extract only text content for get_content
            truncate_content: Truncate content for get_content
            output_format: Format for the response ('markdown', 'json', or 'text')

        Returns:
            Dictionary with operation results and metadata
        """
        # Handle FieldInfo objects
        if isinstance(operation, FieldInfo):
            operation = operation.default
        if isinstance(url, FieldInfo):
            url = url.default
        if isinstance(timestamp, FieldInfo):
            timestamp = timestamp.default
        if isinstance(limit, FieldInfo):
            limit = limit.default
        if isinstance(from_date, FieldInfo):
            from_date = from_date.default
        if isinstance(to_date, FieldInfo):
            to_date = to_date.default
        if isinstance(extract_text_only, FieldInfo):
            extract_text_only = extract_text_only.default
        if isinstance(truncate_content, FieldInfo):
            truncate_content = truncate_content.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default

        start_time = time.time()

        try:
            # Validate parameters
            url, validated_timestamp = self._validate_wayback_parameters(url, timestamp)

            if operation == "list_versions":
                return self._handle_list_versions(url, limit, from_date, to_date, output_format, start_time)
            elif operation == "get_content":
                if not validated_timestamp:
                    return {
                        "error": "Error: 'timestamp' parameter is REQUIRED for 'get_content' operation. Example: tool.run(operation='get_content', url='example.com', timestamp='20230101')",
                        "metadata": WaybackMetadata(
                            url=url,
                            operation="get_content",
                            execution_time=time.time() - start_time,
                            error_type="missing_timestamp",
                        ).model_dump(),
                        "success": False,
                    }
                return self._handle_get_content(url, validated_timestamp, extract_text_only, truncate_content, output_format, start_time)
            else:
                return {
                    "error": f"Error: Unknown operation '{operation}'. Supported operations: 'list_versions', 'get_content'",
                    "metadata": WaybackMetadata(
                        url=url,
                        operation=operation,
                        execution_time=time.time() - start_time,
                        error_type="unknown_operation",
                    ).model_dump(),
                    "success": False,
                }

        except Exception as e:
            error_msg = f"Wayback Machine operation failed: {str(e)}"
            print(f"Error in run: {traceback.format_exc()}")

            return {
                "error": error_msg,
                "metadata": WaybackMetadata(
                    url=url or "unknown",
                    operation=operation or "unknown",
                    execution_time=time.time() - start_time,
                    error_type=type(e).__name__,
                ).model_dump(),
                "success": False,
                "traceback": traceback.format_exc()
            }

    def _handle_list_versions(self, url: str, limit: int, from_date: str | None, to_date: str | None, output_format: str, start_time: float):
        """Handle list_versions operation."""
        print(f"Listing archived versions for: {url}")

        try:
            # Test network connectivity first
            test_response = requests.get("https://web.archive.org", timeout=5)
            if test_response.status_code != 200:
                return {
                    "error": "Network connectivity issue: Wayback Machine service is not accessible. Please check your internet connection and try again.",
                    "metadata": WaybackMetadata(
                        url=url,
                        operation="list_versions",
                        execution_time=time.time() - start_time,
                        error_type="network_unavailable",
                    ).model_dump(),
                    'success': False,
                }
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Network connectivity issue: Cannot reach Wayback Machine service. Error: {str(e)}. Please check your internet connection and firewall settings.",
                "metadata": WaybackMetadata(
                    url=url,
                    operation="list_versions",
                    execution_time=time.time() - start_time,
                    error_type="network_error",
                ).model_dump(),
                'success': False,
                "traceback": traceback.format_exc()
            }

        # Query Wayback Machine CDX API
        try:
            cdx_api = WaybackMachineCDXServerAPI(url, user_agent=self.user_agent)
            all_snapshots = list(cdx_api.snapshots())
        except requests.exceptions.ConnectionError as e:
            return {
                "error": f"Network connectivity issue: Cannot reach Wayback Machine CDX API. Error: {str(e)}. Please check your internet connection and firewall settings.",
                "metadata": WaybackMetadata(
                    url=url,
                    operation="list_versions",
                    execution_time=time.time() - start_time,
                    error_type="cdx_connection_error",
                ).model_dump(),
                'success': False,
                "traceback": traceback.format_exc()
            }

        # Apply date filtering
        if from_date or to_date:
            snapshots = [
                s
                for s in all_snapshots
                if (not from_date or s.timestamp >= from_date) and (not to_date or s.timestamp <= to_date)
            ]
        else:
            snapshots = all_snapshots

        if not snapshots:
            return {
                "error": "No archived versions found for the specified URL and date range.",
                "metadata": WaybackMetadata(
                    url=url,
                    operation="list_versions",
                    total_versions=0,
                    date_range={"from_date": from_date, "to_date": to_date},
                    execution_time=time.time() - start_time,
                    error_type="no_results",
                ).model_dump(),
                'success': False
            }

        # Convert to structured format
        versions = [
            ArchivedVersion(
                timestamp=snapshot.timestamp,
                url=snapshot.archive_url,
                status_code=snapshot.statuscode,
                digest=snapshot.digest,
                length=snapshot.length,
                mime_type=snapshot.mimetype,
            )
            for snapshot in snapshots
        ]

        # Apply limit
        if limit > 0 and len(versions) > limit:
            versions = versions[:limit]

        # Format output
        query_info = {"url": url, "from_date": from_date, "to_date": to_date, "total_found": len(snapshots)}

        if output_format == "json":
            message = [version.model_dump() for version in versions]
        else:
            message = self._format_versions_for_llm(versions, query_info)

        execution_time = time.time() - start_time
        print(f"Found {len(versions)} archived versions in {execution_time:.2f}s")

        return {
            "result": message,
            "metadata": WaybackMetadata(
                url=url,
                operation="list_versions",
                total_versions=len(snapshots),
                date_range={"from_date": from_date, "to_date": to_date},
                execution_time=execution_time,
            ).model_dump(),
            'success': True
        }

    def _handle_get_content(self, url: str, timestamp: str, extract_text_only: bool, truncate_content: bool, output_format: str, start_time: float):
        """Handle get_content operation."""
        print(f"Fetching archived content: {url} at {timestamp}")

        try:
            # Test network connectivity first
            test_response = requests.get("https://web.archive.org", timeout=5)
            if test_response.status_code != 200:
                return {
                    "error": "Network connectivity issue: Wayback Machine service is not accessible. Please check your internet connection and try again.",
                    "metadata": WaybackMetadata(
                        url=url,
                        operation="get_content",
                        timestamp=timestamp,
                        execution_time=time.time() - start_time,
                        error_type="network_unavailable",
                    ).model_dump(),
                    'success': False
                }
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Network connectivity issue: Cannot reach Wayback Machine service. Error: {str(e)}. Please check your internet connection and firewall settings.",
                "metadata": WaybackMetadata(
                    url=url,
                    operation="get_content",
                    timestamp=timestamp,
                    execution_time=time.time() - start_time,
                    error_type="network_error",
                ).model_dump(),
                'success': False,
                "traceback": traceback.format_exc()
            }

        # Query Wayback Machine for closest snapshot
        try:
            cdx_api = WaybackMachineCDXServerAPI(url, user_agent=self.user_agent)
            snapshot = cdx_api.near(wayback_machine_timestamp=timestamp)
        except requests.exceptions.ConnectionError as e:
            return {
                "error": f"Network connectivity issue: Cannot reach Wayback Machine CDX API. Error: {str(e)}. Please check your internet connection and firewall settings.",
                "metadata": WaybackMetadata(
                    url=url,
                    operation="get_content",
                    timestamp=timestamp,
                    execution_time=time.time() - start_time,
                    error_type="cdx_connection_error",
                ).model_dump(),
                'success': False,
                "traceback": traceback.format_exc()
            }

        if not snapshot or not snapshot.archive_url:
            return {
                "error": f"No archived version found for {url} at timestamp {timestamp}",
                "metadata": WaybackMetadata(
                    url=url,
                    operation="get_content",
                    timestamp=timestamp,
                    execution_time=time.time() - start_time,
                    error_type="no_snapshot",
                ).model_dump(),
                'success': False
            }

        # Fetch content
        response = requests.get(snapshot.archive_url, timeout=self.default_timeout)
        response.raise_for_status()
        content = response.text
        original_length = len(content)

        # Extract text if requested
        if extract_text_only:
            soup = BeautifulSoup(content, "html.parser")
            content = soup.get_text(separator=" ", strip=True)

        # Truncate if requested
        truncated = False
        if truncate_content and len(content) > self.max_content_length:
            content = content[: self.max_content_length] + "..."
            truncated = True

        # Prepare content data
        content_data = {
            "url": url,
            "timestamp": timestamp,
            "fetched_timestamp": snapshot.timestamp,
            "content": content,
            "original_content_length": original_length,
            "truncated": truncated,
            "extract_text_only": extract_text_only,
        }

        # Format output
        if output_format == "json":
            message = content_data
        elif output_format == "text":
            message = content
        else:  # markdown
            message = self._format_content_for_llm(content_data, output_format)

        execution_time = time.time() - start_time
        print(f"Retrieved {len(content):,} characters in {execution_time:.2f}s")

        return {
            "result": message,
            "metadata": WaybackMetadata(
                url=url,
                operation="get_content",
                timestamp=snapshot.timestamp,
                content_length=len(content),
                text_extracted=extract_text_only,
                truncated=truncated,
                execution_time=execution_time,
            ).model_dump(),
            'success': True
        }

# Default arguments for testing
if __name__ == "__main__":
    try:
        tool = Archived_Tool()
        tool.embed_tool()
        
    except Exception as e:
        print(f"An error occurred: {e}")
