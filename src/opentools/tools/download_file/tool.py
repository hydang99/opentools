# source code: https://github.com/inclusionAI/AWorld/blob/main/examples/gaia/mcp_collections/tools/download.py
import asyncio, json, os, re, sys, time, traceback, shutil, requests
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
from pydantic import Field, BaseModel
from pydantic.fields import FieldInfo

class DownloadResult(BaseModel):
    """Individual download operation result with structured data."""

    url: str
    file_path: str
    success: bool
    file_size: int | None = None
    duration: str
    timestamp: str
    error_message: str | None = None

class DownloadMetadata(BaseModel):
    """Metadata for download operation results."""

    url: str
    output_path: str
    timeout_seconds: int
    overwrite_enabled: bool
    execution_time: float | None = None
    file_size_bytes: int | None = None
    content_type: str | None = None
    status_code: int | None = None
    error_type: str | None = None
    headers_used: bool = False

class Download_File_Tool(BaseTool):
    # Default args for `opentools test Download_File_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "download_file",
        "file_location": "download_file",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }

    """Download_File_Tool
    ---------------------
    Purpose:
        A comprehensive file download tool that downloads files from HTTP/HTTPS URLs with support for configurable timeouts, path validation, and friendly output in multiple formats (markdown, JSON, text). Provides detailed metadata about download operations, file sizes, execution times, and error handling. Supports file size limits, overwrite protection.

    Core Capabilities:
        - HTTP/HTTPS file downloads
        - Path validation and directory creation
        - File size limits (1GB max)
        - Progress tracking
        - Cross-platform compatibility
        - Async download support

    Intended Use:
        Use this tool when you need to download files from HTTP/HTTPS URLs, including PDF, DOCX, XLSX, PPTX, CSV, ZIP, MP4, etc.

    Limitations:
        - Only supports HTTP and HTTPS URLs
        - Maximum file size limit of 1GB
        - Requires valid URL format with scheme

    """

    def __init__(self) -> None:
        super().__init__(
            type='function',
            name="Download_File_Tool",
            description="""A comprehensive file download tool that downloads files from HTTP/HTTPS URLs with support for configurable timeouts, path validation, and friendly output in multiple formats (markdown, JSON, text). Provides detailed metadata about download operations, file sizes, execution times, and error handling. Supports file size limits, overwrite protection. CAPABILITIES: HTTP/HTTPS file downloads,  path validation and directory creation, file size limits (1GB max),  progress tracking, cross-platform compatibility, async download support. SYNONYMS: file downloader, URL downloader, file fetcher, web downloader, HTTP download tool, file retrieval tool, download manager, file grabber, web file downloader, file transfer tool. EXAMPLES: 'Download this PDF from the URL', 'Save this file to downloads folder', 'Download large file with extended timeout', 'Get file from HTTPS URL'.""",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "HTTP/HTTPS URL of the file to download"
                    },
                    "output_file_path": {
                        "type": "string",
                        "description": "Local path where the file should be saved (absolute or relative to workspace)"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Whether to overwrite existing files (default: False)"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Download timeout in seconds (default: 60)"
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'markdown', 'json', or 'text' (default: json)",
                        "enum": ["markdown", "json", "text"]
                    }
                },
                "required": ["url", "output_file_path"],
                "additionalProperties": False,
            },
            strict=False,
            category="file_operations",
            tags=["file_download", "http_download", "url_download", "file_transfer", "web_download", "download_manager", "file_retrieval","file_operations"],
            limitation="Only supports HTTP and HTTPS URLs, maximum file size limit of 1GB, requires valid URL format with scheme, some servers may block automated downloads, large files may timeout on slow connections",
            agent_type="Download_File-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {    
                "command": "reponse = tool.run(url='https://www.google.com', output_file_path='test.txt', overwrite=True, timeout=60, output_format='markdown')",    
                "description": "Download a file from a URL"
            },
        )
        self.workspace = Path(os.getcwd())
        self.default_timeout = 60 * 3  # 3 minutes timeout
        self.max_file_size = 1024 * 1024 * 1024  # 1GB limit
        self.supported_schemes = {"http", "https"}

        # Initialize a session for cookie persistence and better header handling
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/pdf,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            # Avoid brotli ("br") here: depending on the runtime, urllib3/requests may not
            # transparently decode brotli for streamed downloads, which can lead to saving
            # compressed bytes to disk (e.g., .html files that look "garbled/binary").
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "DNT": "1",
        })


    def _validate_url(self, url: str) -> tuple[bool, str | None]:
        """Validate URL format and scheme.

        Args:
            url: URL to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed = urlparse(url)

            if not parsed.scheme:
                return False, "URL must include a scheme (http:// or https://)"

            if parsed.scheme.lower() not in self.supported_schemes:
                return False, f"Unsupported URL scheme: {parsed.scheme}. Supported: {', '.join(self.supported_schemes)}"

            if not parsed.netloc:
                return False, "URL must include a valid domain"

            return True, None

        except Exception as e:
            return False, f"Invalid URL format: {str(e)}"

    def _resolve_output_path(self, output_path: str) :
        """Resolve and validate output file path.

        Args:
            output_path: Output file path (absolute or relative)

        Returns:
            Resolved Path object
        """
        path = Path(output_path).expanduser()

        if not path.is_absolute():
            path = self.workspace / path

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        return path.resolve()

    def _format_download_output(self, result: DownloadResult, output_format: str = "markdown") -> str:
        """Format download results for LLM consumption.

        Args:
            result: Download execution result
            output_format: Format type ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if output_format == "json":
            return json.dumps(result.model_dump(), indent=2)

        elif output_format == "text":
            output_parts = [
                f"URL: {result.url}",
                f"File Path: {result.file_path}",
                f"Status: {'SUCCESS' if result.success else 'FAILED'}",
                f"Duration: {result.duration}",
                f"Timestamp: {result.timestamp}",
            ]

            if result.file_size is not None:
                output_parts.append(f"File Size: {result.file_size:,} bytes")

            if result.error_message:
                output_parts.append(f"Error: {result.error_message}")

            return "\n".join(output_parts)

        else:  # markdown (default)
            status_emoji = "✅" if result.success else "❌"

            output_parts = [
                f"# File Download {status_emoji}",
                f"**URL:** `{result.url}`",
                f"**File Path:** `{result.file_path}`",
                f"**Status:** {'SUCCESS' if result.success else 'FAILED'}",
                f"**Duration:** {result.duration}",
                f"**Timestamp:** {result.timestamp}",
            ]

            if result.file_size is not None:
                size_mb = result.file_size / (1024 * 1024)
                output_parts.append(f"**File Size:** {result.file_size:,} bytes ({size_mb:.2f} MB)")

            if result.error_message:
                output_parts.extend(["\n## Error Details", f"```\n{result.error_message}\n```"])

            return "\n".join(output_parts)

    def _download_file_async(
        self, url: str, output_path: Path, timeout: int, headers: dict[str, str] | None
    ) -> DownloadResult:
        """Download file asynchronously with comprehensive error handling.

        Args:
            url: URL to download from
            output_path: Local path to save file
            timeout: Request timeout in seconds
            headers: Optional custom headers (merged with session headers)

        Returns:
            DownloadResult with execution details
        """
        start_time = datetime.now()

        try:
            print(f"📥 Starting download: {url}")

            # Try to establish session/cookies by visiting domain root first
            try:
                parsed = urlparse(url)
                domain_root = f"{parsed.scheme}://{parsed.netloc}"
                # Make a lightweight HEAD request to establish session (don't follow redirects)
                self.session.head(domain_root, timeout=5, allow_redirects=False)
            except:
                pass  # Ignore errors in pre-request, proceed with main request

            # Merge additional headers if provided
            request_headers = {}
            if headers:
                request_headers.update(headers)
            
            # Add Referer header if we can determine it
            try:
                parsed = urlparse(url)
                request_headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
            except:
                pass

            # Use session for better cookie handling and bot detection evasion
            with self.session.get(url, stream=True, timeout=timeout, headers=request_headers) as response:
                response.raise_for_status()

                # Check content length if available
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self.max_file_size:
                    raise ValueError(f"File too large: {content_length} bytes (max: {self.max_file_size})")

                # Download file
                with open(output_path, "wb") as f:
                    # Ensure decoding of Content-Encoding (gzip/deflate) while streaming.
                    # Using iter_content is safer than copying response.raw directly.
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)

                file_size = output_path.stat().st_size
                duration = str(datetime.now() - start_time)

                print(f"✅ Download completed: {file_size:,} bytes")

                return DownloadResult(
                    url=url,
                    file_path=str(output_path),
                    success=True,
                    file_size=file_size,
                    duration=duration,
                    timestamp=start_time.isoformat(),
                )

        except requests.exceptions.Timeout:
            duration = str(datetime.now() - start_time)
            error_msg = f"Download timed out after {timeout} seconds"
            print(f"⏰ {error_msg}")

            return DownloadResult(
                url=url,
                file_path=str(output_path),
                success=False,
                duration=duration,
                timestamp=start_time.isoformat(),
                error_message=error_msg,
            )

        except requests.exceptions.RequestException as e:
            duration = str(datetime.now() - start_time)
            error_msg = f"Request failed: {str(e)}"
            print(f"❌ {error_msg}")

            return DownloadResult(
                url=url,
                file_path=str(output_path),
                success=False,
                duration=duration,
                timestamp=start_time.isoformat(),
                error_message=error_msg,
            )

        except Exception as e:
            duration = str(datetime.now() - start_time)
            error_msg = f"Unexpected error: {str(e)}"
            print(f"💥 {error_msg}")

            return DownloadResult(
                url=url,
                file_path=str(output_path),
                success=False,
                duration=duration,
                timestamp=start_time.isoformat(),
                error_message=error_msg,
            )

    def run(
        self,
        url: str = Field(description="HTTP/HTTPS URL of the file to download"),
        output_file_path: str = Field(
            description="Local path where the file should be saved (absolute or relative to workspace)"
        ),
        overwrite: bool = Field(default=False, description="Whether to overwrite existing files (default: False)"),
        timeout: int = Field(default=60, description="Download timeout in seconds (default: 60)"),
        output_format: str = Field(default="json", description="Output format: 'markdown', 'json', or 'text'"),
    ) :
        """Download a file from a URL with comprehensive options and controls.

        This tool provides secure file download capabilities with:
        - HTTP/HTTPS URL support
        - Configurable timeout controls
        - Path validation and directory creation
        - File size limits and safety checks
        - LLM-optimized result formatting

        Args:
            url: The HTTP/HTTPS URL of the file to download
            output_file_path: Local path to save the downloaded file
            overwrite: Whether to overwrite existing files
            timeout: Maximum download time in seconds
            output_format: Format for the response output

        Returns:
            ActionResponse with download results and metadata
        """
        # Handle FieldInfo objects
        if isinstance(url, FieldInfo):
            url = url.default
        if isinstance(output_file_path, FieldInfo):
            output_file_path = output_file_path.default
        if isinstance(overwrite, FieldInfo):
            overwrite = overwrite.default
        if isinstance(timeout, FieldInfo):
            timeout = timeout.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default

        try:
            # Validate URL
            url_valid, url_error = self._validate_url(url)
            if not url_valid:
                return {
                    "message": f"Invalid URL: {url_error}",
                    "success": False
                }

            # Resolve output path
            output_path = self._resolve_output_path(output_file_path)

            # Check if file exists and overwrite setting
            if output_path.exists() and not overwrite:
                existing_size = output_path.stat().st_size
                return {
                    "result": f"File already exists at {output_path} ({existing_size:,} bytes) and overwrite is disabled",
                    "success": True,
                    "status": "file_exists",
                    "file_path": str(output_path),
                    "metadata": {
                        "file_size_bytes": existing_size,
                        "overwrite_enabled": overwrite,
                    },
                }

            # Perform download
            start_time = time.time()
            try:
                # Session already has headers configured, pass None for additional headers
                result = self._download_file_async(url, output_path, timeout, None)
            except Exception as e:
                # Handle async execution errors
                result = DownloadResult(
                    url=url,
                    file_path=str(output_path),
                    success=False,
                    duration="0s",
                    timestamp=datetime.now().isoformat(),
                    error_message=f"Async execution error: {str(e)}"
                )
            execution_time = time.time() - start_time

            # Format output
            formatted_output = self._format_download_output(result, output_format)

            # Create metadata
            metadata = DownloadMetadata(
                url=url,
                output_path=str(output_path),
                timeout_seconds=timeout,
                overwrite_enabled=overwrite,
                execution_time=execution_time,
                file_size_bytes=result.file_size,
                headers_used=self.session is not None and hasattr(self.session, 'headers'),
            )

            if not result.success:
                return {
                    "error": f"Download failed: {result.error_message}",
                    "success": False,
                    "error_type": "download_failure"
                }

            return {"result": formatted_output, "success": True, "metadata": metadata.model_dump()}

        except Exception as e:
            error_msg = f"Failed to download file: {str(e)}"
            print(f"Download error: {traceback.format_exc()}")

            return {
                "error": error_msg,
                "success": False,
                "error_type": "download_failure",
                "traceback": traceback.format_exc()

            }

    def test(self, tool_test: str="download_file", file_location: str="download_file", result_parameter: str="result", search_type: str="exact_match"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)

# Default arguments for testing
if __name__ == "__main__":

    try:
        tool = Download_File_Tool()
        tool.embed_tool()
        tool.run(url="https://www.science.org/doi/pdf/10.1126/sciadv.abi8620", output_file_path="test.pdf", overwrite=True, timeout=60, output_format="markdown")
        # tool.test(tool_test="download_file", file_location="download_file", result_parameter="success", search_type='exact_match')    
    except Exception as e:
        print(f"An error occurred: {e}: {traceback.format_exc()}")
