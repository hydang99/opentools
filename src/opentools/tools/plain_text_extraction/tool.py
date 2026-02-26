import json, os, sys, time, traceback
import gzip
import zlib
from pathlib import Path
from typing import Any, Literal, Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
from marker.converters.pdf import PdfConverter
from pydantic import Field, BaseModel
from pydantic.fields import FieldInfo

class DocumentMetadata(BaseModel):
    """Metadata for extracted document content."""
    file_name: str
    file_size: int
    file_type: str
    absolute_path: str
    page_count: Optional[int] = None
    processing_time: float
    extracted_images: list[str] = []
    extracted_media: list[dict[str, str]] = []
    output_format: str
    ocr_applied: bool
    extracted_text_file_path: Optional[str] = None


class Plain_Text_Extraction_Tool(BaseTool):
    # Default args for `opentools test Plain_Text_Extraction_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "plain_text_extraction",
        "file_location": "plain_text_extraction",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }

    """Plain_Text_Extraction_Tool
    ---------------------
    Purpose:
        A comprehensive text document content extraction tool that extracts text content from various text file formats with encoding detection and analysis. Supports multiple file types including plain text, documentation, programming languages, data formats, configuration files, and logs. Provides LLM-friendly output in multiple formats with comprehensive metadata about the extraction process.

    Core Capabilities:
        - Extracts text content from various text file formats
        - Handles content type detection and analysis
        - Provides LLM-friendly output in multiple formats
        - Supports multiple file types including plain text, documentation, programming languages, data formats, configuration files, and logs

    Intended Use:
        Use this tool when you need to extract text content from various text file formats, including plain text, documentation, programming languages, data formats, configuration files, and logs.

    Limitations:
        - May not handle complex text file formats or content types
    """

    def __init__(self) -> None:
        super().__init__(
            type='function',
            name="Plain_Text_Extraction_Tool",
            description="""A comprehensive text document content extraction tool that extracts text content from various text file formats with encoding detection and analysis. Supports multiple file types including plain text, documentation, programming languages, data formats, configuration files, and logs. Provides LLM-friendly output in multiple formats with comprehensive metadata about the extraction process. CAPABILITIES: Extracts text content from various file formats, handles content type detection and analysis. SYNONYMS: text extractor, document parser, text file reader, encoding detector, text analyzer, content extractor, file processor, text processing tool, document reader, text file analyzer. EXAMPLES: 'Extract content from this text file in markdown format', 'Search for specific keywords in this log file', 'Analyze the encoding and content of this document'.""",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the text document file to extract content from"
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'markdown', 'json', 'html', or 'text' (default: json)",
                        "enum": ["markdown", "json", "html", "text"]
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Specific encoding to use (None for auto-detection)"
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
            strict=False,
            category="document_processing",
            tags=["text_extraction", "document_processing", "encoding_detection", "text_analysis", "content_extraction", "file_processor", "document_reader", "encoding_analyzer"],
            limitation="Search functionality should be used sparingly to avoid scanning entire files. Large text files may exceed token limits when processed entirely. Encoding detection may fail for some binary files. Search queries must be exact strings, not fuzzy matching. File size should be reasonable to avoid memory issues. Some file formats may not render properly in all output formats. Files without extensions must be valid text content.",
            agent_type="File_Extraction-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(file_path='test.txt', output_format='markdown', encoding='utf-8')",
                "description": "Extract content from test.txt in markdown format, search for 'Nature' in the document, use utf-8 encoding"
            }
        )
        self.workspace = Path(os.getcwd())
        self.supported_extensions: set = {
            ".txt",
            ".text",
            ".log",
            ".md",
            ".markdown",
            ".rst",
            ".rtf",
            ".csv",
            ".tsv",
            ".json",
            ".xml",
            ".yaml",
            ".yml",
            ".ini",
            ".cfg",
            ".conf",
            ".properties",
            ".sql",
            ".py",
            ".js",
            ".html",
            ".htm",
            ".css",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".php",
            ".rb",
            ".go",
            ".rs",
            ".sh",
            ".bat",
            ".ps1",
            ".r",
            ".m",
            ".swift",
            ".kt",
            ".scala",
            ".pl",
            ".lua",
            ".vim",
            ".tex",
            ".bib",
        }
        
    def get_mime_type(self, file_path: str, default_mime: str | None = None) -> str:
        """
        Detect MIME type of a file using python-magic if available,
        otherwise fallback to extension-based detection.

        Args:
            file_path: Path to the file
            default_mime: Default MIME type to return if detection fails

        Returns:
            str: Detected MIME type
        """
        # Try using python-magic for accurate MIME type detection
        try:
            mime = magic.Magic(mime=True)
            return mime.from_file(file_path)
        except (AttributeError, IOError):
            # Fallback to extension-based detection
            extension_mime_map = {
                # Audio formats
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".ogg": "audio/ogg",
                ".m4a": "audio/mp4",
                ".flac": "audio/flac",
                # Image formats
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
                ".tiff": "image/tiff",
                # Video formats
                ".mp4": "video/mp4",
                ".avi": "video/x-msvideo",
                ".mov": "video/quicktime",
                ".mkv": "video/x-matroska",
                ".webm": "video/webm",
            }

            ext = Path(file_path).suffix.lower()
            return extension_mime_map.get(ext, default_mime or "application/octet-stream")
    
    def _validate_file_path(self, file_path: str) -> Path:
        """Validate and resolve file path.

        Args:
            file_path: Path to the text document file

        Returns:
            Resolved Path object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace / path

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Also check MIME type for files without extensions or unknown extensions
        mime_type = self.get_mime_type(str(path), default_mime="text/plain")
        is_text_mime = mime_type and mime_type.startswith("text/")

        if path.suffix.lower() not in self.supported_extensions and not is_text_mime:
            # Try to detect if it's a text file by reading a small sample
            try:
                with open(path, "rb") as f:
                    sample = f.read(1024)
                    # Check if the sample contains mostly printable characters
                    if self._is_likely_text(sample):
                        print(f"Detected text file without standard extension: {path.suffix}")
                    else:
                        raise ValueError(
                            f"Unsupported file type: {path.suffix}. "
                            f"Supported types: {', '.join(sorted(self.supported_extensions))} or text MIME types"
                        )
            except Exception as e:
                raise ValueError(
                    f"Cannot determine if file is text: {str(e)}. "
                    f"Supported types: {', '.join(sorted(self.supported_extensions))}"
                ) from e

        return path

    def _is_likely_text(self, data: bytes) -> bool:
        """Check if binary data is likely to be text.

        Args:
            data: Binary data sample

        Returns:
            True if data appears to be text
        """
        if not data:
            return True

        # Check for null bytes (common in binary files)
        if b"\x00" in data:
            return False

        # Try to decode as text
        try:
            data.decode("utf-8")
            return True
        except UnicodeDecodeError:
            pass

        # Check if most bytes are printable ASCII
        printable_count = sum(1 for byte in data if 32 <= byte <= 126 or byte in [9, 10, 13])
        return printable_count / len(data) > 0.7

    def _detect_encoding(self, file_path: Path) -> dict[str, Any]:
        """Detect file encoding and other characteristics.

        Args:
            file_path: Path to the text file

        Returns:
            Dictionary containing encoding information
        """
        encoding_info = {
            "detected_encoding": None,
            "confidence": 0.0,
            "bom_detected": False,
            "line_endings": None,
            "is_binary": False,
        }

        try:
            # Read file in binary mode for encoding detection
            with open(file_path, "rb") as f:
                raw_data = f.read()

            if not raw_data:
                encoding_info["detected_encoding"] = "utf-8"
                encoding_info["confidence"] = 1.0
                return encoding_info

            # Check for BOM (Byte Order Mark)
            if raw_data.startswith(b"\xef\xbb\xbf"):
                encoding_info["bom_detected"] = True
                encoding_info["detected_encoding"] = "utf-8-sig"
                encoding_info["confidence"] = 1.0
            elif raw_data.startswith(b"\xff\xfe"):
                encoding_info["bom_detected"] = True
                encoding_info["detected_encoding"] = "utf-16-le"
                encoding_info["confidence"] = 1.0
            elif raw_data.startswith(b"\xfe\xff"):
                encoding_info["bom_detected"] = True
                encoding_info["detected_encoding"] = "utf-16-be"
                encoding_info["confidence"] = 1.0
            else:
                # Use chardet for encoding detection
                detection_result = chardet.detect(raw_data)
                encoding_info["detected_encoding"] = detection_result.get("encoding", "utf-8")
                encoding_info["confidence"] = detection_result.get("confidence", 0.0)

            # Detect line endings
            if b"\r\n" in raw_data:
                encoding_info["line_endings"] = "CRLF (Windows)"
            elif b"\n" in raw_data:
                encoding_info["line_endings"] = "LF (Unix/Linux/Mac)"
            elif b"\r" in raw_data:
                encoding_info["line_endings"] = "CR (Classic Mac)"
            else:
                encoding_info["line_endings"] = "None detected"

            # Check if file appears to be binary
            encoding_info["is_binary"] = not self._is_likely_text(raw_data[:1024])

        except Exception as e:
            print(f"Failed to detect encoding: {str(e)}")
            encoding_info["detected_encoding"] = "utf-8"
            encoding_info["confidence"] = 0.0

        return encoding_info

    def _detect_encoding_from_bytes(self, raw_data: bytes) -> dict[str, Any]:
        """Detect encoding characteristics from an in-memory byte buffer."""
        encoding_info = {
            "detected_encoding": None,
            "confidence": 0.0,
            "bom_detected": False,
            "line_endings": None,
            "is_binary": False,
        }

        if not raw_data:
            encoding_info["detected_encoding"] = "utf-8"
            encoding_info["confidence"] = 1.0
            encoding_info["line_endings"] = "None detected"
            encoding_info["is_binary"] = False
            return encoding_info

        # Check for BOM (Byte Order Mark)
        if raw_data.startswith(b"\xef\xbb\xbf"):
            encoding_info["bom_detected"] = True
            encoding_info["detected_encoding"] = "utf-8-sig"
            encoding_info["confidence"] = 1.0
        elif raw_data.startswith(b"\xff\xfe"):
            encoding_info["bom_detected"] = True
            encoding_info["detected_encoding"] = "utf-16-le"
            encoding_info["confidence"] = 1.0
        elif raw_data.startswith(b"\xfe\xff"):
            encoding_info["bom_detected"] = True
            encoding_info["detected_encoding"] = "utf-16-be"
            encoding_info["confidence"] = 1.0
        else:
            detection_result = chardet.detect(raw_data)
            encoding_info["detected_encoding"] = detection_result.get("encoding", "utf-8")
            encoding_info["confidence"] = detection_result.get("confidence", 0.0)

        # Detect line endings
        if b"\r\n" in raw_data:
            encoding_info["line_endings"] = "CRLF (Windows)"
        elif b"\n" in raw_data:
            encoding_info["line_endings"] = "LF (Unix/Linux/Mac)"
        elif b"\r" in raw_data:
            encoding_info["line_endings"] = "CR (Classic Mac)"
        else:
            encoding_info["line_endings"] = "None detected"

        # Check if file appears to be binary
        encoding_info["is_binary"] = not self._is_likely_text(raw_data[:1024])
        return encoding_info

    def _maybe_decompress_bytes(self, raw_data: bytes, file_path: Path) -> tuple[bytes, dict[str, Any]]:
        """
        If the payload looks like compressed bytes (often happens when a downloader saved
        Content-Encoding'd HTML as-is), try to decompress it and return (bytes, info).
        """
        info: dict[str, Any] = {
            "decompression_applied": False,
            "compression_type": None,
            "decompression_error": None,
        }

        if not raw_data:
            return raw_data, info

        # gzip magic: 1F 8B
        if raw_data.startswith(b"\x1f\x8b"):
            try:
                return gzip.decompress(raw_data), {**info, "decompression_applied": True, "compression_type": "gzip"}
            except Exception as e:
                return raw_data, {**info, "decompression_error": f"gzip decompress failed: {e}"}

        # zlib/deflate commonly starts with 0x78 0x01/0x9c/0xda (not guaranteed)
        if len(raw_data) >= 2 and raw_data[0] == 0x78 and raw_data[1] in (0x01, 0x9C, 0xDA):
            try:
                return zlib.decompress(raw_data), {**info, "decompression_applied": True, "compression_type": "zlib"}
            except Exception:
                # Might be raw deflate stream; try wbits=-15
                try:
                    return zlib.decompress(raw_data, wbits=-15), {**info, "decompression_applied": True, "compression_type": "deflate"}
                except Exception as e:
                    return raw_data, {**info, "decompression_error": f"zlib/deflate decompress failed: {e}"}

        # brotli has no stable magic header; attempt only for likely-text extensions.
        if file_path.suffix.lower() in {".html", ".htm", ".txt", ".log", ".json", ".xml", ".md", ".csv", ".tsv"}:
            try:
                import brotli  # type: ignore
            except Exception:
                brotli = None

            if brotli is not None:
                try:
                    return brotli.decompress(raw_data), {**info, "decompression_applied": True, "compression_type": "brotli"}
                except Exception:
                    # Not brotli; leave as-is
                    pass

        return raw_data, info

    def _extract_text_content(self, file_path: Path, encoding: str | None = None) -> dict[str, Any]:
        """Extract content from text files.

        Args:
            file_path: Path to the text file
            encoding: Specific encoding to use (None for auto-detection)

        Returns:
            Dictionary containing extracted content and metadata
        """
        start_time = time.time()

        try:
            # Always read bytes first; this lets us handle "binary" inputs that are
            # actually compressed text (gzip/deflate/brotli) saved to disk.
            with open(file_path, "rb") as f:
                raw_data = f.read()

            # Initial detection on raw bytes
            encoding_info = self._detect_encoding_from_bytes(raw_data)

            decompressed_info: dict[str, Any] = {
                "decompression_applied": False,
                "compression_type": None,
                "decompression_error": None,
            }
            payload = raw_data

            # If it looks binary, try decompression before decoding as text.
            if encoding_info.get("is_binary", False):
                payload, decompressed_info = self._maybe_decompress_bytes(raw_data, file_path)
                if decompressed_info.get("decompression_applied"):
                    # Re-detect on decompressed bytes for better encoding signal
                    encoding_info = self._detect_encoding_from_bytes(payload)
                    # Surface decompression info for debugging/traceability
                    encoding_info.update(decompressed_info)
                else:
                    # Still provide decompression attempt info if any
                    encoding_info.update(decompressed_info)

            if encoding:
                target_encoding = encoding
            else:
                target_encoding = encoding_info.get("detected_encoding") or "utf-8"

            content = payload.decode(target_encoding, errors="replace")

            # Analyze content
            lines = content.splitlines()

            # Calculate statistics
            char_count = len(content)
            line_count = len(lines)
            word_count = len(content.split()) if content.strip() else 0

            # Find longest and shortest lines
            line_lengths = [len(line) for line in lines]
            max_line_length = max(line_lengths) if line_lengths else 0
            min_line_length = min(line_lengths) if line_lengths else 0
            avg_line_length = sum(line_lengths) / len(line_lengths) if line_lengths else 0

            # Count empty lines
            empty_lines = sum(1 for line in lines if not line.strip())

            # Detect file type based on content patterns
            content_type = self._detect_content_type(content, file_path)

            processing_time = time.time() - start_time

            return {
                "content": content,
                "encoding_info": encoding_info,
                "statistics": {
                    "character_count": char_count,
                    "line_count": line_count,
                    "word_count": word_count,
                    "empty_lines": empty_lines,
                    "max_line_length": max_line_length,
                    "min_line_length": min_line_length,
                    "avg_line_length": round(avg_line_length, 2),
                },
                "content_type": content_type,
                "processing_time": processing_time,
                "used_encoding": target_encoding,
            }

        except UnicodeDecodeError as e:
            print(f"Failed to decode file with encoding {target_encoding}: {str(e)}")
            # Try with fallback encodings
            fallback_encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

            for fallback_encoding in fallback_encodings:
                if fallback_encoding != target_encoding:
                    try:
                        with open(file_path, "r", encoding=fallback_encoding, errors="replace") as f:
                            content = f.read()

                        # Recalculate with fallback encoding
                        lines = content.splitlines()
                        char_count = len(content)
                        line_count = len(lines)
                        word_count = len(content.split()) if content.strip() else 0

                        processing_time = time.time() - start_time

                        return {
                            "content": content,
                            "encoding_info": encoding_info,
                            "statistics": {
                                "character_count": char_count,
                                "line_count": line_count,
                                "word_count": word_count,
                                "empty_lines": sum(1 for line in lines if not line.strip()),
                                "max_line_length": max(len(line) for line in lines) if lines else 0,
                                "min_line_length": min(len(line) for line in lines) if lines else 0,
                                "avg_line_length": round(sum(len(line) for line in lines) / len(lines), 2)
                                if lines
                                else 0,
                            },
                            "content_type": self._detect_content_type(content, file_path),
                            "processing_time": processing_time,
                            "used_encoding": fallback_encoding,
                            "encoding_fallback": True,
                        }
                    except Exception:
                        continue

            raise ValueError("Unable to decode file with any supported encoding") from e

    def _detect_content_type(self, content: str, file_path: Path) -> str:
        """Detect the type of content based on file extension and content patterns.

        Args:
            content: File content
            file_path: Path to the file

        Returns:
            Detected content type
        """
        extension = file_path.suffix.lower()

        # Map extensions to content types
        extension_map = {
            ".py": "Python source code",
            ".js": "JavaScript source code",
            ".html": "HTML document",
            ".htm": "HTML document",
            ".css": "CSS stylesheet",
            ".json": "JSON data",
            ".xml": "XML document",
            ".yaml": "YAML configuration",
            ".yml": "YAML configuration",
            ".md": "Markdown document",
            ".markdown": "Markdown document",
            ".rst": "reStructuredText document",
            ".csv": "CSV data",
            ".tsv": "TSV data",
            ".sql": "SQL script",
            ".log": "Log file",
            ".ini": "Configuration file",
            ".cfg": "Configuration file",
            ".conf": "Configuration file",
        }

        if extension in extension_map:
            return extension_map[extension]

        # Content-based detection
        content_lower = content.lower().strip()

        if content_lower.startswith("<?xml"):
            return "XML document"
        elif content_lower.startswith("{") and content_lower.endswith("}"):
            return "JSON-like data"
        elif content_lower.startswith("[") and content_lower.endswith("]"):
            return "JSON array or configuration"
        elif "#!/" in content[:50]:
            return "Script file"
        elif content.count(",") > content.count("\n") * 2:
            return "CSV-like data"
        else:
            return "Plain text"

    def _format_content_for_llm(
        self, extraction_result: dict[str, Any], output_format: str
    ) -> str:
        """Format extracted text content to be LLM-friendly.

        Args:
            extraction_result: Result from _extract_text_content
            output_format: Desired output format

        Returns:
            Formatted content string
        """
        content = extraction_result["content"]
        stats = extraction_result["statistics"]
        content_type = extraction_result["content_type"]

        if output_format.lower() == "markdown":
            formatted_parts = []
            formatted_parts.append("# Text Document Content\n")
            formatted_parts.append(f"**File Type:** {content_type}\n")
            formatted_parts.append(f"**Encoding:** {extraction_result['used_encoding']}\n")
            formatted_parts.append("**Statistics:**\n")
            formatted_parts.append(f"- Characters: {stats['character_count']:,}\n")
            formatted_parts.append(f"- Lines: {stats['line_count']:,}\n")
            formatted_parts.append(f"- Words: {stats['word_count']:,}\n")
            formatted_parts.append(f"- Empty lines: {stats['empty_lines']:,}\n")
            formatted_parts.append(f"- Average line length: {stats['avg_line_length']} characters\n\n")

            formatted_parts.append(f"## Content\n\n```\n{content}\n```")

            return "".join(formatted_parts)

        elif output_format.lower() == "json":
            json_data = {
                "document_info": {
                    "content_type": content_type,
                    "encoding": extraction_result["used_encoding"],
                    "statistics": stats,
                },
                "content": content,
            }

            return json.dumps(json_data, indent=2, ensure_ascii=False)

        elif output_format.lower() == "html":
            html_parts = []
            html_parts.append("<html><head><meta charset='utf-8'></head><body>")
            html_parts.append("<h1>Text Document Content</h1>")
            html_parts.append(f"<p><strong>File Type:</strong> {content_type}</p>")
            html_parts.append(f"<p><strong>Encoding:</strong> {extraction_result['used_encoding']}</p>")
            html_parts.append("<h2>Statistics</h2>")
            html_parts.append("<ul>")
            html_parts.append(f"<li>Characters: {stats['character_count']:,}</li>")
            html_parts.append(f"<li>Lines: {stats['line_count']:,}</li>")
            html_parts.append(f"<li>Words: {stats['word_count']:,}</li>")
            html_parts.append(f"<li>Empty lines: {stats['empty_lines']:,}</li>")
            html_parts.append(f"<li>Average line length: {stats['avg_line_length']} characters</li>")
            html_parts.append("</ul>")
            html_parts.append("<h2>Content</h2>")
            html_parts.append(f"<pre><code>{content}</code></pre>")
            html_parts.append("</body></html>")

            return "".join(html_parts)

        else:  # text format
            text_parts = []
            text_parts.append(f"Text Document Content\n{'=' * 50}\n")
            text_parts.append(f"File Type: {content_type}\n")
            text_parts.append(f"Encoding: {extraction_result['used_encoding']}\n")
            text_parts.append("\nStatistics:\n")
            text_parts.append(f"  Characters: {stats['character_count']:,}\n")
            text_parts.append(f"  Lines: {stats['line_count']:,}\n")
            text_parts.append(f"  Words: {stats['word_count']:,}\n")
            text_parts.append(f"  Empty lines: {stats['empty_lines']:,}\n")
            text_parts.append(f"  Average line length: {stats['avg_line_length']} characters\n")
            text_parts.append(f"\nContent:\n{'-' * 30}\n{content}")

            return "".join(text_parts)

    def run(
        self,
        file_path: str = Field(description="Path to the text document file to extract content from"),
        output_format: Literal["markdown", "json", "html", "text"] = Field(
            default="json", description="Output format: 'markdown', 'json', 'html', or 'text'"
        ),
        encoding: str | None = Field(default=None, description="Specific encoding to use (None for auto-detection)"),
    ):
        """Extract content from text documents with encoding detection and analysis.

        This tool provides comprehensive text document content extraction with support for:
        - Various text file formats (TXT, MD, CSV, JSON, XML, source code, etc.)
        - Automatic encoding detection with fallback options
        - Content type detection and analysis
        - Comprehensive text statistics
        - LLM-optimized output formatting
        - Binary file detection and handling

        Args:
            file_path: Path to the text file
            output_format: Desired output format
            encoding: Specific encoding to use

        Returns:
            ActionResponse with extracted content, metadata, and file analysis
        """
        try:
            # Handle FieldInfo objects
            if isinstance(file_path, FieldInfo):
                file_path = file_path.default
            if isinstance(output_format, FieldInfo):
                output_format = output_format.default
            if isinstance(encoding, FieldInfo):
                encoding = encoding.default

            # Validate input file
            file_path: Path = self._validate_file_path(file_path)

            # Extract content from text file
            extraction_result = self._extract_text_content(file_path, encoding)

            # Format content for LLM consumption
            formatted_content = self._format_content_for_llm(extraction_result, output_format)

            file_stats = file_path.stat()
            max_content_length = 100000
            # Create text-specific metadata
            text_metadata = {
                "content_type": extraction_result["content_type"],
                "encoding_info": extraction_result["encoding_info"],
                "text_statistics": extraction_result["statistics"],
                "used_encoding": extraction_result["used_encoding"],
                "encoding_fallback": extraction_result.get("encoding_fallback", False),
                "content_truncated": max_content_length and len(extraction_result["content"]) > max_content_length,
                "original_content_length": extraction_result["statistics"]["character_count"],
            }

            document_metadata = DocumentMetadata(
                file_name=file_path.name,
                file_size=file_stats.st_size,
                file_type=file_path.suffix.lower() or ".txt",
                absolute_path=str(file_path.absolute()),
                page_count=extraction_result["statistics"]["line_count"],  # Use line count as "page" count
                processing_time=extraction_result["processing_time"],
                extracted_images=[],  # Text files don't contain images
                extracted_media=[],  # Text files don't contain media
                output_format=output_format,
                llm_enhanced=False,
                ocr_applied=False,
            )

            # Combine standard and text-specific metadata
            combined_metadata = document_metadata.model_dump()
            combined_metadata.update(text_metadata)
            
            return {'success': True, 'result': formatted_content, 'metadata': combined_metadata}

        except FileNotFoundError as e:
            return {"error": f"File not found: {str(e)}", "success": False, "error_type": "file_not_found", "traceback": traceback.format_exc()}

        except ValueError as e:
            return {"error": f"Invalid input: {str(e)}", "success": False, "error_type": "invalid_input", "traceback": traceback.format_exc()}

        except Exception as e:
            return {"error": f"Text extraction failed: {str(e)}", "success": False, "error_type": "text_extraction_failed", "traceback": traceback.format_exc()}
  
    def test(self, tool_test: str="plain_text_extraction", file_location: str="plain_text_extraction", result_parameter: str="result", search_type: str="exact_match"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)

# Example usage and entry point
if __name__ == "__main__":

    # Initialize and run the text extraction service
    try:
        tool = Plain_Text_Extraction_Tool()
        try:
            tool.embed_tool()
            result = tool.run(file_path="/home/daoqm/opentools_gaia_running/opentools/src/opentools/Benchmark/downloads/Valencia-Mendez_2017_ArticleView_3238.html")
            print(result)
            
            # tool.test(tool_test="plain_text_extraction", file_location="plain_text_extraction", result_parameter="result", search_type="search_pattern")
        except Exception as e:
            print(e)
    except Exception as e:
        print(f"An error occurred: {e}: {traceback.format_exc()}")
