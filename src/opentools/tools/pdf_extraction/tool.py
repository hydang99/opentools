# source code: https://github.com/inclusionAI/AWorld/blob/main/examples/gaia/mcp_collections/documents/pdf.py
import json, os, sys, time, traceback, markdown, pdfplumber
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Optional, List, Dict
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.settings import settings
from pydantic import Field, BaseModel
from pydantic.fields import FieldInfo

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


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
    # Processing flags
    ocr_applied: bool = Field(default=False, description="Whether OCR processing was applied to the PDF document")
    search_performed: bool = Field(default=False, description="Whether text search was performed on the PDF document")
    search_query_used: Optional[str] = Field(default=None, description="The search query used if search was performed")
    search_matches_found: int = Field(default=0, description="Number of search matches found in the PDF document")
    
    # Output file information
    extracted_text_file_path: str | None = Field(
        default=None, description="Absolute path to the extracted text file (if applicable)"
    )

class Pdf_Extraction_Tool(BaseTool):
    # Default args for `opentools test Pdf_Extraction_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "pdf_extraction",
        "file_location": "pdf_extraction",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }

    """Pdf_Extraction_Tool
    ---------------------
    Purpose:
        A comprehensive PDF document content extraction tool that extracts text and images using text processing. Provides friendly output in multiple formats with comprehensive metadata about the extraction process. Only processes files that are present locally on the system; cannot read or extract content from remote, HTTP, or online PDF files (Download_File_Tool is usually required to download the file first).

    Core Capabilities:
        - Extracts text, images, and metadata from PDF documents
        - Handles page range processing
        - Forces OCR processing when needed
        - Saves extracted text to files
        - Extracts and saves embedded images

    Intended Use:
        Use this tool when you need to extract content from PDF documents, including text, images, and metadata.

    Limitations:
        - Only processes files that are present locally on the system
        - Cannot read or extract content from remote, HTTP, or online PDF files (Download_File_Tool is usually required to download the file first).
    """

    def __init__(self) -> None:
        super().__init__(
            type='function',
            name="Pdf_Extraction_Tool",
            description="""A comprehensive PDF document content extraction tool that extracts text and images using text processing. Provides friendly output in multiple formats with comprehensive metadata about the extraction process. Only processes files that are present locally on the system; cannot read or extract content from remote, HTTP, or online PDF files (Download_File_Tool is usually required to download the file first).CAPABILITIES: Extracts text, images, and metadata from PDF documents and search functionality, handles page range processing, forces OCR processing when needed, saves extracted text to files, extracts and saves embedded images. SYNONYMS: PDF extractor, document parser, PDF text extractor, PDF OCR tool, document content extractor, PDF analyzer, PDF reader. EXAMPLES: 'Extract content from this PDF in markdown format', 'Search for specific keywords in this PDF document', 'Extract text from pages 5-10 of this PDF'.""",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Operation to perform. 'extract' extracts PDF content; 'count_pages_with_keywords' scans pages and returns which pages contain the keyword(s) without dumping the full text.",
                        "enum": ["extract", "count_pages_with_keywords"]
                    },
                    "file_path": {
                        "type": "string",
                        "description": "REQUIRED: Local path to the PDF document file to extract content from. The file MUST be available locally on the system - this tool CANNOT read remote URLs. You MUST download the file first using Download_File_Tool before attempting to extract content."
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'markdown', 'json', or 'html' (default: json)",
                        "enum": ["markdown", "json", "html"]
                    },
                    "extract_images": {
                        "type": "boolean",
                        "description": "Whether to extract and save images from the PDF (default: True)"
                    },
                    "save_extracted_text_to_file": {
                        "type": "boolean",
                        "description": "Save extracted text to a local file (default: False)"
                    },
                    "page_range": {
                        "type": "string",
                        "description": "Specific pages to process (e.g., '0,5-10,20'). Using it wisely to avoid token limits (default: None)"
                    },
                    "force_ocr": {
                        "type": "boolean",
                        "description": "Force OCR processing on the entire document (default: True)"
                    },
                    "format_lines": {
                        "type": "boolean",
                        "description": "Reformat lines using local OCR model for better quality (default: True)"
                    },
                    "search_terms": {
                        "type": "string",
                        "description": "Search term/phrase to find in the PDF. Used only when operation='count_pages_with_keywords'. MUST be a single term/phrase (do NOT pass comma/newline-separated lists)."
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the keyword search is case-sensitive (default: false). Used only when operation='count_pages_with_keywords'."
                    },
                    "page_numbering": {
                        "type": "string",
                        "description": "Whether returned page numbers are 'zero_based' (0..N-1) or 'one_based' (1..N). Default: one_based.",
                        "enum": ["zero_based", "one_based"]
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
            strict=False,
            category="document_processing",
            tags=["pdf_extraction", "document_processing", "text_extraction", "ocr", "pdf_parser", "content_extraction", "document_analyzer", "pdf_reader", "text_processing", "document_tools"],
            limitation="PDF files MUST be available locally on the system - this tool CANNOT read remote URLs. You MUST download the file first using Download_File_Tool before attempting to extract content. Search functionality should be used sparingly to avoid scanning entire files. Large PDF files may exceed token limits when processed entirely. OCR processing can be slow for documents with many images. Search queries must be exact strings, not fuzzy matching.",
            agent_type="File_Extraction-Agent",
            accuracy=self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands={
                "command": "reponse = tool.run(file_path='test.pdf', output_format='markdown', extract_images=True, save_extracted_text_to_file=True, page_range='0,5-10,20', force_ocr=True, format_lines=True)",
                "description": "Extract content from test.pdf in markdown format, search for 'Nature' in the document, extract images, save extracted text to a local file, process pages 0, 5-10, and 20"
            }
        )
        self._models_loaded = False
        self._marker_models = None
        self._current_device = None  # Track the device models are loaded on
        self.workspace = Path(os.getcwd())
        
        # Fix: Use Path objects for directory paths
        self._media_output_dir = self.workspace / "extracted_media"
        self._extracted_texts_dir = self.workspace / "extracted_texts"
        
        self.supported_extensions = {".pdf"}

    def _parse_page_range(self, page_range: str | None, page_count: int) -> list[int]:
        """Parse a '0,5-10,20' style range into zero-based page indices, clamped to bounds."""
        if not page_range:
            return list(range(page_count))
        pages: list[int] = []
        for part in page_range.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = map(int, part.split("-"))
                pages.extend(range(start, end + 1))
            else:
                pages.append(int(part))
        return [p for p in pages if 0 <= p < page_count]

    def _count_pages_with_keywords(
        self,
        file_path: Path,
        search_terms: str,
        page_range: str | None,
        case_sensitive: bool,
        page_numbering: Literal["zero_based", "one_based"],
    ) -> dict[str, Any]:
        """
        Scan PDF pages and return page numbers that contain any keyword.

        This is designed to be lightweight: no full-text output, no marker/OCR.
        """
        raw = (search_terms or "").strip()
        if not raw:
            raise ValueError(
                "search_terms is required for operation='count_pages_with_keywords' (single non-empty term/phrase)"
            )
        # Enforce SINGLE term/phrase only.
        # Disallow common list separators so the agent doesn't pack multiple terms into one call.
        if re.search(r"[,;\n]+", raw):
            raise ValueError(
                "search_terms must be a SINGLE term/phrase (no commas/semicolons/newlines). "
                "Make separate tool calls for each term if you need multiple searches."
            )

        key = raw
        key_cmp = key if case_sensitive else key.lower()

        pages_with_hits: list[int] = []
        matches_found = 0
        with pdfplumber.open(str(file_path)) as pdf:
            page_count = len(pdf.pages)
            target_pages = self._parse_page_range(page_range, page_count)
            for p in target_pages:
                page = pdf.pages[p]
                text = page.extract_text() or ""
                hay = text if case_sensitive else text.lower()
                if key_cmp in hay:
                    matches_found += 1
                    if page_numbering == "one_based":
                        pages_with_hits.append(p + 1)
                    else:
                        pages_with_hits.append(p)
        result =  {
            "total_page_number": page_count,
            # Keep the legacy key name for compatibility, but it will always be a single-item list now.
            "search_terms": [key],
            "case_sensitive": case_sensitive,
            "pages_with_keywords": sorted(set(pages_with_hits)),
            "num_pages_with_keywords": len(set(pages_with_hits)),
            "matches_found_pages_scanned": matches_found,
        }
        if matches_found == 0:
            result["result"] = "The document does not contain the search term/phrase"
        return result

    def _load_marker_models(self) -> None:
        """Load marker models for document processing.

        Lazy loading to avoid unnecessary resource consumption.
        """
        if not self._models_loaded:
            try:
                self._marker_models = create_model_dict()
                self._models_loaded = True
            except Exception as e:
                raise ValueError(f"Failed to load marker models: {str(e)}")

    def _extract_content_with_marker(
        self, file_path: Path, page_range: str | None, force_ocr: bool = False
    ) -> dict[str, Any]:
        """Extract content using marker package.

        Args:
            file_path: Path to the document file
            page_range: Specific pages to process (e.g., '0,5-10,20')
            force_ocr: Use OCR to extract text from images if available

        Returns:
            Dictionary containing extracted content and metadata
        """
        start_time = time.time()

        # Prepare marker arguments
        marker_args = {
            "fname": str(file_path),
            "model_lst": self._marker_models,
            "max_pages": None,
            "langs": None,
            "batch_multiplier": 1,
            "force_ocr": force_ocr,
        }

        # Handle page range
        if page_range:
            # Parse page range string (e.g., "0,5-10,20")
            pages = []
            for part in page_range.split(","):
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    pages.extend(range(start, end + 1))
                else:
                    pages.append(int(part))
            marker_args["page_range"] = pages
        converter: PdfConverter = PdfConverter(artifact_dict=self._marker_models)
        try:
            rendered = converter(str(file_path))
            text, _, images = text_from_rendered(rendered)
            text = text.encode(settings.OUTPUT_ENCODING, errors="replace").decode(settings.OUTPUT_ENCODING)
        except Exception as e:
            # Marker uses PDFium under the hood; some valid PDFs still fail to load there
            # (e.g., incremental updates, uncommon xref layouts, malformed objects).
            # Fall back to pdfplumber text extraction so the pipeline can continue.
            err_str = str(e)
            if ("PDFium" in err_str) or ("PdfiumError" in err_str) or ("Data format error" in err_str):
                fallback = self._extract_content_with_pdfplumber(file_path, page_range=page_range)
                fallback["metadata"]["marker_error"] = err_str
                fallback["metadata"]["extraction_backend"] = "pdfplumber_fallback"
                return fallback
            raise

        processing_time = time.time() - start_time
        return {
            "content": text,
            "images": images or {},
            "metadata": defaultdict(),
            "processing_time": processing_time,
        }

    def _extract_content_with_pdfplumber(self, file_path: Path, page_range: str | None) -> dict[str, Any]:
        """Fallback extractor using pdfplumber (pure text, no images)."""
        start_time = time.time()

        pages_to_extract: list[int] | None = None
        if page_range:
            pages: list[int] = []
            for part in page_range.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    pages.extend(range(start, end + 1))
                else:
                    pages.append(int(part))
            pages_to_extract = pages

        extracted_parts: list[str] = []
        page_count: int | None = None
        with pdfplumber.open(str(file_path)) as pdf:
            page_count = len(pdf.pages)
            if pages_to_extract is None:
                target_pages = range(page_count)
            else:
                # Clamp to valid range
                target_pages = [p for p in pages_to_extract if 0 <= p < page_count]
            for i in target_pages:
                page = pdf.pages[i]
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_parts.append(page_text)

        processing_time = time.time() - start_time
        meta = defaultdict()
        meta["page_count"] = page_count
        return {
            "content": "\n\n".join(extracted_parts),
            "images": {},
            "metadata": meta,
            "processing_time": processing_time,
        }

    def _save_extracted_media(self, images: dict[str, Any], file_stem: str) -> list[dict[str, str]]:
        """Save extracted images and return their paths.

        Args:
            images: Dictionary of extracted images from marker
            file_stem: Base name for saving files

        Returns:
            list of dictionaries containing media type and file paths
        """
        saved_media = []
        os.makedirs(self._media_output_dir, exist_ok=True)

        for idx, (page_num, image_data) in enumerate(images.items()):
            try:
                # Generate unique filename
                image_filename = f"{file_stem}_page_{page_num}_img_{idx}.png"
                image_path = self._media_output_dir / image_filename

                # Save image data
                if hasattr(image_data, "save"):
                    # PIL Image object
                    image_data.save(image_path)
                elif isinstance(image_data, bytes):
                    # Raw image bytes
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                else:
                    # Handle other formats
                    continue

                saved_media.append(
                    {"type": "image", "path": str(image_path), "page": str(page_num), "filename": image_filename}
                )


            except Exception as e:
                raise ValueError(f"Failed to save image from page {page_num}: {str(e)}")

        return saved_media

    def _format_content_for_llm(self, content: str, output_format: str) -> str:
        """Format extracted content to be LLM-friendly.

        Args:
            content: Raw extracted content
            output_format: Desired output format

        Returns:
            Formatted content string
        """
        if output_format.lower() == "markdown":
            # Content is already in markdown format from marker
            return content
        elif output_format.lower() == "json":
            # Structure content as JSON

            return json.dumps({"content": content, "format": "structured_text"}, indent=2)
        elif output_format.lower() == "html":
            # Convert markdown to HTML if needed
            try:
                return markdown.markdown(content)
            except ImportError:
                return content
        else:
            return content

    def _validate_file_path(self, file_path: str) -> Path:
        """Validate and normalize file path."""
        if not file_path or not isinstance(file_path, str):
            raise ValueError("File path must be a non-empty string")
            
        path = Path(file_path).expanduser().resolve()
        
        # If not absolute, make it relative to workspace
        if not path.is_absolute():
            path = self.workspace / path

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. Supported types: {', '.join(self.supported_extensions)}"
            )

        return path
            
    def run(
        self,
        operation: Literal["extract", "count_pages_with_keywords"] = Field(
            default="extract",
            description="Operation to perform: 'extract' or 'count_pages_with_keywords'"
        ),
        file_path: str = Field(description="Path to the PDF document file to extract content from"),
        output_format: Literal["markdown", "json", "html"] = Field(
            default="markdown", description="Output format: 'markdown', 'json', or 'html'"
        ), 
        extract_images: bool = Field(default=True, description="Whether to extract and save images from the document"),
        save_extracted_text_to_file: bool = Field(
            default=False, description="Save extracted text to a local file"
        ),
        page_range: str | None = Field(default=None, description="Specific pages to process (e.g., '0,5-10,20')"),
        force_ocr: bool = Field(default=True, description="Force OCR processing on the entire document"),
        format_lines: bool = Field(
            default=True, description="Reformat lines using local OCR model for better quality"
        ),
        search_terms: str | None = Field(
            default=None,
            description="Search term/phrase to find when operation='count_pages_with_keywords'. MUST be a single term/phrase (no comma/newline-separated lists)."
        ),
        case_sensitive: bool = Field(
            default=False,
            description="Case-sensitive keyword matching when operation='count_pages_with_keywords'."
        ),
        page_numbering: Literal["zero_based", "one_based"] = Field(
            default="one_based",
            description="Return page numbers as zero-based (0..N-1) or one-based (1..N). Default: one_based."
        ),
    ) :
        """Extract content from PDF documents using marker package.

        This tool provides comprehensive PDF document content extraction with support for:
        - PDF files (MUST be available locally on the system)
        - Text extraction with proper formatting
        - Image and media extraction
        - Metadata collection
        - LLM-optimized output formatting

        CRITICAL REQUIREMENT:
        - The PDF file MUST be available locally on the system
        - This tool CANNOT read remote URLs or cloud storage links
        - You MUST download the file first using Download_File_Tool before attempting extraction
        - No automatic downloading or URL processing is supported
        - Only local file paths are accepted

        Args:
            args: Document extraction arguments including file path and options

        Returns:
            ActionResponse with extracted content, metadata, and media file paths
        """
        try:            
            if isinstance(operation, FieldInfo):
                operation = operation.default
            if isinstance(file_path, FieldInfo):
                file_path = file_path.default
            if isinstance(output_format, FieldInfo):
                output_format = output_format.default
            if isinstance(extract_images, FieldInfo):
                extract_images = extract_images.default
            if isinstance(save_extracted_text_to_file, FieldInfo):  # Handle new parameter
                save_extracted_text_to_file = save_extracted_text_to_file.default
            if isinstance(page_range, FieldInfo):
                page_range = page_range.default
            if isinstance(force_ocr, FieldInfo):
                force_ocr = force_ocr.default
            if isinstance(format_lines, FieldInfo):
                format_lines = format_lines.default
            if isinstance(search_terms, FieldInfo):
                search_terms = search_terms.default
            if isinstance(case_sensitive, FieldInfo):
                case_sensitive = case_sensitive.default
            if isinstance(page_numbering, FieldInfo):
                page_numbering = page_numbering.default

            # Validate input file
            file_path: Path = self._validate_file_path(file_path)

            # Lightweight keyword scan operation (no marker/OCR, no full-text output)
            if operation == "count_pages_with_keywords":
                if not search_terms or not str(search_terms).strip():
                    raise ValueError("search_terms is required for operation='count_pages_with_keywords'")
                start_time = time.time()
                scan = self._count_pages_with_keywords(
                    file_path=file_path,
                    search_terms=search_terms if search_terms is not None else "",
                    page_range=page_range,
                    case_sensitive=case_sensitive,
                    page_numbering=page_numbering,
                )
                processing_time = time.time() - start_time
                file_stats = file_path.stat()
                document_metadata = DocumentMetadata(
                    file_name=file_path.name,
                    file_size=file_stats.st_size,
                    file_type=file_path.suffix.lower(),
                    absolute_path=str(file_path.absolute()),
                    page_count=scan.get("total_page_number") or scan.get("total_page_count"),
                    processing_time=processing_time,
                    extracted_images=[],
                    extracted_media=[],
                    output_format="json",
                    ocr_applied=False,
                    search_performed=True,
                    search_query_used=(scan.get("search_terms") or [""])[0],
                    search_matches_found=scan.get("num_pages_with_keywords") or scan.get("pages_with_hits_count", 0),
                    extracted_text_file_path=None,
                )
                return {
                    "success": True,
                    "result": json.dumps(scan, indent=2),
                    "metadata": document_metadata.model_dump(),
                }

            # Load marker models if needed
            self._load_marker_models()

            # Extract content using marker
            extraction_result = self._extract_content_with_marker(file_path, page_range, force_ocr)

            # Save extracted media if requested
            saved_media = []
            if extract_images and extraction_result["images"]:
                saved_media = self._save_extracted_media(extraction_result["images"], file_path.stem)

            # Format content for LLM consumption
            formatted_content = self._format_content_for_llm(extraction_result["content"], output_format)

            # Save extracted text to file if requested
            saved_text_path_str: Optional[str] = None
            if save_extracted_text_to_file:
                os.makedirs(self._extracted_texts_dir, exist_ok=True)
                text_file_name = f"{file_path.stem}_extracted_text.txt"
                saved_text_path = self._extracted_texts_dir / text_file_name
                try:
                    with open(saved_text_path, "w", encoding="utf-8") as f:
                        f.write(formatted_content)
                    saved_text_path_str = str(saved_text_path.absolute())
                except Exception as e:
                    raise ValueError(f"Failed to save extracted text to {saved_text_path}: {str(e)}")

            file_stats = file_path.stat()
            document_metadata = DocumentMetadata(
                file_name=file_path.name,
                file_size=file_stats.st_size,
                file_type=file_path.suffix.lower(),
                absolute_path=str(file_path.absolute()),
                page_count=extraction_result["metadata"].get("page_count"),
                processing_time=extraction_result["processing_time"],
                extracted_images=[media["path"] for media in saved_media if media["type"] == "image"],
                extracted_media=saved_media,
                output_format=output_format,
                ocr_applied=force_ocr or format_lines,
                extracted_text_file_path=saved_text_path_str,
            )
            return {'success': True, 'result': formatted_content, 'metadata': document_metadata.model_dump()}

        except FileNotFoundError as e:
            print(f"File not found error: {e}")
            return {"error": "file not found", "success": False, "error_type": "file_not_found", "traceback": traceback.format_exc()}
        except ValueError as e:
            print(f"Invalid input error: {e}")
            return {"error": f"invalid_input: {str(e)}", "success": False, "error_type": "invalid_input", "traceback": traceback.format_exc()}
        except Exception as e:
            print(f"Extraction error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return {"error": f"extraction_error: {str(e)}", "success": False, "error_type": "extraction_error", "traceback": traceback.format_exc()}
    
    def test(self, tool_test: str="pdf_extraction", file_location: str="pdf_extraction", result_parameter: str="result", search_type: str="exact_match"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)
# Example usage and entry point
if __name__ == "__main__":
    try:
        tool = Pdf_Extraction_Tool()
        try:
            print(
                tool.run(
                    file_path="/home/daoqm/opentools_gaia_running/opentools/src/opentools/Benchmark/downloads/IPCC_AR6_SYR_FullVolume.pdf",
                    operation="count_pages_with_keywords",
                    search_terms="nuclear energy",
                    case_sensitive=True,
                    page_numbering="zero_based",
                )
            )
            # tool.embed_tool()
            # tool.test(tool_test="pdf_extraction", file_location="pdf_extraction", result_parameter="result", search_type="search_pattern")

        except Exception as e:
            print(e)
    except Exception as e:
        print(f"An error occurred: {e}: {traceback.format_exc()}")