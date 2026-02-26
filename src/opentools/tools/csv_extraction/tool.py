# source code: https://github.com/inclusionAI/AWorld/blob/main/examples/gaia/mcp_collections/documents/mscsv.py
import json, os, time, traceback, zipfile, sys, chardet
from pathlib import Path
from typing import Any, Literal, Optional
from openpyxl import load_workbook
from pydantic import Field, BaseModel
from pydantic.fields import FieldInfo
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool

class DocumentMetadata(BaseModel):
    """Metadata for extracted Excel document content."""
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
    extracted_text_file_path: Optional[str] = None,
    error_type: Optional[str] = None,


class Csv_Extraction_Tool(BaseTool):
    # Default args for `opentools test Csv_Extraction_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "csv_extraction",
        "file_location": "csv_extraction",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }

    """Csv_Extraction_Tool
    ---------------------
    Purpose:
        A comprehensive CSV document extraction tool that extracts content from CSV, TSV, and delimited text files with support for automatic encoding detection, delimiter detection, statistical analysis, and LLM-friendly output in multiple formats (markdown, JSON, HTML, text). Provides detailed metadata about data structure, dimensions, processing time, and memory usage. Supports selective row processing, statistical summaries, and data visualization generation. Uses pandas with chardet for robust CSV processing and automatic encoding detection.

    Core Capabilities:
        - CSV/TSV/delimited text file processing
        - Automatic encoding detection
        - Automatic delimiter detection
        - Statistical analysis and data profiling
        - Multiple output formats (markdown, JSON, HTML, text)
        - Selective row processing
        - Memory-efficient large file handling
        - Comprehensive metadata collection
        - Cross-platform compatibility

    Intended Use:
        Use this tool when you need to extract content from CSV, TSV, and delimited text files, including automatic encoding detection, delimiter detection, statistical analysis, and LLM-friendly output in multiple formats (markdown, JSON, HTML, text).

    Limitations:
        - May not handle complex CSV files or delimited text files
    """

    def __init__(self) -> None:
        super().__init__(
            type='function',
            name="Csv_Extraction_Tool",
            description="""A comprehensive CSV document extraction tool that extracts content from CSV, TSV, and delimited text files with support for automatic encoding detection, delimiter detection, statistical analysis, and LLM-friendly output in multiple formats (markdown, JSON, HTML, text). Provides detailed metadata about data structure, dimensions, processing time, and memory usage. Supports selective row processing, statistical summaries, and data visualization generation. Uses pandas with chardet for robust CSV processing and automatic encoding detection. CAPABILITIES: CSV/TSV/delimited text file processing, automatic encoding detection, automatic delimiter detection, statistical analysis and data profiling, multiple output formats (Markdown, JSON, HTML, Text), selective row processing, memory-efficient large file handling, comprehensive metadata collection, cross-platform compatibility. SYNONYMS: CSV reader, tabular data extractor, delimiter detector, encoding detector, data profiler, CSV analyzer, tabular data processor, data extraction tool, CSV parser, data profiling tool. EXAMPLES: 'Extract data from this CSV file', 'Convert CSV to markdown format', 'Analyze data structure of this TSV file', 'Generate statistics for this delimited text file'.""",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV document file to extract content from"
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'markdown', 'json', 'html', or 'text' (default: json)",
                        "enum": ["markdown", "json", "html", "text"]
                    },
                    "max_rows": {
                        "type": "integer",
                        "description": "Maximum number of rows to read (None for all rows)"
                    },
                    "include_statistics": {
                        "type": "boolean",
                        "description": "Whether to include statistical summary in output (default: True)"
                    },
                    "generate_visualizations": {
                        "type": "boolean",
                        "description": "Whether to generate and save data visualizations (default: True)"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (auto-detected if None)"
                    },
                    "delimiter": {
                        "type": "string",
                        "description": "CSV delimiter (auto-detected if None)"
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
            strict=False,
            category="data_processing",
            tags=["csv_extraction", "data_processing", "encoding_detection", "delimiter_detection", "data_profiling", "pandas", "data_extraction", "tabular_data"],
            limitation="Image extraction only supported for XLSX files, not XLS, screenshot generation requires GUI environment, large Excel files may consume significant memory, screenshot quality depends on screen resolution, media extraction limited to embedded content, processing time scales with file size and complexity, some Excel features (macros, complex formulas) may not be fully preserved",
            agent_type="File_Extraction-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {    
                "command": "reponse = tool.run(file_path='test.csv', output_format='markdown', max_rows=100, include_statistics=True, generate_visualizations=True, encoding='utf-8', delimiter=',')",    
                "description": "Extract data from a CSV file"
            },
        )
        self.workspace = Path(os.getcwd())
        self._media_output_dir = self.workspace / "extracted_media"
        self.supported_extensions: set = {".csv", ".tsv", ".txt"}

    def _validate_file_path(self, file_path: str) -> Path:
        """Validate and resolve file path. Rely on the predefined supported_extensions class variable.

        Args:
            file_path: Path to the document or media file

        Returns:
            Resolved Path object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = self.workspace / path

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. Supported types: {', '.join(self.supported_extensions)}"
            )

        return path
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet.

        Args:
            file_path: Path to the CSV file

        Returns:
            Detected encoding string
        """
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                encoding = result.get("encoding", "utf-8")
                confidence = result.get("confidence", 0)

                print(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                return encoding if confidence > 0.7 else "utf-8"
        except Exception as e:
            print(f"Encoding detection failed: {e}, using utf-8")
            return "utf-8"

    def _detect_delimiter(self, file_path: Path, encoding: str) -> str:
        """Detect CSV delimiter by analyzing the first few lines.

        Args:
            file_path: Path to the CSV file
            encoding: File encoding

        Returns:
            Detected delimiter character
        """
        try:
            with open(file_path, "r", encoding=encoding) as f:
                sample = f.read(1024)  # Read first 1KB

            # Common delimiters to test
            delimiters = [",", ";", "\t", "|", ":"]
            delimiter_counts = {}

            for delimiter in delimiters:
                count = sample.count(delimiter)
                if count > 0:
                    delimiter_counts[delimiter] = count

            if delimiter_counts:
                detected_delimiter = max(delimiter_counts, key=delimiter_counts.get)
                print(f"Detected delimiter: '{detected_delimiter}'")
                return detected_delimiter
            else:
                return ","
        except Exception as e:
            print(f"Delimiter detection failed: {e}, using comma")
            return ","

    def _extract_csv_content(
        self, file_path: Path, max_rows: int | None = None, encoding: str | None = None, delimiter: str | None = None
    ) -> dict[str, Any]:
        """Extract content from CSV file using pandas.

        Args:
            file_path: Path to the CSV file
            max_rows: Maximum number of rows to read
            encoding: File encoding (auto-detected if None)
            delimiter: CSV delimiter (auto-detected if None)

        Returns:
            Dictionary containing extracted content and metadata
        """
        start_time = time.time()

        # Auto-detect encoding and delimiter if not provided
        if encoding is None:
            encoding = self._detect_encoding(file_path)
        if delimiter is None:
            delimiter = self._detect_delimiter(file_path, encoding)

        try:
            # Read CSV with pandas
            df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, nrows=max_rows, low_memory=False)

            # Get full file info for metadata
            full_df_info = pd.read_csv(
                file_path,
                encoding=encoding,
                delimiter=delimiter,
                nrows=0,  # Just get headers and shape info
            )

            # Count total rows efficiently
            total_rows = sum(1 for _ in open(file_path, "r", encoding=encoding)) - 1  # Subtract header

            processing_time = time.time() - start_time

            return {
                "dataframe": df,
                "total_rows": total_rows,
                "total_columns": len(full_df_info.columns),
                "columns": list(df.columns),
                "encoding": encoding,
                "delimiter": delimiter,
                "processing_time": processing_time,
                "data_types": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
            }

        except Exception as e:
            print(f"Failed to read CSV file: {e}")
            raise

    def _format_content_for_llm(self, df: pd.DataFrame, output_format: str, include_stats: bool = True) -> str:
        """Format extracted CSV content to be LLM-friendly.

        Args:
            df: Pandas DataFrame with CSV data
            output_format: Desired output format
            include_stats: Whether to include statistical summary

        Returns:
            Formatted content string
        """
        if output_format.lower() == "markdown":
            # Convert to markdown table
            content = df.to_markdown(index=False, tablefmt="github")

            if include_stats:
                # Add statistical summary
                stats_content = "\n\n## Data Summary\n\n"
                stats_content += f"- **Rows**: {len(df)}\n"
                stats_content += f"- **Columns**: {len(df.columns)}\n"
                stats_content += f"- **Column Names**: {', '.join(df.columns)}\n\n"

                # Add data types info
                stats_content += "### Column Data Types\n\n"
                for col, dtype in df.dtypes.items():
                    stats_content += f"- **{col}**: {dtype}\n"

                # Add basic statistics for numeric columns
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 0:
                    stats_content += "\n### Numeric Column Statistics\n\n"
                    stats_df = df[numeric_cols].describe()
                    stats_content += stats_df.to_markdown(tablefmt="github")

                content += stats_content

        elif output_format.lower() == "json":
            # Convert to JSON with metadata
            data_dict = {
                "data": df.to_dict(orient="records"),
                "metadata": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns),
                    "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                },
            }
            if include_stats:
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 0:
                    data_dict["statistics"] = df[numeric_cols].describe().to_dict()

            content = json.dumps(data_dict, indent=2, default=str)

        elif output_format.lower() == "html":
            # Convert to HTML table
            content = df.to_html(index=False, classes="table table-striped")

        else:
            # Plain text format
            content = df.to_string(index=False)

        return content

    def run(
        self,
        file_path: str = Field(description="Path to the CSV document file to extract content from"),
        output_format: Literal["markdown", "json", "html", "text"] = Field(
            default="json", description="Output format: 'markdown', 'json', 'html', or 'text'"
        ),
        max_rows: int | None = Field(default=None, description="Maximum number of rows to read (None for all rows)"),
        include_statistics: bool = Field(default=True, description="Whether to include statistical summary in output"),
        generate_visualizations: bool = Field(
            default=True, description="Whether to generate and save data visualizations"
        ),
        encoding: str | None = Field(default=None, description="File encoding (auto-detected if None)"),
        delimiter: str | None = Field(default=None, description="CSV delimiter (auto-detected if None)"),
    ) :
        """Extract content from CSV documents using pandas.

        This tool provides comprehensive CSV document content extraction with support for:
        - CSV, TSV, and delimited text files
        - Automatic encoding and delimiter detection
        - Statistical analysis and data profiling
        - Multiple output formats (Markdown, JSON, HTML, Text)
        - Optional data visualizations
        - Memory-efficient processing for large files

        Args:
            file_path: Path to the CSV file
            output_format: Desired output format
            max_rows: Maximum rows to process (None for all)
            include_statistics: Include statistical summary
            generate_visualizations: Generate data visualizations
            encoding: File encoding (auto-detected if None)
            delimiter: CSV delimiter (auto-detected if None)

        Returns:
            ActionResponse with extracted content, metadata, and optional visualizations
        """
        try:
            # Handle FieldInfo objects from pydantic
            if isinstance(file_path, FieldInfo):
                file_path = file_path.default
            if isinstance(output_format, FieldInfo):
                output_format = output_format.default
            if isinstance(max_rows, FieldInfo):
                max_rows = max_rows.default
            if isinstance(include_statistics, FieldInfo):
                include_statistics = include_statistics.default
            if isinstance(generate_visualizations, FieldInfo):
                generate_visualizations = generate_visualizations.default
                self._media_output_dir.mkdir(exist_ok=True)
            if isinstance(encoding, FieldInfo):
                encoding = encoding.default
            if isinstance(delimiter, FieldInfo):
                delimiter = delimiter.default

            # Validate input file
            file_path: Path = self._validate_file_path(file_path)
            print(f"Processing CSV file: {file_path.name}")

            # Extract CSV content
            extraction_result = self._extract_csv_content(
                file_path, max_rows=max_rows, encoding=encoding, delimiter=delimiter
            )

            df: pd.DataFrame = extraction_result["dataframe"]

            # Format content for LLM consumption
            formatted_content = self._format_content_for_llm(df, output_format, include_stats=include_statistics)
            file_stats = file_path.stat()
            document_metadata = DocumentMetadata(
                            file_name=file_path.name,
                            file_size=file_stats.st_size,
                            file_type=file_path.suffix.lower(),
                            absolute_path=str(file_path.absolute()),
                            page_count=None,  # Not applicable for CSV
                            processing_time=extraction_result["processing_time"],
                            extracted_images=[],  
                            extracted_media=[],
                            output_format=output_format,
                            ocr_applied=False,
                            extracted_text_file_path=None,
                        )
            csv_metadata = {
                "total_rows": extraction_result["total_rows"],
                "total_columns": extraction_result["total_columns"],
                "rows_processed": len(df),
                "columns_processed": len(df.columns),
                "column_names": extraction_result["columns"],
                "data_types": {k: str(v) for k, v in extraction_result["data_types"].items()},
                "encoding": extraction_result["encoding"],
                "delimiter": extraction_result["delimiter"],
                "memory_usage_bytes": int(extraction_result["memory_usage"]),
            }
            final_metadata = {**document_metadata.model_dump(), **csv_metadata}
            return {"result": formatted_content, "success": True, "metadata": final_metadata}

        except FileNotFoundError as e:
            return {"error": f"File not found: {str(e)}", "success": False, "error_type": "file_not_found", "traceback": traceback.format_exc()}
        except ValueError as e:
            return {"error": f"Invalid input: {str(e)}", "success": False, "error_type": "invalid_input", "traceback": traceback.format_exc()}
        except Exception as e:
            return {"error": f"CSV extraction failed: {str(e)}", "success": False, "error_type": "csv_extraction_failed", "traceback": traceback.format_exc()}

    def test(self, tool_test: str="csv_extraction", file_location: str="csv_extraction", result_parameter: str="result", search_type: str="exact_match"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)

# Example usage and entry point
if __name__ == "__main__":

    # Initialize and run the CSV extraction service
    try:
        service = Csv_Extraction_Tool()
        service.embed_tool()
        service.test(tool_test="csv_extraction", file_location="csv_extraction", result_parameter="result", search_type='exact_match')
    except Exception as e:
        print(f"An error occurred: {e}: {traceback.format_exc()}")
