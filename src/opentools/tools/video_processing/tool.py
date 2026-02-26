# source code: https://github.com/inclusionAI/AWorld/blob/main/examples/gaia/mcp_collections/media/video.py
import base64, os, sys, time, traceback, requests, tempfile, magic, cv2, numpy as np, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
from pydantic import Field, BaseModel
from pydantic.fields import FieldInfo
from urllib.parse import urlparse
from openai import OpenAI

def is_url(path_or_url: str) -> bool:
    """
    Check if the given string is a URL.

    Args:
        path_or_url: String to check

    Returns:
        bool: True if the string is a URL, False otherwise
    """
    parsed = urlparse(path_or_url)
    return bool(parsed.scheme and parsed.netloc)

def is_youtube_url(url: str) -> bool:
    """Check if a URL points to YouTube."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return "youtube.com" in host or "youtu.be" in host

def download_youtube_video(url: str, max_size_mb: float = 2500.0) -> tuple[str, str, bytes]:
    """
    Download a YouTube video to a temporary file using yt-dlp.

    Args:
        url: YouTube URL
        max_size_mb: Maximum allowed file size in MB

    Returns:
        Tuple[str, str, bytes]: (file_path, mime_type, empty_content_placeholder)
    """
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise ImportError(
            "Processing YouTube URLs requires the 'yt-dlp' package. "
            "Install it with `pip install yt-dlp`."
        ) from exc

    temp_prefix = Path(tempfile.gettempdir()) / f"yt_video_{int(time.time())}_{os.getpid()}"
    output_tmpl = f"{temp_prefix}.%(ext)s"

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": output_tmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 3,
    }

    def _cleanup_partial_files(prefix: Path) -> None:
        """Clean up partial files created by yt-dlp (including .part, .ytdl, fragments)."""
        try:
            # Find all files matching the prefix pattern
            temp_dir = prefix.parent
            pattern = prefix.name
            for file_path in temp_dir.glob(f"{pattern}*"):
                try:
                    file_path.unlink(missing_ok=True)
                except (OSError, PermissionError):
                    pass  # Ignore errors during cleanup
        except Exception:
            pass  # Ignore cleanup errors

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_path = ydl.prepare_filename(info)
            if info.get("requested_downloads"):
                downloaded_path = info["requested_downloads"][0].get(
                    "filepath", downloaded_path
                )
    except Exception as exc:
        # Clean up any partial files created by yt-dlp before raising
        _cleanup_partial_files(temp_prefix)
        raise ValueError(f"Failed to download YouTube video: {exc}") from exc

    if not os.path.exists(downloaded_path):
        raise ValueError("YouTube download failed to create a video file.")

    max_size_bytes = max_size_mb * 1024 * 1024
    file_size = os.path.getsize(downloaded_path)
    if file_size > max_size_bytes:
        Path(downloaded_path).unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded video size ({file_size / (1024 * 1024):.2f} MB) exceeds "
            f"maximum allowed size ({max_size_mb} MB)"
        )

    mime_type = get_mime_type(downloaded_path, default_mime="video/mp4")
    # Return empty content placeholder to avoid loading large files into memory
    return downloaded_path, mime_type, b""

def get_mime_type(file_path: str, default_mime: str | None = None) -> str:
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
    
def get_file_from_source(
    source: str,
    max_size_mb: float = 100.0,
    timeout: int = 60,
) -> tuple[str, str, bytes]:
    """
    Unified function to get file content from a URL or local path with validation.

    Args:
        source: URL or local file path
        max_size_mb: Maximum allowed file size in MB
        timeout: Timeout for URL requests in seconds

    Returns:
        Tuple[str, str, bytes]: (file_path, mime_type, file_content)
        - For URLs, file_path will be a temporary file path
        - For local files, file_path will be the original path

    Raises:
        ValueError: When file doesn't exist, exceeds size limit, or has invalid MIME type
        IOError: When file cannot be read
        requests.RequestException: When URL request fails
    """
    max_size_bytes = max_size_mb * 1024 * 1024

    if is_url(source):
        # Use yt-dlp for YouTube URLs to fetch the actual video file
        if is_youtube_url(source):
            return download_youtube_video(source, max_size_mb=max_size_mb)

        # Handle URL source
        try:
            # Make a HEAD request first to check content length
            head_response = requests.head(
                source,
                timeout=timeout,
                allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            head_response.raise_for_status()

            # Check content length if available
            content_length = head_response.headers.get("content-length")
            if content_length and int(content_length) > max_size_bytes:
                raise ValueError(
                    f"File size ({int(content_length) / (1024 * 1024):.2f} MB) "
                    f"exceeds maximum allowed size ({max_size_mb} MB)"
                )

            # Download the file
            response = requests.get(
                source,
                timeout=timeout,
                stream=True,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

            # Read content with size checking
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                if len(content) + len(chunk) > max_size_bytes:
                    raise ValueError(f"File size exceeds maximum allowed size ({max_size_mb} MB)")
                content += chunk

            # Create temporary file
            parsed_url = urlparse(source)
            filename = os.path.basename(parsed_url.path) or "downloaded_file"

            # Create temporary file with proper extension
            suffix = Path(filename).suffix or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name

            # Get MIME type
            mime_type = get_mime_type(temp_path)

            # Ensure the downloaded content is a video file
            if (
                mime_type
                and mime_type != "application/octet-stream"
                and not mime_type.startswith("video")
            ):
                Path(temp_path).unlink(missing_ok=True)
                raise ValueError(
                    f"URL does not appear to be a video file. Detected MIME type: {mime_type}"
                )

            return temp_path, mime_type, content

        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to download file from URL: {e}: {traceback.format_exc()}")
        except Exception as e:
            raise IOError(f"Error processing URL: {e}: {traceback.format_exc()}") from e

    else:
        # Handle local file path
        file_path = Path(source)

        # Check if file exists
        if not file_path.exists():
            raise ValueError(f"File does not exist: {source}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {source}")

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > max_size_bytes:
            raise ValueError(
                f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed size ({max_size_mb} MB)"
            )

        # Read file content
        try:
            with open(file_path, "rb") as f:
                content = f.read()
        except Exception as e:
            raise IOError(f"Cannot read file {source}: {e}: {traceback.format_exc()}") from e

        # Get MIME type
        mime_type = get_mime_type(str(file_path))

        return str(file_path), mime_type, content


class VideoAnalysisResult(BaseModel):
    """Video analysis result model with structured data"""

    video_source: str
    analysis_result: str
    frame_count: int
    duration_analyzed: float
    success: bool
    error: str | None = None

class VideoSummaryResult(BaseModel):
    """Video summary result model with structured data"""

    video_source: str
    summary: str
    frame_count: int
    duration_analyzed: float
    success: bool
    error: str | None = None


class VideoMetadata(BaseModel):
    """Metadata for video operation results"""

    operation: str
    video_source: str | None = None
    sample_rate: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    output_directory: str | None = None
    frame_count: int | None = None
    execution_time: float | None = None
    error_type: str | None = None

class Video_Processing_Tool(BaseTool):
    """
    Video_Processing_Tool
    ---------------------
    Purpose:
        A comprehensive video processing tool that analyzes video content using AI, extracts keyframes, and provides detailed video summaries. Supports multiple video formats and AI-powered content analysis with parallel processing capabilities.

    Core Capabilities:
        - AI-powered video content analysis using OpenAI vision models
        - Video summarization with detailed insights
        - Keyframe extraction with scene detection algorithms
        - Multi-format video support (MP4, AVI, MOV, MKV, WebM)
        - Parallel processing for faster analysis
        - Customizable sampling rates and time windows

    Intended Use:
        Use this tool when you need to analyze video content using AI, extract keyframes, and provide detailed video summaries.

    Limitations:
        - Requires OpenAI API key for AI analysis
        - Processing time scales with video length and quality
        - Memory usage increases with high sampling rates
        - Large videos may require significant processing time
        - Some video formats may not be supported on all platforms
        - AI analysis quality depends on video content clarity

    """

    def __init__(self) -> None:
        super().__init__(
            type='function',
            name="Video_Processing_Tool",
            description="""A comprehensive video processing tool that analyzes video content using AI, extracts keyframes, and provides detailed video summaries. Supports multiple video formats and AI-powered content analysis with parallel processing capabilities. CAPABILITIES: AI-powered video content analysis using OpenAI vision models, video summarization with detailed insights, keyframe extraction with scene detection algorithms, multi-format video support (MP4, AVI, MOV, MKV, WebM), parallel processing for faster analysis, customizable sampling rates and time windows. SYNONYMS: video analysis tool, AI video processor, video summarizer, keyframe extractor, video content analyzer, scene detection tool, video AI tool, video processing engine, video analysis platform, intelligent video processor. EXAMPLES: 'Analyze this video and tell me what is happening', 'Summarize the main content of this video', 'Extract keyframes from the 30-second mark with 10-second window'.""",
            parameters={
                "type": "object",
                "properties": {
                    "video_source": {
                        "type": "string",
                        "description": "Path or URL to the video file to process (not a playlist URL)"
                    },
                    "operation": {
                        "type": "string",
                        "description": "Operation to perform: 'analyze' or 'summarize'. Use start_time and end_time parameters if the query asks for a specific time range.",
                        "enum": ["analyze", "summarize"]
                    },
                    "question": {
                        "type": "string",
                        "description": "Question or task for video analysis (required for analyze operation)"
                    },
                    "sample_rate": {
                        "type": "number",
                        "description": "Frame sampling rate (frames per second, default: 1.0)"
                    },
                    "start_time": {
                        "type": "number",
                        "description": "Start time in seconds (default: 0.0). Use this if the query asks for a specific time range or timestamp."
                    },
                    "end_time": {
                        "type": ["number", "null"],
                        "description": "End time in seconds (None for full video). Use this if the query asks for a specific time range or timestamp."
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'markdown', 'json', or 'text' (default: markdown)",
                        "enum": ["markdown", "json", "text"]
                    },
                    "max_workers": {
                        "type": "integer",
                        "description": "Maximum number of parallel workers for analysis (default: 4)"
                    }
                },
                "required": ["video_source", "operation"],
                "additionalProperties": False,
            },
            strict=False,
            category="video_processing",
            tags=["video_analysis", "ai_video_processing", "video_summarization", "scene_detection", "computer_vision", "openai_vision", "parallel_processing", "video_ai", "multimodal_analysis"],
            limitation="Requires OpenAI API key for AI analysis, processing time scales with video length and quality, memory usage increases with high sampling rates, large videos may require significant processing time, some video formats may not be supported on all platforms, AI analysis quality depends on video content clarity",
            agent_type="Media-Agent",
            demo_commands= {
                "command": "reponse = tool.run(video_source='path/to/video', operation='analyze', question='What is happening in the video?')",
                "description": "Analyze the video and tell me what is happening"
            }
        )
        self.supported_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

        # Video analysis prompts
        self.video_analyze_prompt = (
            "Input is a sequence of video frames. Given user's task: {task}. "
            "analyze the video content following these steps:\n"
            "1. Temporal sequence understanding\n"
            "2. Motion and action analysis\n"
            "3. Scene context interpretation\n"
            "4. Object and person tracking\n"
        )

        self.video_summarize_prompt = (
            "Input is a sequence of video frames. "
            "Summarize the main content of the video. "
            "Include key points, main topics, and important visual elements. "
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Setup temporary directory for frame processing
        self.frames_dir = Path(tempfile.gettempdir()) / "video_frames"
        self.frames_dir.mkdir(exist_ok=True)

    def _get_video_frames(
        self,
        video_source: str,
        sample_rate: int = 2,
        start_time: float = 0,
        end_time: float | None = None,
    ) -> list[dict[str, any]]:
        """Extract frames from video with given sample rate.

        Args:
            video_source: Path or URL to the video file
            sample_rate: Number of frames to sample per second
            start_time: Start time of the video segment in seconds
            end_time: End time of the video segment in seconds

        Returns:
            List of dictionaries containing frame data and timestamp

        Raises:
            ValueError: When video file cannot be opened or is not valid
        """
        try:
            # Get file with validation (only video files allowed)
            file_path, mime_type, _ = get_file_from_source(
                video_source,
                max_size_mb=2500.0,  # 2500MB limit for videos
            )

            if (
                mime_type
                and mime_type != "application/octet-stream"
                and not mime_type.startswith("video")
            ):
                raise ValueError(
                    f"Provided source is not a video file (detected MIME type: {mime_type})"
                )

            # Open video file
            video = cv2.VideoCapture(file_path)  # pylint: disable=E1101
            if not video.isOpened():
                raise ValueError(f"Could not open video file: {file_path}")

            fps = video.get(cv2.CAP_PROP_FPS)  # pylint: disable=E1101
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # pylint: disable=E1101
            video_duration = frame_count / fps

            if end_time is None:
                end_time = video_duration

            if start_time > end_time:
                raise ValueError("Start time cannot be greater than end time.")

            if start_time < 0:
                start_time = 0

            if end_time > video_duration:
                end_time = video_duration

            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            all_frames = []
            frames = []

            # Calculate frame interval based on sample rate
            frame_interval = max(1, int(fps / sample_rate))

            # Set the video capture to the start frame
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # pylint: disable=E1101

            for i in range(start_frame, end_frame):
                ret, frame = video.read()
                if not ret:
                    break

                # Resize frame to reduce size (max 512px on longest side)
                height, width = frame.shape[:2]
                max_size = 512
                if max(height, width) > max_size:
                    scale = max_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))  # pylint: disable=E1101

                # Encode frame directly to base64 with optimization
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]  # pylint: disable=E1101
                _, buffer = cv2.imencode(".jpg", frame, encode_params)  # pylint: disable=E1101
                frame_data = base64.b64encode(buffer).decode("utf-8")

                # Add data URL prefix for JPEG image
                frame_data = f"data:image/jpeg;base64,{frame_data}"

                all_frames.append({"data": frame_data, "time": i / fps})

            for i in range(0, len(all_frames), frame_interval):
                frames.append(all_frames[i])

            video.release()

            # Clean up temporary file if it was created for a URL
            if (
                file_path != str(Path(video_source).resolve())
                and Path(file_path).exists()
            ):
                Path(file_path).unlink()

            if not frames:
                raise ValueError(
                    f"Could not extract any frames from video: {video_source}"
                )

            return frames

        except Exception as e:
            raise ValueError(f"Error extracting frames from {video_source}: {str(e)}")
            

    def _create_video_content(
        self, prompt: str, video_frames: list[dict[str, any]]
    ) -> list[dict[str, any]]:
        """Create uniform video format for querying LLM using optimized base64 images."""
        content = [{"type": "text", "text": prompt}]
        content.extend(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": frame["data"], "detail": "low"}
                }
                for frame in video_frames
            ]
        )
        return content

    def _cleanup_frames(self, video_frames: list[dict[str, any]]):
        """Clean up method - no cleanup needed for base64 encoded frames."""
        # No cleanup needed for base64 encoded frames
        pass

    def _format_analysis_output(
        self, result: VideoAnalysisResult, format_type: str = "markdown"
    ) -> str:
        """Format video analysis results for LLM consumption.

        Args:
            result: Video analysis result
            format_type: Output format ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if not result.success:
            return f"Failed to analyze video: {result.error}"

        if format_type == "json":
            return result.model_dump_json(indent=2)

        elif format_type == "text":
            output_parts = [
                "Video Analysis Results",
                f"Source: {result.video_source}",
                f"Frames Analyzed: {result.frame_count}",
                f"Duration: {result.duration_analyzed:.2f} seconds",
                "",
                "Analysis:",
                result.analysis_result,
            ]
            return "\n".join(output_parts)

        else:  # markdown (default)
            output_parts = [
                "# Video Analysis Results ✅",
                "",
                "## Video Information",
                f"**Source:** `{result.video_source}`",
                f"**Frames Analyzed:** {result.frame_count}",
                f"**Duration:** {result.duration_analyzed:.2f} seconds",
                "",
                "## Analysis Results",
                result.analysis_result,
            ]
            return "\n".join(output_parts)

    def _format_summary_output(
        self, result: VideoSummaryResult, format_type: str = "markdown"
    ) -> str:
        """Format video summary results for LLM consumption.

        Args:
            result: Video summary result
            format_type: Output format ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if not result.success:
            return f"Failed to summarize video: {result.error}"

        if format_type == "json":
            return result.model_dump_json(indent=2)

        elif format_type == "text":
            output_parts = [
                "Video Summary",
                f"Source: {result.video_source}",
                f"Frames Analyzed: {result.frame_count}",
                f"Duration: {result.duration_analyzed:.2f} seconds",
                "",
                "Summary:",
                result.summary,
            ]
            return "\n".join(output_parts)

        else:  # markdown (default)
            output_parts = [
                "# Video Summary ✅",
                "",
                "## Video Information",
                f"**Source:** `{result.video_source}`",
                f"**Frames Analyzed:** {result.frame_count}",
                f"**Duration:** {result.duration_analyzed:.2f} seconds",
                "",
                "## Summary",
                result.summary,
            ]
            return "\n".join(output_parts)

    def _analyze_frame_chunk(
        self, chunk_data: tuple[int, list, str]
    ) -> tuple[int, str]:
        """Analyze a chunk of video frames using OpenAI API.

        Args:
            chunk_data: Tuple containing (chunk_index, frames, question)

        Returns:
            Tuple of (chunk_index, analysis_result)
        """
        chunk_index, frames, question = chunk_data

        try:
            content = self._create_video_content(
                self.video_analyze_prompt.format(task=question), frames
            )

            # Use OpenAI API directly for multimodal content
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": content}
                ],
                temperature=0,
            )
            print(f"Prompt tokens:     {response.usage.prompt_tokens}")
            print(f"Completion tokens: {response.usage.completion_tokens}")
            print(f"Total tokens:      {response.usage.total_tokens}")
            analysis_result = response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            # Check if it's a context length error
            if "context_length" in error_msg.lower() or "context window" in error_msg.lower() or "exceeds" in error_msg.lower():
                raise ValueError(f"Video chunk {chunk_index + 1} exceeded context window. Chunk size may be too large. Error: {error_msg}")
            raise ValueError(f"Error analyzing video chunk {chunk_index + 1}: {error_msg}")

        return chunk_index, analysis_result

    def run(
        self,
        video_source: str = Field(description="Path or URL to the video file to process"),
        operation: str = Field(description="Operation to perform: 'analyze' or 'summarize'. Use start_time and end_time parameters if the query asks for a specific time range."),
        question: str = Field(default="", description="Question or task for video analysis (for analyze operation)"),
        sample_rate: float = Field(default=1.0, description="Frame sampling rate (frames per second)"),
        start_time: float = Field(default=0.0, description="Start time in seconds. Use this if the query asks for a specific time range or timestamp."),
        end_time: float | None = Field(default=None, description="End time in seconds (None for full video). Use this if the query asks for a specific time range or timestamp."),
        output_format: str = Field(default="markdown", description="Output format: 'markdown', 'json', or 'text'"),
        max_workers: int = Field(default=4, description="Maximum number of parallel workers for analysis"),
    ):
        """Unified video processing method that activates different functions based on parameters.

        This tool provides comprehensive video processing capabilities including:
        - Analysis: AI-powered video content analysis with custom questions
        - Summarization: Generate detailed video summaries
        Use start_time and end_time parameters if the query asks for a specific time range or timestamp.

        Args:
            video_source: Path or URL to the video file to process
            operation: Type of operation to perform
            question: Question for video analysis (required for analyze operation)
            sample_rate: Frame sampling rate for processing
            start_time: Start time for video segment analysis (use if query asks for specific time range)
            end_time: End time for video segment analysis (use if query asks for specific time range)
            output_format: Format for output results
            max_workers: Number of parallel workers for analysis

        Returns:
            Dictionary containing processed results and metadata
        """
        try:
            # Handle FieldInfo objects
            if isinstance(video_source, FieldInfo):
                video_source = video_source.default
            if isinstance(operation, FieldInfo):
                operation = operation.default
            if isinstance(question, FieldInfo):
                question = question.default
            if isinstance(sample_rate, FieldInfo):
                sample_rate = sample_rate.default
            if isinstance(start_time, FieldInfo):
                start_time = start_time.default
            if isinstance(end_time, FieldInfo):
                end_time = end_time.default
            if isinstance(output_format, FieldInfo):
                output_format = output_format.default
            if isinstance(max_workers, FieldInfo):
                max_workers = max_workers.default

            print(f"Processing video: {video_source} with operation: {operation}")

            # Perform operation based on parameter
            if operation == "analyze":
                if not question:
                    raise ValueError("Question is required for analyze operation")
                return self.mcp_analyze_video(
                    video_url=video_source,
                    question=question,
                    sample_rate=sample_rate,
                    start_time=start_time,
                    end_time=end_time,
                    output_format=output_format,
                    max_workers=max_workers
                )
            
            elif operation == "summarize":
                return self.mcp_summarize_video(
                    video_url=video_source,
                    sample_rate=int(sample_rate),
                    start_time=start_time,
                    end_time=end_time,
                    output_format=output_format
                )
            
            else:
                return {"error": f"Unsupported operation: {operation}", "success": False, "error_type": "unsupported_operation"}

        except Exception as e:
            return {"error": f"Video processing failed: {str(e)}", "success": False, "error_type": "processing_error", "traceback": traceback.format_exc()}

    def mcp_analyze_video(
        self,
        video_url: str = Field(description="Path or URL to the video file to analyze"),
        question: str = Field(description="Question or task for video analysis"),
        sample_rate: float = Field(
            default=1.0, description="Frame sampling rate (frames per second)"
        ),
        start_time: float = Field(default=0.0, description="Start time in seconds"),
        end_time: float | None = Field(
            default=None, description="End time in seconds (None for full video)"
        ),
        output_format: str = Field(
            default="markdown",
            description="Output format: 'markdown', 'json', or 'text'",
        ),
        max_workers: int = Field(
            default=4, description="Maximum number of parallel workers for analysis"
        ),
    ):
        """Analyze video content using AI with parallel processing.

        This tool provides comprehensive video analysis capabilities including:
        - Content understanding and description
        - Object and scene detection
        - Action and movement analysis
        - Temporal event tracking
        - Question-answering about video content
        - Parallel processing for faster analysis

        Args:
            video_url: Path or URL to the video file
            question: Specific question or analysis task
            sample_rate: Frame sampling rate for analysis
            start_time: Start time of the video segment in seconds
            end_time: End time of the video segment in seconds
            output_format: Format for the response output
            max_workers: Maximum number of parallel workers

        Returns:
            ActionResponse with video analysis results and metadata
        """
        start_exec_time = time.time()

        try:
            # Validate video file
            video_path = video_url

            # Extract video frames
            video_frames = self._get_video_frames(
                str(video_path), sample_rate, start_time, end_time
            )

            # Dynamically calculate chunk size based on model's context window
            # GPT-4o-mini has ~128k token context window
            # Each image with "low" detail ≈ 85 tokens
            # Prompt text ≈ 100-500 tokens
            # Leave room for response tokens (~2000 tokens)
            model_context_window = 128000  # GPT-4o-mini context window
            tokens_per_frame_low_detail = 85  # Approximate tokens per frame with "low" detail
            prompt_overhead = 500  # Estimated tokens for prompt text
            response_reserve = 2000  # Reserve tokens for response
            max_frames_per_chunk = int((model_context_window - prompt_overhead - response_reserve) / tokens_per_frame_low_detail)
            
            # Use a conservative chunk size (70% of max to be safe and avoid context errors)
            chunk_size = max(1, int(max_frames_per_chunk * 0.7))
            
            print(f"Calculated chunk size: {chunk_size} frames (max would be {max_frames_per_chunk} frames, total frames: {len(video_frames)})")
            
            chunks = []

            # Create chunks of dynamically calculated size
            for i in range(0, len(video_frames), chunk_size):
                chunk_frames = video_frames[i : i + chunk_size]
                chunks.append((i // chunk_size, chunk_frames, question))

            # Process chunks in parallel
            all_results = [None] * len(chunks)  # Pre-allocate to maintain order

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunk analysis tasks
                future_to_chunk = {
                    executor.submit(self._analyze_frame_chunk, chunk_data): chunk_data[
                        0
                    ]
                    for chunk_data in chunks
                }

                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_index, result = future.result()
                        all_results[chunk_index] = (
                            f"Result of video part {chunk_index + 1}: {result}"
                        )
                    except Exception as e:
                        chunk_index = future_to_chunk[future]
                        all_results[chunk_index] = (
                            f"Result of video part {chunk_index + 1}: Analysis failed - {str(e)}"
                        )

            # Filter out None results and join
            analysis_result = "\n".join(
                [result for result in all_results if result is not None]
            )
            duration_analyzed = (
                end_time - start_time if end_time else len(video_frames) / sample_rate
            )

            # Create result
            result = VideoAnalysisResult(
                video_source=video_url,
                analysis_result=analysis_result,
                frame_count=len(video_frames),
                duration_analyzed=duration_analyzed,
                success=True,
                error=None,
            )

            # Format output for LLM
            message = self._format_analysis_output(result, output_format)
            execution_time = time.time() - start_exec_time

            # Create metadata
            metadata = {
                "video_source": video_url,
                "frame_count": len(video_frames),
                "chunks_processed": len(chunks),
                "chunk_size": chunk_size,
                "parallel_workers": max_workers,
                "duration_analyzed": duration_analyzed,
                "sample_rate": sample_rate,
                "start_time": start_time,
                "end_time": end_time,
                "execution_time": execution_time,
                "output_format": output_format,
                "success": True,
            }

            # Cleanup temporary frame files
            self._cleanup_frames(video_frames)

            return {"result": message, "metadata": metadata, "success": True}

        except Exception as e:
            execution_time = time.time() - start_exec_time
            error_msg = f"Video analysis failed: {str(e)}"
            # Cleanup frames even on error
            try:
                self._cleanup_frames(video_frames)
            except:
                pass
            return {"error": error_msg, "success": False, "error_type": "analysis_error", "traceback": traceback.format_exc()}

    def mcp_summarize_video(
        self,
        video_url: str = Field(
            description="The input video filepath or URL to summarize."
        ),
        sample_rate: int = Field(
            default=1, description="Sample n frames per second (default: 1)."
        ),
        start_time: float = Field(
            default=0,
            description="Start time of the video segment in seconds (default: 0).",
        ),
        end_time: float | None = Field(
            default=None,
            description="End time of the video segment in seconds (default: None).",
        ),
        output_format: str = Field(
            default="markdown",
            description="Output format: 'markdown', 'json', or 'text' (default: markdown).",
        ),
    ) :
        """Summarize the main content of a video using AI analysis.

        This tool provides AI-powered video summarization with:
        - Key point extraction
        - Main topic identification
        - Important visual element recognition
        - LLM-optimized result formatting

        Args:
            video_url: The input video filepath or URL to summarize
            sample_rate: Sample n frames per second
            start_time: Start time of the video segment in seconds
            end_time: End time of the video segment in seconds
            output_format: Format for the response output

        Returns:
            ActionResponse with video summary results and metadata
        """
        start_exec_time = time.time()

        try:
            # Validate video file
            video_path = video_url

            # Extract video frames
            video_frames = self._get_video_frames(
                str(video_path), sample_rate, start_time, end_time
            )

            # Process frames in larger chunks for summarization
            interval = 490
            frame_nums = 500
            all_results = []

            for i in range(0, len(video_frames), interval):
                cur_frames = video_frames[i : i + frame_nums]
                content = self._create_video_content(
                    self.video_summarize_prompt, cur_frames
                )

                try:
                    # Use OpenAI API directly for multimodal content
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "user", "content": content}
                        ],
                        temperature=0,
                        max_tokens=4000,
                    )
                    
                    cur_summary = response.choices[0].message.content
                except Exception as e:
                    raise ValueError(f"Error summarizing video chunk {i // interval + 1}: {str(e)}")
                    cur_summary = (
                        f"Summary failed for video segment {i // interval + 1}"
                    )

                all_results.append(
                    f"Summary of video part {i // interval + 1}: {cur_summary}"
                )

                if i + frame_nums >= len(video_frames):
                    break

            summary_result = "\n".join(all_results)
            duration_analyzed = (
                end_time - start_time if end_time else len(video_frames) / sample_rate
            )

            # Create result
            result = VideoSummaryResult(
                video_source=video_url,
                summary=summary_result,
                frame_count=len(video_frames),
                duration_analyzed=duration_analyzed,
                success=True,
                error=None,
            )

            # Format output for LLM
            message = self._format_summary_output(result, output_format)
            execution_time = time.time() - start_exec_time

            # Create metadata
            metadata = VideoMetadata(
                operation="summarize",
                video_source=video_url,
                sample_rate=sample_rate,
                start_time=start_time,
                end_time=end_time,
                frame_count=len(video_frames),
                execution_time=execution_time,
            ).model_dump()

            # Cleanup temporary frame files
            self._cleanup_frames(video_frames)

            return { "result": message, "metadata": metadata, "success": True}

        except Exception as e:
            error_msg = str(e)
            # Cleanup frames even on error
            try:
                self._cleanup_frames(video_frames)
            except:
                pass
            return { "error": error_msg, "success": False, "error_type": "summarization_error", "traceback": traceback.format_exc()}

# Default arguments for testing
if __name__ == "__main__":

    try:
        tool = Video_Processing_Tool()
        # tool.embed_tool()
        print(tool.run(
            video_source="https://www.youtube.com/watch?v=t7AtQHXCW5s",
            operation="analyze",
            question="a phrase is shown on the screen in white letters on a red background. How many times does the letter \"E\" appear in this phrase?",
            start_time=29,
            end_time=32
        ))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
