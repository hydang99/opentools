# source code: https://github.com/inclusionAI/AWorld/blob/main/examples/gaia/mcp_collections/tools/youtube.py
import os, time, traceback, json, sys
from youtube_transcript_api import YouTubeTranscriptApi
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
from pydantic import Field, BaseModel
from pydantic.fields import FieldInfo

class TranscriptResult(BaseModel):
    """Transcript result model with transcript information"""

    video_id: str
    transcript: object | None = None
    success: bool
    error: str | None = None


class YouTubeMetadata(BaseModel):
    """Metadata model with operation information"""

    operation: str
    url: str | None = None
    video_id: str | None = None
    file_path: str | None = None
    file_name: str | None = None
    file_size: int | None = None
    content_type: str | None = None
    language_code: str | None = None
    translate_to_language: str | None = None
    execution_time: float
    error_type: str | None = None


class Youtube_Tool(BaseTool):
    """Youtube_Tool
    ---------------------
    Purpose:
        A comprehensive YouTube transcript extraction tool that extracts video transcripts directly from YouTube video URLs or video IDs. Supports multiple languages with language code specification, translation capabilities to convert transcripts between different languages, and automatic video ID extraction from full YouTube URLs.

    Core Capabilities:
        - Extract transcripts from YouTube videos using video URLs or video IDs
        - Support for multiple languages with language code specification
        - Translation capabilities to convert transcripts between different languages
        - Automatic video ID extraction from full YouTube URLs
        - LLM-friendly output formatting

    Intended Use:
        Use this tool when you need to extract video transcripts from YouTube videos, including multiple languages, translation, and automatic video ID extraction.

    Limitations:
        - Cannot extract transcripts from videos without available subtitles
        - Translation depends on YouTube's available transcript languages
        - Some videos may have auto-generated transcripts only
        - Requires internet connection for YouTube API access
        - Some videos may have limited transcript availability
        - Some videos may have auto-generated transcripts only
    """
    def __init__(self) -> None:
        super().__init__(
            type='function',
            name="Youtube_Tool",
            description="""YouTube transcript extraction tool for reading video transcripts directly from URLs. 
            This tool focuses solely on transcript extraction. 
            CAPABILITIES: Extract transcripts from YouTube videos using video URLs or video IDs, support for multiple languages 
            with language code specification, translation capabilities to convert transcripts between different languages, 
            automatic video ID extraction from full YouTube URLs. 
            SYNONYMS: YouTube transcript extractor, video transcript tool, YouTube subtitle extractor, video caption tool, 
            YouTube transcript reader, video transcript analyzer, YouTube subtitle tool, transcript extraction tool, 
            YouTube caption extractor, video transcript fetcher. 
            EXAMPLES: 'Extract transcript from YouTube video URL', 'Get transcript in Spanish language', 
            'Translate transcript to French', 'Extract transcript in JSON format'.""",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "YouTube video URL for transcript extraction (e.g., 'https://www.youtube.com/watch?v=dQw4w9WgXcQ')"
                    },
                    "language_code": {
                        "type": "string",
                        "description": "Language code for transcript extraction (default: 'en')"
                    },
                    "translate_to_language": {
                        "type": ["string", "null"],
                        "description": "Translate transcript to this language code if provided"
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'markdown', 'json', or 'text' (default: 'json')",
                        "enum": ["markdown", "json", "text"]
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            strict=False,
            category="media_extraction",
            tags=["youtube_transcript", "transcript_extraction", "video_subtitles", "caption_extraction", "youtube_api", "transcript_analysis", "video_content", "subtitle_tools", "transcript_tools", "youtube_tools"],
            limitation="Cannot extract transcripts from videos without available subtitles, translation depends on YouTube's available transcript languages, some videos may have auto-generated transcripts only, requires internet connection for YouTube API access",
            agent_type="Media-Agent",
            demo_commands= {
                "command": "reponse = tool.run(url='https://www.youtube.com/watch?v=dQw4w9WgXcQ')",
                "description": "Extract transcript from YouTube video URL"
            }
        )

    def _format_transcript_output(self, result: TranscriptResult, format_type: str = "markdown") -> str:
        """Format transcript results for LLM consumption.

        Args:
            result: Transcript extraction result
            format_type: Output format ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if not result.success:
            return f"Failed to extract transcript: {result.error}"

        if format_type == "json":
            return json.dumps(result.model_dump(), indent=2)
        elif format_type == "text":
            if result.transcript:
                transcript_text = " ".join([entry['text'] for entry in result.transcript])
                return f"Video ID: {result.video_id}\n\nTranscript:\n{transcript_text}"
            else:
                return f"No transcript available for video ID: {result.video_id}"
        else:  # markdown (default)
            try:
                if result.transcript:
                    output_parts = [
                        f"# YouTube Transcript - Video ID: {result.video_id}",
                        "",
                        "## Transcript Content",
                        ""
                    ]
                    
                    for entry in result.transcript:
                        timestamp = entry.get('start', 0)
                        text = entry.get('text', '')
                        minutes = int(timestamp // 60)
                        seconds = int(timestamp % 60)
                        output_parts.append(f"**[{minutes:02d}:{seconds:02d}]** {text}")
                    
                    return "\n".join(output_parts)
                else:
                    return f"# YouTube Transcript - Video ID: {result.video_id}\n\nNo transcript available for this video."
            except Exception as e:
                return f"Error formatting transcript as markdown: {str(e)}"

    def run(
        self,
        url: str = Field(description="YouTube video URL for transcript extraction (e.g., 'https://www.youtube.com/watch?v=dQw4w9WgXcQ')"),
        language_code: str = Field(default="en", description="Language code for transcript extraction (default: en)"),
        translate_to_language: str | None = Field(default=None, description="Translate transcript to this language code if provided"),
        output_format: str = Field(default="json", description="Output format: 'markdown', 'json', or 'text' (default: json)"),
    ):
        """Main entry point for YouTube transcript extraction.
        
        Args:
            url: YouTube video URL for transcript extraction
            language_code: Language code for transcript extraction
            translate_to_language: Translate transcript to this language code if provided
            output_format: Output format for the response
            
        Returns:
            Dictionary with transcript data and metadata
        """
        start_time = time.time()

        try:
            # Handle FieldInfo objects
            if isinstance(url, FieldInfo):
                url = url.default
            if isinstance(language_code, FieldInfo):
                language_code = language_code.default
            if isinstance(translate_to_language, FieldInfo):
                translate_to_language = translate_to_language.default
            if isinstance(output_format, FieldInfo):
                output_format = output_format.default

            # Validate URL
            if not url:
                return {
                    "error": "Error: 'url' parameter is REQUIRED. Example: tool.run(url='https://www.youtube.com/watch?v=dQw4w9WgXcQ')",
                    "metadata": YouTubeMetadata(
                        operation="extract_transcript",
                        url=url or "",
                        execution_time=time.time() - start_time,
                        error_type="missing_url",
                    ).model_dump(),
                    "success": False
                }
            
            # Extract transcript using the URL (the function can handle URLs)
            return self.mcp_extract_youtube_transcript(url, language_code, translate_to_language, output_format)

        except Exception as e:
            error_msg = f"YouTube transcript extraction failed: {str(e)}"
            print(f"Error in run: {traceback.format_exc()}")
            return {
                "error": error_msg,
                "metadata": YouTubeMetadata(
                    operation="extract_transcript",
                    url=url,
                    execution_time=time.time() - start_time,
                    error_type=type(e).__name__,
                ).model_dump(),
                "success": False,
                "traceback": traceback.format_exc()
            }

    def mcp_extract_youtube_transcript(
        self,
        video_id: str = Field(description="The YouTube video ID or URL to extract transcript from."),
        language_code: str = Field("en", description="Language code for the transcript (default: en)."),
        translate_to_language: str | None = Field(
            None, description="Translate transcript to this language code if provided."
        ),
        output_format: str = Field(
            "json", description="Output format: 'markdown', 'json', or 'text' (default: json)."
        ),
    ):
        """Extract transcript from a YouTube video given its video ID or URL.

        This tool provides transcript extraction with:
        - Support for multiple languages
        - Translation capabilities
        - URL or video ID input handling
        - LLM-optimized result formatting

        Args:
            video_id: The YouTube video ID or URL to extract transcript from
            language_code: Language code for the transcript
            translate_to_language: Translate transcript to this language code if provided
            output_format: Format for the response output

        Returns:
            Dictionary with transcript data and metadata
        """
        start_time = time.time()

        try:
            # Clean video_id if full URL was provided
            if "youtube.com" in video_id or "youtu.be" in video_id:
                if "?v=" in video_id:
                    video_id = video_id.split("?v=")[-1].split("&")[0]
                elif "youtu.be/" in video_id:
                    video_id = video_id.split("youtu.be/")[-1].split("?")[0]

            print(f"Extracting transcript for video ID: {video_id}")

            # Get transcript in specified language
            if translate_to_language:
                # Get transcript and translate it
                ytt_api = YouTubeTranscriptApi()
                transcript_list = ytt_api.list(video_id)
                transcript = None

                try:
                    # Find transcript and translate it
                    available_transcript = transcript_list.find_transcript([language_code])
                    translated_transcript = available_transcript.translate(translate_to_language)
                    transcript = translated_transcript.fetch().to_raw_data()

                except Exception as e:
                    print(f"Translation failed: {str(e)}")
                    # Fallback to original language
                    ytt_api = YouTubeTranscriptApi()
                    transcript = ytt_api.fetch(video_id, languages=[language_code]).to_raw_data()
            else:
                # Get transcript without translation
                ytt_api = YouTubeTranscriptApi()
                transcript = ytt_api.fetch(video_id, languages=[language_code]).to_raw_data()

            # Check if transcript is valid (not None and not empty list)
            if transcript and (not isinstance(transcript, list) or len(transcript) > 0):
                result = TranscriptResult(video_id=video_id, transcript=transcript, success=True, error=None)
            else:
                print(f"No transcript data found for video ID: {video_id}")
                result = TranscriptResult(video_id=video_id, transcript=None, success=False, error="No transcript available")

        except Exception as e:
            error_msg = str(e)
            print(f"Error during transcript extraction: {error_msg}")
            
            # Check for various transcript unavailable error messages
            if any(phrase in error_msg for phrase in ["Could not retrieve a transcript", "Subtitles are disabled", "No transcript available", "No transcripts were found"]):
                print(f"No transcript data found for video ID: {video_id}")
                result = TranscriptResult(video_id=video_id, transcript=None, success=False, error="No transcript available")
            else:
                result = TranscriptResult(video_id=video_id, transcript=None, success=False, error=error_msg)
        execution_time = time.time() - start_time

        # Format output for LLM
        if not result.success:
            if "error" in result:
                error_msg = f"Failed to extract transcript: {result.error}"
            else:
                error_msg = "Failed to extract transcript: The video does not have a transcript available."
            return {

                "error": error_msg,
                "metadata": YouTubeMetadata(
                    operation="extract_transcript",
                    url=video_id,
                    execution_time=execution_time,
                    error_type=error_msg,
                ).model_dump(),
                "success": False,
                "traceback": traceback.format_exc()
            }

        message = self._format_transcript_output(result, output_format)

        # Create metadata
        metadata = YouTubeMetadata(
            operation="transcript",
            url=None,
            video_id=video_id,
            language_code=language_code,
            translate_to_language=translate_to_language,
            execution_time=execution_time,
            error_type=None,
        ).model_dump()

        return {
            "result": message,
            "metadata": metadata,
            "success": result.success,
        }

# Default arguments for testing
if __name__ == "__main__":
    try:
        tool = Youtube_Tool()
        # tool.embed_tool()
        print(tool.run(url="https://www.youtube.com/watch?v=L1vXCYZAYYM"))
    except Exception as e:
        print(f"Error: {e}")
