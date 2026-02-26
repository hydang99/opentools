# source code: https://github.com/inclusionAI/AWorld/blob/main/examples/gaia/mcp_collections/media/audio.py
import json, os , subprocess, time, traceback, sys
from pathlib import Path
from typing import Any, Literal
from openai import OpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

class AudioMetadata(BaseModel):
    """Metadata extracted from audio processing."""

    file_name: str = Field(description="Original audio file name")
    file_size: int = Field(description="File size in bytes")
    file_type: str = Field(description="Audio file type/extension")
    absolute_path: str = Field(description="Absolute path to the audio file")
    duration: float | None = Field(default=None, description="Duration of audio in seconds")
    sample_rate: int | None = Field(default=None, description="Audio sample rate in Hz")
    channels: int | None = Field(default=None, description="Number of audio channels")
    bitrate: int | None = Field(default=None, description="Audio bitrate in kbps")
    codec: str | None = Field(default=None, description="Audio codec used")
    processing_time: float = Field(description="Time taken to process the audio in seconds")
    output_files: list[str] = Field(default_factory=list, description="Paths to generated output files")
    transcription: str | None = Field(default=None, description="Transcribed text from audio")
    word_count: int | None = Field(default=None, description="Number of words in transcription")
    output_format: str = Field(description="Format of the processed output")


class Audio_Processing_Tool(BaseTool):
    """
    Audio_Processing_Tool
    ---------------------
    Purpose:
        A versatile and comprehensive local audio utility designed for end-to-end speech and sound file processing, supporting a wide range of workflows from transcription to conversion and detailed technical analysis.

    Core Capabilities:
        - **Automatic Speech Transcription**: Transcribes audio content to text using state-of-the-art OpenAI Whisper, enabling both simple (text) and verbose (detailed with timestamps, etc.) output formats.
        - **Metadata Extraction**: Rapidly extracts detailed information from audio files, including duration, sample rate, channels, bitrate, file type, codec, and file size, suitable for cataloging, validation, or QA purposes.
        - **Audio Trimming**: Precisely trims audio files by specifying a start time and duration, creating new audio segments without altering the original file. Beneficial for extracting samples or shortening recordings.
        - **Format Conversion**: Converts audio files between major formats (e.g., MP3, WAV, FLAC, AAC, OGG, M4A, WMA, OPUS, AIFF, AU, RA, AMR). Supports adjusting output sample rate and channels for compatibility or quality optimization.
        - **Audio Analysis**: Analyzes the quality and technical characteristics of audio files, such as bitrate consistency, channel balance, or basic waveform statistics.
        - **Multi-format Output**: Flexible output customization for downstream analysis or machine learning workflows, and supports both summary and rich detailed JSON-style output.

    Features:
        - Local, offline processing: No cloud uploads required for all operations except transcription (if using OpenAI Whisper API).
        - Seamlessly handles most common modern and legacy audio formats.
        - Secure and privacy-local when using local resources for all operations except transcription.
        - Well-suited for batch processing and integration into agent pipelines.

    Intended Use:
        Use this tool for any workflow involving audio file preprocessing, speech-to-text, auditor QA, digital archiving, or format conversion. Particularly valuable for:
            • Transcribing interviews, meetings, podcasts, voice notes, or customer calls.
            • Extracting or analyzing metadata for compliance, search, or sorting.
            • Converting or trimming audio to meet digital publishing or ML dataset requirements.
            • Analyzing audio files before sharing, archiving, or further AI processing.

    Limitations:
        - Requires ffmpeg installed and available on the host system for most operations.
        - Transcription via Whisper requires a valid OpenAI API key and internet connectivity.
        - Audio analysis limited to basic technical metrics; no music recognition or content-based audio analysis.
        - May not handle proprietary or obscure audio codecs; for best results, use widely supported formats.
    """
    # Default args for `opentools test Audio_Processing_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "audio_processing",
        "file_location": "audio_processing",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }

    def __init__(self):
        super().__init__(
            type='function',
            name="Audio_Processing_Tool",
            description="""A comprehensive audio processing tool that supports transcription, metadata extraction, trimming, format conversion, and audio analysis. Uses ffmpeg for audio processing and OpenAI Whisper for transcription. CAPABILITIES: Transcribes audio to text using OpenAI Whisper, extracts detailed audio metadata (duration, sample rate, channels, bitrate, codec), trims audio files to specific time ranges, converts between audio formats (MP3, WAV, FLAC, AAC, OGG, M4A, WMA, OPUS, AIFF, AU, RA, AMR), analyzes audio quality and characteristics. SYNONYMS: audio transcription, speech to text, audio  converter, audio trimmer, audio metadata extractor, audio analyzer, audio format converter, speech recognition, audio processing. EXAMPLES: 'Transcribe this audio file to text', 'Convert this MP3 to WAV format', 'Trim this audio from 10 seconds for 30 seconds', 'Extract metadata from this audio file',  'Analyze the quality of this audio file'.""",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the audio file to process"
                    },
                    "operation": {
                        "type": "string",
                        "description": "Operation to perform: 'transcribe', 'metadata', 'trim', 'convert', 'analyze'",
                        "enum": ["transcribe", "metadata", "trim", "convert", "analyze"]
                    },
                    "output_format": {
                         "type": "string",
                         "description": "Output format for transcription: 'text' or 'detailed'",
                         "enum": ["text", "detailed"]
                     },
                    "start_time": {
                        "type": "number",
                        "description": "Start time in seconds for trimming (optional). Required when operation is 'trim'"
                    },
                    "duration": {
                         "type": "number",
                         "description": "Duration in seconds for trimming (required for trim operation)"
                     },
                    "target_format": {
                        "type": "string",
                        "description": "Target format for conversion (e.g., 'mp3', 'wav', 'flac'). Required when operation is 'convert'"
                    },
                    "sample_rate": {
                        "type": "integer",
                        "description": "Target sample rate for conversion (optional)"
                    },
                    "channels": {
                        "type": "integer",
                        "description": "Number of channels for conversion (optional)"
                    }
                },
                "required": ["file_path", "operation"],
                "additionalProperties": False,
            },
            strict=False,
            category="audio",
            tags=["audio_processing", "transcription", "format_conversion", "audio_analysis", "ffmpeg", "whisper"],
            limitation="Requires ffmpeg to be installed on the system and OpenAI API key for transcription",
            agent_type="Media-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(operation='transcribe', file_path='test.mp3')",
                "description": "Transcribe a test.mp3 file"
            }   
        )
        self.workspace = Path(os.getcwd())
        self._audio_output_dir = self.workspace / "processed_audio"

        # Supported audio formats
        self.supported_extensions = {
            ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", 
            ".opus", ".aiff", ".au", ".ra", ".amr"
        }
        
        # Check ffmpeg availability
        self.ffmpeg_available = self._check_ffmpeg_availability()
        if not self.ffmpeg_available:
            print("Warning: ffmpeg not found. Some audio processing features may not work.")
            print("Please install ffmpeg to enable full audio processing capabilities.")

    def _check_ffmpeg_availability(self) -> bool:
        """Check if ffmpeg is available in the system."""
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10, check=False)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _validate_file_path(self, file_path: str) -> Path:
        """Validate and resolve file path."""
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

    def _get_audio_metadata(self, file_path: Path) -> dict[str, Any]:
        """Extract audio metadata using ffprobe."""
        # If ffmpeg is not available, return basic file info
        if not self.ffmpeg_available:
            return {
                "duration": None,
                "sample_rate": None,
                "channels": None,
                "bitrate": None,
                "codec": None,
            } 
        try:
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)

            if result.returncode == 0:
                metadata = json.loads(result.stdout)
                format_info = metadata.get("format", {})
                streams = metadata.get("streams", [])
                audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

                # Safely extract values with proper type conversion
                duration_str = format_info.get("duration")
                duration = float(duration_str) if duration_str else 0.0
                
                sample_rate_str = audio_stream.get("sample_rate")
                sample_rate = int(sample_rate_str) if sample_rate_str else None
                
                channels_str = audio_stream.get("channels")
                channels = int(channels_str) if channels_str else None
                
                bitrate_str = format_info.get("bit_rate")
                bitrate = int(bitrate_str) // 1000 if bitrate_str else None
                
                codec = audio_stream.get("codec_name")

                return {
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "bitrate": bitrate,
                    "codec": codec,
                }
            else:
                print(f"ffprobe failed with return code {result.returncode}: {result.stderr}")
                return {}

        except FileNotFoundError:
            print("ffprobe not found. Please install ffmpeg to enable audio metadata extraction.")
            return {}
        except Exception as e:
            print(f"Error extracting audio metadata: {str(e)}")
            return {}

    def _prepare_audio_for_transcription(self, file_path: Path) -> Path:
        """Prepare audio file for transcription by converting to optimal format."""
        self._audio_output_dir.mkdir(exist_ok=True)
        output_path = self._audio_output_dir / f"{file_path.stem}_for_transcription.wav"

        cmd = [
            "ffmpeg", "-i", str(file_path), "-ar", "16000", "-ac", "1", 
            "-c:a", "pcm_s16le", "-y", str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)

        if result.returncode != 0:
            raise RuntimeError(f"Audio preparation for transcription failed: {result.stderr}")

        return output_path

    def _transcribe_with_whisper(self, audio_path: Path) -> dict[str, Any]:
        """Transcribe audio using OpenAI Whisper."""
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            with open(audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="text",
                )

            return {"text": transcription.strip() if transcription else ""}
        except Exception as e:
            raise RuntimeError(f"Audio transcription failed: {e}") from e

    def _trim_audio(self, input_path: Path, start_time: float, duration: float | None = None) -> Path:
        """Trim audio file to specified time range."""
        self._audio_output_dir.mkdir(exist_ok=True)
        output_path = self._audio_output_dir / f"{input_path.stem}_trimmed{input_path.suffix}"

        cmd = ["ffmpeg", "-i", str(input_path), "-ss", str(start_time), "-y"]

        if duration is not None:
            cmd.extend(["-t", str(duration)])

        cmd.extend(["-c", "copy", str(output_path)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)

        if result.returncode != 0:
            raise RuntimeError(f"Audio trimming failed: {result.stderr}")

        return output_path

    def _convert_audio_format(self, input_path: Path, target_format: str, sample_rate: int | None = None, channels: int | None = None) -> Path:
        """Convert audio to different format with optional quality settings."""
        self._audio_output_dir.mkdir(exist_ok=True)
        output_path = self._audio_output_dir / f"{input_path.stem}_converted.{target_format}"

        cmd = ["ffmpeg", "-i", str(input_path), "-y"]

        if sample_rate:
            cmd.extend(["-ar", str(sample_rate)])
        if channels:
            cmd.extend(["-ac", str(channels)])

        cmd.append(str(output_path))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)

        if result.returncode != 0:
            raise RuntimeError(f"Audio conversion failed: {result.stderr}")

        return output_path

    def _analyze_audio(self, file_path: Path) -> dict[str, Any]:
        """Analyze audio file for various characteristics."""
        metadata = self._get_audio_metadata(file_path)
        
        # Calculate additional analysis metrics
        analysis = {
            "duration_minutes": metadata.get("duration", 0) / 60 if metadata.get("duration") else 0,
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "quality_score": self._calculate_quality_score(metadata),
            "format_efficiency": self._calculate_format_efficiency(file_path.suffix.lower(), metadata),
        }
        
        return {**metadata, **analysis}

    def _calculate_quality_score(self, metadata: dict) -> float:
        """Calculate a quality score based on audio characteristics."""
        score = 0.0
        
        if metadata.get("sample_rate"):
            if metadata["sample_rate"] >= 44100:
                score += 0.4
            elif metadata["sample_rate"] >= 22050:
                score += 0.2
        
        if metadata.get("channels"):
            if metadata["channels"] >= 2:
                score += 0.3
            else:
                score += 0.1
        
        if metadata.get("bitrate"):
            if metadata["bitrate"] >= 320:
                score += 0.3
            elif metadata["bitrate"] >= 128:
                score += 0.2
            else:
                score += 0.1
        
        return min(score, 1.0)

    def _calculate_format_efficiency(self, format_type: str, metadata: dict) -> str:
        """Calculate format efficiency rating."""
        if format_type in [".flac", ".wav"]:
            return "Lossless"
        elif format_type in [".mp3", ".aac", ".ogg"]:
            return "Compressed"
        else:
            return "Unknown"

    def run(
        self,
        file_path: str = Field(description="Path to the audio file to process"),
        operation: Literal["transcribe", "metadata", "trim", "convert", "analyze"] = Field(
            default="metadata", description="Operation to perform"
        ),
        output_format: Literal["text", "detailed"] = Field(
            default="text", description="Output format for transcription"
        ),
        start_time: float | None = Field(default=None, description="Start time in seconds for trimming"),
        duration: float | None = Field(default=None, description="Duration in seconds for trimming"),
        target_format: str | None = Field(default=None, description="Target format for conversion"),
        sample_rate: int | None = Field(default=None, description="Target sample rate for conversion"),
        channels: int | None = Field(default=None, description="Number of channels for conversion"),
    ):
        """Unified audio processing method that activates different functions based on parameters.

        This tool provides comprehensive audio processing capabilities including:
        - Transcription: Convert speech to text using OpenAI Whisper
        - Metadata extraction: Get detailed audio file information
        - Trimming: Cut audio files to specific time ranges
        - Format conversion: Convert between different audio formats
        - Analysis: Analyze audio characteristics and quality

        Args:
            file_path: Path to the audio file to process
            operation: Type of operation to perform
            output_format: Format for transcription output
            start_time: Start time for trimming operations
            duration: Duration for trimming operations
            target_format: Target format for conversion operations
            sample_rate: Target sample rate for conversion
            channels: Number of channels for conversion

        Returns:
            Dictionary containing processed results, metadata, and output file paths
        """
        try:
            # Handle FieldInfo objects
            if isinstance(file_path, FieldInfo):
                file_path = file_path.default
            if isinstance(operation, FieldInfo):
                operation = operation.default
            if isinstance(output_format, FieldInfo):
                output_format = output_format.default
            if isinstance(start_time, FieldInfo):
                start_time = start_time.default
            if isinstance(duration, FieldInfo):
                duration = duration.default
            if isinstance(target_format, FieldInfo):
                target_format = target_format.default
            if isinstance(sample_rate, FieldInfo):
                sample_rate = sample_rate.default
            if isinstance(channels, FieldInfo):
                channels = channels.default

            start_time_processing = time.time()

            # Validate input file
            file_path: Path = self._validate_file_path(file_path)

            # Get original metadata
            original_metadata = self._get_audio_metadata(file_path)
            file_stats = file_path.stat()

            # Perform operation based on parameter
            if operation == "transcribe":
                return self._handle_transcription(file_path, original_metadata, file_stats, output_format, start_time_processing)
            
            elif operation == "metadata":
                return self._handle_metadata_extraction(file_path, original_metadata, file_stats, start_time_processing)
            
            elif operation == "trim":
                return self._handle_audio_trimming(file_path, original_metadata, file_stats, start_time, duration, start_time_processing)
            
            elif operation == "convert":
                return self._handle_format_conversion(file_path, original_metadata, file_stats, target_format, sample_rate, channels, start_time_processing)
            
            elif operation == "analyze":
                return self._handle_audio_analysis(file_path, original_metadata, file_stats, start_time_processing)
    
            else:
                return {"error": f"Unsupported operation: {operation}", 
                        "metadata": AudioMetadata(
                            operation=operation,
                            error_type="unsupported_operation",
                        ).model_dump(),
                        "success": False
                        }

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Audio processing error: {tb_str}")
            return {
                "error": f"Audio processing failed: {str(e)}",
                "traceback": tb_str,
                "metadata": AudioMetadata(
                    operation=operation,
                    error_type="processing_error",
                ).model_dump(),
                "success": False
            }

    def _handle_transcription(self, file_path: Path, original_metadata: dict, file_stats: Any, output_format: str, start_time: float) -> dict:
        """Handle audio transcription operation."""

        try:
            # Prepare audio for transcription
            prepared_audio = self._prepare_audio_for_transcription(file_path)

            # Perform transcription
            transcription_result = self._transcribe_with_whisper(prepared_audio)

            processing_time = time.time() - start_time
            word_count = len(transcription_result["text"].split()) if transcription_result["text"] else 0

            # Prepare the metadata
            audio_metadata = AudioMetadata(
                file_name=file_path.name,
                file_size=file_stats.st_size,
                file_type=file_path.suffix.lower(),
                absolute_path=str(file_path.absolute()),
                duration=original_metadata.get("duration"),
                sample_rate=original_metadata.get("sample_rate"),
                channels=original_metadata.get("channels"),
                bitrate=original_metadata.get("bitrate"),
                codec=original_metadata.get("codec"),
                processing_time=processing_time,
                output_files=[str(prepared_audio)],
                transcription=transcription_result["text"],
                word_count=word_count,
                output_format=f"transcription_{output_format}",
            )
            # Format output based on requested format
            if output_format == "text":
                result_message = transcription_result["text"]
            elif output_format == "detailed":
                result_message = (
                    f"Transcription Results for {file_path.name}:\n\n"
                    f"**Text:** {transcription_result['text']}\n\n"
                    f"**Word Count:** {word_count}\n"
                    f"**Duration:** {original_metadata.get('duration', 0):.2f} seconds\n"
                    f"**Processing Time:** {processing_time:.2f} seconds"
                )
            else:
                result_message = transcription_result["text"]

            # Clean up temporary file
            try:
                prepared_audio.unlink()
            except Exception:
                pass

            return {"result": result_message, "success": True, "metadata": audio_metadata.model_dump()}
        
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Audio transcription error: {tb_str}")
            return {
                "error": f"Audio transcription failed: {e}",
                "traceback": tb_str,
                "metadata": audio_metadata.model_dump(),
                "success": False
            }

    def _handle_metadata_extraction(self, file_path: Path, original_metadata: dict, file_stats: Any, start_time: float) -> dict:
        """Handle metadata extraction operation."""
        try:
            print(f"Extracting metadata from: {file_path.name}")

            processing_time = time.time() - start_time
            # Prepare the metadata
            audio_metadata = AudioMetadata(
                file_name=file_path.name,
                file_size=file_stats.st_size,
                file_type=file_path.suffix.lower(),
                absolute_path=str(file_path.absolute()),
                duration=original_metadata.get("duration"),
                sample_rate=original_metadata.get("sample_rate"),
                channels=original_metadata.get("channels"),
                bitrate=original_metadata.get("bitrate"),
                codec=original_metadata.get("codec"),
                processing_time=processing_time,
                output_files=[],
                output_format="metadata",
            )
            # Format metadata for output with safe formatting
            duration_str = f"{original_metadata.get('duration', 0):.2f}" if original_metadata.get('duration') is not None else "Unknown"
            sample_rate_str = str(original_metadata.get('sample_rate', 'Unknown'))
            channels_str = str(original_metadata.get('channels', 'Unknown'))
            bitrate_str = str(original_metadata.get('bitrate', 'Unknown'))
            codec_str = str(original_metadata.get('codec', 'Unknown'))
            
            result_message = (
                f"Audio Metadata for {file_path.name}:\n"
                f"Duration: {duration_str} seconds\n"
                f"Sample Rate: {sample_rate_str} Hz\n"
                f"Channels: {channels_str}\n"
                f"Bitrate: {bitrate_str} kbps\n"
                f"Codec: {codec_str}\n"
                f"File Size: {file_stats.st_size / 1024 / 1024:.2f} MB\n"
                f"Format: {file_path.suffix.upper()}"
            )

            return {"result": result_message, "success": True, "metadata": audio_metadata.model_dump()}
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Audio metadata extraction error: {tb_str}")
            return {
                "error": f"Audio metadata extraction failed: {e}",
                "traceback": tb_str,
                "metadata": audio_metadata.model_dump(),
                "success": False
            }

    def _handle_audio_trimming(self, file_path: Path, original_metadata: dict, file_stats: Any, start_time_param: float | None, duration: float | None, start_time: float) -> dict:
        """Handle audio trimming operation."""
        try:
            if start_time_param is None:
                raise ValueError("start_time is required for trimming operation")
            if duration is None:
                raise ValueError("duration is required for trimming operation")

            print(f"Trimming audio: {file_path.name}")

            # Validate time parameters
            if start_time_param < 0:
                raise ValueError("Start time cannot be negative")
            if duration <= 0:
                raise ValueError("Duration must be positive")
            if original_metadata.get("duration") and start_time_param >= original_metadata["duration"]:
                raise ValueError("Start time exceeds audio duration")

            # Trim audio
            output_path = self._trim_audio(file_path, start_time_param, duration)
            trimmed_metadata = self._get_audio_metadata(output_path)
            processing_time = time.time() - start_time
            audio_metadata = AudioMetadata(
                file_name=file_path.name,
                file_size=file_stats.st_size,
                file_type=file_path.suffix.lower(),
                absolute_path=str(file_path.absolute()),
                duration=trimmed_metadata.get("duration"),
                sample_rate=trimmed_metadata.get("sample_rate"),
                channels=trimmed_metadata.get("channels"),
                bitrate=trimmed_metadata.get("bitrate"),
                codec=trimmed_metadata.get("codec"),
                processing_time=processing_time,
                output_files=[str(output_path)],
                output_format="trimmed_audio",
            )
            end_time = start_time_param + duration
            result_message = (
                f"Successfully trimmed {file_path.name}\n"
                f"Original duration: {original_metadata.get('duration', 0):.2f} seconds\n"
                f"Trimmed segment: {start_time_param:.2f}s - {end_time:.2f}s\n"
                f"New duration: {trimmed_metadata.get('duration', 0):.2f} seconds\n"
                f"Output file: {output_path.name}"
            )

            return {"result": result_message, "success": True, "metadata": audio_metadata.model_dump()}
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Audio trimming error: {tb_str}")
            return {
                "error": f"Audio trimming failed: {e}",
                "traceback": tb_str,
                "metadata": audio_metadata.model_dump(),
                "success": False
            }

    def _handle_format_conversion(self, file_path: Path, original_metadata: dict, file_stats: Any, target_format: str | None, sample_rate: int | None, channels: int | None, start_time: float) -> dict:
        """Handle format conversion operation."""
        try:
            if target_format is None:
                raise ValueError("target_format is required for conversion operation")

            print(f"Converting audio: {file_path.name} to {target_format}")

            # Convert audio
            output_path = self._convert_audio_format(file_path, target_format, sample_rate, channels)
            converted_metadata = self._get_audio_metadata(output_path)
            processing_time = time.time() - start_time
            audio_metadata = AudioMetadata(
                file_name=file_path.name,
                file_size=file_stats.st_size,
                file_type=file_path.suffix.lower(),
                absolute_path=str(file_path.absolute()),
                duration=converted_metadata.get("duration"),
                sample_rate=converted_metadata.get("sample_rate"),
                channels=converted_metadata.get("channels"),
                bitrate=converted_metadata.get("bitrate"),
                codec=converted_metadata.get("codec"),
                processing_time=processing_time,
                output_files=[str(output_path)],
                output_format=f"converted_{target_format}",
            )
            result_message = (
                f"Successfully converted {file_path.name} to {target_format.upper()}\n"
                f"Original format: {file_path.suffix.upper()}\n"
                f"New format: {target_format.upper()}\n"
                f"Duration: {converted_metadata.get('duration', 0):.2f} seconds\n"
                f"Sample Rate: {converted_metadata.get('sample_rate', 'Unknown')} Hz\n"
                f"Channels: {converted_metadata.get('channels', 'Unknown')}\n"
                f"Output file: {output_path.name}"
            )

            return {"result": result_message, "success": True, "metadata": audio_metadata.model_dump()}
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Audio format conversion error: {tb_str}")
            return {
                "error": f"Audio format conversion failed: {e}",
                "traceback": tb_str,
                "metadata": audio_metadata.model_dump(),
                "success": False
            }
        
    def _handle_audio_analysis(self, file_path: Path, original_metadata: dict, file_stats: Any, start_time: float) -> dict:
        """Handle audio analysis operation."""
        try:
            print(f"Analyzing audio: {file_path.name}")

            # Analyze audio
            analysis_result = self._analyze_audio(file_path)
            processing_time = time.time() - start_time
            audio_metadata = AudioMetadata(
                        file_name=file_path.name,
                        file_size=file_stats.st_size,
                        file_type=file_path.suffix.lower(),
                        absolute_path=str(file_path.absolute()),
                        duration=analysis_result.get("duration"),
                        sample_rate=analysis_result.get("sample_rate"),
                        channels=analysis_result.get("channels"),
                        bitrate=analysis_result.get("bitrate"),
                        codec=analysis_result.get("codec"),
                        processing_time=processing_time,
                        output_files=[],
                        output_format="analysis",
                    )
            result_message = (
                f"Audio Analysis for {file_path.name}:\n"
                f"Duration: {analysis_result.get('duration', 0):.2f} seconds ({analysis_result.get('duration_minutes', 0):.2f} minutes)\n"
                f"Sample Rate: {analysis_result.get('sample_rate', 'Unknown')} Hz\n"
                f"Channels: {analysis_result.get('channels', 'Unknown')}\n"
                f"Bitrate: {analysis_result.get('bitrate', 'Unknown')} kbps\n"
                f"Codec: {analysis_result.get('codec', 'Unknown')}\n"
                f"File Size: {analysis_result.get('file_size_mb', 0):.2f} MB\n"
                f"Quality Score: {analysis_result.get('quality_score', 0):.2f}/1.0\n"
                f"Format Efficiency: {analysis_result.get('format_efficiency', 'Unknown')}\n"
                f"Format: {file_path.suffix.upper()}"
            )

            return {"result": result_message, "success": True, "metadata": audio_metadata.model_dump()}
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Audio analysis error: {tb_str}")
            return {
                "error": f"Audio analysis failed: {e}",
                "traceback": tb_str,
                "metadata": audio_metadata.model_dump(),
                "success": False
            }
    def test(self, tool_test: str="audio_processing", file_location: str="audio_processing", result_parameter: str="result", search_type: str="search_pattern"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)

# Example usage and entry point
if __name__ == "__main__":
    # Initialize and run the audio processing service
    try:
        service = Audio_Processing_Tool()
        service.embed_tool()
        service.test(tool_test="audio_processing", file_location='audio_processing', result_parameter='result', search_type='search_pattern')
        
    except Exception as e:
        print(f"An error occurred: {e}: {traceback.format_exc()}")
