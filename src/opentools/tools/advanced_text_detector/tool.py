# source code: https://github.com/octotools/octotools/blob/main/octotools/tools/text_detector/tool.py
"""
Compared to the source code, we improved the translation function to be more robust and accurate,
and reorganized the text detection routine to ensure output is returned in proper reading order (left-to-right, top-to-bottom for supported scripts).
This enhances both usability and fidelity in multi-language, multi-region OCR tasks.
"""


import os, time, sys, re , requests, warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
warnings.filterwarnings("ignore")

class Advanced_Text_Detector_Tool(BaseTool):
    """
    Advanced_Text_Detector_Tool: High-Accuracy Local OCR with Multilingual and Auto-Translation Support

    Purpose:
        Provides fast, privacy-preserving Optical Character Recognition (OCR) for images, enabling the extraction of all visible printed text content. Designed for robust recognition in diverse visual scenarios such as photos, scanned documents, complex layouts (multi-column, tabular, formulas), and small font sizes.

    Core Capabilities:
        - Accurately detects printed text in various languages, including (but not limited to) English, Simplified/Traditional Chinese,etc.
        - Automatically translates detected Chinese text into English, streamlining mixed-language workflows.
        - Delivers high accuracy even with challenging fonts, dense layouts, or technical imagery.
        - Supports batch and multi-language extraction by specifying multiple target languages.
        - Handles both CPU and GPU processing seamlessly, with built-in CUDA memory management and retrying for large or high-resolution images.
        - Completely local execution: no uploads, no calling cloud APIs—ideal for privacy-sensitive or offline scenarios.

    Intended Use:
        Employ this tool any time high-precision extraction of printed (not handwritten) text is needed from photos, scans, or screenshots. Useful for digitizing signage, forms, academic papers, technical diagrams, product labels, and images requiring multilingual understanding or on-the-fly translation of Chinese into English.

    Limitations:
        - Not suitable for handwriting recognition or images with extremely poor resolution.
        - Output is a list of detected text blocks, with post-processing for Chinese-to-English translation automatically applied.
    """
    # Default args for `opentools test Advanced_Text_Detector_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "text_detector",
        "file_location": "advanced_text_detector",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }

    def __init__(self):
        super().__init__(
            type = 'function',
            name="Advanced_Text_Detector_Tool",
            description="""Extracts all visible printed text from supported image formats and returns it as a list of detected text blocks. Supports multiple languages including English, Chinese, French, etc. Do NOT use for handwriting. SYNONYMS: OCR, extract text from image, read text in photo, recognize text, copy text from picture, scan text, translate Chinese label. EXAMPLES: 'Read the text in this image',  'Translate the Chinese text on this sign to English', 'Extract printed text from this photo'.""",
            parameters={
                "type": "object", 
                "properties": {
                    "image_path": {
                        "type": "string", 
                        "description": "Absolute or relative file path to a PNG or JPG image containing printed text. High-resolution images produce more accurate results."
                    },
                    "languages": {
                        "type": "array", 
                        "description": "A list of language codes for the OCR model. Each code should be a string corresponding to a supported language by EasyOCR (e.g., 'en' for English, 'ch_sim' for Simplified Chinese, 'fr' for French, etc.). Refer to EasyOCR documentation for the full list of supported language codes.",
                        "items": {
                            "type": "string",
                            "description": "Language code such as 'en', 'fr', 'ch_sim', etc."
                        },
                        "minItems": 1
                    },
                },
                "required": ["image_path", "languages"],
                "additionalProperties": False,
            },
            strict=True,
            category="image",
            tags=["text_extraction"],
            limitation="Do NOT use for handwriting, extremely low-resolution images, or general image Q&A",
            agent_type="Visual-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(image_path='path/to/image', languages=['en'])",
                "description": "Detect the text in the image"
            }
        )
        self.output="list - A list of detected text blocks, with Chinese automatically translated to English."

    def build_tool(self, languages=None):
        """
        Builds and returns the EasyOCR reader model.

        Parameters:
            languages (list): A list of language codes for the OCR model.

        Returns:
            easyocr.Reader: An initialized EasyOCR Reader object.
        """
        languages = languages or ["en"]  # Default to English if no languages provided
        try:
            import easyocr
            reader = easyocr.Reader(languages)
            return reader
        except ImportError:
            raise ImportError("Please install the EasyOCR package using 'pip install easyocr'.")
        except Exception as e:
            print(f"Error building the OCR tool: {e}")
            return None
    
    def detect_text(self, image, languages=['en'], max_retries=10, retry_delay=5, clear_cuda_cache=False, **kwargs):
        """
        detect_texts the OCR tool to detect text in the provided image.

        Parameters:
            image (str): The path to the image file.
            languages (list): A list of language codes for the OCR model.
            max_retries (int): Maximum number of retry attempts.
            retry_delay (int): Delay in seconds between retry attempts.
            clear_cuda_cache (bool): Whether to clear CUDA cache on out-of-memory errors.
            **kwargs: Additional keyword arguments for the OCR reader.

        Returns:
            list: A list of detected text blocks.
        """
        for attempt in range(max_retries):
            try:
                reader = self.build_tool(languages)
                if reader is None:
                    raise ValueError("Failed to build the OCR tool.")
                
                result = reader.readtext(image, **kwargs)
                try:
                    # detail = 1: Convert numpy types to standard Python types
                    cleaned_result = " ".join([
                        (self.translate_chinese(item[1]) if self.is_chinese(item[1]) else item[1])
                        for item in result
                    ])
                    return {"result": cleaned_result, "success": True}
                except Exception as e:
                    return {"error": f"Error detecting text: {e}", "success": False, "error_type": "text_detection_failed", "traceback": traceback.format_exc()}

            except RuntimeError as e:
                import torch
                if "CUDA out of memory" in str(e):
                    print(f"CUDA out of memory error on attempt {attempt + 1}.")
                    if clear_cuda_cache:
                        print("Clearing CUDA cache and retrying...")
                        torch.cuda.empty_cache()
                    else:
                        print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"Runtime error: {e}: {traceback.format_exc()}")
                    break
            except Exception as e:
                print(f"Error detecting text: {e}: {traceback.format_exc()}")
                break
        
        print(f"Failed to detect text after {max_retries} attempts.")
        return {"error": f"Failed to detect text after {max_retries} attempts.", "success": False, "error_type": "text_detection_failed", "traceback": traceback.format_exc()}
    
    def run(self, image_path: str, languages=['en']):
        try:
            return self.detect_text(image=image_path, languages=languages)
        except Exception as e:
            return {"error": f"Error detecting text: {e}", "success": False, "error_type": "text_detection_failed", "traceback": traceback.format_exc()}

    def is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        return bool(chinese_pattern.search(text))

    def translate_chinese(self, text: str) -> str:
        """Translate Chinese text to English using multiple fallback methods."""
        try:
            try:
                url = "https://translate.googleapis.com/translate_a/single"
                params = {
                    'client': 'gtx',
                    'sl': 'zh',
                    'tl': 'en',
                    'dt': 't',
                    'q': text
                }
                response = requests.get(url, params=params, timeout=3)
                if response.status_code == 200:
                    result = response.json()
                    if result and len(result) > 0 and len(result[0]) > 0:
                        translated = result[0][0][0]
                        return f"{text} → {translated}"
            except Exception as e:
                print(f"Google Translate API error: {e}")
            return f"{text} → [Translation unavailable]"
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def test(self, tool_test: str="text_detector", file_location: str="advanced_text_detector", result_parameter: str="result", search_type: str="similarity_eval"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)

if __name__ == "__main__":
    try:
        tool = Advanced_Text_Detector_Tool()
        tool.embed_tool()
        tool.test(tool_test="text_detector", file_location="text_detector", result_parameter="result",search_type= "similarity_eval" )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
