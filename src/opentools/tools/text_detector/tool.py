import os, sys, re, requests, logging, traceback
from paddleocr import PaddleOCR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
class Text_Detector_Tool(BaseTool):
    """Text_Detector_Tool
    ---------------------
    Purpose:
        A tool that extracts all visible printed text from supported image formats and returns it in natural reading order. Automatically detects and translates Chinese text to English. Supports English and Chinese text only. Do NOT use for handwriting. This tool is advanced by utilizing high-end deep learning models to extract text with superior accuracy and performance compared to traditional OCR methods.

    Core Capabilities:
        - Extracts all visible printed text from supported image formats
        - Automatically detects and translates Chinese text to English
        - Supports English and Chinese text only
        - Do NOT use for handwriting
        - Uses high-end deep learning models for superior accuracy and performance

    Intended Use:
        Use this tool when you need to extract text from images, including printed text, handwritten text, and Chinese text.

    Limitations:
        - Do NOT use for handwriting
        - Extremely low-resolution images may not be detected
        - General image Q&A may not be supported

    """
    # Default args for `opentools test Text_Detector_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "text_detector",
        "file_location": "text_detector",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }

    def __init__(self):
        super().__init__(
            type = 'function',
            name="Text_Detector_Tool",
            description="""Extracts all visible printed text from supported image formats and returns it in natural reading order.Automatically detects and translates Chinese text to English. Supports English and Chinese text only. Do NOT use for handwriting. This tool is advanced by utilizing high-end deep learning models to extract text with superior accuracy and performance compared to traditional OCR methods. SYNONYMS: OCR, extract text from image,  read text in photo, recognize text, scan text, translate Chinese label. EXAMPLES: 'Read the text in this image',  'Translate the Chinese text on this sign to English', 'Extract printed text from this photo'.""",
            parameters={
                "type": "object", 
                "properties": {
                    "image_path": {
                        "type": "string", 
                        "description": "Absolute or relative file path to a PNG or JPG image containing printed text. High-resolution images produce more accurate results."
                    },
                },
                "required": ["image_path"],
                "additionalProperties": False, 
            },
            strict=True,
            category="image",
            tags=["text_extraction"],
            limitation= "Do NOT use for handwriting, extremely low-resolution images, or general image Q&A",
            agent_type="Visual-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(image_path='path/to/image')",
                "description": "Extract the text from given image path"
            }   
        )
        # Initialize PaddleOCR
        self.ocr = PaddleOCR(
          use_mp=False,
          use_angle_cls=False,
          show_log=False,
        )
        # Disable logging
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        logging.getLogger("paddleocr").setLevel(logging.ERROR)
        logging.getLogger("paddlex").setLevel(logging.ERROR)
        for name in ("ppocr", "paddleocr", "paddlex"):
            log = logging.getLogger(name)
            log.setLevel(logging.CRITICAL)
            log.propagate = False
            for handler in log.handlers:
                handler.setLevel(logging.CRITICAL)

    def is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        return bool(chinese_pattern.search(text))

    def translate_chinese(self, text: str) -> str:
        """ Using Google Translate API to translate Chinese text to English. """
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

    def run(self, image_path: str):
        try:
            # Get the absolute path of the image
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.path.dirname(__file__), image_path)
            if not os.path.exists(image_path):
                return {"message": f"File not found: {image_path}", "success": False}
            # Start OCR detection
            result = self.ocr.ocr(img=image_path)

            if isinstance(result, list) and result and not isinstance(result[0], dict):
                raw_lines = result[0]                         
            elif isinstance(result, list) and result and isinstance(result[0], dict):
                d = result[0]                               
                raw_lines = [                               
                    (poly, (txt, score))
                    for poly, txt, score in zip(
                        d["rec_polys"], d["rec_texts"], d["rec_scores"]
                    )
                ]
            # Sort the lines
            lines = sorted(raw_lines, key=lambda r: (r[0][0][1], r[0][0][0]))
            # Translate the Chinese text to English
            out = []
            for poly, (text, score) in lines:
                if score is not None and score < 0.0:
                    continue                   
                if len(text.strip()) < 2:
                    continue

                if self.is_chinese(text):
                    out.append(self.translate_chinese(text))
                out.append(text)
            # Return the result
            return {"result": ", ".join(out), "success": True}
        except Exception as e:
            return {"error": f"Error: {e}", "success": False, "traceback": traceback.format_exc()}

    def test(self, tool_test: str="text_detector", file_location: str="text_detector", result_parameter: str="result", search_type: str="similarity_eval"):
        return super().test(tool_test=tool_test, file_location=file_location, result_parameter=result_parameter, search_type=search_type)

if __name__ == "__main__":
    try:
        tool = Text_Detector_Tool()
        tool.test(tool_test="text_detector", file_location='text_detector', result_parameter='result', search_type='similarity_eval')
    except Exception as e:
        print(f"Error: {e}")
