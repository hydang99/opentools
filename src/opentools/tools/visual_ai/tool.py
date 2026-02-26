import sys, os, traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from opentools.core.base import BaseTool

class Visual_AI_Tool(BaseTool):
    """Visual_AI_Tool
    ---------------------
    Purpose:
        Advanced visual AI using OpenAI's multimodal models with intelligent optimization.

    Core Capabilities:
        - Generates detailed captions, answers questions about images, and analyzes visual content
        - Automatically optimizes images for token efficiency
        - Supports custom prompts for specialized visual analysis needs

    Intended Use:
        Use this tool when you need to generate detailed captions, answer questions about images, and analyze visual content.

    Limitations:
        - Requires OpenAI API key for AI analysis
        - Processing time scales with image size and complexity
        - Memory usage increases with high resolution images
        - Large images may require significant processing time
        - Some image formats may not be supported on all platforms
        - AI analysis quality depends on image content clarity

    """
    # Default args for `opentools test Visual_AI_Tool` (uses test_file/data.json)
    DEFAULT_TEST_ARGS = {
        "tool_test": "text_detector",
        "file_location": "visual_ai",
        "result_parameter": "result",
        "search_type": "similarity_eval",
    }

    require_llm_engine = True

    def __init__(self, model_string="gpt-4o-mini", llm_engine=None):
        super().__init__(
            type='function',
            name="Visual_AI_Tool",
            description="""Advanced visual tool that can generate captions, answer questions about images, and analyze visual contents. Supports custom prompts for specialized visual analysis. This tool extract visual information with superior accuracy and natural language understanding  compared to traditional image analysis methods. CAPABILITIES: Generates captions, answers specific questions about images, analyzes visual content,      identifies objects and attributes. SYNONYMS: visual AI, image analysis, image caption, describe image, answer questions about image, visual question  answering, image understanding, visual intelligence. EXAMPLES: 'Describe this image in detail', 'Does the man have a beard?', 'Count the people in this photo',  'What emotions are shown?', 'Analyze the technical aspects of this diagram'.""",
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute or relative file path to a PNG, JPG, or JPEG image. High-resolution images produce more detailed and accurate analysis."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Custom prompt to guide the visual analysis. Default: 'Describe this image in detail.' Use specific prompts for specialized analysis (e.g., 'Does the man have a beard?', 'Count the number of people', 'What emotions are shown?', 'Describe this image in a poetic style')."
                    },
                },
                "required": ["image_path", "prompt"],
                "additionalProperties": False,
            },
            strict=True,
            category="image",
            tags=["visual_ai", "image_analysis", "image_captioning", "image_question_answering"],
            limitation="Requires OpenAI API key for AI analysis, processing time scales with image size and complexity, memory usage increases with high resolution images, large images may require significant processing time, some image formats may not be supported on all platforms, AI analysis quality depends on image content clarity",
            agent_type="Visual-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(image_path='path/to/image', prompt='Describe this image in detail.')",
                "description": "Describe the image in detail"
            },
            require_llm_engine=True,
            llm_engine=llm_engine,
        )
        self.output = "str - A detailed response based on the prompt: caption, answer to question, or analysis of the visual content."
    
    def run(self, image_path: str, prompt: str = "Extract the text from the image."):
        """
        Generate a response for the provided image using the specified prompt."""
        try:
            if (not os.path.isabs(image_path)):
                image_path = os.path.abspath(image_path)
            response = self.llm_engine.generate(image=image_path, query=prompt)
            if isinstance(response, dict):
                response = response.get('text')
            else:
                response = str(response)
            return {"result": response, "success": True, "token_usage": self.llm_engine.get_token_usage()}
        except Exception as e:
            return {"error": f"Error generating response: {str(e)}", "success": False, "error_type": "visual_ai_failed", "traceback": traceback.format_exc()}
    def test(
            self,
            tool_test: str = "text_detector",
            file_location: str = "visual_ai",
            result_parameter: str = "result",
            search_type: str = "similarity_eval",
            count_token: bool = True,
        ):
            """Run the base tool test with text_detector-specific defaults."""
            return super().test(
                tool_test=tool_test,
                file_location=file_location,
                result_parameter=result_parameter,
                search_type=search_type,
                count_token=count_token,
            )
    

if __name__ == "__main__":
    # Example usage of the Visual AI Tool
    tool = Visual_AI_Tool()
    tool.embed_tool()
    tool.test(tool_test="text_detector", file_location="visual_ai", result_parameter="result", search_type="similarity_eval", count_token=True)
