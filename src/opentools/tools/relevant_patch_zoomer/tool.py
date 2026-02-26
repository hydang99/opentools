# Relevant Patch Zoomer Tool
# Analyzes images and identifies relevant patches to zoom
# source code: https://github.com/octotools/octotools/blob/main/octotools/tools/relevant_patch_zoomer/tool.py
import os, sys, re, json, traceback, cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
from pydantic import BaseModel

class PatchZoomerResponse(BaseModel):
    analysis: str
    patch: list[str]

class Relevant_Patch_Zoomer_Tool(BaseTool):
    """Relevant_Patch_Zoomer_Tool
    ---------------------
    Purpose:
        A tool that analyzes an image, divides it into 5 regions (4 quarters + center), and identifies the most relevant patches based on a question. The returned patches are zoomed in by a factor of 2.

    Core Capabilities:
        - Image analysis with LLM
        - Divides images into 5 regions (4 quarters + center)
        - Identifies relevant patches for answering questions
        - Zooms patches by configurable factor
        - Saves zoomed patches as separate images

    Intended Use:
        Use this tool when you need to analyze an image, divide it into 5 regions (4 quarters + center), and identify the most relevant patches based on a question. The returned patches are zoomed in by a factor of 2.

    Limitations:
        - Requires a valid OpenAI API key and internet connectivity
        - May not handle complex image analysis or patch identification
    """
    require_llm_engine = True

    def __init__(self, model_string="gpt-4o", llm_engine=None):
        super().__init__(
            type='function',
            name="Relevant_Patch_Zoomer_Tool",
            description="""A tool that analyzes an image, divides it into 5 regions (4 quarters + center), and identifies the most relevant patches based on a question. The returned patches are zoomed in by a factor of 2. CAPABILITIES: Image analysis with LLM, divides images into 5 regions (4 quarters + center), identifies relevant patches for answering questions, zooms patches by configurable factor, saves zoomed patches as separate images. SYNONYMS: patch zoomer, image region analyzer, relevant region extractor, image patch tool, region zoom tool, image area zoomer. EXAMPLES: 'Find the relevant region showing the car color', 'Identify the patch containing text', 'Zoom in on the most relevant area for this question'.""",
            parameters={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "The path to the image file to analyze."
                    },
                    "question": {
                        "type": "string",
                        "description": "The question about the image content to guide patch selection."
                    },
                    "zoom_factor": {
                        "type": "integer",
                        "description": "The factor by which to zoom the selected patches (default: 2).",
                    }
                },
                "required": ["image", "question"],
                "additionalProperties": False,
            },
            strict=False,
            category="image",
            tags=["image_analysis", "patch_extraction", "image_zooming", "region_detection", "multimodal_ai", "image_processing"],
            limitation="Requires LLM engine with multimodal capabilities, may not always select optimal patches, zoom factor affects image quality, sensitive to image resolution, requires clear question to guide patch selection.",
            agent_type="Visual-Agent",
            demo_commands={
                "command": 'execution = tool.run(image="path/to/image.jpg", question="What is the color of the car?")',
                "description": "Analyze image and return relevant zoomed patches that show the car's color."
            },
            require_llm_engine=True,
            llm_engine=llm_engine,
            model_string=model_string,
            is_multimodal=True,
        )
        self.matching_dict = {
            "A": "top-left",
            "B": "top-right",
            "C": "bottom-left",
            "D": "bottom-right",
            "E": "center"
        }
        
    def _save_patch(self, image_path, patch, save_path, zoom_factor=2):
        """Extract and save a specific patch from the image with 10% margins."""
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        quarter_h = height // 2
        quarter_w = width // 2
        
        margin_h = int(quarter_h * 0.1)
        margin_w = int(quarter_w * 0.1)
        
        patch_coords = {
            'A': ((max(0, 0 - margin_w), max(0, 0 - margin_h)),
                  (min(width, quarter_w + margin_w), min(height, quarter_h + margin_h))),
            'B': ((max(0, quarter_w - margin_w), max(0, 0 - margin_h)),
                  (min(width, width + margin_w), min(height, quarter_h + margin_h))),
            'C': ((max(0, 0 - margin_w), max(0, quarter_h - margin_h)),
                  (min(width, quarter_w + margin_w), min(height, height + margin_h))),
            'D': ((max(0, quarter_w - margin_w), max(0, quarter_h - margin_h)),
                  (min(width, width + margin_w), min(height, height + margin_h))),
            'E': ((max(0, quarter_w//2 - margin_w), max(0, quarter_h//2 - margin_h)),
                  (min(width, quarter_w//2 + quarter_w + margin_w), 
                   min(height, quarter_h//2 + quarter_h + margin_h)))
        }
        
        (x1, y1), (x2, y2) = patch_coords[patch]
        patch_img = img[y1:y2, x1:x2]
        
        zoomed_patch = cv2.resize(patch_img, 
                                (patch_img.shape[1] * zoom_factor, 
                                 patch_img.shape[0] * zoom_factor), 
                                interpolation=cv2.INTER_LINEAR)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, zoomed_patch)
        return save_path

    def run(self, image, question, zoom_factor=2):
        """
        Main execution method for the tool.
        
        Parameters:
            image (str): The path to the image file.
            question (str): The question about the image content.
            zoom_factor (int): The factor by which to zoom the selected patches.
        
        Returns:
            dict: A dictionary with detection results and success status.
        """
        try:
            if not self.llm_engine:
                return {"error": "LLM engine not initialized. Please provide a valid model_string.", "success": False}
            
            # Prepare the prompt
            prompt = f"""
Analyze this image to identify the most relevant region(s) for answering the question:

Question: {question}

The image is divided into 5 regions:
- (A) Top-left quarter
- (B) Top-right quarter
- (C) Bottom-left quarter
- (D) Bottom-right quarter
- (E) Center region (1/4 size, overlapping middle section)

Instructions:
1. First describe what you see in each of the five regions.
2. Then select the most relevant region(s) to answer the question.
3. Choose only the minimum necessary regions - avoid selecting redundant areas that show the same content. For example, if one patch contains the entire object(s), do not select another patch that only shows a part of the same object(s).

Response format:
<analysis>: Describe the image and five patches first. Then analyze the question and select the most relevant patch or list of patches.
<patch>: List of letters (A-E)
"""

            
            # Get response from LLM
            response = self.llm_engine.generate(prompt, image=image, response_format=PatchZoomerResponse)
            
            # Handle response - it might be a string or a Pydantic model
            if isinstance(response, str):
                # If response is a string, try to parse it as JSON
                try:
                    # Try to parse as JSON first
                    response_dict = json.loads(response)
                    response = PatchZoomerResponse(**response_dict)
                except (json.JSONDecodeError, TypeError):
                    # If JSON parsing fails, try to extract from text format
                    # Look for pattern like "<analysis>: ... <patch>: ..."
                    analysis_match = re.search(r'<analysis>:(.*?)<patch>:', response, re.DOTALL)
                    patch_match = re.search(r'<patch>:(.*?)$', response, re.DOTALL)
                    
                    if analysis_match and patch_match:
                        analysis = analysis_match.group(1).strip()
                        patch_str = patch_match.group(1).strip()
                        # Extract patch letters (A-E)
                        patches = re.findall(r'[A-E]', patch_str.upper())
                        if not patches:
                            # If no letters found, try to parse as list
                            patches = [p.strip().strip('[]"\'') for p in patch_str.split(',') if p.strip()]
                        response = PatchZoomerResponse(analysis=analysis, patch=patches)
                    else:
                        raise ValueError(f"Could not parse response: {response}")
            
            # Ensure response is a PatchZoomerResponse instance
            if not isinstance(response, PatchZoomerResponse):
                raise ValueError(f"Expected PatchZoomerResponse, got {type(response)}")
            
            # Save patches
            image_name = os.path.splitext(os.path.basename(image))[0]
            
            # Update the return structure
            patch_info = []
            for patch in response.patch:
                patch_name = self.matching_dict[patch]
                save_path = os.path.join(self.output_dir, 
                                       f"{image_name}_{patch_name}_zoomed_{zoom_factor}x.png")
                saved_path = self._save_patch(image, patch, save_path, zoom_factor)
                save_path = os.path.abspath(saved_path)
                patch_info.append({
                    "path": save_path,
                    "description": f"The {self.matching_dict[patch]} region of the image: {image}."
                })
            
            return {
                "result": {
                    "analysis": response.analysis,
                    "patches": patch_info
                },
                "success": True
            }
            
        except Exception as e:
            print(f"Error in patch zooming: {e}")
            print(traceback.format_exc())
            return {"error": f"Error in patch zooming: {e}", "success": False, "error_type": "patch_zooming_failed", "traceback": traceback.format_exc()}

    def execute(self, image, question, zoom_factor=2):
        """Legacy method name for backward compatibility. Use run() instead."""
        return self.run(image, question, zoom_factor)

    def get_metadata(self):
        return super().get_metadata()

if __name__ == "__main__":
    tool = Relevant_Patch_Zoomer_Tool()
    result = tool.run(image="/home/daoqm/opentools/src/opentools/Benchmark/hallusion-vd/data/images/ocr/9_0.png", question="What is the color of the car?")
    print(result)