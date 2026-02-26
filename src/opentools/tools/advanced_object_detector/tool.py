# Grounding DINO Object Detection Tool
# https://huggingface.co/IDEA-Research/grounding-dino
# source code: https://github.com/octotools/octotools/tree/main/octotools/tools/advanced_object_detector
import os, time, sys, traceback, warnings, torch
warnings.filterwarnings("ignore")
from PIL import Image, ImageOps
from transformers import pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool

class Advanced_Object_Detector_Tool(BaseTool):
    """
    Advanced_Object_Detector_Tool
    --------------------------------------
    This tool provides advanced, local object detection powered by the Grounding DINO model from HuggingFace. Its primary purpose is to identify and isolate specific objects within images, as specified by the user, supporting robust zero-shot detection of a wide range of object labels.

    Key Capabilities:
    - Detects one or multiple user-specified objects within an image (e.g., "car", "person", "baseball")
    - Automatically crops each detected object and saves it as a new image file, with configurable empty padding around the object for better image separation
    - Supports batch detection and multi-label object identification in a single pass
    - Offers selectable model sizes ("tiny", "base", "swinb") to balance speed and detection accuracy according to your needs
    - Runs on both CPU and GPU, with automatic CUDA memory management and automatic retries for optimal performance on available hardware
    - Suitable for local, privacy-respecting object detection workflows, without the need for cloud APIs or third-party uploads

    Intended Usage:
    Use this tool when you need fast, adaptable, and local object detection for images—whether for pre-processing vision datasets, extracting objects for further analysis, or enabling intelligent agents to reason about visual content in real-time.

    Limitations:
        - May not handle complex image analysis or object detection
    """

    def __init__(self):
        super().__init__(
            type='function',
            name="Advanced_Object_Detector_Tool",
            description="""Detects objects in an image using the Grounding DINO model and saves individual object images with empty padding. CAPABILITIES: Local object detection with configurable confidence threshold and model size, automatic object cropping and padding, support for multiple labels, GPU acceleration with CUDA memory management. SYNONYMS: object detection, detect objects in image, find objects, locate objects, identify objects, zero-shot object detection. EXAMPLES: 'Detect all cars and people in this image', 'Find baseball and basket in the photo', 'Locate objects in this picture'.""",
            parameters={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "The path to the image file or URL of the image to analyze."
                    },
                    "labels": {
                        "type": "array",
                        "description": "A list of object labels to detect in the image (e.g., ['car', 'person', 'baseball']).",
                        "items": {
                            "type": "string"
                        }
                    },
                    "threshold": {
                        "type": "number",
                        "description": "The confidence threshold for detection (default: 0.35). Higher values require more confidence."
                    },
                    "model_size": {
                        "type": "string",
                        "description": "Model size to use: 'tiny', 'base', or 'swinb' (default: 'tiny'). Larger models are more accurate but slower.",
                        "enum": ["tiny", "base", "swinb"]
                    },
                    "padding": {
                        "type": "integer",
                        "description": "The number of pixels to add as empty padding around detected objects (default: 20)."
                    }
                },
                "required": ["image", "labels"],
                "additionalProperties": False,
            },
            strict=False,
            category="image",
            tags=["object_detection", "image_analysis", "computer_vision"],
            limitation="The model may not always detect objects accurately, and its performance can vary depending on the input image and the associated labels. It typically struggles with detecting small objects, objects that are uncommon, or objects with limited or specific attributes. Requires transformers and torch libraries. GPU memory may be limited for large images or models. For improved accuracy or better detection in certain situations, consider using supplementary tools or image processing techniques to provide additional information for verification.",
            agent_type="Visual-Agent",
            demo_commands={
                "command": 'execution = tool.execute(image="path/to/image.png", labels=["baseball", "basket"])',
                "description": "Detect baseball and basket in an image, save the detected objects with default empty padding, and return their paths."
            }
        )

    def preprocess_caption(self, caption):
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."

    def build_tool(self, model_size='tiny'):
        """
        Builds and returns the Grounding DINO pipeline model.
        
        Parameters:
            model_size (str): Model size to use ('tiny', 'base', or 'swinb').
        
        Returns:
            pipeline: An initialized Grounding DINO pipeline object.
        """
        if pipeline is None or torch is None:
            raise ImportError("Please install the required packages: 'pip install transformers torch'")
        
        model_name = f"IDEA-Research/grounding-dino-{model_size}"
        device = "cuda:7" if torch.cuda.is_available() else "cpu"
        try:
            pipe = pipeline(model=model_name, task="zero-shot-object-detection", device=device)
            return pipe
        except Exception as e:
            print(f"Error building the Object Detection tool: {e}")
            return None

    def save_detected_object(self, image, box, image_name, label, index, padding):
        object_image = image.crop(box)
        padded_image = ImageOps.expand(object_image, border=padding, fill='white')
        
        filename = f"{image_name}_{label}_{index}.png"
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, filename)
        
        padded_image.save(save_path)
        return save_path

    def detect_objects(self, image, labels, threshold=0.35, model_size='tiny', padding=20, max_retries=10, retry_delay=5, clear_cuda_cache=False, **kwargs):
        """
        Detects objects in the provided image using Grounding DINO.
        
        Parameters:
            image (str): The path to the image file.
            labels (list): A list of object labels to detect.
            threshold (float): The confidence threshold for detection.
            model_size (str): Model size to use ('tiny', 'base', or 'swinb').
            padding (int): The number of pixels to add as empty padding around detected objects.
            max_retries (int): Maximum number of retry attempts.
            retry_delay (int): Delay in seconds between retry attempts.
            clear_cuda_cache (bool): Whether to clear CUDA cache on out-of-memory errors.
            **kwargs: Additional keyword arguments for the pipeline.
        
        Returns:
            dict: A dictionary with 'result' (list of detected objects) and 'success' status.
        """
        for attempt in range(max_retries):
            try:
                pipe = self.build_tool(model_size)
                if pipe is None:
                    raise ValueError("Failed to build the Object Detection tool.")
                
                preprocessed_labels = [self.preprocess_caption(label) for label in labels]
                results = pipe(image, candidate_labels=preprocessed_labels, threshold=threshold, **kwargs)
                
                formatted_results = []
                original_image = Image.open(image)
                image_name = os.path.splitext(os.path.basename(image))[0]
                
                object_counts = {}

                for result in results:
                    box = tuple(result["box"].values())
                    label = result["label"]
                    score = round(result["score"], 2)
                    if label.endswith("."):
                        label = label[:-1]
                    
                    object_counts[label] = object_counts.get(label, 0) + 1
                    index = object_counts[label]
                    
                    save_path = self.save_detected_object(original_image, box, image_name, label, index, padding)
            
                    formatted_results.append({
                        "label": label,
                        "confidence score": score,
                        "box": box,
                        "saved_image_path": save_path
                    })

                return {"result": formatted_results, "success": True}
            
            except RuntimeError as e:
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
                print(f"Error detecting objects: {e}: {traceback.format_exc()}")
                break
        
        print(f"Failed to detect objects after {max_retries} attempts.")
        return {"error": f"Failed to detect objects after {max_retries} attempts.", "success": False, "error_type": "object_detection_failed", "traceback": traceback.format_exc()}
    
    def run(self, image: str, labels: list, threshold: float = 0.35, model_size: str = 'tiny', padding: int = 20):
        """
        Main execution method for the tool.
        
        Parameters:
            image (str): The path to the image file.
            labels (list): A list of object labels to detect.
            threshold (float): The confidence threshold for detection.
            model_size (str): Model size to use ('tiny', 'base', or 'swinb').
            padding (int): The number of pixels to add as empty padding around detected objects.
        
        Returns:
            dict: A dictionary with detection results and success status.
        """
        try:
            return self.detect_objects(image=image, labels=labels, threshold=threshold, model_size=model_size, padding=padding)
        except Exception as e:
            return {"error": f"Error detecting objects: {e}", "success": False, "error_type": "object_detection_failed", "traceback": traceback.format_exc()}
    
    def execute(self, image, labels, threshold=0.35, model_size='tiny', padding=20, max_retries=10, retry_delay=5, clear_cuda_cache=False):
        """Legacy method name for backward compatibility. Use run() instead."""
        return self.detect_objects(image, labels, threshold, model_size, padding, max_retries, retry_delay, clear_cuda_cache)

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    # Test command:
    tool = Advanced_Object_Detector_Tool()
    result = tool.run(image="/home/daoqm/opentools/src/opentools/Benchmark/hallusion-vd/data/images/ocr/9_0.png", labels=["baseball", "basket", "cake"])
    print(result)