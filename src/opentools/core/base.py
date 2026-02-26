# opentools/tools/base.py
"""
Base tool class for the OpenTools framework.
This module provides the foundational BaseTool class that all tools in the OpenTools
framework should inherit from. It provides common functionality for tool metadata,
execution, and integration with the framework.
"""
import json, os , dotenv, traceback
from typing import Any, Dict, List, Optional
from datetime import datetime
from .config import config
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .factory import create_llm_engine
dotenv.load_dotenv()

class BaseTool:
    """
    A base class for building tool classes that perform specific tasks.
    
    This class provides a standardized interface for tools in the OpenTools framework,
    including metadata management, execution handling, and integration capabilities.
    """
    
    require_llm_engine: bool = False  # Default is False, tools that need LLM should set this to True

    def __init__(
        self, 
        name: str, 
        description: str,
        category: str,
        tags: List[str],
        parameters: dict, 
        agent_type: str,
        demo_commands: dict,
        limitation: str = None,
        type: str = 'function',
        strict: bool = True,
        accuracy: dict = None,
        model_string: Optional[str] = "gpt-4o-mini",
        required_api_keys: Optional[List[str]] = None,
        is_multimodal: bool = False,
        llm_engine = None,
        require_llm_engine: bool = False,
    ):

        """
        Initialize a BaseTool instance with metadata and configuration.

        Parameters:
            name (str): The name of the tool.
            description (str): A description of the tool's purpose and functionality.
            category (str): The category under which this tool is grouped.
            tags (List[str]): A list of tags describing the tool's features or use cases.
            parameters (dict): The input parameter schema for the tool.
            agent_type (str): The type of agent this tool is intended for (e.g., "image", "text").
            demo_commands (dict): Example commands demonstrating tool usage.
            type (str, optional): The type of tool, default is 'function'.
            strict (bool, optional): Whether to enforce strict parameter validation. Default is True.
            limitation (str, optional): Any limitations of the tool.
            accuracy (dict, optional): The accuracy score or metric for the tool, if available.
            model_string (str, optional): Model identifier string, used if the tool requires an LLM engine.
            required_api_keys (List[str], optional): List of required API key names (e.g., ['WEATHER_API_KEY']).
            is_multimodal (bool, optional): Whether the tool is multimodal.
            llm_engine (LLMEngine, optional): The LLM engine to use for the tool.
            require_llm_engine (bool, optional): Whether the tool requires an LLM engine.
        This constructor sets up the tool's metadata and configuration for use within the OpenTools framework.
        """
        self.type = type
        self.name = name
        self.description = description
        self.parameters = parameters
        self.category = category
        self.tags = tags
        self.strict = strict

        self.limitation = limitation
        self.demo_commands = demo_commands
        self.accuracy = accuracy

        self.output_dir = os.curdir
        self.agent_type = agent_type
        self.model_string = model_string
        self.required_api_keys = required_api_keys
        if require_llm_engine:
            self.llm_engine = llm_engine if llm_engine else create_llm_engine(model_string, is_multimodal=is_multimodal)
        else:
            self.llm_engine = None

    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get API key for a specific service.
        
        Args:
            key_name: The API key name (e.g., 'WEATHER_API_KEY', 'GOOGLE_API_KEY')
            
        Returns:
            The API key for the service, or None if not found
        """
        return config.get_api_key(key_name)

    def set_custom_output_dir(self, output_dir):
        """
        Set a custom output directory for the tool.

        Parameters:
            output_dir (str): The new output directory path.
        """
        self.output_dir = output_dir
    
    def require_api_key(self, key_name: str) -> str:
        """
        Get a required API key for a specific service. Raises an error if not found.
        
        Args:
            key_name: The API key name (e.g., 'WEATHER_API_KEY')
            
        Returns:
            The API key for the service
            
        Raises:
            ValueError: If the API key is not found
        """
        api_key = self.get_api_key(key_name)
        if not api_key:
            raise ValueError(f"API key '{key_name}' is required but not found. "
                           f"Please set the {key_name} environment variable "
                           f"or add it to your configuration.")
        return api_key
    
    def check_required_api_keys(self) -> None:
        """
        Check if all required API keys are available.
        
        Raises:
            ValueError: If any required API key is missing
        """
        missing_keys = []
        for key_name in self.required_api_keys:
            if not config.has_api_key(key_name):
                missing_keys.append(key_name)
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {missing_keys}. "
                           f"Please set the corresponding environment variables or add them to your configuration.")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns the metadata for the tool.

        Returns:
            A dictionary containing the tool's metadata.
        """
        metadata = {
            "type": self.type,
            "name": self.name,
            "category": self.category,
            "tags": self.tags,
            "description": self.description,
            "parameters": self.parameters,  
            "demo_commands": self.demo_commands,
            "strict": self.strict,
            "limitation": self.limitation,
            "agent_type": self.agent_type,
            "accuracy": self.accuracy,
        }
        return metadata

    def set_llm_engine(self, model_string: str) -> None:
        """
        Set the LLM engine for the tool.

        Args:
            model_string: The model string for the LLM engine.
        """
        self.model_string = model_string

    def embed_tool(self):
        """Create and store tool embeddings using the LLM engine."""
        try:
            # Create dynamic path for tool embeddings file
            embeddings_dir = os.path.join(os.path.dirname(__file__), '..', 'agents', 'embeddings')
            os.makedirs(embeddings_dir, exist_ok=True)
            embeddings_file = os.path.join(embeddings_dir, 'tool_embeddings.json')
            if os.path.exists(embeddings_file):
                with open(embeddings_file, 'r') as f:
                    data = json.load(f)
            else:
                # Initialize file if it doesn't exist
                with open(embeddings_file, 'w') as f:
                    json.dump({}, f)
            # Check if the tool is already embedded
            if self.name in data:
                print(f"Tool {self.name} already embedded")
                return
            meta = self.get_metadata()
            # Get the tool metadata
            tool = f"{meta['name']}\n{meta['description']}\n{meta['category']}\n{meta['tags']}"
            # Embed the tool
            response = self.llm_engine.embed_text(tool)
            # Save the embeddings
            self.save_embeddings(meta['name'], response.data[0].embedding, embeddings_file)
            return response.total_tokens
        except Exception as e:
            print(f"❌ Embedding tool failed: {e}")
            return None
        
    def save_embeddings(self, tool_name: str, tool_embedding: list, file_path: str):
        """Save tool embeddings to a file."""
        try:
            # Load existing data or create new dict if file doesn't exist
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
            else:
                data = {}   
            data[tool_name] = tool_embedding
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Embeddings saved to {file_path}")
        except Exception as e:
            print(f"❌ Failed to save embeddings: {e}")
            
    def find_accuracy(self,filepath):
        """Get the accuracy of the tool from a file."""
        try:
            with open(filepath, 'r', encoding="utf-8") as f:
                data = json.load(f)
            return data['Final_Accuracy']
        except:
            pass
            
    def test(self, tool_test: str, file_location: str , result_parameter: str=  None, search_type: str = None, count_token: bool = False) -> Any:
        """Run test cases for the tool."""
        try:
            # 1. Get the data from the test file
            file_test = os.path.join(os.path.dirname(__file__), '..', 'tools', 'test_file', 'data.json')
            with open(file_test, encoding='utf-8') as f:
                data = json.load(f)[tool_test]
            
            # 2. Prepare the result file with timestamp and test_results directory
            base_path = os.path.dirname(__file__)  # opentools/core/
            tool_dir = os.path.abspath(os.path.join(base_path, '..', 'tools', file_location))
            test_results_dir = os.path.join(tool_dir, 'test_results')
            os.makedirs(test_results_dir, exist_ok=True)
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_result_{timestamp}.json"
            file_result = os.path.join(test_results_dir, filename)
            
            total_token = {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0, "call_count": 0, "average_tokens_per_call": 0}
            with open(file_result, "w") as output_file:
                test_result = {}
                # 3. Add metadata
                test_result['metadata'] = {
                    "tool_name": self.name,
                    "test_timestamp": datetime.now().isoformat(),
                    "test_file": tool_test,
                    "file_location": file_location,
                    "result_parameter": result_parameter,
                    "search_type": search_type,
                    "count_token": count_token,
                    "result_file": filename,
                }
                # 4. Write the test file length
                test_result['Test-File length'] = len(data)
                run_accuracy = {'run_1': 0, 'run_2': 0, 'run_3': 0}
                # 4. Iterate through the data from the data set
                # The data set is a list of dictionaries, each dictionary contains the parameters and the answer
                for i in range (0,len(data)):
                    test = data[i]
                    # 5. Ensure the image path and file path are absolute
                    if 'image_path' in test:    
                        if not os.path.isabs(test['image_path']):
                            test['image_path'] = os.path.join(os.path.dirname(__file__), '..', 'tools', 'test_file', test['image_path'])
                    if 'image' in test:    
                        if not os.path.isabs(test['image']):
                            test['image'] = os.path.join(os.path.dirname(__file__), '..', 'tools', 'test_file', test['image'])
                    if 'file_path' in test:
                        if not os.path.isabs(test['file_path']):
                            test['file_path'] = os.path.join(os.path.dirname(__file__), '..', 'tools', 'test_file', test['file_path'])
                    # 6. Prepare the parameters to run the tool, the parameters are the keys of the dictionary except for the answer, id, and category
                    parameters = {} 
                    for key, value in test.items(): 
                        if key != 'answer' and key != 'id' and key != 'category':
                            parameters[key] = value
                    # 7. Prepare the question result
                    question_result = {"id": f"{tool_test}_{i + 1}" }
                    if search_type:
                        question_result['evaluation_metrics'] = search_type
                    else:
                        print("No search type is provided")
                    if 'query' in test:
                        question_result['query'] = test['query']
                    if 'answer' in test:
                        question_result['answer'] = test['answer']
                    for j in range(0,3):
                        run_result = {}                        
                        # 8. Run the tool and retrieve the result
                        result = self.run(**parameters)
                        print("Result: ", result)
                        if count_token:
                            total_token["total_prompt_tokens"] += result['token_usage']['total_prompt_tokens']
                            total_token["total_completion_tokens"] += result['token_usage']['total_completion_tokens']
                            total_token["total_tokens"] += result['token_usage']['total_tokens']
                            total_token["call_count"] += result['token_usage']['call_count']
                            total_token["average_tokens_per_call"] = total_token["total_tokens"] / total_token["call_count"]
                        # 9. Check if the result is successful
                        if result['success'] == False:
                            run_result['tool_call_pass'] = False
                            run_result['result'] = result
                            run_result['accuracy'] = 0
                            question_result[f'run_{j + 1}'] = run_result
                            continue
                        elif result['success'] == 'file_exists':
                            run_result['tool_call_pass'] = True
                            run_result['result'] = result
                            run_result['accuracy'] = 1
                            run_accuracy[f'run_{j + 1}'] += 1
                            question_result[f'run_{j + 1}'] = run_result
                            continue
                        else: 
                            run_result['tool_call_pass'] = True
                            run_result['result'] = result
                        # 10. Get the response from the result
                        response = result[result_parameter]
                        # 11. Check if the search type is exact match
                        if search_type == 'exact_match':
                            # 12. Check if the result is correct by comparing the result with the expected answer
                            if response == test['answer']:
                                run_accuracy[f'run_{j + 1}'] += 1
                                run_result['accuracy'] = 1
                            else:
                                run_result['accuracy'] = 0
                        # 12. Check if the search type is similarity eval
                        elif search_type == 'similarity_eval':
                            # 13. Calculate the accuracy
                            try:
                                acc = self.eval_accuracy(response, test['answer'])
                                run_accuracy[f'run_{j + 1}'] += acc
                                run_result['accuracy'] = acc
                            except Exception as e:
                                print(f"Error: {e}")
                                print(traceback.format_exc())
                                # for search tool, the answer is the query
                                acc = self.eval_accuracy(response, test['query'])
                                run_accuracy[f'run_{j + 1}'] += acc
                                run_result['accuracy'] = acc
                        # 13. Check if the search type is search pattern
                        elif search_type == 'search_pattern':
                            if test['answer'].lower() in response.lower():
                                run_accuracy[f'run_{j + 1}'] += 1
                                run_result['accuracy'] = 1
                            else:
                                run_result['accuracy'] = 0
                        # 14. Check if the search type is not supported
                        else: 
                            print(f"Search type {search_type} is not supported")
                        print(f"Finish query: {i + 1}")
                        question_result[f'run_{j + 1}'] = run_result
                    test_result[f'Q{i + 1}'] = question_result
                    print("Finish test: ", i + 1)
                # 15. Calculate the accuracy of the tool
                test_result['Final_Accuracy'] = {'run_1': run_accuracy['run_1']*100/len(data), 'run_2': run_accuracy['run_2']*100/len(data), 'run_3': run_accuracy['run_3']*100/len(data)}
                if count_token:
                    total_token["average_tokens_per_call"] /= len(data)
                    test_result['token_usage'] = total_token
                
                # Update metadata with final results
                test_result['metadata']['final_accuracy'] = test_result['Final_Accuracy']
                test_result['metadata']['total_questions'] = len(data)
                
                print(test_result['Final_Accuracy'])
                print(f"📁 Test result saved to: {file_result}")
                json.dump(test_result, output_file, indent=2, default=str)

        # 16. Handle the exception
        except Exception as e:
            print(f"❌ Failed to test the tool with this error: {e}")
            return False
        return True
    
    def eval_accuracy(self, result: str, expected_result: str):
        """Evaluate the accuracy of the tool."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        v1  = model.encode(result)
        v2 =  model.encode(expected_result)
        # Convert numpy float32 to Python float for JSON serialization
        return float(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0,0])
    
    