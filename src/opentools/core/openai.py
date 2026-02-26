# Reference: https://github.com/zou-group/textgrad/blob/main/textgrad/engine/openai.py
# Reference: https://github.com/octotools/octotools/blob/main/octotools/engine/openai.py
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, and add 'OPENAI_API_KEY' to your environment variables.")
import os
import openai
import traceback
import platformdirs
import tempfile
import shutil
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union, Tuple
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
class DefaultFormat(BaseModel):
    response: str
# Reference: https://github.com/zou-group/textgrad/blob/main/textgrad/engine/base.py
import hashlib
import diskcache as dc
from abc import ABC, abstractmethod
# Reference https://github.com/octotools/octotools/blob/main/octotools/engine/openai.py
class EngineLM(ABC):
    system_prompt: str = "You are a helpful, creative, and smart assistant."
    model_string: str
    @abstractmethod
    def generate(self, prompt, system_prompt=None, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


class CachedEngine:
    def __init__(self, cache_path):
        super().__init__()
        self.cache_path = cache_path
        self.cache = dc.Cache(cache_path)

    def _hash_prompt(self, prompt: str):
        return hashlib.sha256(f"{prompt}".encode()).hexdigest()

    def _check_cache(self, prompt: str):
        if prompt in self.cache:
            return self.cache[prompt]
        else:
            return None

    def _save_cache(self, prompt: str, response: str):
        self.cache[prompt] = response

    def __getstate__(self):
        # Remove the cache from the state before pickling
        state = self.__dict__.copy()
        del state['cache']
        return state

    def __setstate__(self, state):
        # Restore the cache after unpickling
        self.__dict__.update(state)
        self.cache = dc.Cache(self.cache_path)
        
def validate_structured_output_model(model_string: str):
    # Ref: https://platform.openai.com/docs/guides/structured-outputs
    Structure_Output_Models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
    return any(x in model_string for x in Structure_Output_Models)

def validate_chat_model(model_string: str):
    # Treat general GPT chat models as chat, but exclude explicit reasoning model gpt-5
    return ("gpt" in model_string) and ("gpt-5" not in model_string)

def validate_reasoning_model(model_string: str):
    # Ref: https://platform.openai.com/docs/guides/reasoning
    # Include gpt-5 as a non-pro reasoning model
    return (("gpt-5" in model_string) or any(x in model_string for x in ["o1", "o3", "o4"])) and not validate_pro_reasoning_model(model_string)

def validate_pro_reasoning_model(model_string: str):
    # Ref: https://platform.openai.com/docs/guides/reasoning
    return any(x in model_string for x in ["o1-pro", "o3-pro", "o4-pro", "gpt-5-pro"])

class ChatOpenAI(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    def __init__(
        self,
        model_string="gpt-4o-mini",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool= True,
        use_cache: bool=True,
        **kwargs):
        """
        :param model_string:
        :param system_prompt:
        :param is_multimodal:
        :param tools:
        :param tool_choice:
        """

        self.model_string = model_string
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal
        # Tool choice: auto or required
        self.tools = None
        self.tool_choice = None
        # Token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

        self.support_structured_output = validate_structured_output_model(self.model_string)
        self.is_chat_model = validate_chat_model(self.model_string)
        self.is_reasoning_model = validate_reasoning_model(self.model_string)
        self.is_pro_reasoning_model = validate_pro_reasoning_model(self.model_string)

        if self.use_cache:
            root = platformdirs.user_cache_dir("src")
            cache_path = os.path.join(root, f"cache_openai_{self.model_string}.db")
            self.image_cache_dir = os.path.join(root, "image_cache")
            os.makedirs(self.image_cache_dir, exist_ok=True)
            super().__init__(cache_path=cache_path)
        
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("Please set the OPENAI_API_KEY environment variable if you'd like to use OpenAI models.")
        
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    # Embed text for vector search (retrieval)
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def embed_text(self, text: str):
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        self._track_tokens(response)
        return response
        
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def embed_with_normalization(self, text: str):
        try:
            response = self.embed_text(text)
            import numpy as np  
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.reshape(1, -1)
        except Exception as e:
            print(f"Error embedding text with normalization: {str(e)}")
            return None

    # Generate output both text only or image with text
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self,  query, image=None, file=None, system_prompt=None, tools=None, tool_choice=None, **kwargs):
        self.set_tools(tools, tool_choice)
        try:
            if image is not None:
                return self._generate_multimodal(query, image=image, system_prompt=system_prompt, **kwargs)
            if file is not None:
                return self._generate_multimodal(query, file=file, system_prompt=system_prompt, **kwargs)
            else:
                return self._generate_text(query, system_prompt=system_prompt, **kwargs)
        except openai.LengthFinishReasonError as e:
            print(f"Token limit exceeded: {str(e)}")
            print(f"Tokens used - Completion: {e.completion.usage.completion_tokens},\n Prompt: {e.completion.usage.prompt_tokens},\n Total: {e.completion.usage.total_tokens}")
            return {
                "error": "token_limit_exceeded",
                "message": str(e),
                "details": {
                    "completion_tokens": e.completion.usage.completion_tokens,
                    "prompt_tokens": e.completion.usage.prompt_tokens,
                    "total_tokens": e.completion.usage.total_tokens
                }
            }
        except openai.RateLimitError as e:
            print(f"Rate limit error encountered: {str(e)}")
            return {
                "error": "rate_limit",
                "message": str(e),
                "details": getattr(e, 'args', None)
            }
        except Exception as e:
            print(f"Type of image: {str(type(image))}")
            print(f"Error in generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            print(f"Trace: {traceback.format_exc()}")
            return {
                "error": type(e).__name__,
                "message": str(e),
                "details": getattr(e, 'args', None),
                "traceback": traceback.format_exc()
            }

    def set_tools(self, tools: List[dict] = None, tool_choice: str="auto"):
        self.tools = tools
        self.tool_choice = tool_choice if tools is not None else None

    def extract_response(self, api_response):
        response = {}
        has_any_output = False

        for choice in api_response.output:
            has_any_output = True
            # Accept function calls with "completed" status
            if choice.type == "function_call" and choice.status == "completed":
                if "tool_calls" not in response:
                    response["tool_calls"] = []
                response["tool_calls"].append({"name": choice.name, "arguments": choice.arguments})
            # Also accept function calls with "in_progress" status if they have name and arguments
            elif choice.type == "function_call" and choice.status == "in_progress":
                try:
                    if hasattr(choice, 'name') and hasattr(choice, 'arguments') and choice.name and choice.arguments:
                        if "tool_calls" not in response:
                            response["tool_calls"] = []
                        response["tool_calls"].append({"name": choice.name, "arguments": choice.arguments})
                except Exception:
                    pass
            elif choice.type != "reasoning" and choice.status == "completed":
                try:
                    texts = [block.text for block in choice.content]
                except Exception:
                    texts = [block.output_text for block in choice.content]
                if "text" not in response:
                    response["text"] = ""
                response["text"] += "".join(texts)
        if not response:
            # Provide more informative error message
            if not has_any_output:
                raise RuntimeError("No output found in reasoning model response")
            else:
                print(f"API response: {api_response}")
                output_types = [choice.type for choice in api_response.output]
                output_statuses = [choice.status for choice in api_response.output]
                # Special-case: Responses API truncated output due to max_output_tokens.
                # Return a structured error so the caller can handle/retry instead of crashing.
                if getattr(api_response, "status", None) == "incomplete":
                    incomplete_details = getattr(api_response, "incomplete_details", None)
                    reason = getattr(incomplete_details, "reason", None) if incomplete_details is not None else None
                    if reason == "max_output_tokens":
                        output_types = [getattr(choice, "type", None) for choice in getattr(api_response, "output", [])]
                        output_statuses = [getattr(choice, "status", None) for choice in getattr(api_response, "output", [])]
                        self._track_tokens(api_response)
                        return {
                            "error": "max_output_tokens",
                            "message": "Response incomplete: max_output_tokens reached.",
                            "details": {
                                "output_types": output_types,
                                "output_statuses": output_statuses,
                            },
                        }

                # Default: raise the same informative error as before.
                raise RuntimeError(
                    f"No non-reasoning and completed output found in reasoning model. "
                    f"Output types: {output_types}, Statuses: {output_statuses}"
                )

        self._track_tokens(api_response)
        if "tool_calls" not in response:
            return response["text"]
        else:
            return response
    
    def _generate_text(
        self, prompt, system_prompt=None, temperature=0, max_tokens=16384, 
        top_p=0.99, response_format=None, reasoning_effort="medium"
    ):
        """
        Generate text output using the OpenAI API. Response format and temperature are deprecated from gpt-5. 
        """
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        # Allow runtime override to prevent truncation/incomplete Responses.
        # NOTE: This controls Responses API "max_output_tokens".
        try:
            max_tokens = int(os.getenv("OPENTOOLS_MAX_OUTPUT_TOKENS", str(max_tokens)))
        except Exception:
            pass

        if self.use_cache:
            cache_key = sys_prompt_arg + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        if response_format is not None:
            print(f"Using structured output model: {self.model_string}")
            if "gpt-5" in self.model_string:
                max_tokens = 120000
            try:
                response = self.client.responses.parse(
                    model=self.model_string,
                    input= prompt,
                    instructions=sys_prompt_arg,
                    text_format=response_format,
                    max_output_tokens=max_tokens,
                )
                self._track_tokens(response)
                # Check if output_parsed is None (shouldn't happen, but handle it)
                if response.output_parsed is None:
                    print(f"Warning: response.output_parsed is None, this should not happen")
                    return {
                        "error": "ValidationError",
                        "message": "Structured output parsing returned None",
                        "details": "response.output_parsed is None"
                    }
                return response.output_parsed
            except Exception as parse_error:
                # Re-raise to be caught by outer exception handler in generate()
                # This will return an error dict instead of None
                raise
                
        # Reasoning models first (e.g., gpt-5, o1, o3, o4)
        elif self.is_reasoning_model and not self.is_pro_reasoning_model:
            print(f"Using reasoning model: {self.model_string}")
            api_response = self.client.responses.create(
                model=self.model_string,
                input= prompt,
                instructions=sys_prompt_arg,
                reasoning={
                    "effort": reasoning_effort
                },
                tools=self.tools,
                tool_choice=self.tool_choice,
                max_output_tokens=max_tokens,
            )
            response = self.extract_response(api_response)
        # Reasoning models: Pro models using v1/completions
        elif self.is_pro_reasoning_model:
            api_response = self.client.responses.create(
                model=self.model_string,
                input= prompt,
                instructions=sys_prompt_arg,              
                reasoning={
                    "effort": "medium"
                },
                max_output_tokens=max_tokens,
            )
            response = self.extract_response(api_response)

        # Chat models given structured output format
        elif self.is_chat_model:
            api_response = self.client.responses.create(
                model=self.model_string,
                input= prompt,
                instructions=sys_prompt_arg,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens,
                tools=self.tools,
                tool_choice=self.tool_choice,
            )
            response = self.extract_response(api_response)

        if self.use_cache:
            self._save_cache(cache_key, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
    
    def _track_tokens(self, response):
        """Track token usage for either Chat Completions or Responses API."""
        u = getattr(response, "usage", None)
        if not u:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0}

        # Chat Completions schema
        if hasattr(u, "prompt_tokens") and hasattr(u, "completion_tokens"):
            pt = u.prompt_tokens or 0
            ct = u.completion_tokens or 0
            tt = u.total_tokens or (pt + ct)

        # Responses API schema
        elif hasattr(u, "input_tokens") and hasattr(u, "output_tokens"):
            pt = u.input_tokens or 0
            ct = u.output_tokens or 0
            tt = u.total_tokens or (pt + ct)

        else:
            return

        self.total_prompt_tokens += pt
        self.total_completion_tokens += ct
        self.total_tokens += tt
        self.call_count += 1   

    def get_token_usage(self):
        """Get current token usage statistics"""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "average_tokens_per_call": self.total_tokens / self.call_count if self.call_count > 0 else 0
        }
    
    def reset_token_counters(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
    
    def _format_content(self, query: str, image: str = None, file: str = None, local: str = True):
        """
        Format content for multimodal API calls.
        
        Returns:
            tuple: (formatted_content, file_id) where file_id is the uploaded file ID
                   that should be deleted after use to avoid storage costs.
        """
        formatted_content = []
        formatted_content.append({
            "type": "input_text",
            "text": query
        })
        if not local:
            if image is not None:
                formatted_content.append({
                    "type": "input_image",
                    "image_url": image,
                    "detail": "high"
                })
            elif file is not None:
                formatted_content.append({
                    "type": "input_file",
                    "file_url": file,
                })
            else:
                raise ValueError("No image or file provided in _format_content")
            return formatted_content, None
        try:
            if image is not None:
                f = self.client.files.create(
                    file=open(image, "rb"),
                    purpose="vision"
                )
                formatted_content.append({
                    "type": "input_image",
                    "file_id": f.id,
                    "detail": "high"
                })  
            elif file is not None:
                # OpenAI API is case-sensitive for file extensions - normalize to lowercase
                file_path = Path(file)
                if file_path.suffix and file_path.suffix != file_path.suffix.lower():
                    # Create temp file with lowercase extension
                    temp_suffix = file_path.suffix.lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix) as temp_file:
                        shutil.copy2(file, temp_file.name)
                        temp_file_path = temp_file.name
                    try:
                        f = self.client.files.create(
                            file=open(temp_file_path, "rb"),
                            purpose="assistants"
                        )
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                else:
                    f = self.client.files.create(
                        file=open(file, "rb"),
                        purpose="assistants"
                    )
                formatted_content.append({
                    "type": "input_file",
                    "file_id": f.id,
                })
            else:
                raise ValueError("No image or file provided in _format_content")
            return formatted_content, f.id
        except Exception as e:
            raise ValueError(f"Failed to format content in _format_content: {e}: {traceback.format_exc()}") from e

    def _format_calling (self, query: str, image: str = None, file: str = None) -> List[dict]:
        try:
            try:
                if image is not None:
                    if not str(image).lower().startswith(("http://", "https://")) and os.path.exists(image):
                        formatted_content = self._format_content(query, image=image, local=True)
                    else:
                        formatted_content = self._format_content(query, image=image, local=False)
                elif file is not None:
                    if not str(file).lower().startswith(("http://", "https://")) and os.path.exists(file):
                        formatted_content = self._format_content(query, file=file, local=True)
                    else:
                        formatted_content = self._format_content(query, file=file, local=False)
                else:
                    raise ValueError(f"No image or file provided in _format_calling: {traceback.format_exc()}") from e
            except Exception as e:
                raise ValueError(f"Failed to format content in _format_calling: {e}: {traceback.format_exc()}") from e
            return formatted_content
        except Exception as e:
            raise ValueError(f"Failed to format content in _format_calling: {e}: {traceback.format_exc()}") from e

    def _generate_multimodal(
        self, query: str, image: str = None, file: str = None, system_prompt=None, temperature=0, max_tokens=16384, top_p=0.99, response_format=None
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content, file_id = self._format_calling(query, image=image, file=file)
        
        # Check for error in file upload
        if isinstance(formatted_content, dict) and formatted_content.get("error"):
            return formatted_content
        
        if self.use_cache:
            # Create cache key from text content only (excluding file_ids for efficiency)
            text_content = [item.get("text", "") for item in formatted_content if item.get("type") == "input_text"]
            cache_key = sys_prompt_arg + "".join(text_content)
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                # Delete file even if using cache to avoid storage costs
                if file_id:
                    try:
                        self.client.files.delete(file_id)
                    except Exception as e:
                        print(f"Warning: Failed to delete file {file_id}: {e}")
                return cache_or_none

        try:
            # Use Responses API for better token efficiency with multimodal content
            if self.is_chat_model:
                api_response = self.client.responses.create(
                    model=self.model_string,
                    input=[{
                        "role": "user",
                        "content": formatted_content
                    }],
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_tokens,
                )
                response_text = self.extract_response(api_response)

            # Reasoning models: currently only supports base response
            elif self.is_reasoning_model:
                if "gpt-5" in self.model_string: 
                    api_response = self.client.responses.create(
                        model=self.model_string,
                        input=[{
                            "role": "user",
                            "content": formatted_content
                        }],
                        max_output_tokens=max_tokens,
                    )
                    response_text = self.extract_response(api_response)
                else:
                    api_response = self.client.chat.completions.create(
                        model=self.model_string,
                        messages=[{
                            "role": "user",
                            "content": formatted_content
                        }],
                        max_completion_tokens=max_tokens,
                        reasoning_effort="medium"
                    )
                    self._track_tokens(api_response)
                    # Workaround for handling length finish reason
                    if "finishreason" in api_response.choices[0] and api_response.choices[0].finishreason == "length":
                        response_text = "Token limit exceeded"
                    else:
                        response_text = api_response.choices[0].message.content

            # Reasoning models: Pro models using v1/completions
            elif self.is_pro_reasoning_model:
                api_response = self.client.responses.create(
                    model=self.model_string,
                    messages=[{
                        "role": "user",
                        "content": formatted_content
                    }],
                    reasoning={
                        "effort": "medium"
                    },
                )
                # Note: Pro reasoning models may not have standard usage info
                response_text = self.extract_response(api_response)

            if file_id:
                try:
                    self.client.files.delete(file_id)
                    print(f"Deleted uploaded file {file_id} to avoid storage costs")
                except Exception as e:
                    print(f"Warning: Failed to delete file {file_id}: {e}")
            if self.use_cache:
                self._save_cache(cache_key, response_text)
            return response_text
        except Exception as e:
            raise ValueError(f"Failed to generate multimodal response: {e}: {traceback.format_exc()}") from e