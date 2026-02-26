# Ref: https://github.com/zou-group/textgrad/blob/main/textgrad/engine/gemini.py
# Ref: https://ai.google.dev/gemini-api/docs/quickstart?lang=python
# Changed to use the new google-genai package May 25, 2025
# Ref: https://ai.google.dev/gemini-api/docs/migrate
# source code: https://github.com/octotools/octotools/blob/main/octotools/engine/gemini.py

try:
    # import google.generativeai as genai
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("If you'd like to use Gemini models, please install the google-genai package by running `pip install google-genai`, and add 'GOOGLE_API_KEY' to your environment variables.")

import os
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import base64
import json
from typing import List, Union
# from .base import EngineLM, CachedEngine
import io
from PIL import Image
import hashlib
import diskcache as dc
from abc import ABC, abstractmethod

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
        
class ChatGemini(EngineLM, CachedEngine):
    SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    def __init__(
        self,
        model_string="gemini-pro",
        use_cache: bool=False,
        system_prompt=SYSTEM_PROMPT,
        is_multimodal: bool=False,
    ):
        self.use_cache = use_cache
        self.model_string = model_string
        self.system_prompt = system_prompt
        assert isinstance(self.system_prompt, str)
        self.is_multimodal = is_multimodal

        if self.use_cache:
            root = platformdirs.user_cache_dir("opentools")
            cache_path = os.path.join(root, f"cache_gemini_{model_string}.db")
            super().__init__(cache_path=cache_path)
            
        if os.getenv("GOOGLE_API_KEY") is None:
            raise ValueError("Please set the GOOGLE_API_KEY environment variable if you'd like to use Gemini models.")
        
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def model_support_function_calling(self, model_string: str):
        return model_string in ["gemini-3-pro", "gemini-2.5-pro", "gemini-2.5-flash","gemini-2.5-flash-lite","gemini-2.0-flash", "gemini-2.0-flash-lite","gemini-live-2.5-flash-preview-native-audio-09-2025", "gemini-2.0-flash-live-preview-04-09"]


    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        try:
            if isinstance(content, str):
                return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)
            
            elif isinstance(content, list):
                has_multimodal_input = any(isinstance(item, bytes) for item in content)
                if (has_multimodal_input) and (not self.is_multimodal):
                    raise NotImplementedError("Multimodal generation is only supported for Gemini Pro Vision.")
                
                return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)
        
        except Exception as e:
            print(f"Error in Gemini generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            
            # Return a proper error response instead of raising
            return {
                "error": type(e).__name__,
                "message": str(e),
                "details": getattr(e, 'args', None)
            }

    def create_tool_set(self, tools: List[str]):
        tool_set = []
        for tool in tools:
            tool_set.append(FunctionDeclaration(name=tool.get('name'), description=tool.get('description'), parameters=tool.get('parameters')))
        return tool_set

    def _generate_from_single_prompt(
        self, prompt: str, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99,tools=None, **kwargs
    ):
        try:
            sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
            tool_set = self.create_tool_set(tools)
            if self.use_cache:
                cache_or_none = self._check_cache(sys_prompt_arg + prompt)
                if cache_or_none is not None:
                    return cache_or_none

            # messages = [{'role': 'user', 'parts': [prompt]}]
            messages = [prompt]
            
            print(f"Calling Gemini API with model: {self.model_string}")
            print(f"Prompt length: {len(prompt)} characters")
            # if tools is not None:
            #     response = self.client.models.generate_content(
            #         model=self.model_string,
            #         contents=messages,
            #         config=types.GenerateContentConfig(
            #             system_instruction=sys_prompt_arg,
            #             max_output_tokens=max_tokens,
            #             temperature=temperature,
            #             top_p=top_p,
            #             candidate_count=1
            #         ),
            #         tools = [Tool(function_declaration=tool_set)]
            #     )
            #     ouptut_response = response.candidates.content.parts
            #     tool_calls = ouptut_response.get('function_calls')
            #     _tool_calls = {"name": tool_calls.get('name')}
            #     _tool_calls["arguments"] = tool_calls.get('args').get('fields')
            #     response_text = {"text": ouptut_response.get('text')}
            # else:
            response = self.client.models.generate_content(
                model=self.model_string,
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt_arg,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    candidate_count=1
                )
            )
            response_text = response.text

            if self.use_cache:
                self._save_cache(sys_prompt_arg + prompt, response_text)
            return response_text
            
        except Exception as e:
            print(f"Error in _generate_from_single_prompt: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            raise e

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                image_obj = Image.open(io.BytesIO(item))
                formatted_content.append(image_obj)
            elif isinstance(item, str):
                formatted_content.append(item)
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, **kwargs
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        if self.use_cache:
            cache_key = sys_prompt_arg + json.dumps(formatted_content)
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        response = self.client.models.generate_content(
            model=self.model_string,
            contents=formatted_content,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt_arg,
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                candidate_count=1
            )
        )
        response_text = response.text

        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text