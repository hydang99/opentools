from typing import Any

def create_llm_engine(model_string: str, use_cache: bool = False, is_multimodal: bool = True, **kwargs) -> Any:
    """
    Factory function to create appropriate LLM engine instance.
    """
    if any(x in model_string for x in ["gpt", "o1", "o3", "o4"]):
        from .openai import ChatOpenAI
        return ChatOpenAI(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)

    elif "gemini" in model_string:
        from .gemini import ChatGemini
        return ChatGemini(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)

    
    else:
        raise ValueError(f"Engine {model_string} not supported. If you are using Together models, please ensure have the prefix 'together-' in the model string. If you are using VLLM models, please ensure have the prefix 'vllm-' in the model string. For other custom engines, you can edit the factory.py file and add its interface file to add support for your engine. Your pull request will be warmly welcomed!")
    