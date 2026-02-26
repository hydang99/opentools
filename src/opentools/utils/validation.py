"""
Validation utilities for OpenTools framework.
"""

import re
from typing import Optional


def validate_api_key(api_key: str, key_name: str) -> bool:
    """Validate an API key format.
    
    Args:
        api_key: The API key to validate.
        key_name: The API key name (e.g., 'WEATHER_API_KEY', 'GOOGLE_API_KEY').
        
    Returns:
        True if the API key format is valid, False otherwise.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Remove whitespace
    api_key = api_key.strip()
    
    if not api_key:
        return False
    
    return True


def get_api_key_help(key_name: str) -> str:
    """Get help information for a specific API key.
    
    Args:
        key_name: The API key name (e.g., 'WEATHER_API_KEY').
        
    Returns:
        Help information for the API key.
    """
    help_info = {
        'WEATHER_API_KEY': "Get your API key from https://www.weatherapi.com/",
        'GOOGLE_API_KEY': "Get your API key from https://developers.google.com/custom-search/v1/introduction",
        'OPENAI_API_KEY': "Get your API key from https://platform.openai.com/api-keys",
        'HUGGINGFACE_API_KEY': "Get your token from https://huggingface.co/settings/tokens",
        'SERPAPI_API_KEY': "Get your API key from https://serpapi.com/search-api",
        'AMADEUS_ID_API_KEY': "Get your ID from https://developers.amadeus.com/",
        'AMADEUS_KEY_API_KEY': "Get your key from https://developers.amadeus.com/",
        'BAIDU_TRANSLATE_API_KEY': "Get your key from https://fanyi-api.baidu.com/",
        'BAIDU_SECRET_API_KEY': "Get your secret from https://fanyi-api.baidu.com/",
        'STEAMSHIP_API_KEY': "Get your API key from https://steamship.com/account/api",
    }
    
    return help_info.get(key_name, f"Please check the documentation for {key_name} requirements.")


def sanitize_input(text: str, max_length: Optional[int] = None) -> str:
    """Sanitize user input.
    
    Args:
        text: Input text to sanitize.
        max_length: Maximum allowed length.
        
    Returns:
        Sanitized text.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove leading/trailing whitespace
    sanitized = text.strip()
    
    # Truncate if max_length is specified
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized 