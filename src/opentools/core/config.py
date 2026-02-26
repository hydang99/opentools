"""
Configuration management for OpenTools framework.
"""

import os
from typing import Dict, Optional
from pydantic import BaseModel, Field

# Try to import python-dotenv, but don't fail if it's not installed
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None


class OpenToolsConfig(BaseModel):
    """Configuration class for OpenTools framework."""
    
    api_keys: Dict[str, str] = Field(default_factory=dict, description="Dynamic API keys for various services")
    
    default_timeout: int = Field(default=30, description="Default timeout for API calls")
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")
    
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    enable_cache: bool = Field(default=True, description="Enable caching for API responses")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        return self.api_keys.get(key_name)
    
    def set_api_key(self, key_name: str, api_key: str) -> None:
        self.api_keys[key_name] = api_key
    
    def has_api_key(self, key_name: str) -> bool:
        return self.get_api_key(key_name) is not None
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "OpenToolsConfig":
        if env_file and DOTENV_AVAILABLE:
            load_dotenv(env_file)
        elif env_file and not DOTENV_AVAILABLE:
            raise ImportError(
                "python-dotenv is required to load .env files. "
                "Install it with: pip install python-dotenv"
            )
        elif DOTENV_AVAILABLE:
            load_dotenv()

        api_keys = {key: value for key, value in os.environ.items() if key.endswith('_API_KEY') and value}
        return cls(
            api_keys=api_keys,
            log_level=os.getenv("OPENTOOLS_LOG_LEVEL", "INFO"),
            log_file=os.getenv("OPENTOOLS_LOG_FILE"),
        )


# Global configuration instance
config = OpenToolsConfig.from_env() 