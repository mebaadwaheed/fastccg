from typing import Type
from fastccg.core.model_base import ModelBase
from fastccg.core.terminal import run_terminal

_api_keys = {}

def add_openai_key(key: str) -> str:
    """Set OpenAI API key."""
    _api_keys["openai"] = key
    return key

def add_gemini_key(key: str) -> str:
    """Set Google Gemini key."""
    _api_keys["gemini"] = key
    return key

def add_claude_key(key: str) -> str:
    """Set Claude key."""
    _api_keys["anthropic"] = key
    return key

def add_mistral_key(key: str) -> str:
    """Set Mistral key."""
    _api_keys["mistral"] = key
    return key

def init_model(model_class: Type[ModelBase], **kwargs) -> ModelBase:
    """Create a model instance with the appropriate API key."""
    provider = model_class.provider
    api_key = kwargs.pop("api_key", _api_keys.get(provider))
    
    if not api_key:
        raise ValueError(f"API key for {provider} not set. Use add_{provider}_key() or pass it as api_key.")
        
    return model_class(api_key=api_key, **kwargs)

__all__ = [
    "add_openai_key",
    "add_gemini_key",
    "add_claude_key",
    "add_mistral_key",
    "init_model",
    "run_terminal",
    "ModelBase",
]