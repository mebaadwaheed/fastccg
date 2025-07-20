from typing import Type
import json
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

from fastccg.core.model_base import ModelBase

# Ensure all models are imported so subclasses are registered
from fastccg.models.gpt import gpt_4o, gpt_3_5_turbo
from fastccg.models.gemini import gemini_flash_1_5, gemini_pro_1_5
from fastccg.models.claude import claude_3_sonnet
from fastccg.models.mistral import mistral_tiny, mistral_small, mistral_medium


def all_subclasses(cls):
    subclasses = set()
    for subclass in cls.__subclasses__():
        subclasses.add(subclass)
        subclasses.update(all_subclasses(subclass))
    return subclasses


def load_model(path: str, api_key: str) -> ModelBase:
    import json

    with open(path, "r") as f:
        state = json.load(f)

    provider = state["provider"]
    model_name = state["model_name"]

    print("Looking for model:", provider, model_name)
    print("Known subclasses:")
    for cls in all_subclasses(ModelBase):
        print("-", cls.__name__, cls.provider, getattr(cls, "model_name", "?"))

    for cls in all_subclasses(ModelBase):
        if cls.matches(provider, model_name):
            return cls.load(path, api_key)

    raise ValueError(
        f"Could not find a model class for provider '{provider}' and model '{model_name}'."
    )



__all__ = [
    "add_openai_key",
    "add_gemini_key",
    "add_claude_key",
    "add_mistral_key",
    "init_model",
    "load_model",
    "run_terminal",
    "ModelBase",
]