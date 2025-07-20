import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Callable, List, Optional, Type

from fastccg.types.prompt import ModelPrompt
from fastccg.types.response import ModelResponse


class ModelBase(ABC):
    """Abstract base class for all models."""

    provider: str = "unknown"

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.history: List[ModelPrompt] = []
        self._sys_prompt: Optional[ModelPrompt] = None
        self._reply_filter: Optional[Callable[[str], str]] = None
        self._temperature: Optional[float] = None
        self._max_tokens: Optional[int] = None

    # --- Abstract Methods for Subclasses --- #

    @abstractmethod
    def _ask(self, prompt: str) -> ModelResponse:
        """Abstract method to send a prompt to the model."""
        pass

    @abstractmethod
    async def _ask_async(self, prompt: str) -> ModelResponse:
        """Abstract async method to send a prompt to the model."""
        pass

    @abstractmethod
    async def _ask_stream(self, prompt: str) -> AsyncGenerator[ModelResponse, None]:
        """Abstract method for streaming responses."""
        yield  # This makes it a generator

    # --- Public-Facing API --- #

    def append_response(self, msg: str) -> "ModelBase":
        self.history.append(ModelPrompt(role="assistant", content=msg))
        return self

    def ask(self, prompt: str) -> ModelResponse:
        """Send a message and get a blocking response."""
        return asyncio.run(self.ask_async(prompt))

    async def ask_async(self, prompt: str) -> ModelResponse:
        """Send a message and get an async response."""
        self.append_prompt(prompt)
        response = await self._ask_async(prompt)
        if self._reply_filter:
            response.content = self._reply_filter(response.content)
        self.append_response(response.content)  # ✅ new
        return response

    async def ask_stream(
        self, prompt: str
    ) -> AsyncGenerator[ModelResponse, None]:
        """Stream the model's response chunk by chunk."""
        self.append_prompt(prompt)
        full_response = ""
        
        async for response in self._ask_stream(prompt):
            if self._reply_filter:
                response.content = self._reply_filter(response.content)
            full_response += response.content  # ✅ accumulate chunks
            yield response

        self.append_response(full_response)  # ✅ save full response


    # --- Configuration Methods --- #

    def sys_prompt(self, msg: str) -> "ModelBase":
        """Set the system-level behavior prompt."""
        self._sys_prompt = ModelPrompt(role="system", content=msg)
        return self

    def reply_filter(self, fn: Callable[[str], str]) -> "ModelBase":
        """Add a post-processing function for outputs."""
        self._reply_filter = fn
        return self

    def temperature(self, val: float) -> "ModelBase":
        """Set response creativity (temperature)."""
        self._temperature = val
        return self

    def max_tokens(self, n: int) -> "ModelBase":
        """Limit max response tokens."""
        self._max_tokens = n
        return self

    # --- History and State Management --- #

    def append_prompt(self, msg: str) -> "ModelBase":
        """Add a user prompt without triggering a reply."""
        self.history.append(ModelPrompt(role="user", content=msg))
        return self

    def get_history(self) -> List[ModelPrompt]:
        """Return a copy of the conversation history."""
        return self.history.copy()

    def reset(self) -> "ModelBase":
        """Reset conversation history and all configurations."""
        self.history = []
        self._sys_prompt = None
        self._reply_filter = None
        self._temperature = None
        self._max_tokens = None
        return self

    def model_info(self) -> dict:
        """Return model name, provider, and config."""
        return {
            "model": self.model_name,
            "provider": self.provider,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    def save(self, path: str) -> None:
        """Save the model's state to a JSON file."""
        state = {
            "model_name": self.model_name,
            "provider": self.provider,
            "history": [p.to_dict() for p in self.history],
            "sys_prompt": self._sys_prompt.to_dict() if self._sys_prompt else None,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls: Type["ModelBase"], path: str, api_key: str) -> "ModelBase":
        """Load a model's state from a JSON file."""
        with open(path, "r") as f:
            state = json.load(f)

        # Recreate the instance
        instance = cls(api_key=api_key, model_name=state["model_name"])

        # Restore state
        instance.history = [
            ModelPrompt(**p) for p in state.get("history", [])
        ]
        if state.get("sys_prompt"):
            instance._sys_prompt = ModelPrompt(**state["sys_prompt"])
        instance._temperature = state.get("temperature")
        instance._max_tokens = state.get("max_tokens")

        return instance
    
    @classmethod
    def matches(cls, provider: str, model_name: str) -> bool:
        """Override in each subclass to support dynamic loading."""
        return False