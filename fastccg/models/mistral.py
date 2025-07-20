import asyncio
from typing import AsyncGenerator

from mistralai import Mistral
from fastccg.core.model_base import ModelBase
from fastccg.types.response import ModelResponse
from fastccg.types.prompt import ModelPrompt


class _MistralModel(ModelBase):
    """Base class for Mistral models using the new client."""

    provider = "mistral"

    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key=api_key, model_name=model_name)
        self.client = Mistral(api_key=api_key)

    def _build_messages(self) -> list:
        """Build chat messages for Mistral format."""
        messages = []
        if self._sys_prompt:
            messages.append({"role": "system", "content": self._sys_prompt.content})
        for p in self.history:
            messages.append({"role": p.role, "content": p.content})
        return messages

    def _build_params(self) -> dict:
        return {
            "model": self.model_name,
            "messages": self._build_messages(),
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    def _ask(self, prompt: str) -> ModelResponse:
        """Sync wrapper."""
        return asyncio.run(self._ask_async(prompt))

    async def _ask_async(self, prompt: str) -> ModelResponse:
        """Async call using new Mistral client."""
        self.append_prompt(prompt)
        params = self._build_params()
        response = await self.client.chat.complete_async(**params)

        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        self.history.append(ModelPrompt(role="assistant", content=content))

        return ModelResponse(
            content=content,
            tokens_used=tokens_used,
            provider=self.provider,
            raw=response,
        )

    async def _ask_stream(self, prompt: str) -> AsyncGenerator[ModelResponse, None]:
        """Streaming async chat using new client."""
        self.append_prompt(prompt)
        params = self._build_params()
        stream = await self.client.chat.stream_async(**params)

        full_response = ""
        async for chunk in stream:
            content = chunk.data.choices[0].delta.content or ""
            full_response += content
            yield ModelResponse(content=content, provider=self.provider, raw=chunk)

        self.history.append(ModelPrompt(role="assistant", content=content))


# Model subclasses
class mistral_tiny(_MistralModel):
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, model_name="mistral-tiny")


class mistral_small(_MistralModel):
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, model_name="mistral-small")


class mistral_medium(_MistralModel):
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, model_name="mistral-medium")
