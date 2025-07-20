import asyncio
from typing import AsyncGenerator

import google.generativeai as genai
from fastccg.core.model_base import ModelBase
from fastccg.types.response import ModelResponse


class _GeminiModel(ModelBase):
    """Base class for Google Gemini models."""

    provider = "gemini"

    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key=api_key, model_name=model_name)
        genai.configure(api_key=api_key)

    def _build_config_and_prompt(self) -> tuple[genai.types.GenerationConfig, str]:
        """Helper to build the generation config and the full prompt."""
        config_params = {}
        if self._temperature is not None:
            config_params["temperature"] = self._temperature
        if self._max_tokens is not None:
            config_params["max_output_tokens"] = self._max_tokens
        generation_config = genai.types.GenerationConfig(**config_params)

        full_prompt = ""
        if self._sys_prompt:
            full_prompt += self._sys_prompt.content + "\n\n"
        for p in self.history:
            full_prompt += f"{p.role}: {p.content}\n"

        return generation_config, full_prompt

    def _ask(self, prompt: str) -> ModelResponse:
        """Sync wrapper for the async ask method."""
        return asyncio.run(self._ask_async(prompt))

    async def _ask_async(self, prompt: str) -> ModelResponse:
        """Async method to send a prompt to the model."""
        generation_config, full_prompt = self._build_config_and_prompt()
        model = genai.GenerativeModel(
            self.model_name, generation_config=generation_config
        )
        chat = model.start_chat(history=[])

        response = await chat.send_message_async(full_prompt)

        return ModelResponse(
            content=response.text,
            provider=self.provider,
            raw=response,
        )

    async def _ask_stream(self, prompt: str) -> AsyncGenerator[ModelResponse, None]:
        """Stream the model's response."""
        generation_config, full_prompt = self._build_config_and_prompt()
        model = genai.GenerativeModel(
            self.model_name, generation_config=generation_config
        )

        response_stream = await model.generate_content_async(full_prompt, stream=True)

        async for chunk in response_stream:
            yield ModelResponse(
                content=chunk.text, provider=self.provider, raw=chunk
            )


class gemini_pro_1_5(_GeminiModel):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro-latest"):
        super().__init__(api_key=api_key, model_name=model_name)

    @classmethod
    def matches(cls, provider: str, model_name: str) -> bool:
        return provider == "gemini" and model_name == "gemini-1.5-pro-latest"


class gemini_flash_1_5(_GeminiModel):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash-latest"):
        super().__init__(api_key=api_key, model_name=model_name)

    @classmethod
    def matches(cls, provider: str, model_name: str) -> bool:
        return provider == "gemini" and model_name == "gemini-1.5-flash-latest"