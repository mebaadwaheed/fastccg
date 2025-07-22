import asyncio
from typing import AsyncGenerator

import openai
from openai import RateLimitError, NotFoundError, APIConnectionError

from fastccg.core.model_base import ModelBase
from fastccg.types.response import ModelResponse
from fastccg.types.prompt import ModelPrompt
from fastccg.errors import QuotaExceeded, ModelUnavailable, APIRequestFailed


class _OpenAIModel(ModelBase):
    """Base class for OpenAI models."""

    provider = "openai"

    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key=api_key, model_name=model_name)
        self.client = openai.AsyncOpenAI(api_key=api_key)

    def _build_params(self) -> dict:
        messages = []
        if self._sys_prompt:
            messages.append({"role": "system", "content": self._sys_prompt.content})
        for p in self.memory.history:
            messages.append({"role": p.role, "content": p.content})

        params = {"model": self.model_name, "messages": messages}
        if self._temperature is not None:
            params["temperature"] = self._temperature
        if self._max_tokens is not None:
            params["max_tokens"] = self._max_tokens
        return params

    def _ask(self, prompt: str) -> ModelResponse:
        return asyncio.run(self._ask_async(prompt))

    async def _ask_async(self, prompt: str) -> ModelResponse:
        try:
            response = await self.client.chat.completions.create(**self._build_params())
            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
            return ModelResponse(
                content=content,
                tokens_used=tokens_used,
                provider=self.provider,
                raw=response,
            )
        except RateLimitError:
            raise QuotaExceeded()
        except NotFoundError:
            raise ModelUnavailable(f"The model `{self.model_name}` was not found or is unavailable.")
        except APIConnectionError as e:
            raise APIRequestFailed(f"Connection error: {str(e)}")
        except Exception as e:
            raise APIRequestFailed(f"Unexpected error: {str(e)}")

    async def _ask_stream(self, prompt: str) -> AsyncGenerator[ModelResponse, None]:
        try:
            stream = await self.client.chat.completions.create(
                **self._build_params(), stream=True
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                yield ModelResponse(content=content, provider=self.provider, raw=chunk)
        except RateLimitError:
            raise QuotaExceeded()
        except NotFoundError:
            raise ModelUnavailable(f"The model `{self.model_name}` was not found or is unavailable.")
        except APIConnectionError as e:
            raise APIRequestFailed(f"Connection error: {str(e)}")
        except Exception as e:
            raise APIRequestFailed(f"Unexpected error: {str(e)}")

class gpt_4o(_OpenAIModel):
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, model_name="gpt-4o")


class gpt_3_5_turbo(_OpenAIModel):
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, model_name="gpt-3.5-turbo")