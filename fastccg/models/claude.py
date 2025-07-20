import asyncio
from typing import AsyncGenerator

import anthropic
from anthropic import AsyncAnthropic, APIStatusError, RateLimitError, APIConnectionError

from fastccg.core.model_base import ModelBase
from fastccg.types.response import ModelResponse
from fastccg.types.prompt import ModelPrompt
from fastccg.errors import (
    QuotaExceeded,
    ModelUnavailable,
    APIRequestFailed,
)


class _ClaudeModel(ModelBase):
    """Base class for Anthropic Claude models."""

    provider = "anthropic"

    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key=api_key, model_name=model_name)
        self.client = AsyncAnthropic(api_key=api_key)

    def _build_params(self) -> dict:
        messages = [{"role": p.role, "content": p.content} for p in self.history]

        params = {
            "model": self.model_name,
            "max_tokens": self._max_tokens or 1024,
            "messages": messages,
        }
        if self._sys_prompt:
            params["system"] = self._sys_prompt.content
        if self._temperature is not None:
            params["temperature"] = self._temperature

        return params

    def _ask(self, prompt: str) -> ModelResponse:
        return asyncio.run(self._ask_async(prompt))

    async def _ask_async(self, prompt: str) -> ModelResponse:
        self.append_prompt(prompt)

        try:
            response = await self.client.messages.create(**self._build_params())

            content = response.content[0].text
            tokens_used = (
                response.usage.input_tokens + response.usage.output_tokens
                if response.usage else 0
            )

            self.history.append(ModelPrompt(role="assistant", content=content))

            return ModelResponse(
                content=content,
                tokens_used=tokens_used,
                provider=self.provider,
                raw=response,
            )

        except RateLimitError:
            raise QuotaExceeded()
        except APIStatusError as e:
            if e.status_code == 404:
                raise ModelUnavailable(f"The model `{self.model_name}` is unavailable.")
            raise APIRequestFailed(f"Anthropic status error: {e}")
        except APIConnectionError as e:
            raise APIRequestFailed(f"Anthropic connection error: {e}")
        except Exception as e:
            raise APIRequestFailed(f"Unexpected Claude error: {e}")

    async def _ask_stream(self, prompt: str) -> AsyncGenerator[ModelResponse, None]:
        self.append_prompt(prompt)
        full_response = ""

        try:
            async with self.client.messages.stream(**self._build_params()) as stream:
                async for text in stream.text_stream:
                    full_response += text
                    yield ModelResponse(
                        content=text,
                        provider=self.provider,
                        raw=text,
                    )

            self.history.append(ModelPrompt(role="assistant", content=full_response))

        except RateLimitError:
            raise QuotaExceeded()
        except APIStatusError as e:
            if e.status_code == 404:
                raise ModelUnavailable(f"The model `{self.model_name}` is unavailable.")
            raise APIRequestFailed(f"Anthropic status error: {e}")
        except APIConnectionError as e:
            raise APIRequestFailed(f"Anthropic connection error: {e}")
        except Exception as e:
            raise APIRequestFailed(f"Unexpected Claude stream error: {e}")



class claude_3_sonnet(_ClaudeModel):
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, model_name="claude-3-sonnet-20240229")