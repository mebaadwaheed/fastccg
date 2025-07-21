import asyncio
from typing import AsyncGenerator

from fastccg.core.model_base import ModelBase
from fastccg.types.response import ModelResponse
from fastccg.types.prompt import ModelPrompt

class MockModel(ModelBase):
    """Mock model for testing purposes."""

    provider = "mock"

    def __init__(self, api_key: str = "mock_key", model_name: str = "mock_model"):
        super().__init__(api_key=api_key, model_name=model_name)

    def _ask(self, prompt: str) -> ModelResponse:
        return asyncio.run(self._ask_async(prompt))

    async def _ask_async(self, prompt: str) -> ModelResponse:
        content = f"This is a mock response to: {prompt}"
        return ModelResponse(
            content=content,
            tokens_used=10,
            provider=self.provider,
            raw=None,
        )

    async def _ask_stream(self, prompt: str) -> AsyncGenerator[ModelResponse, None]:
        stream_chunks = ["This ", "is ", "a ", "mock ", "streamed ", "response."]
        for chunk in stream_chunks:
            yield ModelResponse(
                content=chunk,
                provider=self.provider,
                raw=chunk,
            )
            await asyncio.sleep(0.1)
