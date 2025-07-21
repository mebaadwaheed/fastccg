from typing import List, Union

from openai import AsyncOpenAI, RateLimitError, APIStatusError, APIConnectionError

from fastccg.embedding.base import EmbeddingBase
from fastccg.errors import QuotaExceeded, ModelUnavailable, APIRequestFailed


class _OpenAIEmbedding(EmbeddingBase):
    """Base class for OpenAI embedding models."""

    provider = "openai"

    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key=api_key, model_name=model_name)
        self.client = AsyncOpenAI(api_key=api_key)

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Creates embeddings for a list of texts using the OpenAI API."""
        if isinstance(texts, str):
            texts = [texts]

        try:
            response = await self.client.embeddings.create(
                input=texts, model=self.model_name
            )
            return [item.embedding for item in response.data]
        except RateLimitError:
            raise QuotaExceeded()
        except APIStatusError as e:
            if e.status_code == 404:
                raise ModelUnavailable(f"The model `{self.model_name}` is unavailable.")
            raise APIRequestFailed(f"OpenAI status error: {e}")
        except APIConnectionError as e:
            raise APIRequestFailed(f"OpenAI connection error: {e}")
        except Exception as e:
            raise APIRequestFailed(f"Unexpected OpenAI embedding error: {e}")


class text_embedding_3_small(_OpenAIEmbedding):
    """OpenAI's highly efficient `text-embedding-3-small` model."""

    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, model_name="text-embedding-3-small")
