from abc import ABC, abstractmethod
from typing import List, Union
import asyncio

class EmbeddingBase(ABC):
    """Abstract base class for all embedding models."""

    provider: str = "unknown"

    def __init__(self, api_key: str, model_name: str):
        """Initializes the embedding model.

        Args:
            api_key (str): The API key for the embedding provider.
            model_name (str): The name of the embedding model.
        """
        self.api_key = api_key
        self.model_name = model_name

    @abstractmethod
    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Creates embeddings for a list of texts.

        Args:
            texts (Union[str, List[str]]): A single text or a list of texts to embed.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        pass

    def embed_sync(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Synchronous wrapper for the embed method.

        Args:
            texts (Union[str, List[str]]): A single text or a list of texts to embed.

        Returns:
            List[List[float]]: A list of embeddings.
        """
        return asyncio.run(self.embed(texts))
