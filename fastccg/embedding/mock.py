import asyncio
import hashlib
from typing import List, Union, Optional

from .base import EmbeddingBase


class MockEmbedding(EmbeddingBase):
    """
    A mock embedding model that generates deterministic vectors based on the input text's hash.
    This provides more realistic mock behavior, where the same text always produces the same vector.
    """
    provider = "mock"
    model = "mock-embedding-sha256"
    dimensions = 128

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key, model_name=self.model)

    async def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Generates a deterministic vector from the SHA256 hash of the input text."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        embeddings = []
        for item in texts:
            hasher = hashlib.sha256(item.encode('utf-8'))
            hash_bytes = hasher.digest()

            # Use the hash bytes to create a deterministic list of floats
            vector = []
            for i in range(self.dimensions):
                # Each float is derived from a 4-byte chunk of the hash
                start = (i * 4) % len(hash_bytes)
                end = start + 4
                # Wrap around if we need more bytes than the hash provides
                chunk = hash_bytes[start:end] if end <= len(hash_bytes) else hash_bytes[start:] + hash_bytes[:end - len(hash_bytes)]
                
                # Convert 4 bytes to a signed integer and normalize to a float between -1 and 1
                val = int.from_bytes(chunk, 'big', signed=True)
                normalized_val = val / (2**31 - 1)  # Normalize to [-1, 1]
                vector.append(normalized_val)
            
            embeddings.append(vector)
        
        await asyncio.sleep(0.01)  # Simulate network latency
        return embeddings
