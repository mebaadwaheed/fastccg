import math
import json
from typing import List, Dict, Any, Tuple, Optional

from .base import VectorStoreBase


class InMemoryVectorStore(VectorStoreBase):
    """An in-memory vector store for simple RAG applications."""

    def __init__(self):
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def add(self, doc_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Adds a document and its vector to the store."""
        self._vectors[doc_id] = vector
        self._metadata[doc_id] = metadata or {}

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculates cosine similarity between two vectors."""
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        norm_vec1 = math.sqrt(sum(v**2 for v in vec1))
        norm_vec2 = math.sqrt(sum(v**2 for v in vec2))
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)

    def similarity_search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Finds the most similar documents to a query vector."""
        if not self._vectors:
            return []

        # Calculate similarities for all documents
        similarities = []
        for doc_id, vector in self._vectors.items():
            score = self._cosine_similarity(query_vector, vector)
            similarities.append((doc_id, score, self._metadata.get(doc_id, {})))

        # Sort by similarity score in descending order and return the top_k results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def save(self, filepath: str, pretty_print: bool = False) -> None:
        """Saves the vector store to a file using JSON serialization."""
        store = {doc_id: (self._vectors[doc_id], self._metadata.get(doc_id, {})) for doc_id in self._vectors}
        with open(filepath, 'w') as f:
            if pretty_print:
                json.dump(store, f, indent=4)
            else:
                json.dump(store, f)

    def load(self, filepath: str) -> None:
        """Loads the vector store from a file."""
        with open(filepath, 'r') as f:
            store = json.load(f)
            self._vectors = {doc_id: vector for doc_id, (vector, _) in store.items()}
            self._metadata = {doc_id: metadata for doc_id, (_, metadata) in store.items()}
