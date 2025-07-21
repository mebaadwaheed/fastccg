from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional


class VectorStoreBase(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, doc_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Adds a document and its vector embedding to the store.

        Args:
            doc_id: A unique identifier for the document.
            vector: The vector embedding of the document.
            metadata: Optional dictionary of metadata associated with the document.
        """
        pass

    @abstractmethod
    def similarity_search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Performs a similarity search to find the most relevant documents.

        Args:
            query_vector: The vector embedding of the query.
            top_k: The number of top results to return.

        Returns:
            A list of tuples, where each tuple contains the document ID, 
            the similarity score, and the document's metadata.
        """
        pass

    @abstractmethod
    def save(self, filepath: str, pretty_print: bool = False) -> None:
        """Saves the vector store to a file."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Loads the vector store from a file."""
        pass
