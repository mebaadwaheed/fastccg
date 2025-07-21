"""Embedding models for the fastccg library."""

from .base import EmbeddingBase
from .openai import text_embedding_3_small
from .mock import MockEmbedding
from .google import GeminiEmbedding

__all__ = ["EmbeddingBase", "text_embedding_3_small", "MockEmbedding", "GeminiEmbedding"]
