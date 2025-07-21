from typing import List, Union, Optional
import google.generativeai as genai
from .base import EmbeddingBase


class GeminiEmbedding(EmbeddingBase):
    """Google Gemini embedding model."""
    provider = "gemini"
    model = "text-embedding-004"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key, model_name=self.model)
        genai.configure(api_key=self.api_key)

    async def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Asynchronously embeds a string or list of strings."""
        if isinstance(text, str):
            text = [text]
        
        try:
            result = await genai.embed_content_async(
                model=self.model,
                content=text,
                task_type="retrieval_document",
                title="fastccg embedding"
            )
            return result['embedding']
        except Exception as e:
            # Handle potential API errors gracefully
            print(f"An error occurred with the Gemini API: {e}")
            # Return a list of empty lists to match the expected output format
            return [[] for _ in text]
