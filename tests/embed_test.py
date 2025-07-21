import pytest
import fastccg
from fastccg.embedding.mock import MockEmbedding

# 1. Add the mock key. This sets up the mock provider.
api = fastccg.add_mock_key()

@pytest.fixture
def mock_embedding_model() -> MockEmbedding:
    """Fixture to provide a fresh mock embedding model for each test."""
    return fastccg.init_embedding(MockEmbedding, api_key=api)

@pytest.mark.asyncio
async def test_mock_embedding(mock_embedding_model: MockEmbedding):
    """Tests that the mock embedding model works as expected."""
    texts = ["hello", "world"]
    
    # Get embeddings
    embeddings = await mock_embedding_model.embed(texts)
    
    # 1. Check that we got the right number of embeddings
    assert len(embeddings) == 2
    
    # 2. Check that each embedding has the correct size
    assert all(len(e) == mock_embedding_model.embedding_size for e in embeddings)
    
    # 3. Check for determinism: the same input should always yield the same output
    embedding_hello_1 = await mock_embedding_model.embed("hello")
    embedding_hello_2 = await mock_embedding_model.embed("hello")
    assert embedding_hello_1 == embedding_hello_2

    print(f"\\nSuccessfully generated {len(embeddings)} mock embeddings.")
    print(f"Embedding size: {len(embeddings[0])}")
