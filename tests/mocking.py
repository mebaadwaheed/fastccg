import pytest
import fastccg
from fastccg import init_model
from fastccg.models.mock import MockModel

# Add the mock key. This can be done once for your test suite.
fastccg.add_mock_key()

@pytest.fixture
def mock_model() -> MockModel:
    """Fixture to provide a fresh mock model for each test."""
    # Initialize the model using the central init_model function
    return init_model(MockModel)

def test_mock_model_ask(mock_model: MockModel):
    """Tests the synchronous ask method of the mock model."""
    prompt = "Hello, mock model!"
    response = mock_model.ask(prompt)
    
    assert response.provider == "mock"
    assert "This is a mock response to: Hello, mock model!" in response.content
    print(f"\nResponse: {response.content}")


async def test_mock_model_ask_stream(mock_model: MockModel):
    """Tests the streaming functionality of the mock model."""
    prompt = "Stream a response."
    response_chunks = []
    
    async for chunk in mock_model.ask_stream(prompt):
        response_chunks.append(chunk.content)
        
    full_response = "".join(response_chunks)
    assert full_response == "This is a mock streamed response."
    print(f"\nStreamed Response: {full_response}")