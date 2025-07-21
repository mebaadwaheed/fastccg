# Advanced Embedding Techniques

FastCCG's embedding system is built to be both flexible and robust, supporting asynchronous operations and multiple embedding providers. This guide covers advanced patterns for initializing, using, and managing embedding models in your applications.

## 1. Asynchronous Embedding

All embedding models in FastCCG are designed to be non-blocking. The `embed()` method is an `async` function that returns a list of vectors. This is crucial for high-performance applications, as it allows your program to perform other tasks while waiting for potentially slow API responses.

Always use `await` when calling the `embed` method.

```python
import asyncio
import fastccg
from fastccg.embedding import MockEmbedding

async def main():
    api_key = fastccg.add_mock_key()
    embedder = fastccg.init_embedding(MockEmbedding, api_key=api_key)

    # Embed a single string
    single_vector = (await embedder.embed("Hello world"))[0]

    # Embed a list of strings in a single call
    multiple_vectors = await embedder.embed(["First sentence.", "Second sentence."])

    print(f"Got {len(multiple_vectors)} vectors.")

asyncio.run(main())
```

## 2. Initializing Embedding Models

The `fastccg.init_embedding()` function is the standard way to create an embedding model instance. It takes the model class and an optional API key as arguments.

### Using Mock Models

For testing and development, the `MockEmbedding` model is ideal. It runs locally, requires no API key, and produces deterministic (predictable) vectors.

```python
from fastccg.embedding.mock import MockEmbedding

# The mock key is a placeholder and has no real value
api_key = fastccg.add_mock_key()
embedder = fastccg.init_embedding(MockEmbedding, api_key=api_key)
```

### Using Production Models (OpenAI, Gemini, etc.)

When using a production model, you must provide a valid API key. It is strongly recommended to load keys from environment variables rather than hardcoding them in your source code.

```python
import os
from fastccg.embedding.gemini import GeminiEmbedding

# Load the key securely from the environment
gemini_api_key = os.getenv("GEMINI_API_KEY")

if gemini_api_key:
    # Initialize the model with the key
    embedder = fastccg.init_embedding(GeminiEmbedding, api_key=gemini_api_key)
else:
    print("Error: GEMINI_API_KEY not found.")
```

## 3. API Key Management

FastCCG provides helper functions to manage API keys, but you are free to use your own system (like `python-dotenv`).

-   `fastccg.add_openai_key(key)`
-   `fastccg.add_gemini_key(key)`
-   `fastccg.add_mock_key()`

The `init_embedding` function can accept the key directly, which is the most explicit and recommended approach.

## 4. CLI for Quick Embeddings

For quick tests or one-off tasks, you can generate embeddings directly from the command line. This is useful for debugging or getting a feel for a model's output without writing any Python code.

### List Available Models

```bash
fastccg list-embeddings
```

### Generate an Embedding

```bash
# Using the mock model
fastccg embed "This is a test." --model MockEmbedding

# Using a production model (requires API key to be set in the environment)
fastccg embed "This is a test." --model text_embedding_3_small
```
