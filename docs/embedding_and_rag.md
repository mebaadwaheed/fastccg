# Embedding and RAG Support in FastCCG

FastCCG includes powerful, easy-to-use, and asynchronous features for creating text embeddings and building Retrieval-Augmented Generation (RAG) pipelines. This guide provides a high-level overview of these capabilities.

For detailed guides, see:
-   [**Advanced Embedding Techniques**](./advanced_embedding.md)
-   [**Advanced RAG Features**](./advanced_rag.md)

## 1. What are Embeddings and RAG?

-   **Embeddings** are numerical representations (vectors) of text that capture its semantic meaning. Similar texts will have similar vectors.
-   **RAG** is a technique that enhances language models by retrieving relevant information from a knowledge base and providing it as context when generating a response. This makes the model's answers more accurate and up-to-date.

## 2. Command-Line Interface (CLI) Usage

The CLI provides quick access to embedding features, which is useful for testing and one-off tasks.

### Listing Available Embedding Models

```bash
fastccg list-embeddings
```

### Generating Embeddings from Text

```bash
# Using a mock model (no API key needed)
fastccg embed "Hello, world!" --model MockEmbedding

# Using a production model (requires API key in environment)
fastccg embed "This is a test." --model text_embedding_3_small
```

For more details, see the [Advanced Embedding Techniques](./advanced_embedding.md) guide.

## 3. Python API Usage

The Python API provides full control for building custom RAG pipelines. The core components are:

-   **Embedding Models**: Asynchronous models for converting text to vectors.
-   **Vector Stores**: Simple, in-memory storage for your vectors.
-   **The `RAGModel`**: A high-level class that orchestrates the entire retrieval and generation process.

### Basic RAG Pipeline Example

This example demonstrates the end-to-end process of indexing documents and asking a question.

```python
import asyncio
import fastccg
from fastccg.embedding.mock import MockEmbedding
from fastccg.models.mock import MockModel
from fastccg.vector_store.in_memory import InMemoryVectorStore
from fastccg.rag import RAGModel

async def main():
    # 1. Setup
    api = fastccg.add_mock_key()
    embedder = fastccg.init_embedding(MockEmbedding, api_key=api)
    llm = fastccg.init_model(MockModel, api_key=api)
    store = InMemoryVectorStore()

    # 2. Indexing
    documents = {"doc1": "The capital of France is Paris."}
    doc_vectors = await embedder.embed(list(documents.values()))
    for i, doc_id in enumerate(documents.keys()):
        store.add(doc_id, doc_vectors[i], metadata={"text": list(documents.values())[i]})

    # 3. Initialize and use the RAGModel
    rag = RAGModel(llm=llm, embedder=embedder, store=store)
    question = "What is the capital of France?"
    response = await rag.ask_async(question)
    print(response.content)

asyncio.run(main())
```

### Advanced Features

The `RAGModel` also supports advanced features like debug tracing, automatic prompt selection, and saving/loading the knowledge base. For a complete guide, please see the [**Advanced RAG Features**](./advanced_rag.md) documentation.
