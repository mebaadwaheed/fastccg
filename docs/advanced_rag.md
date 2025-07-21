# Advanced RAG Features

The `RAGModel` in FastCCG is designed for both simplicity and power. Beyond the basic setup, it offers several advanced features to enhance debugging, streamline prompt engineering, and manage your knowledge base efficiently.

This guide covers:
-   **Debug Tracing**: Get detailed, real-time insights into the RAG process.
-   **Automatic Prompt Templating**: Let the model intelligently select the best prompt structure.
-   **Saving and Loading**: Persist your indexed knowledge base to a file and load it back later.

## 1. Enabling Debug Tracing

When building a RAG pipeline, it's crucial to understand what's happening under the hood. By enabling the `trace` flag during initialization, the `RAGModel` will print detailed logs for each step of the `ask_async` process.

-   The question being asked.
-   The `top_k` value used for retrieval.
-   The documents retrieved from the vector store, along with their similarity scores.
-   The final, augmented prompt sent to the language model.

```python
from fastccg.rag import RAGModel

rag = RAGModel(
    llm=llm,
    embedder=embedding_model,
    store=vector_store,
    trace=True,  # Enable debug tracing
)
```

This is invaluable for debugging why the model is giving a certain response or ensuring the right context is being retrieved.

## 2. Automatic Prompt Templating

Different language models perform best with different prompt structures. Instead of manually crafting prompts, you can set `template="auto"`. FastCCG will automatically select an optimized prompt template based on the model you are using.

```python
from fastccg.rag import RAGModel

rag = RAGModel(
    llm=llm,
    embedder=embedding_model,
    store=vector_store,
    template="auto",  # Auto-select the best prompt template
)
```

This feature simplifies development and helps you get better performance from your chosen LLM without extra effort.

## 3. Saving and Loading the Knowledge Base

Indexing documents can be time-consuming and costly, as it often involves making API calls to an embedding model. To avoid re-indexing every time you start your application, you can save the state of your `RAGModel` (including the populated vector store) to a file.

### Saving the State

Use the [save()](cci:1://file:///d:/Ebaad/Projects/FastCCG/fastccg/vector_store/in_memory.py:45:4-52:35) method after you have indexed your documents. It saves the contents of the vector store to a compact [.fcvs](cci:7://file:///d:/Ebaad/Projects/FastCCG/my_knowledge.fcvs:0:0-0:0) file. You can also pass `pretty_print=True` to save the file in a human-readable, indented format, which is useful for debugging.

```python
# After indexing documents...
# Save in a compact format (default)
rag.save("my_knowledge.fcvs")

# Save in a human-readable format
rag.save("my_knowledge_pretty.fcvs", pretty_print=True)

### Loading the State

To restore your knowledge base, create a new `RAGModel` with an empty vector store and then call the `load()` method. This will populate the store with the data from your saved file, making it ready to answer questions immediately.

```python
from fastccg.vector_store.in_memory import InMemoryVectorStore
from fastccg.rag import RAGModel

# Create a new RAG instance with a fresh, empty store
new_vector_store = InMemoryVectorStore()
loaded_rag = RAGModel(llm=llm, store=new_vector_store)

# Load the previously saved knowledge
loaded_rag.load("my_knowledge.fcvs")
```

## Full Example: Advanced RAG Pipeline

The following script, adapted from `tests/advanced_rag_example.py`, demonstrates all these features working together.

```python
import asyncio
import os
from rich import print

import fastccg
from fastccg.embedding.mock import MockEmbedding
from fastccg.models.mock import MockModel
from fastccg.vector_store.in_memory import InMemoryVectorStore
from fastccg.rag import RAGModel

async def main():
    """Demonstrates the advanced features of the RAGModel in fastccg."""

    # --- 1. Setup ---
    print("--- 1. Initializing models and vector store... ---")
    api = fastccg.add_mock_key()
    embedding_model = fastccg.init_embedding(MockEmbedding, api_key=api)
    llm = fastccg.init_model(MockModel, api_key=api)
    vector_store = InMemoryVectorStore()
    print("[yellow]Running with Mock Models...[/]")

    # --- 2. Indexing ---
    print("\n--- 2. Indexing documents... ---")
    documents = {
        "doc1": "The sky is blue.",
        "doc2": "Photosynthesis is how plants make food.",
        "doc3": "The capital of France is Paris."
    }
    doc_vectors = await embedding_model.embed(list(documents.values()))
    for i, doc_id in enumerate(documents.keys()):
        vector_store.add(doc_id, doc_vectors[i], metadata={"text": list(documents.values())[i]})

    # --- 3. Create RAGModel with Advanced Features ---
    print("\n--- 3. Creating RAGModel with trace=True and template='auto'... ---")
    rag = RAGModel(
        llm=llm,
        embedder=embedding_model,
        store=vector_store,
        trace=True,
        template="auto",
        top_k=1
    )

    # --- 4. Ask a question and save the knowledge base ---
    print("\n--- 4. Asking a question and saving the store... ---")
    question = "What is the capital of France?"
    await rag.ask_async(question)

    save_path = "my_knowledge.fcvs"
    rag.save(save_path)
    print(f"\nKnowledge base saved to [yellow]{save_path}[/].")

    # --- 5. Load the knowledge into a new RAG instance ---
    print("\n--- 5. Loading knowledge into a new RAG model... ---")
    new_llm = fastccg.init_model(MockModel, api_key=api)
    new_vector_store = InMemoryVectorStore()
    loaded_rag = RAGModel(
        llm=new_llm,
        embedder=embedding_model,
        store=new_vector_store,
        top_k=1,
        template="auto"
    )
    loaded_rag.load(save_path)

    # --- 6. Ask the same question with the loaded model ---
    print("\n--- 6. Asking the same question with the loaded model... ---")
    loaded_response = await loaded_rag.ask_async(question)
    print(f"\n[bold green]Loaded RAG Answer:[/] {loaded_response.content}")

    os.remove(save_path)

if __name__ == "__main__":
    asyncio.run(main())
```
