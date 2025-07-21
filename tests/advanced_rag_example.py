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

    # --- 1. Setup: Initialize Models and Vector Store ---
    print("--- 1. Initializing models and vector store... ---")
    api = fastccg.add_mock_key()
    embedding_model = fastccg.init_embedding(MockEmbedding, api_key=api)
    llm = fastccg.init_model(MockModel, api_key=api)
    vector_store = InMemoryVectorStore()
    print("[yellow]Running with Mock Models...[/]")
    print("Setup complete.\n")

    # --- 2. Indexing: Add documents to the vector store ---
    print("--- 2. Indexing documents... ---")
    documents = {
        "doc1": "The sky is blue because of Rayleigh scattering.",
        "doc2": "Photosynthesis is the process used by plants to convert light into energy.",
        "doc3": "The capital of France is Paris, known for the Eiffel Tower."
    }
    doc_texts = list(documents.values())
    doc_ids = list(documents.keys())
    doc_vectors = await embedding_model.embed(doc_texts)
    for i, doc_id in enumerate(doc_ids):
        vector_store.add(doc_id, doc_vectors[i], metadata={"text": doc_texts[i]})
    print("Indexing complete.\n")

    # --- 3. Create RAGModel with Advanced Features ---
    print("--- 3. Creating RAGModel with trace=True and template='auto'... ---")
    rag = RAGModel(
        llm=llm,
        embedder=embedding_model,
        store=vector_store,
        trace=True,  # Enable debug tracing
        template="auto",  # Auto-select the best prompt template
        top_k=1
    )

    # --- 4. Ask a question and save the knowledge base ---
    print("\n--- 4. Asking a question and saving the store... ---")
    question = "What is the main city in France?"
    final_response = await rag.ask_async(question)
    print(f"\n[bold green]RAG Answer:[/] {final_response.content}")

    # Save the populated vector store to a file
    save_path = "my_knowledge.fcvs"
    rag.save(save_path, pretty_print=True)
    print(f"\nKnowledge base saved to [yellow]{save_path}[/].")

    # --- 5. Load the knowledge base into a new RAG instance ---
    print("\n--- 5. Loading knowledge into a new RAG model... ---")
    new_llm = fastccg.init_model(MockModel, api_key=api)
    new_vector_store = InMemoryVectorStore() # Create an empty store

    # Create a new RAG model (without trace this time)
    loaded_rag = RAGModel(
        llm=new_llm,
        embedder=embedding_model,
        store=new_vector_store,
        top_k=1,             # 🔧 Add this to match original behavior
        template="auto"      # 🔧 Optional, for consistency
    )
    loaded_rag.load(save_path)    # Load the previously saved knowledge

    print("\n--- 6. Asking the same question with the loaded model... ---")
    loaded_response = await loaded_rag.ask_async(question)
    print(f"\n[bold green]Loaded RAG Answer:[/] {loaded_response.content}")

    # Clean up the created file
    os.remove(save_path)

if __name__ == "__main__":
    asyncio.run(main())
