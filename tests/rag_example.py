import asyncio
import os
from rich import print

import fastccg
from fastccg.embedding import MockEmbedding, GeminiEmbedding
from fastccg.models.mock import MockModel
from fastccg.models.gemini import gemini_pro_1_5
from fastccg.vector_store.in_memory import InMemoryVectorStore


async def main():
    """Demonstrates a full, simple RAG pipeline using fastccg."""

    # --- 1. Setup: Initialize Models and Vector Store ---
    print("--- 1. Initializing models and vector store... ---")

    # --- Mock Example (for testing without API keys) ---
    api = fastccg.add_mock_key()
    embedding_model = fastccg.init_embedding(MockEmbedding, api_key=api)
    llm = fastccg.init_model(MockModel, api_key=api)
    print("[yellow]Running with Mock Models...[/]")

    # --- Production Example (comment out the mock section and uncomment this) ---
    # gemini_api_key = os.getenv("GEMINI_API_KEY")
    # if not gemini_api_key:
    #     print("[bold red]Error:[/] GEMINI_API_KEY environment variable not set.")
    #     return
    #
    # fastccg.add_gemini_key(gemini_api_key)
    # embedding_model = fastccg.init_embedding(GeminiEmbedding)
    # llm = fastccg.init_model(gemini_pro_1_5)
    # print("[green]Running with Gemini Models...[/]")

    vector_store = InMemoryVectorStore()
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
        print(f'  - Added "{doc_texts[i][:30]}..." to vector store.')
    print("Indexing complete.\n")

    # --- 3. Retrieval: Perform a similarity search ---
    query = "What is the main city in France?"
    print(f"--- 3. Searching for documents related to: '{query}' ---")

    query_vector = (await embedding_model.embed(query))[0]

    search_results = vector_store.similarity_search(query_vector, top_k=1)
    
    if not search_results:
        print("[bold red]No relevant documents found.[/]")
        return

    retrieved_doc = search_results[0]
    retrieved_text = retrieved_doc[2]['text']
    print(f"  - Most relevant document found: '{retrieved_text}' (Score: {retrieved_doc[1]:.2f})\n")

    # --- 4. Generation: Augment the prompt and ask the LLM ---
    print("--- 4. Generating a response with retrieved context... ---")

    augmented_prompt = f"""
    Context: {retrieved_text}
    
    Question: {query}
    
    Answer the question based on the provided context.
    """

    print(f"  - Augmented Prompt:\n{augmented_prompt}")

    final_response = await llm.ask_async(augmented_prompt)

    print(f"\n[bold green]Final RAG Answer:[/] {final_response.content}")


if __name__ == "__main__":
    asyncio.run(main())
