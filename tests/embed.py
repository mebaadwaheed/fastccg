import asyncio
import fastccg
from fastccg.embedding.mock import MockEmbedding

# 1. Set up the mock provider. 
# This allows us to use mock models without a real API key.
api = fastccg.add_mock_key()

# 2. Initialize the mock embedding model.
embedding_model = fastccg.init_embedding(MockEmbedding, api_key=api)

# 3. Define the texts you want to embed.
texts_to_embed = [
    "What is the capital of France?",
    "The sky is blue.",
    "Fast, minimalist, multi-model SDK for LLMs."
]

async def main():
    """The main function to run the embedding example."""
    print(f"--- Using Mock Embedding Model: {embedding_model.model_name} ---")
    
    # 4. Get the embeddings asynchronously.
    embeddings = await embedding_model.embed(texts_to_embed)
    
    # 5. Print the results.
    print(f"Successfully generated {len(embeddings)} embeddings.\n")
    for text, embedding in zip(texts_to_embed, embeddings):
        print(f'Text: "{text}"')
        # Showing only the first 5 dimensions for brevity
        print(f"  -> Embedding (first 5 dims): {embedding[:5]}")
        print(f"  -> Total dimensions: {len(embedding)}\n")

# 6. Run the main async function.
if __name__ == "__main__":
    asyncio.run(main())