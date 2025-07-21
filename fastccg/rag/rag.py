import warnings
from typing import List, Optional

from rich import print as rich_print

from ..core.model_base import ModelBase, ModelResponse
from ..embedding.base import EmbeddingBase
from ..vector_store.base import VectorStoreBase
from .prompts import get_prompt_template


class RAGModel:
    """
    A high-level model for performing Retrieval-Augmented Generation (RAG).

    This class orchestrates the process of embedding a query, retrieving
    relevant documents from a vector store, augmenting a prompt with the
    retrieved context, and generating a response from a language model.
    """

    def __init__(
        self,
        llm: ModelBase,
        embedder: EmbeddingBase,
        store: VectorStoreBase,
        template: str = "auto",
        top_k: int = 3,
        strict_mode: bool = False,
        trace: bool = False,
    ):
        """
        Initializes the RAGModel.

        Args:
            llm: The language model to use for generation.
            embedder: The embedding model to use for creating vectors.
            store: The vector store to use for document retrieval.
            template: The name of the prompt template to use, or 'auto' to select based on the LLM.
            top_k: The number of top documents to retrieve for context.
            strict_mode: If True, warns when no documents are found.
            trace: If True, prints debugging information during the RAG process.
        """
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.strict_mode = strict_mode
        self.trace = trace

        if template == "auto":
            model_name = llm.model_name.lower()
            if "claude" in model_name:
                template = "qa_block"
            else:  # Default for GPT and others
                template = "context_question"
            
            if self.trace:
                rich_print(f"[bold cyan][RAG TRACE][/] Auto-selected prompt template: '[yellow]{template}[/]' for model '[yellow]{llm.model_name}[/]'" )

        self.prompt_template_str = get_prompt_template(template)

    async def ask_async(self, question: str) -> ModelResponse:
        """
        Asks a question using the RAG pipeline.

        Args:
            question: The question to ask.

        Returns:
            A ModelResponse object containing the generated answer.
        """
        if self.trace:
            rich_print(f"[bold cyan][RAG TRACE][/] Asking question: '[yellow]{question}[/]'")
            rich_print(f"[bold cyan][RAG TRACE][/] Using top_k: [yellow]{self.top_k}[/]")

        # 1. Embed the query
        query_vector = (await self.embedder.embed(question))[0]

        # 2. Retrieve relevant documents
        search_results = self.store.similarity_search(query_vector, top_k=self.top_k)
        if self.trace:
            rich_print("[bold cyan][RAG TRACE][/] Retrieved documents:")
            rich_print(search_results)

        # 3. Augment the prompt
        context = "\n\n".join([result[2].get("text", "") for result in search_results])
        
        if not context.strip():
            if self.strict_mode:
                warnings.warn("No documents found for the query. The RAG model is proceeding without context.")
            context = "No relevant context found."

        augmented_prompt = self.prompt_template_str.format(
            context=context,
            question=question
        )

        if self.trace:
            rich_print("[bold cyan][RAG TRACE][/] Augmented prompt:")
            rich_print(f"[grey50]{augmented_prompt}[/]")

        # 4. Generate the final response
        response = await self.llm.ask_async(augmented_prompt)
        return response

    def save(self, filepath: str, pretty_print: bool = False) -> None:
        """
        Saves the underlying vector store to a file.

        Args:
            filepath: The path to the file where the store will be saved.
            pretty_print: If True, saves the store in a human-readable format.
        """
        if self.trace:
            rich_print(f"[bold cyan][RAG TRACE][/] Saving vector store to '[yellow]{filepath}[/]'")
        self.store.save(filepath, pretty_print=pretty_print)

    def load(self, filepath: str) -> None:
        """
        Loads the underlying vector store from a file.

        Args:
            filepath: The path to the file from which to load the store.
        """
        if self.trace:
            rich_print(f"[bold cyan][RAG TRACE][/] Loading vector store from '[yellow]{filepath}[/]'")
        self.store.load(filepath)
