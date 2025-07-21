from typing import Dict

# Pre-defined prompt templates for different RAG use cases
PROMPT_TEMPLATES: Dict[str, str] = {
    # A standard, effective template for question-answering
    "context_question": (
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer the question based on the provided context."
    ),
    
    # A more structured block-style template
    "qa_block": (
        "--- BEGIN CONTEXT ---\n"
        "{context}\n"
        "--- END CONTEXT ---\n\n"
        "--- BEGIN QUESTION ---\n"
        "{question}\n"
        "--- END QUESTION ---\n\n"
        "Please provide a detailed answer based on the context above."
    ),
    
    # A simple template that just provides the context and question
    "plain": "{context}\n\n{question}",
}

def get_prompt_template(template_name: str) -> str:
    """
    Retrieves a prompt template by name.

    Args:
        template_name: The name of the template to retrieve.

    Returns:
        The prompt template string.

    Raises:
        ValueError: If the template name is not found.
    """
    template = PROMPT_TEMPLATES.get(template_name)
    if template is None:
        raise ValueError(f"Prompt template '{template_name}' not found. "
                         f"Available templates: {list(PROMPT_TEMPLATES.keys())}")
    return template
